"""Depth-controlled N-Queens message-passing experiment for RYS.

Mirrors ``scripts/run_sat_messagepassing_experiment.py``.  For each trained depth:

- baselines (valid board rate) on val/test/OOD;
- a validity-vs-rounds curve (read out after each round);
- the validation row-state CKA connectome;
- strict RYS ΔValidity matrices (half-open windows, >=2 duplicated rounds).

RYS uses ``rys.surgery.apply_rys`` half-open windows ``[start, end)`` with
``end < n_rounds`` (a downstream round must receive the doubled stream), matching
``run_sat_theory_validation.py`` and ``predict_cka_under_rys``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

from rys.cka import cka_matrix
from rys.nqueens_data import (
    QueensDataset,
    make_queens_examples,
    make_queens_splits,
    soft_nqueens_loss,
    verify_boards_tensor,
)
from rys.nqueens_message_passing import QueensMessagePassingModel, QueensMPConfig
from rys.surgery import apply_rys
from rys.training.device_cache import cache_batches_on_device
from rys.training.modules import NqueensMPLitModule
from rys.training.rys_logging import log_rys_progress
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(45, help="Random seed."),
    depths: str = typer.Option("8,16,32", help="Comma-separated message-passing depths (rounds) to sweep."),
    n: int = typer.Option(8, help="Board size N (in-distribution)."),
    ood_n: int = typer.Option(8, help="Board size N for the OOD split."),
    n_givens: int = typer.Option(4, help="Revealed queens per instance."),
    ood_givens: int = typer.Option(
        2,
        help="Givens for the OOD split. Fewer givens at the same N = harder completion "
        "(more free rows, more reasoning), avoiding the untrained-vocabulary confound of larger N.",
    ),
    n_train: int = typer.Option(4096, help="Training examples."),
    n_val: int = typer.Option(1024, help="Validation examples."),
    n_test: int = typer.Option(1024, help="Test (and OOD) examples."),
    batch_size: int = typer.Option(128, help="Batch size."),
    epochs: int = typer.Option(20, help="Training epochs."),
    lr: float = typer.Option(5e-4, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-2, help="AdamW weight decay."),
    d_model: int = typer.Option(64, help="Model width (must be divisible by n-heads)."),
    n_heads: int = typer.Option(4, help="Attention heads per round."),
    d_mlp: int = typer.Option(128, help="MLP hidden width."),
    given_ce_weight: float = typer.Option(0.25, help="Cross-entropy weight anchoring given rows."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    skip_rys: bool = typer.Option(
        False, "--skip-rys/--no-skip-rys", help="Train and export curves/CKA without RYS matrices."
    ),
    output_dir: Path = typer.Option(Path("results/nqueens_messagepassing"), help="Run output directory."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_depths(value: str) -> list[int]:
    depths = [int(p.strip()) for p in value.split(",") if p.strip()]
    if not depths or any(d < 2 for d in depths):
        raise ValueError("Depths must contain integers >= 2.")
    return depths


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], int]:
    max_n = max(args.n, args.ood_n)
    splits = make_queens_splits(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n=args.n, n_givens=args.n_givens, seed=args.seed,
    )
    splits["ood"] = make_queens_examples(
        args.n_test, n=args.ood_n, n_givens=args.ood_givens, seed=args.seed + 3, prefix="ood",
    )
    datasets = {k: QueensDataset(v, max_n=max_n) for k, v in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, max_n


def build_model(args: argparse.Namespace, *, n_rounds: int, max_n: int) -> tuple[QueensMessagePassingModel, dict]:
    config = QueensMPConfig(
        max_n=max_n, d_model=args.d_model, n_rounds=n_rounds,
        n_heads=args.n_heads, d_mlp=args.d_mlp, pre_norm=True,
    )
    return QueensMessagePassingModel(config), config.__dict__


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def on_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "given_col": batch["given_col"].to(device),
        "is_given": batch["is_given"].to(device),
        "row_mask": batch["row_mask"].to(device),
        "col_mask": batch["col_mask"].to(device),
    }


def board_metrics(logits, row_mask, given_col, is_given, labels) -> dict[str, int]:
    cols = logits.argmax(dim=-1)
    valid = verify_boards_tensor(cols, row_mask, given_col, is_given)
    cell_correct = ((cols == labels) & row_mask).sum()
    return {
        "valid_correct": int(valid.sum().detach().cpu()),
        "cell_correct": int(cell_correct.detach().cpu()),
        "cell_total": int(row_mask.sum().detach().cpu()),
        "example_total": int(labels.shape[0]),
    }


def accumulate(counts: dict[str, int]) -> dict[str, float]:
    return {
        "valid_board_rate": counts["valid_correct"] / counts["example_total"],
        "cell_accuracy": counts["cell_correct"] / counts["cell_total"],
    }


def empty_counts() -> dict[str, int]:
    return {"valid_correct": 0, "cell_correct": 0, "cell_total": 0, "example_total": 0}


def train_epoch(model, loader, optimizer, device, *, given_ce_weight: float) -> dict[str, float]:
    model.train()
    counts = empty_counts()
    for batch in loader:
        kw = on_device(batch, device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(**kw, return_round_logits=True)
        terms = [
            soft_nqueens_loss(rl, kw["row_mask"], kw["col_mask"], given_col=kw["given_col"],
                              is_given=kw["is_given"], given_ce_weight=given_ce_weight)
            for rl in out["round_logits"]
        ]
        loss = torch.stack(terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k, v in board_metrics(out["logits"], kw["row_mask"], kw["given_col"], kw["is_given"], labels).items():
            counts[k] += v
    return accumulate(counts)


@torch.inference_mode()
def evaluate(model, loader, device, *, window: tuple[int, int] | None = None, n_repeats: int = 2) -> dict[str, float]:
    model.eval()
    counts = empty_counts()
    for batch in loader:
        kw = on_device(batch, device)
        labels = batch["labels"].to(device)
        if window is None:
            logits = model(**kw)["logits"]
        else:
            with apply_rys(model, window=window, n_repeats=n_repeats):
                logits = model(**kw)["logits"]
        for k, v in board_metrics(logits, kw["row_mask"], kw["given_col"], kw["is_given"], labels).items():
            counts[k] += v
    return accumulate(counts)


@torch.inference_mode()
def validity_vs_rounds(model, loader, device, *, n_rounds: int) -> pd.DataFrame:
    model.eval()
    per_round = [empty_counts() for _ in range(n_rounds)]
    for batch in loader:
        kw = on_device(batch, device)
        labels = batch["labels"].to(device)
        out = model(**kw, return_round_logits=True)
        for r, logits in enumerate(out["round_logits"]):
            for k, v in board_metrics(logits, kw["row_mask"], kw["given_col"], kw["is_given"], labels).items():
                per_round[r][k] += v
    rows = [{"round": r + 1, **accumulate(c)} for r, c in enumerate(per_round)]
    return pd.DataFrame(rows)


@torch.inference_mode()
def capture_row_states(model, loader, device, *, max_batches: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        kw = on_device(batch, device)
        trace = model(**kw, return_hidden_states=True)["hidden_states"]
        mask = batch["row_mask"].bool()
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        for layer_idx, hidden in enumerate(trace):
            for row_idx, pid in enumerate(prompt_ids):
                rows.append({
                    "prompt_id": pid,
                    "layer": layer_idx,
                    "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                    "strategy": "row",
                })
    return pd.DataFrame(rows)


def strict_windows(n_rounds: int, min_span: int = 2) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_rounds) for j in range(i + min_span, n_rounds)]


def delta_matrix(model, loader, device, n_rounds) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    cached = cache_batches_on_device(loader, device)
    base = evaluate(model, cached, device)["valid_board_rate"]
    mat = np.full((n_rounds, n_rounds), np.nan)
    rows = []
    for start, end in log_rys_progress(strict_windows(n_rounds), device=device, depth=n_rounds):
        rys = evaluate(model, cached, device, window=(start, end))["valid_board_rate"]
        mat[start, end] = rys - base
        rows.append({"start": start, "end": end, "baseline": base, "rys": rys, "delta_valid": rys - base})
    return pd.DataFrame(mat), base, pd.DataFrame(rows)


def save_curve(curves: dict[str, pd.DataFrame], path: Path, *, depth: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, frame in curves.items():
        ax.plot(frame["round"], frame["valid_board_rate"], marker="o", label=split)
    ax.set_title(f"Valid board rate vs rounds (L={depth})")
    ax.set_xlabel("round")
    ax.set_ylabel("valid board rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(matrix: pd.DataFrame, path: Path, *, title: str, cbar: str, cmap="RdBu_r", diverging=True) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite values for {path}.")
    if diverging:
        bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("end round (half-open)")
    ax.set_ylabel("start round")
    fig.colorbar(im, ax=ax, label=cbar)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cka_band_summary(cka: pd.DataFrame) -> dict[str, float]:
    n = cka.shape[0]
    early = cka.iloc[: n // 3, : n // 3].to_numpy()
    late = cka.iloc[2 * n // 3:, 2 * n // 3:].to_numpy()
    return {
        "cka_early_mean": float(np.nanmean(early)),
        "cka_late_mean": float(np.nanmean(late)),
        "cka_global_mean": float(np.nanmean(cka.to_numpy())),
    }


def train_one_depth(args, *, n_rounds, loaders, max_n, device, run_dir) -> dict:
    depth_dir = run_dir / f"L{n_rounds:02d}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    model, config = build_model(args, n_rounds=n_rounds, max_n=max_n)
    model.to(device)
    params = count_parameters(model)
    print(json.dumps({"depth": n_rounds, "n_params_total": params["total"], "n_params_trainable": params["trainable"]}))
    lit = NqueensMPLitModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        given_ce_weight=args.given_ce_weight,
        model_config=config,
    )
    trainer = build_trainer(
        depth_dir,
        max_epochs=args.epochs,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
        extra_log_fields={"depth": n_rounds},
    )
    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    ckpt = best_checkpoint_path(trainer)

    reloaded, _ = load_rys_model(
        ckpt,
        lambda _config: build_model(args, n_rounds=n_rounds, max_n=max_n)[0],
        map_location=device,
    )
    reloaded.to(device)

    curves = {s: validity_vs_rounds(reloaded, loaders[s], device, n_rounds=n_rounds) for s in ("val", "test", "ood")}
    for s, frame in curves.items():
        frame.to_csv(depth_dir / f"validity_vs_rounds_{s}.csv", index=False)
    save_curve(curves, depth_dir / "validity_vs_rounds.png", depth=n_rounds)

    activations = capture_row_states(reloaded, loaders["val"], device, max_batches=args.capture_batches)
    activations.to_pickle(depth_dir / "row_activations.pkl")
    cka = cka_matrix(activations, unbiased=False, device="cpu")
    cka.to_csv(depth_dir / "cka_variable_val.csv")
    save_heatmap(cka, depth_dir / "cka_variable_val.png", title=f"Row-state CKA (L={n_rounds})",
                 cbar="CKA", cmap="viridis", diverging=False)
    bands = cka_band_summary(cka)

    baselines = {s: evaluate(reloaded, loaders[s], device) for s in ("val", "test", "ood")}
    if args.skip_rys:
        summary = {"depth": n_rounds, "config": config, "n_params": params, "checkpoint": str(ckpt), "baselines": baselines,
                   "cka_bands": bands, "best_rows": [], "rys_skipped": True}
        (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    best_rows = []
    for s in ("val", "test", "ood"):
        mat, base, long = delta_matrix(reloaded, loaders[s], device, n_rounds)
        long.to_csv(depth_dir / f"delta_valid_{s}_long.csv", index=False)
        mat.to_csv(depth_dir / f"delta_valid_{s}.csv")
        save_heatmap(mat, depth_dir / f"delta_valid_{s}.png",
                     title=f"RYS ΔValidity {s} (L={n_rounds})", cbar="Δ valid board rate")
        best = long.loc[long["delta_valid"].idxmax()].to_dict()
        best["split"] = s
        best_rows.append(best)

    summary = {"depth": n_rounds, "config": config, "n_params": params, "checkpoint": str(ckpt), "baselines": baselines,
               "cka_bands": bands, "best_rows": best_rows, "rys_skipped": False}
    (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def write_report(run_dir, summaries, args) -> None:
    lines = [
        "# N-Queens Message-Passing Depth Sweep",
        "",
        "Task: complete a partial N-Queens board (givens) to a valid full board.",
        "Primary metric: valid_board_rate (verifier), not exact match to a canonical board.",
        "",
        f"- Depths (rounds): `{args.depths}`",
        f"- Board: N={args.n}, givens={args.n_givens}; OOD N={args.ood_n}, givens={args.ood_givens}",
        f"- Train/val/test/OOD: `{args.n_train}/{args.n_val}/{args.n_test}/{args.n_test}`",
        "",
        "| depth | val valid | test valid | OOD valid | CKA early | CKA late | best val | best test | best OOD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for s in summaries:
        b = s["baselines"]
        best = {r["split"]: r for r in s["best_rows"]}
        bands = s["cka_bands"]

        def _w(split: str, *, current_summary: dict = s, current_best: dict = best) -> str:
            if current_summary.get("rys_skipped"):
                return "skipped"
            r = current_best[split]
            return f"({int(r['start'])},{int(r['end'])}) {r['delta_valid']:+.3f}"

        lines.append(
            f"| {s['depth']} | {b['val']['valid_board_rate']:.3f} | {b['test']['valid_board_rate']:.3f} | "
            f"{b['ood']['valid_board_rate']:.3f} | {bands['cka_early_mean']:.3f} | {bands['cka_late_mean']:.3f} | "
            f"{_w('val')} | {_w('test')} | {_w('ood')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    depths = parse_depths(args.depths)
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders, max_n = make_loaders(args)
    summaries = [train_one_depth(args, n_rounds=d, loaders=loaders, max_n=max_n, device=device, run_dir=run_dir) for d in depths]
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "summaries": summaries}, indent=2, default=str))
    write_report(run_dir, summaries, args)
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
