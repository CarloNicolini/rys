"""Depth-controlled graph k-colouring message-passing experiment for RYS.

Mirrors ``scripts/run_nqueens_messagepassing_experiment.py``.  For each trained
depth:

- baselines (valid colouring rate) on val/test/OOD;
- a validity-vs-rounds curve (read out after each round);
- the validation vertex-state CKA connectome;
- strict RYS ΔValidity matrices (half-open windows, >=2 duplicated rounds).

RYS uses ``rys.surgery.apply_rys`` half-open windows ``[start, end)`` with
``end < n_rounds`` (a downstream round must receive the doubled stream), matching
the SAT and N-Queens experiments and ``predict_cka_under_rys``.

Graph colouring is the first *sparse* constraint graph in the programme (SAT is
bipartite, N-Queens is complete), and the solver is permutation-equivariant with
a fixed colour vocabulary, so the OOD split can grow the graph (more vertices and
edges) without the untrained-vocabulary collapse that forced the N-Queens OOD
split to reduce givens instead.
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
from rys.coloring_data import (
    ColoringDataset,
    make_coloring_examples,
    make_coloring_splits,
    soft_coloring_loss,
    verify_coloring_tensor,
)
from rys.coloring_message_passing import ColoringMessagePassingModel, ColoringMPConfig
from rys.surgery import apply_rys
from rys.training.device_cache import cache_batches_on_device
from rys.training.modules import ColoringMPLitModule
from rys.training.rys_logging import log_rys_progress
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(47, help="Random seed."),
    depths: str = typer.Option("8,16,32", help="Comma-separated message-passing depths (rounds) to sweep."),
    n_vertices: int = typer.Option(12, help="In-distribution vertex count."),
    n_edges: int = typer.Option(24, help="In-distribution edge count."),
    n_colors: int = typer.Option(3, help="Number of colours k."),
    n_givens: int = typer.Option(3, help="Revealed vertices per instance."),
    ood_vertices: int = typer.Option(
        18,
        help="Larger graph OOD (more vertices). Safe here because the solver is "
        "permutation-equivariant with a fixed colour vocabulary.",
    ),
    ood_edges: int = typer.Option(40, help="OOD edge count."),
    ood_givens: int = typer.Option(3, help="OOD revealed vertices per instance."),
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
    max_degree: int = typer.Option(24, help="Max vertex degree embedding index."),
    given_ce_weight: float = typer.Option(0.25, help="Cross-entropy weight anchoring given vertices."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    skip_rys: bool = typer.Option(
        False, "--skip-rys/--no-skip-rys", help="Train and export curves/CKA without RYS matrices."
    ),
    output_dir: Path = typer.Option(Path("results/coloring_messagepassing"), help="Run output directory."),
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
    max_v = max(args.n_vertices, args.ood_vertices)
    splits = make_coloring_splits(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n=args.n_vertices, k=args.n_colors, n_edges=args.n_edges, n_givens=args.n_givens, seed=args.seed,
    )
    splits["ood"] = make_coloring_examples(
        args.n_test, n=args.ood_vertices, k=args.n_colors, n_edges=args.ood_edges,
        n_givens=args.ood_givens, seed=args.seed + 3, prefix="ood",
    )
    datasets = {k: ColoringDataset(v, max_v=max_v, n_colors=args.n_colors) for k, v in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, max_v


def build_model(args: argparse.Namespace, *, n_rounds: int, max_v: int) -> tuple[ColoringMessagePassingModel, dict]:
    config = ColoringMPConfig(
        max_v=max_v, n_colors=args.n_colors, max_degree=args.max_degree, d_model=args.d_model,
        n_rounds=n_rounds, n_heads=args.n_heads, d_mlp=args.d_mlp, pre_norm=True,
    )
    return ColoringMessagePassingModel(config), config.__dict__


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def on_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "given_color": batch["given_color"].to(device),
        "is_given": batch["is_given"].to(device),
        "degree": batch["degree"].to(device),
        "adjacency": batch["adjacency"].to(device),
        "vertex_mask": batch["vertex_mask"].to(device),
    }


def coloring_metrics(logits, vertex_mask, adjacency, given_color, is_given, labels) -> dict[str, int]:
    colors = logits.argmax(dim=-1)
    valid = verify_coloring_tensor(colors, vertex_mask, adjacency, given_color, is_given)
    cell_correct = ((colors == labels) & vertex_mask).sum()
    return {
        "valid_correct": int(valid.sum().detach().cpu()),
        "cell_correct": int(cell_correct.detach().cpu()),
        "cell_total": int(vertex_mask.sum().detach().cpu()),
        "example_total": int(labels.shape[0]),
    }


def accumulate(counts: dict[str, int]) -> dict[str, float]:
    return {
        "valid_coloring_rate": counts["valid_correct"] / counts["example_total"],
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
            soft_coloring_loss(rl, kw["adjacency"], kw["vertex_mask"], given_color=kw["given_color"],
                               is_given=kw["is_given"], given_ce_weight=given_ce_weight)
            for rl in out["round_logits"]
        ]
        loss = torch.stack(terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for k, v in coloring_metrics(out["logits"], kw["vertex_mask"], kw["adjacency"],
                                     kw["given_color"], kw["is_given"], labels).items():
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
        for k, v in coloring_metrics(logits, kw["vertex_mask"], kw["adjacency"],
                                     kw["given_color"], kw["is_given"], labels).items():
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
            for k, v in coloring_metrics(logits, kw["vertex_mask"], kw["adjacency"],
                                         kw["given_color"], kw["is_given"], labels).items():
                per_round[r][k] += v
    rows = [{"round": r + 1, **accumulate(c)} for r, c in enumerate(per_round)]
    return pd.DataFrame(rows)


@torch.inference_mode()
def capture_vertex_states(model, loader, device, *, max_batches: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        kw = on_device(batch, device)
        trace = model(**kw, return_hidden_states=True)["hidden_states"]
        mask = batch["vertex_mask"].bool()
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        for layer_idx, hidden in enumerate(trace):
            for row_idx, pid in enumerate(prompt_ids):
                rows.append({
                    "prompt_id": pid,
                    "layer": layer_idx,
                    "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                    "strategy": "vertex",
                })
    return pd.DataFrame(rows)


def strict_windows(n_rounds: int, min_span: int = 2) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_rounds) for j in range(i + min_span, n_rounds)]


def delta_matrix(model, loader, device, n_rounds) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    cached = cache_batches_on_device(loader, device)
    base = evaluate(model, cached, device)["valid_coloring_rate"]
    mat = np.full((n_rounds, n_rounds), np.nan)
    rows = []
    for start, end in log_rys_progress(strict_windows(n_rounds), device=device, depth=n_rounds):
        rys = evaluate(model, cached, device, window=(start, end))["valid_coloring_rate"]
        mat[start, end] = rys - base
        rows.append({"start": start, "end": end, "baseline": base, "rys": rys, "delta_valid": rys - base})
    return pd.DataFrame(mat), base, pd.DataFrame(rows)


def save_curve(curves: dict[str, pd.DataFrame], path: Path, *, depth: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, frame in curves.items():
        ax.plot(frame["round"], frame["valid_coloring_rate"], marker="o", label=split)
    ax.set_title(f"Valid colouring rate vs rounds (L={depth})")
    ax.set_xlabel("round")
    ax.set_ylabel("valid colouring rate")
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


def train_one_depth(args, *, n_rounds, loaders, max_v, device, run_dir) -> dict:
    depth_dir = run_dir / f"L{n_rounds:02d}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    model, config = build_model(args, n_rounds=n_rounds, max_v=max_v)
    model.to(device)
    params = count_parameters(model)
    print(json.dumps({"depth": n_rounds, "n_params_total": params["total"], "n_params_trainable": params["trainable"]}))
    lit = ColoringMPLitModule(
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
        lambda _config: build_model(args, n_rounds=n_rounds, max_v=max_v)[0],
        map_location=device,
    )
    reloaded.to(device)

    curves = {s: validity_vs_rounds(reloaded, loaders[s], device, n_rounds=n_rounds) for s in ("val", "test", "ood")}
    for s, frame in curves.items():
        frame.to_csv(depth_dir / f"validity_vs_rounds_{s}.csv", index=False)
    save_curve(curves, depth_dir / "validity_vs_rounds.png", depth=n_rounds)

    activations = capture_vertex_states(reloaded, loaders["val"], device, max_batches=args.capture_batches)
    activations.to_pickle(depth_dir / "vertex_activations.pkl")
    cka = cka_matrix(activations, unbiased=False, device="cpu")
    cka.to_csv(depth_dir / "cka_variable_val.csv")
    save_heatmap(cka, depth_dir / "cka_variable_val.png", title=f"Vertex-state CKA (L={n_rounds})",
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
                     title=f"RYS ΔValidity {s} (L={n_rounds})", cbar="Δ valid colouring rate")
        best = long.loc[long["delta_valid"].idxmax()].to_dict()
        best["split"] = s
        best_rows.append(best)

    summary = {"depth": n_rounds, "config": config, "n_params": params, "checkpoint": str(ckpt), "baselines": baselines,
               "cka_bands": bands, "best_rows": best_rows, "rys_skipped": False}
    (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def write_report(run_dir, summaries, args) -> None:
    lines = [
        "# Graph k-Colouring Message-Passing Depth Sweep",
        "",
        "Task: complete a partial graph k-colouring (givens) to a proper colouring.",
        "Primary metric: valid_coloring_rate (verifier), not exact match to the planted colouring.",
        "",
        f"- Depths (rounds): `{args.depths}`",
        f"- Graph: V={args.n_vertices}, E={args.n_edges}, k={args.n_colors}, givens={args.n_givens}; "
        f"OOD V={args.ood_vertices}, E={args.ood_edges}, givens={args.ood_givens}",
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
            f"| {s['depth']} | {b['val']['valid_coloring_rate']:.3f} | {b['test']['valid_coloring_rate']:.3f} | "
            f"{b['ood']['valid_coloring_rate']:.3f} | {bands['cka_early_mean']:.3f} | {bands['cka_late_mean']:.3f} | "
            f"{_w('val')} | {_w('test')} | {_w('ood')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    depths = parse_depths(args.depths)
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders, max_v = make_loaders(args)
    summaries = [train_one_depth(args, n_rounds=d, loaders=loaders, max_v=max_v, device=device, run_dir=run_dir) for d in depths]
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "summaries": summaries}, indent=2, default=str))
    write_report(run_dir, summaries, args)
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
