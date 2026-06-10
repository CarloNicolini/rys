"""Sorted Translation experiment: grokking + RYS OOD-length generalization.

Trains an encoder-only RoPE Transformer to sort short sequences under heavy
weight decay (to force grokking), then uses the post-training RYS hook to repeat
middle layers and sort longer out-of-distribution sequences with no retraining.

Per run it records:
- a grokking metric panel (token / sequence accuracy, sortedness, W_E/W_U norms);
- an accuracy-vs-rounds curve (read out after each layer) for every split;
- the validation token-stream CKA connectome and a rho/phi theory fit;
- an RYS sweep over windows x n_repeats for every OOD length, with the headline
  "sequence accuracy vs n_repeats of the best window" plot.
"""

from __future__ import annotations

import argparse
import json
import random
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
from rys.sorted_translation_data import (
    SortedTranslationDataset,
    make_sorted_examples,
    sortedness,
    verify_sorted_tensor,
)
from rys.sorted_translation_transformer import (
    SortedTranslationConfig,
    SortedTranslationTransformer,
)
from rys.surgery import apply_rys
from rys.theory_validation import rho_phi_table, theory_fit
from rys.training.modules import SortedTranslationLitModule
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(0, help="Random seed."),
    train_len: int = typer.Option(16, help="In-distribution train/validation length."),
    train_lens: str = typer.Option(
        "", help="Comma list of training lengths for mixed-length training. Empty = single --train-len."
    ),
    ood_lens: str = typer.Option("24,32,48", help="Comma list of out-of-distribution lengths."),
    vocab: int = typer.Option(128, help="Token vocabulary size."),
    n_train: int = typer.Option(8192, help="Training examples."),
    n_val: int = typer.Option(2048, help="Validation examples."),
    n_test: int = typer.Option(2048, help="Test (per OOD length) examples."),
    batch_size: int = typer.Option(256, help="Batch size."),
    epochs: int = typer.Option(400, help="Training epochs."),
    lr: float = typer.Option(1e-3, help="AdamW learning rate."),
    weight_decay: float = typer.Option(0.1, help="AdamW weight decay (heavy, to force grokking)."),
    d_model: int = typer.Option(128, help="Model width (must be divisible by n-heads)."),
    n_layers: int = typer.Option(3, help="Number of transformer layers."),
    n_heads: int = typer.Option(4, help="Attention heads per layer."),
    d_mlp: int = typer.Option(512, help="MLP hidden width."),
    dropout: float = typer.Option(0.0, help="Dropout probability."),
    weight_tied: bool = typer.Option(
        False, "--weight-tied/--no-weight-tied", help="Share one round across depth (iterated map)."
    ),
    pre_norm: bool = typer.Option(True, "--pre-norm/--post-norm", help="Pre-norm (additive residual) vs post-norm."),
    max_repeat: int = typer.Option(6, help="Maximum total traversals of a window in the RYS sweep."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    log_every: int = typer.Option(
        10, help="Deprecated under Lightning (validation runs every epoch); kept for CLI compatibility."
    ),
    checkpoint: Path | None = typer.Option(None, help="Load weights and skip training."),
    output_dir: Path = typer.Option(Path("results/sorted_translation"), help="Run output directory."),
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_lens(value: str) -> list[int]:
    lens = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not lens or any(length < 1 for length in lens):
        raise ValueError("ood-lens must be positive integers.")
    return lens


def make_loaders(
    args: argparse.Namespace,
) -> tuple[list[DataLoader], dict[str, DataLoader], list[str], list[int]]:
    """Build mixed-length training loaders and fixed-length evaluation loaders.

    Training uses one loader per length in ``--train-lens`` (or a single
    ``--train-len`` if empty); each batch is internally single-length, so no
    padding is required and RoPE handles the variable lengths. Validation is at
    ``--train-len`` (in-distribution) and each ``--ood-lens`` length is a
    separate extrapolation split.
    """
    train_lens = parse_lens(args.train_lens) if args.train_lens else [args.train_len]
    ood_lens = parse_lens(args.ood_lens)
    per_len = max(1, args.n_train // len(train_lens))

    train_loaders: list[DataLoader] = []
    for offset, length in enumerate(train_lens):
        examples = make_sorted_examples(
            per_len, length, seed=args.seed + offset, vocab=args.vocab, prefix=f"train{length}"
        )
        train_loaders.append(
            DataLoader(SortedTranslationDataset(examples), batch_size=args.batch_size, shuffle=True)
        )

    eval_loaders: dict[str, DataLoader] = {}
    val_examples = make_sorted_examples(
        args.n_val, args.train_len, seed=args.seed + 1000, vocab=args.vocab, prefix="val"
    )
    eval_loaders["val"] = DataLoader(
        SortedTranslationDataset(val_examples), batch_size=args.batch_size, shuffle=False
    )
    for offset, length in enumerate(ood_lens):
        examples = make_sorted_examples(
            args.n_test, length, seed=args.seed + 2000 + offset, vocab=args.vocab, prefix=f"ood{length}"
        )
        eval_loaders[f"ood_{length}"] = DataLoader(
            SortedTranslationDataset(examples), batch_size=args.batch_size, shuffle=False
        )

    eval_splits = ["val"] + [f"ood_{length}" for length in ood_lens]
    return train_loaders, eval_loaders, eval_splits, train_lens


def build_model(args: argparse.Namespace) -> tuple[SortedTranslationTransformer, dict]:
    config = SortedTranslationConfig(
        vocab_in=args.vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
        pre_norm=args.pre_norm,
        weight_tied=args.weight_tied,
    )
    return SortedTranslationTransformer(config), config.__dict__


def metrics_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    token_correct = int((preds == target_ids).sum().item())
    token_total = int(target_ids.numel())
    seq_correct = int(verify_sorted_tensor(preds, input_ids).sum().item())
    seq_total = int(target_ids.shape[0])
    sorted_sum = float(sortedness(preds).sum().item())
    return {
        "token_correct": token_correct,
        "token_total": token_total,
        "seq_correct": seq_correct,
        "seq_total": seq_total,
        "sorted_sum": sorted_sum,
    }


def fold(metrics: dict[str, float], loss_sum: float) -> dict[str, float]:
    return {
        "loss": loss_sum / metrics["seq_total"],
        "token_accuracy": metrics["token_correct"] / metrics["token_total"],
        "sequence_accuracy": metrics["seq_correct"] / metrics["seq_total"],
        "sortedness": metrics["sorted_sum"] / metrics["seq_total"],
    }


@torch.inference_mode()
def evaluate(
    model: SortedTranslationTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    window: tuple[int, int] | None = None,
    n_repeats: int = 1,
) -> dict[str, float]:
    model.eval()
    agg = {"token_correct": 0, "token_total": 0, "seq_correct": 0, "seq_total": 0, "sorted_sum": 0.0}
    loss_sum = 0.0
    use_rys = window is not None and n_repeats > 1

    def run() -> None:
        nonlocal loss_sum
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            out = model(input_ids, target_ids=target_ids)
            loss_sum += float(out["loss"].item()) * target_ids.shape[0]
            for key, value in metrics_from_logits(out["logits"], input_ids, target_ids).items():
                agg[key] += value

    if use_rys:
        with apply_rys(model, window=window, n_repeats=n_repeats):
            run()
    else:
        run()
    return fold(agg, loss_sum)


def train_epoch(
    model: SortedTranslationTransformer,
    train_loaders: list[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    agg = {"token_correct": 0, "token_total": 0, "seq_correct": 0, "seq_total": 0, "sorted_sum": 0.0}
    loss_sum = 0.0
    # Each batch is single-length; interleave batches across lengths and shuffle
    # their order so mixed-length training is balanced within an epoch.
    batches = [batch for loader in train_loaders for batch in loader]
    random.shuffle(batches)
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids, target_ids=target_ids)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.item()) * target_ids.shape[0]
        with torch.no_grad():
            for key, value in metrics_from_logits(out["logits"], input_ids, target_ids).items():
                agg[key] += value
    return fold(agg, loss_sum)


def weight_norms(model: SortedTranslationTransformer) -> dict[str, float]:
    return {
        "w_e_norm": float(model.model.embed.weight.norm().item()),
        "w_u_norm": float(model.unembed.weight.norm().item()),
    }


@torch.inference_mode()
def accuracy_vs_rounds(
    model: SortedTranslationTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    n_layers: int,
) -> pd.DataFrame:
    model.eval()
    per_round = [
        {"token_correct": 0, "token_total": 0, "seq_correct": 0, "seq_total": 0, "sorted_sum": 0.0}
        for _ in range(n_layers)
    ]
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        out = model(input_ids, return_round_logits=True)
        for idx, logits in enumerate(out["round_logits"]):
            for key, value in metrics_from_logits(logits, input_ids, target_ids).items():
                per_round[idx][key] += value
    rows = []
    for idx, agg in enumerate(per_round):
        folded = fold(agg, 0.0)
        rows.append(
            {
                "round": idx + 1,
                "token_accuracy": folded["token_accuracy"],
                "sequence_accuracy": folded["sequence_accuracy"],
                "sortedness": folded["sortedness"],
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def capture_token_states(
    model: SortedTranslationTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        out = model(input_ids, return_hidden_states=True)
        trace = out["hidden_states"]
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        for layer_idx, hidden in enumerate(trace):
            for row_idx, prompt_id in enumerate(prompt_ids):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "layer": layer_idx,
                        "activation": hidden[row_idx].detach().cpu().numpy(),
                        "strategy": "token",
                    }
                )
    return pd.DataFrame(rows)


def rys_windows(n_layers: int) -> list[tuple[int, int]]:
    """Half-open windows ``(start, end)`` valid for apply_rys (``end <= n_layers-1``)."""
    return [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]


def rys_sweep(
    model: SortedTranslationTransformer,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    eval_splits: list[str],
    n_layers: int,
    max_repeat: int,
) -> pd.DataFrame:
    rows = []
    windows = rys_windows(n_layers)
    for split in eval_splits:
        base = evaluate(model, loaders[split], device)
        for window in windows:
            for n_repeats in range(1, max_repeat + 1):
                if n_repeats == 1:
                    metrics = base
                else:
                    metrics = evaluate(model, loaders[split], device, window=window, n_repeats=n_repeats)
                row = {
                    "split": split,
                    "start": window[0],
                    "end": window[1],
                    "n_repeats": n_repeats,
                    "sequence_accuracy": metrics["sequence_accuracy"],
                    "token_accuracy": metrics["token_accuracy"],
                    "sortedness": metrics["sortedness"],
                    "baseline_sequence_accuracy": base["sequence_accuracy"],
                    "delta_sequence_accuracy": metrics["sequence_accuracy"] - base["sequence_accuracy"],
                }
                rows.append(row)
                print(json.dumps(row))
    return pd.DataFrame(rows)


def save_round_curve(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, frame in curves.items():
        ax.plot(frame["round"], frame["sequence_accuracy"], marker="o", label=split)
    ax.set_title("Sequence accuracy vs layer rounds")
    ax.set_xlabel("round (layer read-out)")
    ax.set_ylabel("sequence accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rys_plot(sweep: pd.DataFrame, path: Path) -> dict[str, dict]:
    """Plot sequence accuracy vs n_repeats for each split's best window."""
    best_windows: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in sweep["split"].unique():
        sub = sweep[sweep["split"] == split]
        # Best window = the one with the highest sequence accuracy at max n_repeats.
        max_rep = sub["n_repeats"].max()
        at_max = sub[sub["n_repeats"] == max_rep]
        best_row = at_max.loc[at_max["sequence_accuracy"].idxmax()]
        start, end = int(best_row["start"]), int(best_row["end"])
        curve = sub[(sub["start"] == start) & (sub["end"] == end)].sort_values("n_repeats")
        ax.plot(curve["n_repeats"], curve["sequence_accuracy"], marker="o", label=f"{split} ({start},{end})")
        best_windows[split] = {
            "start": start,
            "end": end,
            "best_sequence_accuracy": float(best_row["sequence_accuracy"]),
            "baseline_sequence_accuracy": float(best_row["baseline_sequence_accuracy"]),
            "delta_sequence_accuracy": float(best_row["delta_sequence_accuracy"]),
            "n_repeats": int(max_rep),
        }
    ax.set_title("RYS: sequence accuracy vs repeats (best window per split)")
    ax.set_xlabel("n_repeats of window")
    ax.set_ylabel("sequence accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return best_windows


def save_cka_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="viridis", vmin=float(values.min()), vmax=float(values.max()))
    ax.set_title("Token-stream CKA (validation)")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label="CKA")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(run_dir: Path, *, args: argparse.Namespace, baselines: dict, best_windows: dict, fit: dict) -> None:
    lines = [
        "# Sorted Translation — RYS-controlled experiment",
        "",
        "Encoder-only RoPE Transformer sorting an unsorted sequence into a disjoint",
        "output vocabulary (`sorted(input)+vocab`). One-hot input embedding, distinct",
        "unembedding head (no weight tying), trained with heavy weight decay to grok.",
        "",
        f"- Train lengths: `{args.train_lens or args.train_len}`; val length: `{args.train_len}`; OOD lengths: `{args.ood_lens}`",
        f"- Layers: `{args.n_layers}`, heads `{args.n_heads}`, d_model `{args.d_model}`, d_mlp `{args.d_mlp}`",
        f"- Weight decay: `{args.weight_decay}`; pre_norm `{args.pre_norm}`; weight_tied `{args.weight_tied}`",
        "",
        "## Baseline accuracy per split",
        "",
        "| split | token acc | sequence acc | sortedness |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split, metrics in baselines.items():
        lines.append(
            f"| {split} | {metrics['token_accuracy']:.4f} | "
            f"{metrics['sequence_accuracy']:.4f} | {metrics['sortedness']:.4f} |"
        )
    lines += [
        "",
        "## Best RYS window per split (sequence accuracy)",
        "",
        "| split | window | repeats | baseline seq acc | RYS seq acc | delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split, info in best_windows.items():
        lines.append(
            f"| {split} | ({info['start']}, {info['end']}) | {info['n_repeats']} | "
            f"{info['baseline_sequence_accuracy']:.4f} | {info['best_sequence_accuracy']:.4f} | "
            f"{info['delta_sequence_accuracy']:+.4f} |"
        )
    lines += [
        "",
        "## rho/phi theory fit (validation token stream)",
        "",
        f"- Pairs (total / plateau): {fit['n_pairs_total']} / {fit['n_pairs_plateau']}",
        f"- Pearson(measured, plateau predictor): {fit['pearson_plateau_pred']:.4f}",
        f"- Spearman(measured, plateau predictor): {fit['spearman_plateau_pred']:.4f}",
        f"- Median R (relative force): {fit['median_R']:.4f}; median cos phi: {fit['median_cos_phi']:.4f}",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loaders, loaders, eval_splits, train_lens = make_loaders(args)
    model, config = build_model(args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"n_params": n_params, "train_lens": train_lens, "config": config, "device": str(device)}, default=str))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        lit = SortedTranslationLitModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_config=config,
        )
        trainer = build_trainer(
            run_dir,
            max_epochs=args.epochs,
            monitor=lit.primary_metric,
            mode=lit.primary_mode,
        )
        trainer.fit(lit, train_dataloaders=train_loaders, val_dataloaders=loaders["val"])
        checkpoint_path = best_checkpoint_path(trainer)

    reloaded, _ = load_rys_model(
        checkpoint_path,
        lambda _config: build_model(args)[0],
        map_location=device,
    )
    reloaded.to(device)

    baselines = {split: evaluate(reloaded, loaders[split], device) for split in eval_splits}
    (run_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))

    curves = {
        split: accuracy_vs_rounds(reloaded, loaders[split], device, n_layers=args.n_layers)
        for split in eval_splits
    }
    for split, frame in curves.items():
        frame.to_csv(run_dir / f"accuracy_vs_rounds_{split}.csv", index=False)
    save_round_curve(curves, run_dir / "accuracy_vs_rounds.png")

    activations = capture_token_states(reloaded, loaders["val"], device, max_batches=args.capture_batches)
    cka = cka_matrix(activations, unbiased=False, device="cpu")
    cka.to_csv(run_dir / "cka_val.csv")
    save_cka_heatmap(cka, run_dir / "cka_val.png")
    rho_phi = rho_phi_table(activations)
    rho_phi.to_csv(run_dir / "rho_phi.csv", index=False)
    fit = theory_fit(rho_phi)
    (run_dir / "theory_fit.json").write_text(json.dumps(fit, indent=2))

    sweep = rys_sweep(
        reloaded,
        loaders,
        device,
        eval_splits=eval_splits,
        n_layers=args.n_layers,
        max_repeat=args.max_repeat,
    )
    sweep.to_csv(run_dir / "rys_sweep.csv", index=False)
    best_windows = save_rys_plot(sweep, run_dir / "rys_sweep.png")

    summary = {
        "args": vars(args),
        "device": str(device),
        "n_params": n_params,
        "config": config,
        "checkpoint": str(checkpoint_path),
        "baselines": baselines,
        "best_windows": best_windows,
        "theory_fit": fit,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(run_dir, args=args, baselines=baselines, best_windows=best_windows, fit=fit)
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
