"""Train once, reload a checkpoint, and sweep strict upper-triangular RYS windows."""

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
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader

from rys.cka import cka_matrix
from rys.sat_data import SatDataset, make_sat_examples, make_sat_splits, sequence_length, vocab_size
from rys.tensor_activations import capture_tensor_residual_stream
from rys.tiny_transformer import (
    FactorizedCNFSatTransformer,
    FactorizedCNFTransformerConfig,
    TinySatTransformer,
    TinyTransformerConfig,
)
from rys.training.device_cache import cache_batches_on_device
from rys.training.modules import ClassifierLitModule
from rys.training.rys_logging import log_rys_progress
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(30, help="Random seed."),
    architecture: str = typer.Option("factorized", help="Model architecture: 'flat' or 'factorized'."),
    n_vars: int = typer.Option(4, help="In-distribution variable count."),
    n_clauses: int = typer.Option(12, help="In-distribution clause count (keep the ratio high enough for UNSAT)."),
    ood_vars: int = typer.Option(5, help="Out-of-distribution variable count."),
    ood_clauses: int = typer.Option(16, help="Out-of-distribution clause count."),
    n_train: int = typer.Option(8192, help="Training examples."),
    n_val: int = typer.Option(2048, help="Validation examples."),
    n_test: int = typer.Option(2048, help="Test (and OOD) examples."),
    batch_size: int = typer.Option(128, help="Batch size."),
    epochs: int = typer.Option(30, help="Training epochs."),
    lr: float = typer.Option(3e-4, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-2, help="AdamW weight decay."),
    d_model: int = typer.Option(64, help="Model width (must be divisible by n-heads)."),
    n_layers: int = typer.Option(32, help="Number of transformer layers."),
    n_heads: int = typer.Option(4, help="Attention heads per layer."),
    d_mlp: int = typer.Option(128, help="MLP hidden width."),
    dropout: float = typer.Option(0.0, help="Dropout probability."),
    max_repeat: int = typer.Option(2, help="Total traversals of the inclusive window."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the static CKA connectome."),
    output_dir: Path = typer.Option(Path("results/rys_accuracy_matrix"), help="Run output directory."),
    checkpoint: Path | None = typer.Option(None, help="Optional checkpoint to load instead of training."),
) -> None:
    if architecture not in ("flat", "factorized"):
        raise typer.BadParameter("architecture must be 'flat' or 'factorized'.")
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_quality(splits: dict[str, list]) -> pd.DataFrame:
    rows = []
    for name, examples in splits.items():
        labels = np.array([ex.label for ex in examples], dtype=int)
        n_sat = int(labels.sum())
        n_total = int(labels.size)
        n_unsat = n_total - n_sat
        rows.append(
            {
                "split": name,
                "n_total": n_total,
                "n_sat": n_sat,
                "n_unsat": n_unsat,
                "sat_rate": n_sat / n_total,
                "majority_baseline": max(n_sat, n_unsat) / n_total,
            }
        )
    return pd.DataFrame(rows)


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], pd.DataFrame, int, int]:
    max_vars = max(args.n_vars, args.ood_vars)
    max_clauses = max(args.n_clauses, args.ood_clauses)
    splits = make_sat_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_vars=args.n_vars,
        n_clauses=args.n_clauses,
        seed=args.seed,
    )
    splits["ood"] = make_sat_examples(
        args.n_test,
        n_vars=args.ood_vars,
        n_clauses=args.ood_clauses,
        seed=args.seed + 3,
        prefix="ood",
    )
    quality = split_quality(splits)
    datasets = {
        name: SatDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
        for name, examples in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, quality, max_vars, max_clauses


def build_model(args: argparse.Namespace, max_vars: int, max_clauses: int) -> tuple[torch.nn.Module, dict]:
    if args.architecture == "flat":
        config = TinyTransformerConfig(
            vocab_size=vocab_size(max_vars),
            max_seq_len=sequence_length(max_clauses),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_mlp=args.d_mlp,
            dropout=args.dropout,
        )
        return TinySatTransformer(config), config.__dict__
    config = FactorizedCNFTransformerConfig(
        max_vars=max_vars,
        max_clauses=max_clauses,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )
    return FactorizedCNFSatTransformer(config), config.__dict__ | {"max_seq_len": config.max_seq_len}


def forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    architecture: str,
    labels: torch.Tensor | None = None,
    rys_window: tuple[int, int] | None = None,
    n_repeats: int = 2,
) -> dict[str, torch.Tensor | None]:
    if rys_window is None or n_repeats == 1:
        if architecture == "flat":
            return model(  # type: ignore[return-value]
                batch["input_ids"].to(device),  # type: ignore[union-attr]
                attention_mask=batch["attention_mask"].to(device),  # type: ignore[union-attr]
                labels=labels,
            )
        return model(  # type: ignore[return-value]
            variable_ids=batch["variable_ids"].to(device),  # type: ignore[union-attr]
            sign_ids=batch["sign_ids"].to(device),  # type: ignore[union-attr]
            clause_ids=batch["clause_ids"].to(device),  # type: ignore[union-attr]
            slot_ids=batch["slot_ids"].to(device),  # type: ignore[union-attr]
            token_type_ids=batch["token_type_ids"].to(device),  # type: ignore[union-attr]
            attention_mask=batch["factor_attention_mask"].to(device),  # type: ignore[union-attr]
            labels=labels,
        )
    logits = forward_logits_with_inclusive_rys(
        model,
        batch,
        device,
        architecture=architecture,
        window=rys_window,
        n_repeats=n_repeats,
    )
    loss = F.cross_entropy(logits, labels) if labels is not None else None
    return {"logits": logits, "loss": loss}


def embed_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    architecture: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if architecture == "flat":
        backbone = model.model  # type: ignore[attr-defined]
        input_ids = batch["input_ids"].to(device)  # type: ignore[union-attr]
        attention_mask = batch["attention_mask"].to(device)  # type: ignore[union-attr]
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        hidden = backbone.token_embedding(input_ids) + backbone.position_embedding(positions)
        return hidden, attention_mask

    backbone = model.model  # type: ignore[attr-defined]
    variable_ids = batch["variable_ids"].to(device)  # type: ignore[union-attr]
    hidden = (
        backbone.variable_embedding(variable_ids)
        + backbone.sign_embedding(batch["sign_ids"].to(device))  # type: ignore[union-attr]
        + backbone.clause_embedding(batch["clause_ids"].to(device))  # type: ignore[union-attr]
        + backbone.slot_embedding(batch["slot_ids"].to(device))  # type: ignore[union-attr]
        + backbone.token_type_embedding(batch["token_type_ids"].to(device))  # type: ignore[union-attr]
    )
    if backbone.position_embedding is not None:
        positions = torch.arange(variable_ids.shape[1], device=device).unsqueeze(0)
        hidden = hidden + backbone.position_embedding(positions)
    hidden = backbone.dropout(backbone.embedding_norm(hidden))
    return hidden, batch["factor_attention_mask"].to(device)  # type: ignore[union-attr]


def forward_logits_with_inclusive_rys(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    architecture: str,
    window: tuple[int, int],
    n_repeats: int,
) -> torch.Tensor:
    start, end = window
    layers = model.model.layers  # type: ignore[attr-defined]
    if start >= end:
        raise ValueError(
            f"RYS window must span at least two layers (start < end); got {window}."
        )
    if not (0 <= start < end < len(layers)):
        raise ValueError(f"Bad inclusive window {window} for L={len(layers)}.")

    hidden, attention_mask = embed_batch(model, batch, device, architecture=architecture)
    for layer_idx, layer in enumerate(layers):
        hidden = layer(hidden, attention_mask=attention_mask)[0]
        if layer_idx == end:
            for _ in range(n_repeats - 1):
                for replay_idx in range(start, end + 1):
                    hidden = layers[replay_idx](hidden, attention_mask=attention_mask)[0]
    hidden = model.model.norm(hidden)  # type: ignore[attr-defined]
    return model.classifier(hidden[:, 0, :])  # type: ignore[attr-defined]


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    architecture: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_batch(model, batch, device, architecture=architecture, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        if not isinstance(loss, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise RuntimeError("Training outputs are missing.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total += labels.numel()
    return {"loss": total_loss / total, "accuracy": total_correct / total}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    architecture: str,
    rys_window: tuple[int, int] | None = None,
    n_repeats: int = 2,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        outputs = forward_batch(
            model,
            batch,
            device,
            architecture=architecture,
            labels=labels,
            rys_window=rys_window,
            n_repeats=n_repeats,
        )
        loss = outputs["loss"]
        logits = outputs["logits"]
        if not isinstance(loss, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise RuntimeError("Evaluation outputs are missing.")
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total += labels.numel()
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def train_or_load(
    args: argparse.Namespace,
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    run_dir: Path,
) -> tuple[list[dict[str, float]], Path]:
    if args.checkpoint is not None:
        return [], args.checkpoint

    lit = ClassifierLitModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_config=getattr(model, "config", None),
    )
    trainer = build_trainer(
        run_dir,
        max_epochs=args.epochs,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
    )
    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    return [], best_checkpoint_path(trainer)


def strict_upper_windows(n_layers: int) -> list[tuple[int, int]]:
    """Inclusive windows (i, j) with i < j — at least two blocks, no diagonal."""
    return [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]


def delta_accuracy_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    architecture: str,
    n_layers: int,
    n_repeats: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    cached = cache_batches_on_device(loader, device)
    baseline = evaluate(model, cached, device, architecture=architecture)
    matrix = np.full((n_layers, n_layers), np.nan, dtype=float)
    rows = []
    for start, end in log_rys_progress(strict_upper_windows(n_layers), device=device, depth=n_layers):
        metrics = evaluate(
            model,
            cached,
            device,
            architecture=architecture,
            rys_window=(start, end),
            n_repeats=n_repeats,
        )
        delta = metrics["accuracy"] - baseline["accuracy"]
        matrix[start, end] = delta
        rows.append(
            {
                "start": start,
                "end": end,
                "n_repeats": n_repeats,
                "baseline_accuracy": baseline["accuracy"],
                "rys_accuracy": metrics["accuracy"],
                "delta_accuracy": delta,
                "rys_loss": metrics["loss"],
            }
        )
        print(json.dumps(rows[-1]))
    index = list(range(n_layers))
    return pd.DataFrame(matrix, index=index, columns=index), baseline, pd.DataFrame(rows)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cmap: str = "RdBu_r",
    cbar_label: str,
    diverging: bool = True,
) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"No finite values to plot for {path}.")
    if diverging:
        bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("end layer j (inclusive)")
    ax.set_ylabel("start layer i")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_cka_connectome(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    architecture: str,
    capture_batches: int,
) -> pd.DataFrame:
    activations = capture_tensor_residual_stream(
        model,
        loader,
        device=device,
        architecture=architecture,
        token_strategy="cls",
        max_batches=capture_batches,
    )
    return cka_matrix(activations, unbiased=False, device="cpu")


def best_window_row(long: pd.DataFrame, *, split: str) -> dict:
    best = long.loc[long["delta_accuracy"].idxmax()].to_dict()
    best["split"] = split
    return best


def best_combined_window(
    long_by_split: dict[str, pd.DataFrame],
    *,
    test_weight: float = 0.5,
    ood_weight: float = 0.5,
) -> dict:
    test_long = long_by_split["test"]
    ood_long = long_by_split["ood"]
    merged = test_long.merge(
        ood_long,
        on=["start", "end", "n_repeats"],
        suffixes=("_test", "_ood"),
    )
    merged["combined_delta"] = (
        test_weight * merged["delta_accuracy_test"]
        + ood_weight * merged["delta_accuracy_ood"]
    )
    best = merged.loc[merged["combined_delta"].idxmax()].to_dict()
    return {
        "start": int(best["start"]),
        "end": int(best["end"]),
        "combined_delta": float(best["combined_delta"]),
        "delta_test": float(best["delta_accuracy_test"]),
        "delta_ood": float(best["delta_accuracy_ood"]),
        "rys_accuracy_test": float(best["rys_accuracy_test"]),
        "rys_accuracy_ood": float(best["rys_accuracy_ood"]),
    }


def write_report(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    baselines: dict[str, dict[str, float]],
    best_rows: list[dict],
    combined_best: dict,
    cka_path: Path,
) -> None:
    n_windows = len(strict_upper_windows(args.n_layers))
    lines = [
        "# RYS Accuracy Matrix Report",
        "",
        f"- Architecture: `{args.architecture}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Layers: `{args.n_layers}`",
        (
            "- RYS convention: inclusive window `(i, j)` with **`i < j`** "
            "duplicates blocks `i..j` (≥2 layers). Diagonal and lower triangle are omitted."
        ),
        f"- Windows swept: `{n_windows}` strict upper-triangular pairs",
        f"- Repeats: `{args.max_repeat}` total traversals",
        f"- CKA connectome: `{cka_path}` (`{args.capture_batches}` val batches, CLS token)",
        "",
        "## Baselines",
        "",
        "| split | accuracy | loss |",
        "| --- | ---: | ---: |",
    ]
    for split, metrics in baselines.items():
        lines.append(f"| {split} | {metrics['accuracy']:.4f} | {metrics['loss']:.4f} |")
    lines.extend(
        [
            "",
            "## Best ΔAccuracy Per Split (ex-post on that split)",
            "",
            "| split | window | RYS accuracy | delta |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in best_rows:
        lines.append(
            f"| {row['split']} | ({row['start']}, {row['end']}) | "
            f"{row['rys_accuracy']:.4f} | {row['delta_accuracy']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Best Window by Test+OOD Combined Score",
            "",
            (
                f"Weighted `0.5 * Δ_test + 0.5 * Δ_ood`: "
                f"**({combined_best['start']}, {combined_best['end']})** "
                f"(Δ_test={combined_best['delta_test']:+.4f}, "
                f"Δ_ood={combined_best['delta_ood']:+.4f}, "
                f"combined={combined_best['combined_delta']:+.4f})"
            ),
            "",
            "Use this row when choosing a **single** RYS window for deployment; "
            "per-split optima above can disagree (e.g. Cycle 2 had different test vs OOD winners).",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, quality, max_vars, max_clauses = make_loaders(args)
    quality.to_csv(run_dir / "dataset_quality.csv", index=False)
    print(quality.to_json(orient="records"))

    model, config = build_model(args, max_vars, max_clauses)
    model.to(device)
    _, checkpoint_path = train_or_load(args, model, loaders, device, run_dir)

    # Reload from disk before the intervention sweep so the matrix is explicitly
    # computed from the fixed checkpointed weights.
    reloaded_model, _ = load_rys_model(
        checkpoint_path,
        lambda _config: build_model(args, max_vars, max_clauses)[0],
        map_location=device,
    )
    reloaded_model.to(device)

    cka = compute_cka_connectome(
        reloaded_model,
        loaders["val"],
        device,
        architecture=args.architecture,
        capture_batches=args.capture_batches,
    )
    cka_path = run_dir / "cka_val.csv"
    cka.to_csv(cka_path)
    save_heatmap(
        cka,
        run_dir / "cka_val.png",
        title="Linear CKA (validation, CLS)",
        cmap="viridis",
        cbar_label="CKA",
        diverging=False,
    )

    baselines: dict[str, dict[str, float]] = {}
    best_rows: list[dict] = []
    long_by_split: dict[str, pd.DataFrame] = {}
    for split in ("val", "test", "ood"):
        matrix, baseline, long = delta_accuracy_matrix(
            reloaded_model,
            loaders[split],
            device,
            architecture=args.architecture,
            n_layers=args.n_layers,
            n_repeats=args.max_repeat,
        )
        baselines[split] = baseline
        long_by_split[split] = long
        matrix.to_csv(run_dir / f"delta_accuracy_{split}.csv")
        long.to_csv(run_dir / f"delta_accuracy_{split}_long.csv", index=False)
        save_heatmap(
            matrix,
            run_dir / f"delta_accuracy_{split}.png",
            title=f"RYS ΔAccuracy ({split}, i<j only)",
            cbar_label="Δ accuracy vs baseline",
        )
        best_rows.append(best_window_row(long, split=split))

    combined_best = best_combined_window(long_by_split)

    summary = {
        "args": vars(args),
        "device": str(device),
        "config": config,
        "checkpoint": str(checkpoint_path),
        "dataset_quality": quality.to_dict(orient="records"),
        "baselines": baselines,
        "best_rows": best_rows,
        "combined_best_test_ood": combined_best,
        "n_rys_windows": len(strict_upper_windows(args.n_layers)),
        "cka_val_path": str(cka_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(
        run_dir,
        args=args,
        checkpoint_path=checkpoint_path,
        baselines=baselines,
        best_rows=best_rows,
        combined_best=combined_best,
        cka_path=cka_path,
    )
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
