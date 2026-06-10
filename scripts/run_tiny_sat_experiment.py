"""Train a tiny 3-SAT Transformer and evaluate RYS windows.

This is the first controlled PI/postdoc-loop experiment: train a small local
model on brute-force-labelled 3-SAT, compute CKA/residual diagnostics, then ask
whether pre-benchmark window scores enrich for useful RYS interventions.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader

from rys.cka import cka_matrix
from rys.residual_force import residual_force_long, residual_force_matrices
from rys.sat_data import SatDataset, make_sat_examples, make_sat_splits, sequence_length, vocab_size
from rys.surgery import apply_rys
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
    seed: int = typer.Option(0, help="Random seed."),
    n_vars: int = typer.Option(6, help="In-distribution variable count."),
    n_clauses: int = typer.Option(18, help="In-distribution clause count (keep the ratio high enough for UNSAT)."),
    ood_vars: int = typer.Option(7, help="Out-of-distribution variable count."),
    ood_clauses: int = typer.Option(24, help="Out-of-distribution clause count."),
    n_train: int = typer.Option(1024, help="Training examples."),
    n_val: int = typer.Option(256, help="Validation examples."),
    n_test: int = typer.Option(256, help="Test (and OOD) examples."),
    batch_size: int = typer.Option(64, help="Batch size."),
    epochs: int = typer.Option(8, help="Training epochs."),
    lr: float = typer.Option(3e-4, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-2, help="AdamW weight decay."),
    architecture: str = typer.Option("factorized", help="Model architecture: 'flat' or 'factorized'."),
    d_model: int = typer.Option(64, help="Model width (must be divisible by n-heads)."),
    n_layers: int = typer.Option(6, help="Number of transformer layers."),
    n_heads: int = typer.Option(4, help="Attention heads per layer."),
    d_mlp: int = typer.Option(128, help="MLP hidden width."),
    dropout: float = typer.Option(0.0, help="Dropout probability."),
    min_window: int = typer.Option(2, help="Minimum RYS window length to score."),
    max_window: int = typer.Option(4, help="Maximum RYS window length to score."),
    max_repeat: int = typer.Option(4, help="Maximum total traversals of a window."),
    top_k_windows: int = typer.Option(8, help="Number of top-scoring windows to evaluate."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    output_dir: Path = typer.Option(Path("results/tiny_sat"), help="Run output directory."),
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
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("Training loss is missing.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        logits = outputs["logits"]
        assert isinstance(logits, torch.Tensor)
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
    window: tuple[int, int] | None = None,
    n_repeats: int = 1,
) -> dict[str, float]:
    model.eval()
    if window is not None and n_repeats > 1:
        with apply_rys(model, window, n_repeats=n_repeats):
            return _evaluate_no_context(model, loader, device, architecture=architecture)
    return _evaluate_no_context(model, loader, device, architecture=architecture)


def _evaluate_no_context(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    architecture: str,
) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        labels = batch["labels"].to(device)
        outputs = forward_batch(model, batch, device, architecture=architecture, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        if not isinstance(loss, torch.Tensor) or not isinstance(logits, torch.Tensor):
            raise RuntimeError("Evaluation outputs are missing.")
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total_correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total += labels.numel()
    return {"loss": total_loss / total, "accuracy": total_correct / total}


@torch.inference_mode()
def base_to_rys_kl(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    architecture: str,
    window: tuple[int, int],
    n_repeats: int,
) -> float:
    model.eval()
    total = 0
    weighted_kl = 0.0
    for batch in loader:
        labels = batch["labels"].to(device)
        base_logits = forward_batch(model, batch, device, architecture=architecture, labels=labels)["logits"]
        with apply_rys(model, window, n_repeats=n_repeats):
            rys_logits = forward_batch(model, batch, device, architecture=architecture, labels=labels)["logits"]
        if not isinstance(base_logits, torch.Tensor) or not isinstance(rys_logits, torch.Tensor):
            raise RuntimeError("KL logits are missing.")
        base_probs = F.softmax(base_logits, dim=-1)
        rys_log_probs = F.log_softmax(rys_logits, dim=-1)
        kl = F.kl_div(rys_log_probs, base_probs, reduction="batchmean")
        weighted_kl += float(kl.detach().cpu()) * labels.shape[0]
        total += labels.shape[0]
    return weighted_kl / total


def forward_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    architecture: str,
    labels: torch.Tensor | None = None,
    return_hidden_states: bool = False,
) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
    """Forward helper shared by flat and factorized architectures."""
    if architecture == "flat":
        return model(  # type: ignore[return-value]
            batch["input_ids"].to(device),  # type: ignore[union-attr]
            attention_mask=batch["attention_mask"].to(device),  # type: ignore[union-attr]
            labels=labels,
            return_hidden_states=return_hidden_states,
        )
    if architecture == "factorized":
        return model(  # type: ignore[return-value]
            variable_ids=batch["variable_ids"].to(device),  # type: ignore[union-attr]
            sign_ids=batch["sign_ids"].to(device),  # type: ignore[union-attr]
            clause_ids=batch["clause_ids"].to(device),  # type: ignore[union-attr]
            slot_ids=batch["slot_ids"].to(device),  # type: ignore[union-attr]
            token_type_ids=batch["token_type_ids"].to(device),  # type: ignore[union-attr]
            attention_mask=batch["factor_attention_mask"].to(device),  # type: ignore[union-attr]
            labels=labels,
            return_hidden_states=return_hidden_states,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def candidate_windows(n_layers: int, min_len: int, max_len: int) -> list[tuple[int, int]]:
    """Windows accepted by the current hook: ``end`` must name an existing layer."""
    windows: list[tuple[int, int]] = []
    for length in range(min_len, max_len + 1):
        for start in range(0, n_layers - length):
            end = start + length
            if end < n_layers:
                windows.append((start, end))
    return windows


def mean_offdiag(M: pd.DataFrame, start: int, end: int) -> float:
    block = M.loc[start : end - 1, start : end - 1].to_numpy(dtype=float)
    if block.shape[0] < 2:
        return float("nan")
    mask = ~np.eye(block.shape[0], dtype=bool)
    return float(block[mask].mean())


def score_windows(
    M: pd.DataFrame,
    bundle: dict[str, pd.DataFrame],
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    raw_rows = []
    activities = []
    for start, end in windows:
        endpoint = end - 1
        activity = float(bundle["S_norm"].at[start, endpoint]) if endpoint != start else 0.0
        activities.append(activity)
        plateau = 1.0 - mean_offdiag(M, start, end)
        q_value = float(bundle["Q"].at[start, endpoint]) if endpoint != start else 0.0
        raw_rows.append(
            {
                "start": start,
                "end": end,
                "length": end - start,
                "plateau": plateau,
                "activity": activity,
                "Q": q_value,
            }
        )

    median_activity = float(np.median([x for x in activities if x > 0])) if any(x > 0 for x in activities) else 1.0
    rows = []
    for row in raw_rows:
        activity_term = row["activity"] / (row["activity"] + median_activity)
        plateau_term = math.exp(-5.0 * max(row["plateau"], 0.0))
        row["score"] = plateau_term * activity_term
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def split_quality(splits: dict[str, list]) -> pd.DataFrame:
    """Summarize class balance for each split before training."""
    rows = []
    for name, examples in splits.items():
        labels = np.array([ex.label for ex in examples], dtype=int)
        n_pos = int(labels.sum())
        n_total = int(labels.size)
        n_neg = n_total - n_pos
        majority = max(n_pos, n_neg) / n_total
        rows.append(
            {
                "split": name,
                "n_total": n_total,
                "n_sat": n_pos,
                "n_unsat": n_neg,
                "sat_rate": n_pos / n_total,
                "majority_baseline": majority,
            }
        )
    return pd.DataFrame(rows)


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

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
    print(quality.to_json(orient="records"))
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

    config = TinyTransformerConfig(
        vocab_size=vocab_size(max_vars),
        max_seq_len=sequence_length(max_clauses),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )
    factorized_config = FactorizedCNFTransformerConfig(
        max_vars=max_vars,
        max_clauses=max_clauses,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )
    if args.architecture == "flat":
        model = TinySatTransformer(config).to(device)
        saved_config = config.__dict__
    else:
        model = FactorizedCNFSatTransformer(factorized_config).to(device)
        saved_config = factorized_config.__dict__ | {"max_seq_len": factorized_config.max_seq_len}
    t0 = time.time()
    lit = ClassifierLitModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        model_config=saved_config,
    )
    trainer = build_trainer(
        run_dir,
        max_epochs=args.epochs,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
    )
    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    checkpoint_path = best_checkpoint_path(trainer)
    model, _ = load_rys_model(
        checkpoint_path,
        lambda _config: (
            TinySatTransformer(config)
            if args.architecture == "flat"
            else FactorizedCNFSatTransformer(factorized_config)
        ),
        map_location=device,
    )
    model.to(device)

    base_metrics = {
        split: evaluate(model, loader, device, architecture=args.architecture)
        for split, loader in loaders.items()
        if split != "train"
    }

    capture_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    activations = capture_tensor_residual_stream(
        model,
        capture_loader,
        device=device,
        architecture=args.architecture,
        token_strategy="cls",
        max_batches=args.capture_batches,
    )
    M = cka_matrix(activations, unbiased=False, device="cpu")
    bundle = residual_force_matrices(activations, device="cpu")
    force_long = residual_force_long(activations, device="cpu")

    windows = candidate_windows(args.n_layers, args.min_window, args.max_window)
    window_scores = score_windows(M, bundle, windows)
    selected = [
        (int(row.start), int(row.end))
        for row in window_scores.head(args.top_k_windows).itertuples(index=False)
    ]

    cached_eval = {
        split: cache_batches_on_device(loaders[split], device) for split in ("val", "test", "ood")
    }
    rys_rows = []
    for window in log_rys_progress(selected, device=device, n_layers=args.n_layers):
        for n_repeats in range(2, args.max_repeat + 1):
            val = evaluate(
                model,
                cached_eval["val"],
                device,
                architecture=args.architecture,
                window=window,
                n_repeats=n_repeats,
            )
            test = evaluate(
                model,
                cached_eval["test"],
                device,
                architecture=args.architecture,
                window=window,
                n_repeats=n_repeats,
            )
            ood = evaluate(
                model,
                cached_eval["ood"],
                device,
                architecture=args.architecture,
                window=window,
                n_repeats=n_repeats,
            )
            rys_rows.append(
                {
                    "start": window[0],
                    "end": window[1],
                    "n_repeats": n_repeats,
                    "val_accuracy": val["accuracy"],
                    "test_accuracy": test["accuracy"],
                    "ood_accuracy": ood["accuracy"],
                    "val_loss": val["loss"],
                    "kl_base_to_rys": base_to_rys_kl(
                        model,
                        cached_eval["val"],
                        device,
                        architecture=args.architecture,
                        window=window,
                        n_repeats=n_repeats,
                    ),
                }
            )
            print(json.dumps(rys_rows[-1]))

    elapsed = time.time() - t0
    quality.to_csv(run_dir / "dataset_quality.csv", index=False)
    pd.DataFrame(rys_rows).to_csv(run_dir / "rys_window_metrics.csv", index=False)
    window_scores.to_csv(run_dir / "window_scores.csv", index=False)
    force_long.to_parquet(run_dir / "residual_force_long.parquet", index=False)
    activations.to_parquet(run_dir / "activations.parquet", index=False)
    M.to_parquet(run_dir / "cka.parquet")

    payload = {
        "args": vars(args),
        "device": str(device),
        "config": saved_config,
        "checkpoint": str(checkpoint_path),
        "dataset_quality": quality.to_dict(orient="records"),
        "base_metrics": base_metrics,
        "selected_windows": selected,
        "elapsed_s": elapsed,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
