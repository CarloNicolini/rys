"""Depth-controlled SAT assignment-generation experiment for RYS."""

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
from rys.sat_data import (
    SatAssignmentDataset,
    make_sat_assignment_examples,
    make_sat_assignment_splits,
    verify_assignment_tensor,
)
from rys.tiny_transformer import (
    FactorizedAssignmentTransformerConfig,
    FactorizedCNFAssignmentTransformer,
)
from rys.training.modules import AssignmentLitModule
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(41, help="Random seed."),
    depths: str = typer.Option("8,16,32", help="Comma-separated transformer depths (layers) to sweep."),
    n_vars: int = typer.Option(6, help="In-distribution variable count."),
    n_clauses: int = typer.Option(24, help="In-distribution clause count."),
    ood_vars: int = typer.Option(8, help="Out-of-distribution variable count."),
    ood_clauses: int = typer.Option(34, help="Out-of-distribution clause count."),
    n_train: int = typer.Option(4096, help="Training examples."),
    n_val: int = typer.Option(1024, help="Validation examples."),
    n_test: int = typer.Option(1024, help="Test (and OOD) examples."),
    batch_size: int = typer.Option(128, help="Batch size."),
    epochs: int = typer.Option(20, help="Training epochs."),
    lr: float = typer.Option(3e-4, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-2, help="AdamW weight decay."),
    d_model: int = typer.Option(64, help="Model width (must be divisible by n-heads)."),
    n_heads: int = typer.Option(4, help="Attention heads per layer."),
    d_mlp: int = typer.Option(128, help="MLP hidden width."),
    dropout: float = typer.Option(0.0, help="Dropout probability."),
    max_repeat: int = typer.Option(2, help="Total traversals of each RYS window."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    sat_loss_weight: float = typer.Option(1.0, help="Weight on the soft-SAT loss term."),
    ce_loss_weight: float = typer.Option(0.0, help="Weight on the canonical-assignment cross-entropy term."),
    skip_rys: bool = typer.Option(
        False, "--skip-rys/--no-skip-rys", help="Train and export CKA/baselines without RYS matrices."
    ),
    output_dir: Path = typer.Option(Path("results/sat_assignment_generation"), help="Run output directory."),
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_depths(value: str) -> list[int]:
    depths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not depths or any(depth < 2 for depth in depths):
        raise ValueError("Depths must contain integers >= 2.")
    return depths


def split_quality(splits: dict[str, list]) -> pd.DataFrame:
    rows = []
    for split, examples in splits.items():
        assignments = np.array([ex.assignment for ex in examples], dtype=float)
        rows.append(
            {
                "split": split,
                "n_total": len(examples),
                "n_vars": examples[0].n_vars,
                "n_clauses": examples[0].n_clauses,
                "assignment_one_rate": float(assignments.mean()),
            }
        )
    return pd.DataFrame(rows)


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], pd.DataFrame, int, int]:
    max_vars = max(args.n_vars, args.ood_vars)
    max_clauses = max(args.n_clauses, args.ood_clauses)
    splits = make_sat_assignment_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_vars=args.n_vars,
        n_clauses=args.n_clauses,
        seed=args.seed,
    )
    splits["ood"] = make_sat_assignment_examples(
        args.n_test,
        n_vars=args.ood_vars,
        n_clauses=args.ood_clauses,
        seed=args.seed + 3,
        prefix="ood",
    )
    quality = split_quality(splits)
    datasets = {
        name: SatAssignmentDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
        for name, examples in splits.items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, quality, max_vars, max_clauses


def build_model(
    args: argparse.Namespace,
    *,
    n_layers: int,
    max_vars: int,
    max_clauses: int,
) -> tuple[FactorizedCNFAssignmentTransformer, dict]:
    config = FactorizedAssignmentTransformerConfig(
        max_vars=max_vars,
        max_clauses=max_clauses,
        d_model=args.d_model,
        n_layers=n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )
    return FactorizedCNFAssignmentTransformer(config), config.__dict__ | {
        "formula_seq_len": config.formula_seq_len,
        "max_seq_len": config.max_seq_len,
    }


def model_kwargs(batch: dict[str, torch.Tensor | list[str]], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "variable_ids": batch["variable_ids"].to(device),  # type: ignore[union-attr]
        "sign_ids": batch["sign_ids"].to(device),  # type: ignore[union-attr]
        "clause_ids": batch["clause_ids"].to(device),  # type: ignore[union-attr]
        "slot_ids": batch["slot_ids"].to(device),  # type: ignore[union-attr]
        "token_type_ids": batch["token_type_ids"].to(device),  # type: ignore[union-attr]
        "attention_mask": batch["factor_attention_mask"].to(device),  # type: ignore[union-attr]
    }


def assignment_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    clause_variable_ids: torch.Tensor,
    clause_sign_ids: torch.Tensor,
    clause_mask: torch.Tensor,
) -> dict[str, int]:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels) & mask
    bit_total = int(mask.sum().detach().cpu())
    exact = ((preds == labels) | ~mask).all(dim=1)
    valid = verify_assignment_tensor(preds, clause_variable_ids, clause_sign_ids, clause_mask)
    return {
        "bit_correct": int(correct.sum().detach().cpu()),
        "bit_total": bit_total,
        "exact_correct": int(exact.sum().detach().cpu()),
        "valid_correct": int(valid.sum().detach().cpu()),
        "example_total": int(labels.shape[0]),
    }


def accumulate_metrics(total_loss: float, counts: dict[str, int]) -> dict[str, float]:
    return {
        "loss": total_loss / counts["example_total"],
        "bit_accuracy": counts["bit_correct"] / counts["bit_total"],
        "exact_match": counts["exact_correct"] / counts["example_total"],
        "valid_assignment_rate": counts["valid_correct"] / counts["example_total"],
    }


def soft_sat_loss(
    logits: torch.Tensor,
    clause_variable_ids: torch.Tensor,
    clause_sign_ids: torch.Tensor,
    clause_mask: torch.Tensor,
) -> torch.Tensor:
    """Differentiable relaxation of the hard SAT verifier."""
    p_true = logits.softmax(dim=-1)[..., 1]
    gather_idx = (clause_variable_ids - 1).clamp_min(0)
    expanded = p_true[:, None, :].expand(-1, gather_idx.shape[1], -1)
    literal_true_prob = torch.gather(expanded, dim=2, index=gather_idx)
    literal_sat_prob = torch.where(clause_sign_ids == 1, literal_true_prob, 1.0 - literal_true_prob)
    clause_unsat_prob = torch.prod(1.0 - literal_sat_prob, dim=-1)
    clause_sat_prob = (1.0 - clause_unsat_prob).clamp_min(1e-6)
    losses = -torch.log(clause_sat_prob) * clause_mask
    return losses.sum() / clause_mask.sum().clamp_min(1)


def train_epoch(
    model: FactorizedCNFAssignmentTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    sat_loss_weight: float,
    ce_loss_weight: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    counts = {
        "bit_correct": 0,
        "bit_total": 0,
        "exact_correct": 0,
        "valid_correct": 0,
        "example_total": 0,
    }
    for batch in loader:
        labels = batch["assignment_labels"].to(device)
        mask = batch["assignment_mask"].to(device)
        clause_variable_ids = batch["clause_variable_ids"].to(device)
        clause_sign_ids = batch["clause_sign_ids"].to(device)
        clause_mask = batch["clause_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**model_kwargs(batch, device))
        logits = outputs["logits"]
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Training outputs are missing.")
        sat_loss = soft_sat_loss(logits, clause_variable_ids, clause_sign_ids, clause_mask)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss = sat_loss_weight * sat_loss + ce_loss_weight * ce_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        for key, value in assignment_metrics(
            logits,
            labels,
            mask,
            clause_variable_ids,
            clause_sign_ids,
            clause_mask,
        ).items():
            counts[key] += value
    return accumulate_metrics(total_loss, counts)


@torch.inference_mode()
def evaluate(
    model: FactorizedCNFAssignmentTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    rys_window: tuple[int, int] | None = None,
    n_repeats: int = 2,
    sat_loss_weight: float = 1.0,
    ce_loss_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    counts = {
        "bit_correct": 0,
        "bit_total": 0,
        "exact_correct": 0,
        "valid_correct": 0,
        "example_total": 0,
    }
    for batch in loader:
        labels = batch["assignment_labels"].to(device)
        mask = batch["assignment_mask"].to(device)
        clause_variable_ids = batch["clause_variable_ids"].to(device)
        clause_sign_ids = batch["clause_sign_ids"].to(device)
        clause_mask = batch["clause_mask"].to(device)
        if rys_window is None or n_repeats == 1:
            outputs = model(**model_kwargs(batch, device))
            logits = outputs["logits"]
        else:
            logits = forward_logits_with_inclusive_rys(
                model,
                batch,
                device,
                window=rys_window,
                n_repeats=n_repeats,
            )
        if not isinstance(logits, torch.Tensor):
            raise RuntimeError("Evaluation logits are missing.")
        sat_loss = soft_sat_loss(logits, clause_variable_ids, clause_sign_ids, clause_mask)
        ce_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 2),
            labels.reshape(-1),
            ignore_index=-100,
        )
        loss = sat_loss_weight * sat_loss + ce_loss_weight * ce_loss
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        for key, value in assignment_metrics(
            logits,
            labels,
            mask,
            clause_variable_ids,
            clause_sign_ids,
            clause_mask,
        ).items():
            counts[key] += value
    return accumulate_metrics(total_loss, counts)


def embed_batch(
    model: FactorizedCNFAssignmentTransformer,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    backbone = model.model
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
    model: FactorizedCNFAssignmentTransformer,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    window: tuple[int, int],
    n_repeats: int,
) -> torch.Tensor:
    start, end = window
    layers = model.model.layers
    if start >= end:
        raise ValueError(f"RYS window must have start < end; got {window}.")
    if not (0 <= start < end < len(layers)):
        raise ValueError(f"Bad inclusive window {window} for L={len(layers)}.")

    hidden, attention_mask = embed_batch(model, batch, device)
    for layer_idx, layer in enumerate(layers):
        hidden = layer(hidden, attention_mask=attention_mask)[0]
        if layer_idx == end:
            for _ in range(n_repeats - 1):
                for replay_idx in range(start, end + 1):
                    hidden = layers[replay_idx](hidden, attention_mask=attention_mask)[0]
    hidden = model.model.norm(hidden)
    query_start = model.config.formula_seq_len
    query_states = hidden[:, query_start : query_start + model.config.max_vars, :]
    return model.assignment_head(query_states)


def strict_upper_windows(n_layers: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]


def delta_matrices(
    model: FactorizedCNFAssignmentTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    n_layers: int,
    n_repeats: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], pd.DataFrame]:
    baseline = evaluate(model, loader, device)
    exact_matrix = np.full((n_layers, n_layers), np.nan, dtype=float)
    bit_matrix = np.full((n_layers, n_layers), np.nan, dtype=float)
    valid_matrix = np.full((n_layers, n_layers), np.nan, dtype=float)
    rows = []
    for start, end in strict_upper_windows(n_layers):
        metrics = evaluate(model, loader, device, rys_window=(start, end), n_repeats=n_repeats)
        delta_exact = metrics["exact_match"] - baseline["exact_match"]
        delta_bit = metrics["bit_accuracy"] - baseline["bit_accuracy"]
        delta_valid = metrics["valid_assignment_rate"] - baseline["valid_assignment_rate"]
        exact_matrix[start, end] = delta_exact
        bit_matrix[start, end] = delta_bit
        valid_matrix[start, end] = delta_valid
        row = {
            "start": start,
            "end": end,
            "n_repeats": n_repeats,
            "baseline_bit_accuracy": baseline["bit_accuracy"],
            "baseline_exact_match": baseline["exact_match"],
            "baseline_valid_assignment_rate": baseline["valid_assignment_rate"],
            "rys_bit_accuracy": metrics["bit_accuracy"],
            "rys_exact_match": metrics["exact_match"],
            "rys_valid_assignment_rate": metrics["valid_assignment_rate"],
            "delta_bit_accuracy": delta_bit,
            "delta_exact_match": delta_exact,
            "delta_valid_assignment_rate": delta_valid,
            "rys_loss": metrics["loss"],
        }
        rows.append(row)
        print(json.dumps(row))
    matrices = {
        "exact_match": pd.DataFrame(exact_matrix),
        "bit_accuracy": pd.DataFrame(bit_matrix),
        "valid_assignment_rate": pd.DataFrame(valid_matrix),
    }
    return matrices, baseline, pd.DataFrame(rows)


@torch.inference_mode()
def capture_query_residual_stream(
    model: FactorizedCNFAssignmentTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> pd.DataFrame:
    rows = []
    model.eval()
    query_start = model.config.formula_seq_len
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        outputs = model(**model_kwargs(batch, device), return_hidden_states=True)
        hidden_states = outputs["hidden_states"]
        if not isinstance(hidden_states, list):
            raise RuntimeError("Hidden states are missing.")
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        mask = batch["assignment_mask"].bool()
        for layer_idx, hidden in enumerate(hidden_states):
            query_hidden = hidden[:, query_start : query_start + model.config.max_vars, :]
            for row_idx, prompt_id in enumerate(prompt_ids):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "layer": layer_idx,
                        "activation": query_hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                        "strategy": "query",
                    }
                )
    return pd.DataFrame(rows)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
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


def train_one_depth(
    args: argparse.Namespace,
    *,
    n_layers: int,
    loaders: dict[str, DataLoader],
    quality: pd.DataFrame,
    max_vars: int,
    max_clauses: int,
    device: torch.device,
    run_dir: Path,
) -> dict:
    depth_dir = run_dir / f"L{n_layers:02d}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(depth_dir / "dataset_quality.csv", index=False)
    model, config = build_model(args, n_layers=n_layers, max_vars=max_vars, max_clauses=max_clauses)
    model.to(device)
    lit = AssignmentLitModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sat_loss_weight=args.sat_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
        model_config=config,
    )
    trainer = build_trainer(
        depth_dir,
        max_epochs=args.epochs,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
        extra_log_fields={"depth": n_layers},
    )
    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    checkpoint_path = best_checkpoint_path(trainer)

    reloaded, _ = load_rys_model(
        checkpoint_path,
        lambda _config: build_model(args, n_layers=n_layers, max_vars=max_vars, max_clauses=max_clauses)[0],
        map_location=device,
    )
    reloaded.to(device)

    activations = capture_query_residual_stream(
        reloaded,
        loaders["val"],
        device,
        max_batches=args.capture_batches,
    )
    activations.to_pickle(depth_dir / "query_activations.pkl")
    cka = cka_matrix(activations, unbiased=False, device="cpu")
    cka.to_csv(depth_dir / "cka_query_val.csv")
    save_heatmap(
        cka,
        depth_dir / "cka_query_val.png",
        title=f"Query CKA validation (L={n_layers})",
        cbar_label="CKA",
        cmap="viridis",
        diverging=False,
    )

    baselines = {
        split: evaluate(
            reloaded,
            loaders[split],
            device,
            sat_loss_weight=args.sat_loss_weight,
            ce_loss_weight=args.ce_loss_weight,
        )
        for split in ("val", "test", "ood")
    }
    if args.skip_rys:
        summary = {
            "depth": n_layers,
            "config": config,
            "checkpoint": str(checkpoint_path),
            "baselines": baselines,
            "best_rows": [],
            "n_rys_windows": 0,
            "rys_skipped": True,
        }
        (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    best_rows = []
    for split in ("val", "test", "ood"):
        matrices, baseline, long = delta_matrices(
            reloaded,
            loaders[split],
            device,
            n_layers=n_layers,
            n_repeats=args.max_repeat,
        )
        baselines[split] = baseline
        matrices["exact_match"].to_csv(depth_dir / f"delta_exact_{split}.csv")
        matrices["bit_accuracy"].to_csv(depth_dir / f"delta_bit_{split}.csv")
        matrices["valid_assignment_rate"].to_csv(depth_dir / f"delta_valid_{split}.csv")
        long.to_csv(depth_dir / f"delta_assignment_{split}_long.csv", index=False)
        save_heatmap(
            matrices["valid_assignment_rate"],
            depth_dir / f"delta_valid_{split}.png",
            title=f"RYS ΔValid Assignment {split} (L={n_layers}, i<j)",
            cbar_label="Δ valid assignment rate",
        )
        save_heatmap(
            matrices["exact_match"],
            depth_dir / f"delta_exact_{split}.png",
            title=f"RYS ΔExact Match {split} (L={n_layers}, i<j)",
            cbar_label="Δ exact match",
        )
        save_heatmap(
            matrices["bit_accuracy"],
            depth_dir / f"delta_bit_{split}.png",
            title=f"RYS ΔBit Accuracy {split} (L={n_layers}, i<j)",
            cbar_label="Δ bit accuracy",
        )
        best = long.loc[long["delta_valid_assignment_rate"].idxmax()].to_dict()
        best["split"] = split
        best_rows.append(best)

    summary = {
        "depth": n_layers,
        "config": config,
        "checkpoint": str(checkpoint_path),
        "baselines": baselines,
        "best_rows": best_rows,
        "n_rys_windows": len(strict_upper_windows(n_layers)),
        "rys_skipped": False,
    }
    (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def write_report(run_dir: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# SAT Assignment Generation Depth Sweep",
        "",
        "Task: produce any assignment that satisfies a satisfiable 3-SAT formula.",
        "Inputs include fixed query tokens, one per variable, so the final layers must prepare per-variable logits.",
        "The primary metric is valid-assignment rate; exact match to the canonical brute-force assignment is diagnostic only.",
        "",
        f"- Depths: `{args.depths}`",
        f"- Train/val/test/OOD: `{args.n_train}/{args.n_val}/{args.n_test}/{args.n_test}`",
        f"- In-distribution: `{args.n_vars}` vars, `{args.n_clauses}` clauses",
        f"- OOD: `{args.ood_vars}` vars, `{args.ood_clauses}` clauses",
        f"- Loss: `{args.sat_loss_weight}` * soft SAT + `{args.ce_loss_weight}` * canonical CE",
        "",
        "| depth | val valid | test valid | OOD valid | val exact | test exact | OOD exact | best val window | best test window | best OOD window |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for summary in summaries:
        best = {row["split"]: row for row in summary["best_rows"]}
        baselines = summary["baselines"]

        def _window(split: str, *, current_summary: dict = summary, current_best: dict = best) -> str:
            if current_summary.get("rys_skipped"):
                return "skipped"
            row = current_best[split]
            return f"({int(row['start'])}, {int(row['end'])}) Δ={row['delta_valid_assignment_rate']:+.3f}"

        lines.append(
            f"| {summary['depth']} | "
            f"{baselines['val']['valid_assignment_rate']:.3f} | "
            f"{baselines['test']['valid_assignment_rate']:.3f} | "
            f"{baselines['ood']['valid_assignment_rate']:.3f} | "
            f"{baselines['val']['exact_match']:.3f} | "
            f"{baselines['test']['exact_match']:.3f} | "
            f"{baselines['ood']['exact_match']:.3f} | "
            f"{_window('val')} | {_window('test')} | {_window('ood')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    depths = parse_depths(args.depths)
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, quality, max_vars, max_clauses = make_loaders(args)
    quality.to_csv(run_dir / "dataset_quality.csv", index=False)
    summaries = []
    for depth in depths:
        summaries.append(
            train_one_depth(
                args,
                n_layers=depth,
                loaders=loaders,
                quality=quality,
                max_vars=max_vars,
                max_clauses=max_clauses,
                device=device,
                run_dir=run_dir,
            )
        )
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "summaries": summaries}, indent=2, default=str))
    write_report(run_dir, summaries, args)
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
