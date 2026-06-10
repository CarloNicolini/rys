"""Depth-controlled clause-variable message-passing SAT experiment for RYS.

The solver is trained to produce *any* satisfying assignment (validity), not a
canonical one.  For each trained depth the script records:

- a validity-vs-rounds curve (assignment quality read out after each round);
- the validation CKA connectome over variable states;
- strict upper-triangular RYS Delta matrices (validity, bit, exact).

By default, formulas are sampled synthetically.  Pass ``--labels-csv`` to train on
real RandSATBench 3-SAT instances (CaDiCaL labels); CNF files are resolved under
``--data-root`` (defaults to the labels file's parent directory).
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
from rys.randsat_data import make_randsat_assignment_splits
from rys.sat_data import (
    SatAssignmentDataset,
    make_sat_assignment_examples,
    make_sat_assignment_splits,
    soft_sat_loss,
    verify_assignment_tensor,
)

_DEFAULT_RANDSAT_ROOT = Path("~/workspace/RandSATBench/datasets/3SAT").expanduser()
from rys.sat_message_passing import MessagePassingConfig, MessagePassingSatModel
from rys.training.modules import SatMPLitModule
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(43, help="Random seed."),
    depths: str = typer.Option("8,16,32", help="Comma-separated message-passing depths (rounds) to sweep."),
    n_vars: int = typer.Option(6, help="In-distribution variable count."),
    n_clauses: int = typer.Option(24, help="In-distribution clause count."),
    ood_vars: int = typer.Option(8, help="Out-of-distribution variable count."),
    ood_clauses: int = typer.Option(34, help="Out-of-distribution clause count."),
    n_train: int = typer.Option(4096, help="Training examples."),
    n_val: int = typer.Option(1024, help="Validation examples."),
    n_test: int = typer.Option(1024, help="Test (and OOD) examples."),
    batch_size: int = typer.Option(128, help="Batch size."),
    epochs: int = typer.Option(20, help="Training epochs."),
    lr: float = typer.Option(5e-4, help="AdamW learning rate."),
    weight_decay: float = typer.Option(1e-2, help="AdamW weight decay."),
    d_model: int = typer.Option(64, help="Model width."),
    d_mlp: int = typer.Option(128, help="MLP hidden width."),
    dropout: float = typer.Option(0.0, help="Dropout probability."),
    max_repeat: int = typer.Option(2, help="Total traversals of each RYS window."),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    deep_supervision: bool = typer.Option(
        True, "--deep-supervision/--no-deep-supervision", help="Average the soft-SAT loss over every round."
    ),
    skip_rys: bool = typer.Option(
        False, "--skip-rys/--no-skip-rys", help="Train and export curves/CKA without RYS matrices."
    ),
    checkpoint: Path | None = typer.Option(
        None,
        help="Load these weights and skip training (single --depths value). "
        "Lets you re-run the RYS sweep with a different --max-repeat without retraining.",
    ),
    labels_csv: Path | None = typer.Option(
        None,
        help="RandSATBench labels CSV (e.g. train_labels.csv). When set, load real 3-SAT "
        "instances instead of synthetic formulas.",
    ),
    data_root: Path | None = typer.Option(
        None,
        help="RandSATBench dataset root for CNF resolution (defaults to labels CSV parent).",
    ),
    indist_vars: str = typer.Option(
        "16,32",
        help="Variable counts N for the in-distribution pool (RandSATBench mode only).",
    ),
    randsat_ood_vars: str = typer.Option(
        "64",
        help="Variable counts N for the OOD split (RandSATBench mode only).",
    ),
    val_frac: float = typer.Option(0.1, help="Validation fraction of the in-dist pool (RandSATBench only)."),
    test_frac: float = typer.Option(0.1, help="Test fraction of the in-dist pool (RandSATBench only)."),
    max_indist: int | None = typer.Option(
        None,
        help="Cap on in-distribution examples, not variables (RandSATBench only).",
    ),
    max_ood: int | None = typer.Option(
        None,
        help="Cap on OOD examples, not variables (RandSATBench only).",
    ),
    num_workers: int = typer.Option(4, help="DataLoader worker processes (RandSATBench only)."),
    pin_memory: bool = typer.Option(
        True, "--pin-memory/--no-pin-memory", help="Pin host memory for CUDA (RandSATBench only)."
    ),
    output_dir: Path = typer.Option(Path("results/sat_messagepassing"), help="Run output directory."),
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


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one integer in {value!r}.")
    return values


def split_quality(splits: dict[str, list]) -> pd.DataFrame:
    rows = []
    for split, examples in splits.items():
        if not examples:
            raise ValueError(f"Split {split!r} is empty.")
        n_vars_values = sorted({ex.n_vars for ex in examples})
        n_clauses_values = [ex.n_clauses for ex in examples]
        assignments = [value for ex in examples for value in ex.assignment]
        rows.append(
            {
                "split": split,
                "n_total": len(examples),
                "n_vars_values": ",".join(str(v) for v in n_vars_values),
                "n_clauses_min": min(n_clauses_values),
                "n_clauses_max": max(n_clauses_values),
                "assignment_one_rate": float(sum(assignments) / len(assignments)),
            }
        )
    return pd.DataFrame(rows)


def _resolve_data_root(args: argparse.Namespace) -> Path:
    if args.data_root is not None:
        return args.data_root
    if args.labels_csv is not None:
        return args.labels_csv.parent
    return _DEFAULT_RANDSAT_ROOT


def _build_dataloaders(
    datasets: dict[str, SatAssignmentDataset],
    args: argparse.Namespace,
    *,
    randsat_mode: bool,
) -> dict[str, DataLoader]:
    loader_kwargs: dict = {}
    if randsat_mode:
        pin_memory = args.pin_memory and torch.cuda.is_available()
        loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin_memory}
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
    return {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, **loader_kwargs),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False, **loader_kwargs),
    }


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], int, int, pd.DataFrame | None]:
    if args.labels_csv is not None:
        data_root = _resolve_data_root(args)
        splits = make_randsat_assignment_splits(
            data_root,
            labels_csv=args.labels_csv,
            indist_vars=parse_int_list(args.indist_vars),
            ood_vars=parse_int_list(args.randsat_ood_vars),
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            max_indist=args.max_indist,
            max_ood=args.max_ood,
            seed=args.seed,
        )
        quality = split_quality(splits)
        max_vars = max(ex.n_vars for split in splits.values() for ex in split)
        max_clauses = max(ex.n_clauses for split in splits.values() for ex in split)
        datasets = {
            name: SatAssignmentDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
            for name, examples in splits.items()
        }
        loaders = _build_dataloaders(datasets, args, randsat_mode=True)
        return loaders, max_vars, max_clauses, quality

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
    datasets = {
        name: SatAssignmentDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
        for name, examples in splits.items()
    }
    loaders = _build_dataloaders(datasets, args, randsat_mode=False)
    return loaders, max_vars, max_clauses, None


def build_model(args: argparse.Namespace, *, n_rounds: int, max_vars: int, max_clauses: int) -> tuple[MessagePassingSatModel, dict]:
    config = MessagePassingConfig(
        max_vars=max_vars,
        max_clauses=max_clauses,
        d_model=args.d_model,
        n_rounds=n_rounds,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
    )
    return MessagePassingSatModel(config), config.__dict__


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def clause_tensors(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["clause_variable_ids"].to(device),
        batch["clause_sign_ids"].to(device),
        batch["clause_mask"].to(device),
    )


def batch_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    clause_variable_ids: torch.Tensor,
    clause_sign_ids: torch.Tensor,
    clause_mask: torch.Tensor,
) -> dict[str, int]:
    preds = logits.argmax(dim=-1)
    bit_correct = int(((preds == labels) & mask).sum().detach().cpu())
    exact = ((preds == labels) | ~mask).all(dim=1)
    valid = verify_assignment_tensor(preds, clause_variable_ids, clause_sign_ids, clause_mask)
    return {
        "bit_correct": bit_correct,
        "bit_total": int(mask.sum().detach().cpu()),
        "exact_correct": int(exact.sum().detach().cpu()),
        "valid_correct": int(valid.sum().detach().cpu()),
        "example_total": int(labels.shape[0]),
    }


def accumulate(total_loss: float, counts: dict[str, int]) -> dict[str, float]:
    return {
        "loss": total_loss / counts["example_total"],
        "bit_accuracy": counts["bit_correct"] / counts["bit_total"],
        "exact_match": counts["exact_correct"] / counts["example_total"],
        "valid_assignment_rate": counts["valid_correct"] / counts["example_total"],
    }


def empty_counts() -> dict[str, int]:
    return {"bit_correct": 0, "bit_total": 0, "exact_correct": 0, "valid_correct": 0, "example_total": 0}


def supervised_loss(
    round_logits: list[torch.Tensor],
    final_logits: torch.Tensor,
    cvi: torch.Tensor,
    csi: torch.Tensor,
    cm: torch.Tensor,
    *,
    deep_supervision: bool,
) -> torch.Tensor:
    if deep_supervision and round_logits:
        terms = [soft_sat_loss(logit, cvi, csi, cm) for logit in round_logits]
        return torch.stack(terms).mean()
    return soft_sat_loss(final_logits, cvi, csi, cm)


def train_epoch(
    model: MessagePassingSatModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    deep_supervision: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    counts = empty_counts()
    for batch in loader:
        labels = batch["assignment_labels"].to(device)
        mask = batch["assignment_mask"].to(device)
        cvi, csi, cm = clause_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            clause_variable_ids=cvi,
            clause_sign_ids=csi,
            clause_mask=cm,
            return_round_logits=deep_supervision,
        )
        logits = outputs["logits"]
        loss = supervised_loss(
            outputs["round_logits"] or [],
            logits,
            cvi,
            csi,
            cm,
            deep_supervision=deep_supervision,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        for key, value in batch_metrics(logits, labels, mask, cvi, csi, cm).items():
            counts[key] += value
    return accumulate(total_loss, counts)


@torch.inference_mode()
def evaluate(
    model: MessagePassingSatModel,
    loader: DataLoader,
    device: torch.device,
    *,
    rys_window: tuple[int, int] | None = None,
    n_repeats: int = 2,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    counts = empty_counts()
    for batch in loader:
        labels = batch["assignment_labels"].to(device)
        mask = batch["assignment_mask"].to(device)
        cvi, csi, cm = clause_tensors(batch, device)
        if rys_window is None or n_repeats == 1:
            logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm)["logits"]
        else:
            logits = forward_logits_with_inclusive_rys(model, cvi, csi, cm, window=rys_window, n_repeats=n_repeats)
        loss = soft_sat_loss(logits, cvi, csi, cm)
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        for key, value in batch_metrics(logits, labels, mask, cvi, csi, cm).items():
            counts[key] += value
    return accumulate(total_loss, counts)


def forward_logits_with_inclusive_rys(
    model: MessagePassingSatModel,
    cvi: torch.Tensor,
    csi: torch.Tensor,
    cm: torch.Tensor,
    *,
    window: tuple[int, int],
    n_repeats: int,
) -> torch.Tensor:
    start, end = window
    backbone = model.model
    layers = backbone.layers
    if start >= end:
        raise ValueError(f"RYS window must have start < end; got {window}.")
    if not (0 <= start < end < len(layers)):
        raise ValueError(f"Bad inclusive window {window} for L={len(layers)}.")

    context = backbone.build_context(cvi, csi, cm)
    batch = cvi.shape[0]
    d = backbone.config.d_model
    var_states = backbone.var_init.view(1, 1, d).expand(batch, backbone.config.max_vars, d)
    clause_states = backbone.clause_init.view(1, 1, d).expand(batch, backbone.config.max_clauses, d)
    hidden = torch.cat([var_states, clause_states], dim=1).contiguous()

    for idx, layer in enumerate(layers):
        layer.ctx = context
        hidden = layer(hidden, attention_mask=None)[0]
        if idx == end:
            for _ in range(n_repeats - 1):
                for replay_idx in range(start, end + 1):
                    hidden = layers[replay_idx](hidden, attention_mask=None)[0]
    var_states = backbone.norm(hidden[:, : backbone.config.max_vars])
    return model.assignment_head(var_states)


@torch.inference_mode()
def validity_vs_rounds(
    model: MessagePassingSatModel,
    loader: DataLoader,
    device: torch.device,
    *,
    n_rounds: int,
) -> pd.DataFrame:
    model.eval()
    per_round = [empty_counts() for _ in range(n_rounds)]
    for batch in loader:
        labels = batch["assignment_labels"].to(device)
        mask = batch["assignment_mask"].to(device)
        cvi, csi, cm = clause_tensors(batch, device)
        outputs = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, return_round_logits=True)
        for round_idx, logits in enumerate(outputs["round_logits"]):
            for key, value in batch_metrics(logits, labels, mask, cvi, csi, cm).items():
                per_round[round_idx][key] += value
    rows = []
    for round_idx, counts in enumerate(per_round):
        metrics = accumulate(0.0, counts)
        rows.append(
            {
                "round": round_idx + 1,
                "valid_assignment_rate": metrics["valid_assignment_rate"],
                "bit_accuracy": metrics["bit_accuracy"],
                "exact_match": metrics["exact_match"],
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def capture_variable_states(
    model: MessagePassingSatModel,
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
        cvi, csi, cm = clause_tensors(batch, device)
        outputs = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, return_hidden_states=True)
        trace = outputs["hidden_states"]
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        mask = batch["assignment_mask"].bool()
        for layer_idx, hidden in enumerate(trace):
            for row_idx, prompt_id in enumerate(prompt_ids):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "layer": layer_idx,
                        "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                        "strategy": "variable",
                    }
                )
    return pd.DataFrame(rows)


def strict_upper_windows(n_layers: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]


def delta_matrices(
    model: MessagePassingSatModel,
    loader: DataLoader,
    device: torch.device,
    *,
    n_layers: int,
    n_repeats: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, float], pd.DataFrame]:
    baseline = evaluate(model, loader, device)
    matrices = {key: np.full((n_layers, n_layers), np.nan) for key in ("valid_assignment_rate", "bit_accuracy", "exact_match")}
    rows = []
    for start, end in strict_upper_windows(n_layers):
        metrics = evaluate(model, loader, device, rys_window=(start, end), n_repeats=n_repeats)
        for key in matrices:
            matrices[key][start, end] = metrics[key] - baseline[key]
        rows.append(
            {
                "start": start,
                "end": end,
                "n_repeats": n_repeats,
                "baseline_valid_assignment_rate": baseline["valid_assignment_rate"],
                "rys_valid_assignment_rate": metrics["valid_assignment_rate"],
                "delta_valid_assignment_rate": metrics["valid_assignment_rate"] - baseline["valid_assignment_rate"],
                "delta_bit_accuracy": metrics["bit_accuracy"] - baseline["bit_accuracy"],
                "delta_exact_match": metrics["exact_match"] - baseline["exact_match"],
                "rys_loss": metrics["loss"],
            }
        )
        print(json.dumps(rows[-1]))
    frames = {key: pd.DataFrame(value) for key, value in matrices.items()}
    return frames, baseline, pd.DataFrame(rows)


def save_heatmap(matrix: pd.DataFrame, path: Path, *, title: str, cbar_label: str, cmap: str = "RdBu_r", diverging: bool = True) -> None:
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
    ax.set_xlabel("end round j (inclusive)")
    ax.set_ylabel("start round i")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_round_curve(curves: dict[str, pd.DataFrame], path: Path, *, depth: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, frame in curves.items():
        ax.plot(frame["round"], frame["valid_assignment_rate"], marker="o", label=split)
    ax.set_title(f"Validity vs message-passing rounds (L={depth})")
    ax.set_xlabel("round")
    ax.set_ylabel("valid assignment rate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_one_depth(
    args: argparse.Namespace,
    *,
    n_rounds: int,
    loaders: dict[str, DataLoader],
    max_vars: int,
    max_clauses: int,
    device: torch.device,
    run_dir: Path,
) -> dict:
    depth_dir = run_dir / f"L{n_rounds:02d}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    model, config = build_model(args, n_rounds=n_rounds, max_vars=max_vars, max_clauses=max_clauses)
    model.to(device)
    params = count_parameters(model)
    print(json.dumps({"depth": n_rounds, "n_params_total": params["total"], "n_params_trainable": params["trainable"]}))

    checkpoint_path = depth_dir / "best.ckpt"
    if args.checkpoint is not None:
        # Inference-only: load existing weights and skip training so the RYS
        # sweep can be re-run with a different --max-repeat at no training cost.
        model, _ = load_rys_model(
            args.checkpoint,
            lambda _config: build_model(args, n_rounds=n_rounds, max_vars=max_vars, max_clauses=max_clauses)[0],
            map_location=device,
        )
        checkpoint_path = args.checkpoint
        model.to(device)
        print(json.dumps({"depth": n_rounds, "loaded_checkpoint": str(args.checkpoint)}))
    else:
        lit = SatMPLitModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            deep_supervision=args.deep_supervision,
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
        checkpoint_path = best_checkpoint_path(trainer)

    reloaded, _ = load_rys_model(
        checkpoint_path,
        lambda _config: build_model(args, n_rounds=n_rounds, max_vars=max_vars, max_clauses=max_clauses)[0],
        map_location=device,
    )
    reloaded.to(device)

    curves = {
        split: validity_vs_rounds(reloaded, loaders[split], device, n_rounds=n_rounds)
        for split in ("val", "test", "ood")
    }
    for split, frame in curves.items():
        frame.to_csv(depth_dir / f"validity_vs_rounds_{split}.csv", index=False)
    save_round_curve(curves, depth_dir / "validity_vs_rounds.png", depth=n_rounds)

    activations = capture_variable_states(reloaded, loaders["val"], device, max_batches=args.capture_batches)
    activations.to_pickle(depth_dir / "variable_activations.pkl")
    cka = cka_matrix(activations, unbiased=False, device="cpu")
    cka.to_csv(depth_dir / "cka_variable_val.csv")
    save_heatmap(
        cka,
        depth_dir / "cka_variable_val.png",
        title=f"Variable-state CKA (L={n_rounds})",
        cbar_label="CKA",
        cmap="viridis",
        diverging=False,
    )

    baselines = {split: evaluate(reloaded, loaders[split], device) for split in ("val", "test", "ood")}
    if args.skip_rys:
        summary = {
            "depth": n_rounds,
            "config": config,
            "n_params": params,
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
        frames, baseline, long = delta_matrices(reloaded, loaders[split], device, n_layers=n_rounds, n_repeats=args.max_repeat)
        baselines[split] = baseline
        long.to_csv(depth_dir / f"delta_{split}_long.csv", index=False)
        frames["valid_assignment_rate"].to_csv(depth_dir / f"delta_valid_{split}.csv")
        save_heatmap(
            frames["valid_assignment_rate"],
            depth_dir / f"delta_valid_{split}.png",
            title=f"RYS ΔValidity {split} (L={n_rounds}, i<j)",
            cbar_label="Δ valid assignment rate",
        )
        best = long.loc[long["delta_valid_assignment_rate"].idxmax()].to_dict()
        best["split"] = split
        best_rows.append(best)

    summary = {
        "depth": n_rounds,
        "config": config,
        "n_params": params,
        "checkpoint": str(checkpoint_path),
        "baselines": baselines,
        "best_rows": best_rows,
        "n_rys_windows": len(strict_upper_windows(n_rounds)),
        "rys_skipped": False,
    }
    (depth_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def write_report(run_dir: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    if args.labels_csv is not None:
        data_root = _resolve_data_root(args)
        lines = [
            "# Clause-Variable Message-Passing SAT Depth Sweep (RandSATBench)",
            "",
            "Task: produce any assignment that satisfies a satisfiable 3-SAT formula.",
            "Solver: permutation-equivariant clause-variable message passing, trained with the soft SAT loss.",
            f"Data source: RandSATBench under `{data_root}` with labels from `{args.labels_csv}`.",
            "Primary metric: valid-assignment rate (verified, not canonical exact match).",
            "",
            f"- Depths (rounds): `{args.depths}`",
            f"- In-distribution vars: `{args.indist_vars}`",
            f"- OOD vars: `{args.randsat_ood_vars}`",
            f"- Val/test fractions: `{args.val_frac}` / `{args.test_frac}`",
            f"- Max in-distribution / OOD examples: `{args.max_indist}` / `{args.max_ood}`",
            f"- Deep supervision: `{args.deep_supervision}`",
            "",
            "| depth | val valid | test valid | OOD valid | best val window | best test window | best OOD window |",
            "| ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    else:
        lines = [
            "# Clause-Variable Message-Passing SAT Depth Sweep",
            "",
            "Task: produce any assignment that satisfies a satisfiable 3-SAT formula.",
            "Primary metric: valid-assignment rate (verified, not canonical exact match).",
            "",
            f"- Depths (rounds): `{args.depths}`",
            f"- Train/val/test/OOD: `{args.n_train}/{args.n_val}/{args.n_test}/{args.n_test}`",
            f"- In-distribution: `{args.n_vars}` vars, `{args.n_clauses}` clauses",
            f"- OOD: `{args.ood_vars}` vars, `{args.ood_clauses}` clauses",
            f"- Deep supervision: `{args.deep_supervision}`",
            "",
            "| depth | val valid | test valid | OOD valid | best val window | best test window | best OOD window |",
            "| ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    for summary in summaries:
        baselines = summary["baselines"]
        best = {row["split"]: row for row in summary["best_rows"]}

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
            f"{_window('val')} | {_window('test')} | {_window('ood')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    depths = parse_depths(args.depths)
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, max_vars, max_clauses, quality = make_loaders(args)
    if quality is not None:
        quality.to_csv(run_dir / "dataset_quality.csv", index=False)
    summaries = []
    for depth in depths:
        summaries.append(
            train_one_depth(
                args,
                n_rounds=depth,
                loaders=loaders,
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
