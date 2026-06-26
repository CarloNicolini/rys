"""Grokking / overfitting probe for the clause-variable message-passing solver.

The validity ceiling (~0.52) seen in Cycle 5/6 has train ~= val, i.e. no
generalization gap. Grokking is *delayed* generalization and requires the model
to first fit the training set (train validity ~ 1.0). This script tests the
precondition directly: take a *tiny fixed* set of satisfiable 3-SAT formulas and
train for many epochs, watching whether train validity can reach ~1.0 and
whether a held-out split later catches up.

Two outcomes settle the question:

- train validity -> ~1.0 while val lags: a memorization regime exists, so
  grokking is testable (watch for a delayed val jump).
- train validity plateaus well below 1.0 even on a handful of formulas: the
  ceiling is a representational/optimization limit, not training duration, and
  grokking is ruled out.

Progress is shown as a percentage bar with live train/val validity.
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
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader

from rys.sat_data import (
    SatAssignmentDataset,
    make_sat_assignment_examples,
    soft_sat_loss,
    verify_assignment_tensor,
)
from rys.sat_message_passing import MessagePassingConfig, MessagePassingSatModel
from rys.training.modules import SatMPLitModule
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(50, help="Random seed."),
    n_train: int = typer.Option(64, help="Size of the tiny fixed train set."),
    n_val: int = typer.Option(512, help="Held-out split (same distribution)."),
    n_vars: int = typer.Option(6, help="Variable count."),
    n_clauses: int = typer.Option(24, help="Clause count."),
    n_rounds: int = typer.Option(16, help="Message-passing depth (rounds)."),
    d_model: int = typer.Option(64, help="Model width."),
    d_mlp: int = typer.Option(128, help="MLP hidden width."),
    epochs: int = typer.Option(4000, help="Training epochs (long, to probe delayed generalization)."),
    lr: float = typer.Option(5e-4, help="AdamW learning rate."),
    weight_decay: float = typer.Option(
        0.0,
        help="0.0 maximises ability to memorise (pure capacity probe). "
        "Grokking proper usually needs a non-zero value, e.g. 1e-2.",
    ),
    batch_size: int = typer.Option(0, help="0 = full-batch over the tiny set."),
    deep_supervision: bool = typer.Option(
        True, "--deep-supervision/--no-deep-supervision", help="Average the soft-SAT loss over every round."
    ),
    variable_id_embeddings: bool = typer.Option(
        False,
        "--variable-id-embeddings/--no-variable-id-embeddings",
        help="Give each variable a distinct learned initial state, breaking "
        "permutation-equivariance (tests the Weisfeiler-Leman ceiling).",
    ),
    ce_weight: float = typer.Option(
        0.0,
        help="Weight of a cross-entropy term toward the canonical assignment, "
        "added to the soft-SAT loss (tests the rounding-gap hypothesis).",
    ),
    eval_every: int = typer.Option(
        10, help="Deprecated under Lightning (validation runs every epoch); kept for CLI compatibility."
    ),
    output_dir: Path = typer.Option(Path("results/sat_grokking_probe"), help="Run output directory."),
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


def clause_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch["clause_variable_ids"].to(device),
        batch["clause_sign_ids"].to(device),
        batch["clause_mask"].to(device),
    )


def ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked cross-entropy toward the canonical satisfying assignment."""
    per_bit = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction="none").reshape(labels.shape)
    return (per_bit * mask).sum() / mask.sum().clamp_min(1)


def supervised_loss(
    round_logits: list[torch.Tensor],
    final_logits: torch.Tensor,
    cvi: torch.Tensor,
    csi: torch.Tensor,
    cm: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    deep_supervision: bool,
    ce_weight: float,
) -> torch.Tensor:
    logits_for_loss = round_logits if (deep_supervision and round_logits) else [final_logits]
    soft_term = torch.stack([soft_sat_loss(logit, cvi, csi, cm) for logit in logits_for_loss]).mean()
    if ce_weight == 0.0:
        return soft_term
    ce_term = torch.stack([ce_loss(logit, labels, mask) for logit in logits_for_loss]).mean()
    return soft_term + ce_weight * ce_term


@torch.inference_mode()
def evaluate(model: MessagePassingSatModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    valid_correct = 0
    bit_correct = 0
    bit_total = 0
    example_total = 0
    for batch in loader:
        labels = batch["assignment_labels"].to(device)
        mask = batch["assignment_mask"].to(device)
        cvi, csi, cm = clause_tensors(batch, device)
        logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm)["logits"]
        loss = soft_sat_loss(logits, cvi, csi, cm)
        preds = logits.argmax(dim=-1)
        valid = verify_assignment_tensor(preds, cvi, csi, cm)
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        valid_correct += int(valid.sum().detach().cpu())
        bit_correct += int(((preds == labels) & mask).sum().detach().cpu())
        bit_total += int(mask.sum().detach().cpu())
        example_total += int(labels.shape[0])
    return {
        "loss": total_loss / example_total,
        "bit_accuracy": bit_correct / bit_total,
        "valid_assignment_rate": valid_correct / example_total,
    }


@torch.inference_mode()
def marginal_diagnostic(
    model: MessagePassingSatModel, loader: DataLoader, device: torch.device, run_dir: Path
) -> dict[str, float]:
    """Confidence |P(x=1) - 0.5| of variables, split by solved vs unsolved formula.

    The rounding-gap hypothesis predicts that variables in *unsolved* formulas
    sit near 0.5 (ambiguous), so the hard argmax is an arbitrary coin flip that
    can falsify a clause.
    """
    model.eval()
    conf_solved: list[float] = []
    conf_unsolved: list[float] = []
    for batch in loader:
        mask = batch["assignment_mask"].to(device).bool()
        cvi, csi, cm = clause_tensors(batch, device)
        logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm)["logits"]
        prob_true = torch.softmax(logits, dim=-1)[..., 1]
        confidence = (prob_true - 0.5).abs()
        valid = verify_assignment_tensor(logits.argmax(dim=-1), cvi, csi, cm)
        for row in range(logits.shape[0]):
            vals = confidence[row][mask[row]].cpu().tolist()
            (conf_solved if bool(valid[row]) else conf_unsolved).extend(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    if conf_solved:
        ax.hist(conf_solved, bins=20, range=(0, 0.5), density=True, alpha=0.6,
                color="#1f77b4", label=f"solved-formula vars (n={len(conf_solved)})")
    if conf_unsolved:
        ax.hist(conf_unsolved, bins=20, range=(0, 0.5), density=True, alpha=0.6,
                color="#c2410c", label=f"unsolved-formula vars (n={len(conf_unsolved)})")
    ax.set_xlabel("|P(x=1) - 0.5|   (0 = maximally ambiguous, 0.5 = confident)")
    ax.set_ylabel("density")
    ax.set_title("Variable marginal confidence on the train set")
    ax.legend()
    fig.savefig(run_dir / "marginal_confidence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def _frac_ambiguous(values: list[float]) -> float:
        return float(np.mean([v < 0.1 for v in values])) if values else float("nan")

    return {
        "mean_conf_solved": _mean(conf_solved),
        "mean_conf_unsolved": _mean(conf_unsolved),
        "frac_ambiguous_solved": _frac_ambiguous(conf_solved),
        "frac_ambiguous_unsolved": _frac_ambiguous(conf_unsolved),
        "n_vars_solved": len(conf_solved),
        "n_vars_unsolved": len(conf_unsolved),
    }


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tiny *fixed* train set and a disjoint same-distribution validation set.
    train_examples = make_sat_assignment_examples(
        args.n_train, n_vars=args.n_vars, n_clauses=args.n_clauses, seed=args.seed, prefix="grok_tr"
    )
    val_examples = make_sat_assignment_examples(
        args.n_val, n_vars=args.n_vars, n_clauses=args.n_clauses, seed=args.seed + 9991, prefix="grok_va"
    )
    train_ds = SatAssignmentDataset(train_examples, max_vars=args.n_vars, max_clauses=args.n_clauses)
    val_ds = SatAssignmentDataset(val_examples, max_vars=args.n_vars, max_clauses=args.n_clauses)
    batch_size = args.batch_size or args.n_train
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    config = MessagePassingConfig(
        max_vars=args.n_vars,
        max_clauses=args.n_clauses,
        d_model=args.d_model,
        n_rounds=args.n_rounds,
        d_mlp=args.d_mlp,
        var_id_embeddings=args.variable_id_embeddings,
    )
    model = MessagePassingSatModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "device": str(device),
                "n_params": n_params,
                "n_train": args.n_train,
                "n_val": args.n_val,
                "n_rounds": args.n_rounds,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "deep_supervision": args.deep_supervision,
            }
        )
    )

    lit = SatMPLitModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        deep_supervision=args.deep_supervision,
        ce_weight=args.ce_weight,
        model_config=config.__dict__,
    )
    trainer = build_trainer(
        run_dir,
        max_epochs=args.epochs,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    checkpoint_path = best_checkpoint_path(trainer)
    model, _ = load_rys_model(
        checkpoint_path,
        lambda _config: MessagePassingSatModel(config),
        map_location=device,
    )
    model.to(device)

    history_df = pd.read_csv(run_dir / "train_history.csv")
    train_metrics = evaluate(model, train_eval_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    final = {
        "epoch": int(history_df["epoch"].max()) if not history_df.empty else args.epochs,
        "train_loss": train_metrics["loss"],
        "train_bit_accuracy": train_metrics["bit_accuracy"],
        "train_valid_assignment_rate": train_metrics["valid_assignment_rate"],
        "val_loss": val_metrics["loss"],
        "val_bit_accuracy": val_metrics["bit_accuracy"],
        "val_valid_assignment_rate": val_metrics["valid_assignment_rate"],
    }
    best_train_valid = float(history_df.get("train_valid_assignment_rate", pd.Series([final["train_valid_assignment_rate"]])).max())
    best_val_valid = float(history_df.get("val_valid_assignment_rate", pd.Series([final["val_valid_assignment_rate"]])).max())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_valid_assignment_rate"], label="train validity", color="#c2410c")
    ax.plot(history_df["epoch"], history_df["val_valid_assignment_rate"], label="val validity", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("epoch (log scale)")
    ax.set_ylabel("valid assignment rate")
    variant = "id-embed" if args.variable_id_embeddings else "anonymous"
    ax.set_title(
        f"Grokking probe ({variant}): {args.n_train} fixed formulas, "
        f"L={args.n_rounds}, wd={args.weight_decay}, ce={args.ce_weight}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(run_dir / "grokking_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    diagnostic = marginal_diagnostic(model, train_eval_loader, device, run_dir)

    summary = {
        "args": vars(args),
        "n_params": n_params,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "final": final,
        "best_train_valid_assignment_rate": best_train_valid,
        "best_val_valid_assignment_rate": best_val_valid,
        "memorized_train": best_train_valid >= 0.99,
        "generalization_gap": best_train_valid - best_val_valid,
        "marginal_diagnostic": diagnostic,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary["final"], indent=2, default=str))
    print(json.dumps({"marginal_diagnostic": diagnostic}, indent=2, default=str))
    print(
        f"best train validity {best_train_valid:.3f} | best val validity {best_val_valid:.3f} | "
        f"memorized={summary['memorized_train']}"
    )
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
