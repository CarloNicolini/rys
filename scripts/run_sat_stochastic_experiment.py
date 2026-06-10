"""Cycle 7: stochastic transitions, width (sampling) vs depth (RYS).

Trains a clause-variable SAT solver with GRAM-style stochastic latent
transitions and compares it to the deterministic baseline on three axes:

1. single-sample valid-assignment rate (deterministic mean path);
2. width scaling — valid@N and coverage from N sampled trajectories;
3. depth scaling — RYS replay of a late window, with and without sampling.

The deterministic operator F + mu carries the RYS surgery and the rho/phi
geometry; the Gaussian noise is the extra width axis on top.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import DataLoader

from rys.sat_data import (
    SatAssignmentDataset,
    make_sat_assignment_examples,
    make_sat_assignment_splits,
    multisample_validity,
    soft_sat_loss,
    verify_assignment_tensor,
)
from rys.sat_message_passing import MessagePassingConfig, MessagePassingSatModel
from rys.surgery import apply_rys
from rys.training.modules import SatStochasticLitModule
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(45, help="Random seed."),
    depth: int = typer.Option(16, help="Message-passing depth (rounds)."),
    modes: str = typer.Option("deterministic,stochastic", help="Comma list of modes to compare."),
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
    n_samples: int = typer.Option(20, help="N for valid@N / coverage."),
    sigma_floor: float = typer.Option(0.1, help="Target min std; penalise sigma below it."),
    sigma_reg: float = typer.Option(1e-2, help="Weight of the variance-floor penalty."),
    rys_window: str = typer.Option("13,15", help="Half-open late window for the RYS probe (end < depth)."),
    output_dir: Path = typer.Option(Path("results/sat_stochastic"), help="Run output directory."),
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


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], int, int]:
    max_vars = max(args.n_vars, args.ood_vars)
    max_clauses = max(args.n_clauses, args.ood_clauses)
    splits = make_sat_assignment_splits(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n_vars=args.n_vars, n_clauses=args.n_clauses, seed=args.seed,
    )
    splits["ood"] = make_sat_assignment_examples(
        args.n_test, n_vars=args.ood_vars, n_clauses=args.ood_clauses, seed=args.seed + 3, prefix="ood",
    )
    datasets = {n: SatAssignmentDataset(ex, max_vars=max_vars, max_clauses=max_clauses) for n, ex in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, max_vars, max_clauses


def tensors(batch, device):
    return (
        batch["clause_variable_ids"].to(device),
        batch["clause_sign_ids"].to(device),
        batch["clause_mask"].to(device),
        batch["assignment_labels"].to(device),
        batch["assignment_mask"].to(device),
    )


def train(model, loaders, optimizer, device, args, stochastic) -> list[dict]:
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            cvi, csi, cm, _, _ = tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm,
                        return_round_logits=True, sample=stochastic)
            loss = torch.stack([soft_sat_loss(rl, cvi, csi, cm) for rl in out["round_logits"]]).mean()
            if stochastic and out["mean_log_sigma"] is not None:
                # Penalise the std falling below the floor (keeps exploration alive).
                sigma = out["mean_log_sigma"].exp()
                loss = loss + args.sigma_reg * torch.relu(args.sigma_floor - sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = single_sample_validity(model, loaders["val"], device)
        history.append({"epoch": epoch, "val_valid": val})
        print(json.dumps({"stochastic": stochastic, **history[-1]}))
    return history


@torch.inference_mode()
def single_sample_validity(model, loader, device, *, window=None) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        cvi, csi, cm, _, _ = tensors(batch, device)
        if window is None:
            logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm)["logits"]
        else:
            with apply_rys(model, window=window, n_repeats=2):
                logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm)["logits"]
        valid = verify_assignment_tensor(logits.argmax(dim=-1), cvi, csi, cm)
        correct += int(valid.sum())
        total += valid.shape[0]
    return correct / total


@torch.inference_mode()
def multisample_metrics(model, loader, device, *, n_samples, window=None) -> dict[str, float]:
    model.eval()
    valid_any = n_valid = coverage = total = 0
    for batch in loader:
        cvi, csi, cm, _, amask = tensors(batch, device)
        draws = []
        for _ in range(n_samples):
            if window is None:
                logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, sample=True)["logits"]
            else:
                with apply_rys(model, window=window, n_repeats=2):
                    logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, sample=True)["logits"]
            draws.append(logits.argmax(dim=-1))
        out = multisample_validity(torch.stack(draws), cvi, csi, cm, assignment_mask=amask)
        valid_any += int(out["valid_any"].sum())
        n_valid += int(out["n_valid"].sum())
        coverage += int(out["coverage"].sum())
        total += cvi.shape[0]
    return {
        "valid_at_n": valid_any / total,
        "mean_valid_fraction": n_valid / (total * n_samples),
        "mean_coverage": coverage / total,
    }


def run_mode(args, mode, loaders, max_vars, max_clauses, device, run_dir) -> dict:
    stochastic = mode == "stochastic"
    config = MessagePassingConfig(
        max_vars=max_vars, max_clauses=max_clauses, d_model=args.d_model, n_rounds=args.depth,
        d_mlp=args.d_mlp, pre_norm=True, stochastic=stochastic,
    )
    mode_dir = run_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    model = MessagePassingSatModel(config).to(device)
    lit = SatStochasticLitModule(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        sigma_floor=args.sigma_floor,
        sigma_reg=args.sigma_reg,
        stochastic_sample=stochastic,
        model_config=config.__dict__,
    )
    trainer = build_trainer(
        mode_dir,
        max_epochs=args.epochs,
        monitor=lit.primary_metric,
        mode=lit.primary_mode,
        extra_log_fields={"mode": mode, "stochastic": stochastic},
    )
    trainer.fit(lit, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    checkpoint_path = best_checkpoint_path(trainer)
    model, _ = load_rys_model(
        checkpoint_path,
        lambda _config: MessagePassingSatModel(config),
        map_location=device,
    )
    model.to(device)

    window = tuple(int(x) for x in args.rys_window.split(","))
    result = {"mode": mode, "stochastic": stochastic, "checkpoint": str(checkpoint_path), "rys_window": list(window), "splits": {}}
    for split in ("val", "test", "ood"):
        entry = {
            "single_sample_valid": single_sample_validity(model, loaders[split], device),
            "single_sample_valid_rys": single_sample_validity(model, loaders[split], device, window=window),
        }
        if stochastic:
            entry["multisample"] = multisample_metrics(model, loaders[split], device, n_samples=args.n_samples)
            entry["multisample_rys"] = multisample_metrics(model, loaders[split], device, n_samples=args.n_samples, window=window)
        result["splits"][split] = entry
        print(json.dumps({"mode": mode, "split": split, **entry}))
    (mode_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def save_bar(results: list[dict], path: Path) -> None:
    splits = ["val", "test", "ood"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(splits))
    width = 0.2
    det = next((r for r in results if not r["stochastic"]), None)
    sto = next((r for r in results if r["stochastic"]), None)
    if det:
        ax.bar(x - 1.5 * width, [det["splits"][s]["single_sample_valid"] for s in splits], width, label="det single")
    if sto:
        ax.bar(x - 0.5 * width, [sto["splits"][s]["single_sample_valid"] for s in splits], width, label="stoch single")
        ax.bar(x + 0.5 * width, [sto["splits"][s]["multisample"]["valid_at_n"] for s in splits], width, label="stoch valid@N")
        ax.bar(x + 1.5 * width, [sto["splits"][s]["multisample"]["mean_coverage"] for s in splits], width, label="stoch coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("rate / coverage")
    ax.set_title("Deterministic vs stochastic: validity, valid@N, coverage")
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders, max_vars, max_clauses = make_loaders(args)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = [run_mode(args, m, loaders, max_vars, max_clauses, device, run_dir) for m in modes]
    save_bar(results, run_dir / "validity_comparison.png")
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2, default=str))
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
