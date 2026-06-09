"""Cycle 8: try to break the deterministic mode-collapse ceiling.

Two training regimes for the stochastic clause-variable solver are compared:

- ``floor``: the literal "raise the variance floor + diversity reward" recipe —
  mean soft-SAT loss, a high target std, and a reward that maximises the spread
  of per-bit probabilities across samples;
- ``best_of_k``: GRAM-inspired multiple-choice / winner-take-all training — draw
  K trajectories per formula and back-propagate only the *best* one's soft-SAT
  loss, so identical samples waste the minimum and the model is rewarded for
  *productive* diversity (reaching different valid assignments).

Both are scored on single-sample validity, valid@N, and coverage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--regimes", type=str, default="floor,best_of_k")
    parser.add_argument("--n-vars", type=int, default=6)
    parser.add_argument("--n-clauses", type=int, default=24)
    parser.add_argument("--ood-vars", type=int, default=8)
    parser.add_argument("--ood-clauses", type=int, default=34)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-mlp", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--train-k", type=int, default=4, help="K trajectories for best_of_k / diversity.")
    parser.add_argument("--floor-sigma", type=float, default=0.7, help="Target std for the 'floor' regime.")
    parser.add_argument("--floor-reg", type=float, default=0.2, help="Variance-floor penalty weight.")
    parser.add_argument("--diversity-weight", type=float, default=0.1, help="Per-bit spread reward weight.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sat_coverage"))
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(args):
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
        batch["assignment_mask"].to(device),
    )


def diversity_reward(prob_samples: torch.Tensor, amask: torch.Tensor) -> torch.Tensor:
    """Mean per-active-bit variance of P(x=1) across the K samples (to maximise)."""
    var = prob_samples.var(dim=0, unbiased=False)  # (B, V)
    return (var * amask).sum() / amask.sum().clamp_min(1)


def train(model, loaders, optimizer, device, args, regime) -> list[dict]:
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loaders["train"]:
            cvi, csi, cm, amask = tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            sample_losses = []
            sample_probs = []
            sigma_terms = []
            for _ in range(args.train_k):
                out = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, sample=True)
                sample_losses.append(soft_sat_loss(out["logits"], cvi, csi, cm, reduction="none"))  # (B,)
                sample_probs.append(out["logits"].softmax(dim=-1)[..., 1])
                if out["mean_log_sigma"] is not None:
                    sigma_terms.append(out["mean_log_sigma"])
            stacked = torch.stack(sample_losses)  # (K, B)
            if regime == "best_of_k":
                loss = stacked.min(dim=0).values.mean()
            else:  # floor regime: mean loss + explicit diversity + variance floor
                loss = stacked.mean()
            if args.diversity_weight > 0:
                div = diversity_reward(torch.stack(sample_probs), amask)
                loss = loss - args.diversity_weight * div
            if regime == "floor" and sigma_terms:
                sigma = torch.stack(sigma_terms).mean().exp()
                loss = loss + args.floor_reg * torch.relu(args.floor_sigma - sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = single_sample_validity(model, loaders["val"], device)
        history.append({"epoch": epoch, "val_valid": val})
        print(json.dumps({"regime": regime, **history[-1]}))
    return history


@torch.inference_mode()
def single_sample_validity(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        cvi, csi, cm, _ = tensors(batch, device)
        logits = model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm)["logits"]
        valid = verify_assignment_tensor(logits.argmax(dim=-1), cvi, csi, cm)
        correct += int(valid.sum())
        total += valid.shape[0]
    return correct / total


@torch.inference_mode()
def multisample_metrics(model, loader, device, *, n_samples) -> dict[str, float]:
    model.eval()
    valid_any = n_valid = coverage = total = 0
    for batch in loader:
        cvi, csi, cm, amask = tensors(batch, device)
        draws = [model(clause_variable_ids=cvi, clause_sign_ids=csi, clause_mask=cm, sample=True)["logits"].argmax(dim=-1)
                 for _ in range(n_samples)]
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


def run_regime(args, regime, loaders, max_vars, max_clauses, device, run_dir) -> dict:
    config = MessagePassingConfig(
        max_vars=max_vars, max_clauses=max_clauses, d_model=args.d_model, n_rounds=args.depth,
        d_mlp=args.d_mlp, pre_norm=True, stochastic=True, log_sigma_init=-1.0,
    )
    model = MessagePassingSatModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = train(model, loaders, optimizer, device, args, regime)
    rdir = run_dir / regime
    rdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(rdir / "train_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__, "args": vars(args)}, rdir / "checkpoint.pt")
    result = {"regime": regime, "splits": {}}
    for split in ("val", "test", "ood"):
        entry = {
            "single_sample_valid": single_sample_validity(model, loaders[split], device),
            "multisample": multisample_metrics(model, loaders[split], device, n_samples=args.n_samples),
        }
        result["splits"][split] = entry
        print(json.dumps({"regime": regime, "split": split, **entry}))
    (rdir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def save_bar(results, path):
    splits = ["val", "test", "ood"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(splits))
    n = len(results)
    width = 0.8 / (n * 3)
    for ri, r in enumerate(results):
        base = (ri - n / 2) * 3 * width
        ax.bar(x + base, [r["splits"][s]["single_sample_valid"] for s in splits], width, label=f"{r['regime']} single")
        ax.bar(x + base + width, [r["splits"][s]["multisample"]["valid_at_n"] for s in splits], width, label=f"{r['regime']} valid@N")
        ax.bar(x + base + 2 * width, [r["splits"][s]["multisample"]["mean_coverage"] for s in splits], width, label=f"{r['regime']} coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("rate / coverage")
    ax.set_title("Breaking the ceiling: variance-floor vs best-of-K")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders, max_vars, max_clauses = make_loaders(args)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    results = [run_regime(args, r, loaders, max_vars, max_clauses, device, run_dir) for r in regimes]
    save_bar(results, run_dir / "coverage_comparison.png")
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2, default=str))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
