"""Validate the rho/phi RYS theory on a pre-norm graph-colouring solver.

Mirrors ``scripts/run_sat_theory_validation.py``.  Pipeline for one trained
depth:

1. Train a pre-norm (additive residual stream) adjacency-masked colouring solver.
2. Capture the raw vertex residual stream and measure the rho/phi table; test the
   plateau prediction ``1 - CKA ~ 1/2 rho^2 sin^2 phi`` against measured CKA.
3. Sweep strict half-open RYS windows ``(start, end)`` (>= 2 duplicated rounds)
   and record Delta valid-colouring rate on val/test/OOD.
4. For the best windows, capture RYS activations and test the doubling
   prediction (``S_norm ~ x2``, ``1 - CKA ~ x4``), plus junction mismatch and
   block-Jacobian spectral radius, then link every diagnostic to Delta validity.

All RYS here uses the half-open convention of ``rys.surgery.apply_rys`` so the
theory and the behavioural sweep share one convention.
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
from rys.theory_validation import (
    block_jacobian_sigma_max,
    junction_mismatch,
    rho_phi_table,
    rys_amplification_summary,
    theory_fit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--weight-tied", action="store_true")
    parser.add_argument("--n-vertices", type=int, default=12)
    parser.add_argument("--n-edges", type=int, default=24)
    parser.add_argument("--n-colors", type=int, default=3)
    parser.add_argument("--n-givens", type=int, default=3)
    parser.add_argument("--ood-vertices", type=int, default=18)
    parser.add_argument("--ood-edges", type=int, default=40)
    parser.add_argument("--ood-givens", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-mlp", type=int, default=128)
    parser.add_argument("--max-degree", type=int, default=24)
    parser.add_argument("--given-ce-weight", type=float, default=0.25)
    parser.add_argument("--capture-batches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--jacobian-batch", type=int, default=16, help="0 disables the Jacobian estimate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/coloring_theory_validation"))
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
    datasets = {n: ColoringDataset(ex, max_v=max_v, n_colors=args.n_colors) for n, ex in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, max_v


def kwargs_on(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "given_color": batch["given_color"].to(device),
        "is_given": batch["is_given"].to(device),
        "degree": batch["degree"].to(device),
        "adjacency": batch["adjacency"].to(device),
        "vertex_mask": batch["vertex_mask"].to(device),
    }


def metrics(logits, labels, vertex_mask, adjacency, given_color, is_given) -> dict[str, int]:
    colors = logits.argmax(dim=-1)
    valid = verify_coloring_tensor(colors, vertex_mask, adjacency, given_color, is_given)
    return {
        "valid_correct": int(valid.sum().detach().cpu()),
        "cell_correct": int(((colors == labels) & vertex_mask).sum().detach().cpu()),
        "cell_total": int(vertex_mask.sum().detach().cpu()),
        "example_total": int(labels.shape[0]),
    }


def train(model, loaders, optimizer, device, epochs, given_ce_weight) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders["train"]:
            kw = kwargs_on(batch, device)
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
        val = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch, "val_valid": val["valid_coloring_rate"], "val_cell": val["cell_accuracy"]})
        print(json.dumps(history[-1]))
    return history


@torch.inference_mode()
def evaluate(model, loader, device, *, window: tuple[int, int] | None = None) -> dict[str, float]:
    model.eval()
    counts = {"valid_correct": 0, "cell_correct": 0, "cell_total": 0, "example_total": 0}
    for batch in loader:
        kw = kwargs_on(batch, device)
        labels = batch["labels"].to(device)
        if window is None:
            logits = model(**kw)["logits"]
        else:
            with apply_rys(model, window=window, n_repeats=2):
                logits = model(**kw)["logits"]
        for k, v in metrics(logits, labels, kw["vertex_mask"], kw["adjacency"], kw["given_color"], kw["is_given"]).items():
            counts[k] += v
    return {
        "valid_coloring_rate": counts["valid_correct"] / counts["example_total"],
        "cell_accuracy": counts["cell_correct"] / counts["cell_total"],
    }


@torch.inference_mode()
def capture_states(model, loader, device, *, window: tuple[int, int] | None = None, max_batches: int = 4) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        kw = kwargs_on(batch, device)
        mask = batch["vertex_mask"].bool()
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        if window is None:
            trace = model(**kw, return_hidden_states=True)["hidden_states"]
        else:
            with apply_rys(model, window=window, n_repeats=2):
                trace = model(**kw, return_hidden_states=True)["hidden_states"]
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


def delta_validity_matrix(model, loader, device, n_rounds) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    base = evaluate(model, loader, device)["valid_coloring_rate"]
    mat = np.full((n_rounds, n_rounds), np.nan)
    rows = []
    for start, end in strict_windows(n_rounds):
        rys = evaluate(model, loader, device, window=(start, end))["valid_coloring_rate"]
        mat[start, end] = rys - base
        rows.append({"start": start, "end": end, "baseline": base, "rys": rys, "delta_valid": rys - base})
    return pd.DataFrame(mat), base, pd.DataFrame(rows)


def save_scatter(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(table["one_minus_cka_plateau"], table["one_minus_cka_full"], s=10, alpha=0.5, label="pairs")
    lim = max(table["one_minus_cka_full"].max(), table["one_minus_cka_plateau"].max(), 1e-6)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel(r"predicted $\frac{1}{2}\rho^2\sin^2\phi$")
    ax.set_ylabel(r"measured $1-\mathrm{CKA}$")
    ax.set_title("rho/phi plateau prediction vs measured")
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(matrix: pd.DataFrame, path: Path, *, title: str) -> None:
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_title(title)
    ax.set_xlabel("end round (half-open)")
    ax.set_ylabel("start round")
    fig.colorbar(im, ax=ax, label="Δ valid colouring rate")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_block_map(model, batch, device, window):
    """Return a closure mapping vertex hidden -> post-block vertex hidden."""
    kw = kwargs_on(batch, device)
    backbone = model.model
    context = backbone.build_context(kw["adjacency"], kw["vertex_mask"])
    start, end = window
    layers = backbone.layers
    for layer in layers:
        layer.ctx = context
    hidden = backbone.embed(kw["given_color"], kw["is_given"], kw["degree"])
    for idx in range(start):
        hidden = layers[idx](hidden, attention_mask=None)[0]
    hidden_in = hidden.detach()

    def block_map(h: torch.Tensor) -> torch.Tensor:
        out = h
        for idx in range(start, end):
            out = layers[idx](out, attention_mask=None)[0]
        return out

    return block_map, hidden_in


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, max_v = make_loaders(args)
    config = ColoringMPConfig(
        max_v=max_v, n_colors=args.n_colors, max_degree=args.max_degree, d_model=args.d_model,
        n_rounds=args.depth, n_heads=args.n_heads, d_mlp=args.d_mlp, pre_norm=True, weight_tied=args.weight_tied,
    )
    model = ColoringMessagePassingModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = train(model, loaders, optimizer, device, args.epochs, args.given_ce_weight)
    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__, "args": vars(args)}, run_dir / "checkpoint.pt")

    # 1) rho/phi theory on base activations
    base_states = capture_states(model, loaders["val"], device, max_batches=args.capture_batches)
    table = rho_phi_table(base_states)
    table.to_csv(run_dir / "rho_phi_long.csv", index=False)
    fit = theory_fit(table)
    save_scatter(table, run_dir / "rho_phi_prediction.png")
    cka = cka_matrix(base_states, unbiased=False, device="cpu")
    cka.to_csv(run_dir / "cka_variable_val.csv")

    # 2) Delta validity matrices
    baselines, long_by_split = {}, {}
    for split in ("val", "test", "ood"):
        mat, base, long = delta_validity_matrix(model, loaders[split], device, args.depth)
        baselines[split] = base
        long_by_split[split] = long
        mat.to_csv(run_dir / f"delta_valid_{split}.csv")
        long.to_csv(run_dir / f"delta_valid_{split}_long.csv", index=False)
        save_heatmap(mat, run_dir / f"delta_valid_{split}.png", title=f"RYS ΔValidity {split} (L={args.depth})")

    merged = long_by_split["test"].merge(long_by_split["ood"], on=["start", "end"], suffixes=("_test", "_ood"))
    merged["combined"] = 0.5 * merged["delta_valid_test"] + 0.5 * merged["delta_valid_ood"]
    merged = merged.merge(long_by_split["val"][["start", "end", "delta_valid"]].rename(columns={"delta_valid": "delta_valid_val"}), on=["start", "end"])
    top = merged.nlargest(args.top_k, "combined")

    # 3) theory diagnostics linked to the best windows
    link_rows = []
    for _, row in top.iterrows():
        window = (int(row["start"]), int(row["end"]))
        rys_states = capture_states(model, loaders["val"], device, window=window, max_batches=args.capture_batches)
        _, amp = rys_amplification_summary(base_states, rys_states, window)
        jmis = junction_mismatch(base_states, window)
        sigma = float("nan")
        if args.jacobian_batch > 0:
            jbatch = next(iter(DataLoader(loaders["val"].dataset, batch_size=args.jacobian_batch, shuffle=False)))
            block_map, hidden_in = make_block_map(model, jbatch, device, window)
            sigma = block_jacobian_sigma_max(block_map, hidden_in)
        link_rows.append({
            "start": window[0], "end": window[1], "span": window[1] - window[0],
            "delta_valid_val": float(row["delta_valid_val"]),
            "delta_valid_test": float(row["delta_valid_test"]),
            "delta_valid_ood": float(row["delta_valid_ood"]),
            "combined": float(row["combined"]),
            "S_norm_ratio": amp["S_norm_ratio_mean"],
            "one_minus_cka_ratio": amp["one_minus_cka_ratio_mean"],
            "cos_phi_diff": amp["cos_phi_diff_mean"],
            "junction_mismatch": jmis,
            "jacobian_sigma_max": sigma,
        })
        print(json.dumps(link_rows[-1]))
    link = pd.DataFrame(link_rows)
    link.to_csv(run_dir / "theory_link.csv", index=False)

    summary = {
        "args": vars(args),
        "device": str(device),
        "config": config.__dict__,
        "baselines": baselines,
        "theory_fit": fit,
        "best_windows": link_rows,
        "convention": "half-open RYS window (start, end); duplicated rounds start..end-1; n_repeats=2",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
