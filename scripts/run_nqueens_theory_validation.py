"""Cycle 3: validate the rho/phi RYS theory on a pre-norm N-Queens solver.

Mirrors ``scripts/run_sat_theory_validation.py``:

1. Train a pre-norm N-Queens message-passing solver.
2. Capture the raw row residual stream, measure the rho/phi table, and test the
   plateau prediction ``1 - CKA ~ 1/2 rho^2 sin^2 phi`` against measured CKA.
3. Sweep strict half-open RYS windows and record Delta valid-board rate.
4. For the best windows, test the doubling prediction and the dynamical
   diagnostics (junction mismatch, block-Jacobian spectral radius), linking each
   to Delta validity in ``theory_link.csv``.
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
from rys.nqueens_data import (
    QueensDataset,
    make_queens_examples,
    make_queens_splits,
    soft_nqueens_loss,
    verify_boards_tensor,
)
from rys.nqueens_message_passing import QueensMessagePassingModel, QueensMPConfig
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
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--weight-tied", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--ood-n", type=int, default=8)
    parser.add_argument("--n-givens", type=int, default=4)
    parser.add_argument("--ood-givens", type=int, default=2)
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
    parser.add_argument("--given-ce-weight", type=float, default=0.25)
    parser.add_argument("--capture-batches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--jacobian-batch", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("results/nqueens_theory_validation"))
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
    max_n = max(args.n, args.ood_n)
    splits = make_queens_splits(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        n=args.n, n_givens=args.n_givens, seed=args.seed,
    )
    splits["ood"] = make_queens_examples(
        args.n_test, n=args.ood_n, n_givens=args.ood_givens, seed=args.seed + 3, prefix="ood",
    )
    datasets = {k: QueensDataset(v, max_n=max_n) for k, v in splits.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False),
    }
    return loaders, max_n


def on_device(batch, device):
    return {
        "given_col": batch["given_col"].to(device),
        "is_given": batch["is_given"].to(device),
        "row_mask": batch["row_mask"].to(device),
        "col_mask": batch["col_mask"].to(device),
    }


def valid_rate(model, loader, device, *, window=None, n_repeats=2) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for batch in loader:
            kw = on_device(batch, device)
            if window is None:
                logits = model(**kw)["logits"]
            else:
                with apply_rys(model, window=window, n_repeats=n_repeats):
                    logits = model(**kw)["logits"]
            valid = verify_boards_tensor(logits.argmax(-1), kw["row_mask"], kw["given_col"], kw["is_given"])
            correct += int(valid.sum().detach().cpu())
            total += int(logits.shape[0])
    return correct / total


def train(model, loaders, optimizer, device, epochs, given_ce_weight) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loaders["train"]:
            kw = on_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(**kw, return_round_logits=True)
            terms = [soft_nqueens_loss(rl, kw["row_mask"], kw["col_mask"], given_col=kw["given_col"],
                                       is_given=kw["is_given"], given_ce_weight=given_ce_weight)
                     for rl in out["round_logits"]]
            loss = torch.stack(terms).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        v = valid_rate(model, loaders["val"], device)
        history.append({"epoch": epoch, "val_valid": v})
        print(json.dumps(history[-1]))
    return history


@torch.inference_mode()
def capture_row_states(model, loader, device, *, window=None, max_batches=4) -> pd.DataFrame:
    model.eval()
    rows = []
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        kw = on_device(batch, device)
        mask = batch["row_mask"].bool()
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        if window is None:
            trace = model(**kw, return_hidden_states=True)["hidden_states"]
        else:
            with apply_rys(model, window=window, n_repeats=2):
                trace = model(**kw, return_hidden_states=True)["hidden_states"]
        for layer_idx, hidden in enumerate(trace):
            for row_idx, pid in enumerate(prompt_ids):
                rows.append({"prompt_id": pid, "layer": layer_idx,
                             "activation": hidden[row_idx, mask[row_idx]].detach().cpu().numpy(),
                             "strategy": "row"})
    return pd.DataFrame(rows)


def strict_windows(n_rounds, min_span=2):
    return [(i, j) for i in range(n_rounds) for j in range(i + min_span, n_rounds)]


def delta_matrix(model, loader, device, n_rounds):
    base = valid_rate(model, loader, device)
    mat = np.full((n_rounds, n_rounds), np.nan)
    rows = []
    for start, end in strict_windows(n_rounds):
        rys = valid_rate(model, loader, device, window=(start, end))
        mat[start, end] = rys - base
        rows.append({"start": start, "end": end, "baseline": base, "rys": rys, "delta_valid": rys - base})
    return pd.DataFrame(mat), base, pd.DataFrame(rows)


def save_scatter(table, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(table["one_minus_cka_plateau"], table["one_minus_cka_full"], s=10, alpha=0.5, label="pairs")
    lim = max(table["one_minus_cka_full"].max(), table["one_minus_cka_plateau"].max(), 1e-6)
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel(r"predicted $\frac{1}{2}\rho^2\sin^2\phi$")
    ax.set_ylabel(r"measured $1-\mathrm{CKA}$")
    ax.set_title("rho/phi plateau prediction vs measured (N-Queens)")
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(matrix, path, *, title):
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_title(title)
    ax.set_xlabel("end round (half-open)")
    ax.set_ylabel("start round")
    fig.colorbar(im, ax=ax, label="Δ valid board rate")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_block_map(model, batch, device, window):
    kw = on_device(batch, device)
    backbone = model.model
    context_mask = kw["row_mask"]
    start, end = window
    layers = backbone.layers
    from rys.nqueens_message_passing import _RowContext
    ctx = _RowContext(row_mask=context_mask)
    for layer in layers:
        layer.ctx = ctx
    hidden = backbone.embed(kw["given_col"], kw["is_given"])
    for idx in range(start):
        hidden = layers[idx](hidden, attention_mask=None)[0]
    hidden_in = hidden.detach()

    def block_map(h):
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
    loaders, max_n = make_loaders(args)
    config = QueensMPConfig(max_n=max_n, d_model=args.d_model, n_rounds=args.depth,
                            n_heads=args.n_heads, d_mlp=args.d_mlp, pre_norm=True, weight_tied=args.weight_tied)
    model = QueensMessagePassingModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = train(model, loaders, optimizer, device, args.epochs, args.given_ce_weight)
    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__, "args": vars(args)}, run_dir / "checkpoint.pt")

    base_states = capture_row_states(model, loaders["val"], device, max_batches=args.capture_batches)
    table = rho_phi_table(base_states)
    table.to_csv(run_dir / "rho_phi_long.csv", index=False)
    fit = theory_fit(table)
    save_scatter(table, run_dir / "rho_phi_prediction.png")
    cka = cka_matrix(base_states, unbiased=False, device="cpu")
    cka.to_csv(run_dir / "cka_variable_val.csv")

    baselines, long_by_split = {}, {}
    for s in ("val", "test", "ood"):
        mat, base, long = delta_matrix(model, loaders[s], device, args.depth)
        baselines[s] = base
        long_by_split[s] = long
        mat.to_csv(run_dir / f"delta_valid_{s}.csv")
        long.to_csv(run_dir / f"delta_valid_{s}_long.csv", index=False)
        save_heatmap(mat, run_dir / f"delta_valid_{s}.png", title=f"RYS ΔValidity {s} (L={args.depth})")

    merged = long_by_split["test"].merge(long_by_split["ood"], on=["start", "end"], suffixes=("_test", "_ood"))
    merged["combined"] = 0.5 * merged["delta_valid_test"] + 0.5 * merged["delta_valid_ood"]
    merged = merged.merge(long_by_split["val"][["start", "end", "delta_valid"]].rename(columns={"delta_valid": "delta_valid_val"}), on=["start", "end"])
    top = merged.nlargest(args.top_k, "combined")

    link_rows = []
    for _, row in top.iterrows():
        window = (int(row["start"]), int(row["end"]))
        rys_states = capture_row_states(model, loaders["val"], device, window=window, max_batches=args.capture_batches)
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
    pd.DataFrame(link_rows).to_csv(run_dir / "theory_link.csv", index=False)

    summary = {
        "args": vars(args), "device": str(device), "config": config.__dict__,
        "baselines": baselines, "theory_fit": fit, "best_windows": link_rows,
        "convention": "half-open RYS window (start, end); duplicated rounds start..end-1; n_repeats=2",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
