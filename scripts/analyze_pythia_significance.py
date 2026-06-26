"""Significance analysis for the Pythia RYS sweep: luck vs. signal.

Re-scores each frozen model per problem (not just in aggregate) so we can attach
error bars to the RYS claims:

- a per-window McNemar test on the paired (baseline-correct, RYS-correct) counts,
  giving the real (paired) noise floor instead of the independent upper bound;
- a paired bootstrap over the GSM8K problems for the geometry->effect rank
  correlations Spearman(Delta, rho) and Spearman(Delta, CKA), so the functional
  law is shown robust to *which* problems were sampled;
- the pure-noise expectation for the best-of-W window, to check whether the small
  positive gains exceed what chance alone would produce.

It reuses the geometry (rho, CKA) from each run's saved ``rho_phi_table.csv`` and
writes ``significance.json`` (and the per-problem ``correctness.npz``) next to it.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, spearmanr

from rys.eval_core import resolve_device, swept_windows
from rys.gsm8k_mc import load_pythia, make_gsm8k_mc, prepare_mc_batches, score_prepared
from rys.surgery import apply_rys


def correctness_vector(scores: pd.DataFrame, mc_df: pd.DataFrame) -> np.ndarray:
    """Per-problem boolean: did the gold answer win the per-token-mean ranking."""
    idx = scores.groupby("prompt_id")["loglik_mean"].idxmax()
    win = scores.loc[idx].set_index("prompt_id")["is_gold"]
    return mc_df["prompt_id"].map(win).to_numpy().astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pythia-70m,pythia-160m,pythia-410m,pythia-1b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--fewshot", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--root", default="results/pythia_gsm8k_rys")
    args = ap.parse_args()

    device = resolve_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rng = np.random.default_rng(args.seed)

    for tag in args.models.split(","):
        hf = f"EleutherAI/{tag}"
        run_dir = sorted(glob.glob(f"{args.root}/{tag}/*/"))[-1]
        rp = pd.read_csv(f"{run_dir}/rho_phi_table.csv")
        print(f"\n{'='*70}\n{tag}  ({run_dir})", flush=True)

        model, tok = load_pythia(hf, device=device, dtype=dtype)
        L = len(model.model.layers)
        mc = make_gsm8k_mc(n=args.n, k_distractors=args.k, fewshot=args.fewshot, seed=args.seed)
        batches = prepare_mc_batches(tok, mc, batch_size=args.batch_size)

        base = correctness_vector(score_prepared(model, batches, device=device), mc)
        windows = swept_windows(L)
        rys = np.zeros((len(windows), len(base)), dtype=bool)
        for w, (i, j) in enumerate(windows):
            with apply_rys(model, (i, j), n_repeats=2):
                rys[w] = correctness_vector(score_prepared(model, batches, device=device), mc)
            if (w + 1) % 25 == 0:
                print(f"  {w + 1}/{len(windows)} windows", flush=True)

        n = len(base)
        delta = rys.mean(1) - base.mean()
        wdf = pd.DataFrame({"start": [i for i, _ in windows], "end": [j for _, j in windows], "delta": delta})
        wdf = wdf.merge(rp, left_on=["start", "end"], right_on=["layer_i", "layer_j"])
        rho = wdf["R"].to_numpy()
        cka = wdf["cka_full"].to_numpy()
        dl = wdf["delta"].to_numpy()
        order = [windows.index((int(r.start), int(r.end))) for r in wdf.itertuples(index=False)]
        rys_m = rys[order]

        # per-window McNemar (paired) -> real noise floor and significance counts
        b = (base[None, :] & ~rys_m).sum(1)
        c = (~base[None, :] & rys_m).sum(1)
        disc = b + c
        se = np.sqrt(np.maximum(disc, 1)) / n
        pvals = np.array([binomtest(int(cc), int(dd), 0.5).pvalue if dd > 0 else 1.0
                          for cc, dd in zip(c, disc, strict=True)])
        sig_pos = int(((dl > 0) & (pvals < 0.05)).sum())
        sig_neg = int(((dl < 0) & (pvals < 0.05)).sum())

        # pure-noise expectation for best-of-W (using median paired SE as sigma)
        sigma = float(np.median(se))
        W = len(dl)
        exp_max_noise = sigma * np.sqrt(2 * np.log(W))

        # paired bootstrap over problems for the geometry->effect correlations
        sp_rho = spearmanr(dl, rho).statistic
        sp_cka = spearmanr(dl, cka).statistic
        boot_rho = np.empty(args.boot)
        boot_cka = np.empty(args.boot)
        boot_best = np.empty(args.boot)
        for t in range(args.boot):
            idx = rng.integers(0, n, n)
            db = rys_m[:, idx].mean(1) - base[idx].mean()
            boot_rho[t] = spearmanr(db, rho).statistic
            boot_cka[t] = spearmanr(db, cka).statistic
            boot_best[t] = db.max()

        def ci(a):
            return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

        out = {
            "model": tag, "L": L, "n_problems": n, "n_windows": W,
            "baseline_acc": round(float(base.mean()), 3),
            "mean_delta": round(float(dl.mean()), 3),
            "best_delta": round(float(dl.max()), 3),
            "worst_delta": round(float(dl.min()), 3),
            "median_paired_SE": round(sigma, 4),
            "expected_best_under_noise": round(float(exp_max_noise), 3),
            "n_windows_sig_pos_p05": sig_pos,
            "n_windows_sig_neg_p05": sig_neg,
            "spearman_delta_rho": round(float(sp_rho), 3),
            "spearman_delta_rho_ci95": ci(boot_rho),
            "spearman_delta_cka": round(float(sp_cka), 3),
            "spearman_delta_cka_ci95": ci(boot_cka),
            "best_delta_ci95": ci(boot_best),
        }
        print(json.dumps(out, indent=2), flush=True)
        Path(f"{run_dir}/significance.json").write_text(json.dumps(out, indent=2))
        np.savez_compressed(f"{run_dir}/correctness.npz", base=base, rys=rys_m,
                            start=wdf["start"].to_numpy(), end=wdf["end"].to_numpy(),
                            rho=rho, cka=cka, delta=dl)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
