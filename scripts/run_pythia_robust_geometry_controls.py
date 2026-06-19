"""Test rho/phi RYS prediction after suppressing massive-activation geometry.

Pythia residual streams at 410M+ are dominated by a few high-variance dimensions
and sometimes by first-token/sink-token rows. Raw linear CKA can therefore
saturate near one for the wrong reason: it mostly measures the shared nuisance
subspace rather than the bulk residual dynamics.

This script keeps the already-computed downstream RYS deltas fixed and recomputes
only the geometry under *global* controls that preserve the residual telescoping:

- raw;
- drop_first_token;
- drop_top_dims;
- clamp_top_dims_to_mean;
- global_standardize;
- global_standardize plus the sink/top-dimension controls.

Each transform is fixed across all layers. That is important: if ``T`` is fixed,
then ``T(h_j - h_i) = T h_j - T h_i``, so the rho/phi residual-force theory is
still being tested in a different inner product. Per-layer transforms would not
have this property.

Outputs land in ``results/robust_geometry_controls/<model>/`` and include one
summary JSON, one per-variant rho/phi table, one per-variant CKA matrix, and a
single long CSV comparing all variants against the saved RYS deltas.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix
from rys.gsm8k_mc import load_pythia
from rys.guesstimation import make_guesstimation_questions
from rys.residual_force import residual_force_long
from rys.theory_validation import theory_fit


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def latest_delta_long(model_tag: str) -> Path | None:
    runs = sorted(glob.glob(f"results/pythia_guesstimation_rys/{model_tag}/*/delta_score_long.csv"))
    return Path(runs[-1]) if runs else None


def arrays(acts: pd.DataFrame) -> list[np.ndarray]:
    return [np.atleast_2d(np.asarray(x)).astype(np.float64, copy=False) for x in acts["activation"]]


def global_stats(acts: pd.DataFrame) -> dict[str, object]:
    xs = arrays(acts)
    all_x = np.concatenate(xs, axis=0)
    mean = all_x.mean(axis=0)
    var = all_x.var(axis=0)
    std = np.sqrt(var + 1e-8)
    order = np.argsort(var)[::-1]
    total = float(var.sum())
    if total <= 0:
        fractions = {"top1_dim_var_frac": float("nan"), "top5_dim_var_frac": float("nan"), "top10_dim_var_frac": float("nan")}
    else:
        fractions = {
            "top1_dim_var_frac": float(var[order[0]] / total),
            "top5_dim_var_frac": float(var[order[:5]].sum() / total),
            "top10_dim_var_frac": float(var[order[:10]].sum() / total),
        }
    return {
        "mean": mean,
        "std": std,
        "order": order,
        "d_model": int(var.size),
        "top_dims": order[:10].astype(int).tolist(),
        **fractions,
    }


def transform_activations(
    acts: pd.DataFrame,
    variant: str,
    *,
    stats: dict[str, object],
    top_k_dims: int,
) -> pd.DataFrame:
    top_dims = np.asarray(stats["order"][:top_k_dims], dtype=int)
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])

    def transform_one(x: np.ndarray) -> np.ndarray:
        y = np.array(x, dtype=np.float64, copy=True)
        if "drop_first_token" in variant and y.shape[0] > 1:
            y = y[1:]
        if "clamp_top_dims" in variant and top_k_dims > 0:
            y[:, top_dims] = mean[top_dims]
        if "global_standardize" in variant:
            y = y / std
        if "drop_top_dims" in variant and top_k_dims > 0:
            y = np.delete(y, top_dims, axis=1)
        return y

    out = acts.copy()
    if variant == "raw":
        return out
    out["activation"] = [transform_one(x) for x in arrays(acts)]
    return out


def median_offdiag(cka: pd.DataFrame) -> float:
    m = cka.to_numpy(dtype=float)
    iu = np.triu_indices_from(m, k=1)
    return float(np.nanmedian(m[iu]))


def functional_corr(delta_long: pd.DataFrame, rp: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    mg = delta_long.merge(rp, left_on=["start", "end"], right_on=["layer_i", "layer_j"], how="inner")

    def corr(column: str) -> float:
        if len(mg) < 3 or mg["delta_score"].std() == 0 or mg[column].std() == 0:
            return float("nan")
        return float(spearmanr(mg["delta_score"], mg[column]).statistic)

    mg = mg.copy()
    mg["Q2"] = mg["Q"] ** 2
    summary = {
        "n_windows": int(len(mg)),
        "spearman_delta_rho": corr("R"),
        "spearman_delta_cka": corr("cka_full"),
        "spearman_delta_plateau_pred": corr("one_minus_cka_plateau"),
        "spearman_delta_Qpsi": corr("one_minus_cka_Qpsi"),
        "spearman_delta_Rphi": corr("one_minus_cka_Rphi"),
        "spearman_delta_Q2": corr("Q2"),
        "best_delta": float(mg["delta_score"].max()) if len(mg) else float("nan"),
    }
    if len(mg):
        best = mg.loc[mg["delta_score"].idxmax()]
        summary["best_window_start"] = int(best["start"])
        summary["best_window_end"] = int(best["end"])
    return mg, summary


def variants_for(top_k_dims: int) -> list[str]:
    variants = [
        "raw",
        "drop_first_token",
        "global_standardize",
        "global_standardize_drop_first_token",
    ]
    if top_k_dims > 0:
        variants.extend(
            [
                "drop_top_dims",
                "clamp_top_dims",
                "global_standardize_drop_top_dims",
                "global_standardize_clamp_top_dims",
                "global_standardize_drop_top_dims_drop_first_token",
            ]
        )
    return variants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "int4"])
    ap.add_argument("--capture-n", type=int, default=44)
    ap.add_argument("--capture-batch-size", type=int, default=8)
    ap.add_argument("--top-k-dims", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/robust_geometry_controls")
    args = ap.parse_args()

    device = resolve_device()
    load_in_4bit = args.dtype == "int4"
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int4": None}[args.dtype]
    tag = args.model.split("/")[-1]
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_pythia(args.model, device=device, dtype=torch_dtype, load_in_4bit=load_in_4bit)
    cka_device = device if device.type == "cuda" else "cpu"

    questions = make_guesstimation_questions(seed=args.seed)[: args.capture_n]
    prompts = pd.DataFrame(
        {"prompt_id": [f"q{i}" for i in range(len(questions))], "prompt": [q["question"] for q in questions]}
    )
    acts = capture_residual_stream(
        model,
        tok,
        prompts,
        batch_size=args.capture_batch_size,
        device=device,
    )
    stats = global_stats(acts)
    stats_public = {
        key: value
        for key, value in stats.items()
        if key not in {"mean", "std", "order"}
    }
    (out / "massive_activation_stats.json").write_text(json.dumps(stats_public, indent=2))

    delta_path = latest_delta_long(tag)
    delta_long = pd.read_csv(delta_path) if delta_path is not None else None

    summary: dict[str, object] = {
        "model": args.model,
        "dtype": args.dtype,
        "capture_n": len(questions),
        "top_k_dims": args.top_k_dims,
        "delta_source": str(delta_path) if delta_path is not None else None,
        "massive_activation_stats": stats_public,
        "variants": {},
    }
    all_merged = []

    for variant in variants_for(args.top_k_dims):
        print(f"{tag}: {variant}", flush=True)
        acts_v = transform_activations(acts, variant, stats=stats, top_k_dims=args.top_k_dims)
        rp = residual_force_long(acts_v, upper_only=True, dtype=torch.float64)
        cka = cka_matrix(acts_v, unbiased=False, device=cka_device)
        rp.to_csv(out / f"rho_phi_{variant}.csv", index=False)
        cka.to_csv(out / f"cka_{variant}.csv")

        variant_summary: dict[str, object] = {
            "median_offdiag_cka": median_offdiag(cka),
            "theory_fit": theory_fit(rp),
        }
        if delta_long is not None:
            merged, functional = functional_corr(delta_long, rp)
            merged.insert(0, "variant", variant)
            all_merged.append(merged)
            variant_summary["functional"] = functional
        summary["variants"][variant] = variant_summary

    if all_merged:
        pd.concat(all_merged, ignore_index=True).to_csv(out / "delta_geometry_by_variant.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
