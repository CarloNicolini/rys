"""Generic robust geometry controls for decoder-only Hugging Face LMs.

This is the model-agnostic companion of ``run_pythia_robust_geometry_controls.py``.
It recomputes only representation geometry from a fixed prompt set, under global
linear controls that preserve residual telescoping:

- raw;
- drop_first_token;
- drop_top_dims;
- clamp_top_dims_to_mean;
- global_standardize;
- global_standardize plus the sink/top-dimension controls.

If a matching ``delta_score_long.csv`` exists, or ``--delta-long`` is provided,
the script also joins every geometry variant to the already-computed RYS deltas.
No RYS window is reswept here.
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
from rys.gsm8k_mc import load_causal_lm
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
    patterns = [
        f"results/qwen_guesstimation_rys/{model_tag}/*/delta_score_long.csv",
        f"results/pythia_guesstimation_rys/{model_tag}/*/delta_score_long.csv",
        f"results/pythia_guesstimation_rys/pythia-{model_tag}/*/delta_score_long.csv",
        f"results/*guesstimation*/{model_tag}/*/delta_score_long.csv",
    ]
    runs = sorted(path for pattern in patterns for path in glob.glob(pattern))
    return Path(runs[-1]) if runs else None


def arrays(activations: pd.DataFrame) -> list[np.ndarray]:
    return [np.atleast_2d(np.asarray(x)).astype(np.float64, copy=False) for x in activations["activation"]]


def global_stats(activations: pd.DataFrame) -> dict[str, object]:
    xs = arrays(activations)
    all_x = np.concatenate(xs, axis=0)
    mean = all_x.mean(axis=0)
    var = all_x.var(axis=0)
    std = np.sqrt(var + 1e-8)
    order = np.argsort(var)[::-1]
    total = float(var.sum())
    if total <= 0:
        fractions = {
            "top1_dim_var_frac": float("nan"),
            "top5_dim_var_frac": float("nan"),
            "top10_dim_var_frac": float("nan"),
        }
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
    activations: pd.DataFrame,
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

    out = activations.copy()
    if variant == "raw":
        return out
    out["activation"] = [transform_one(x) for x in arrays(activations)]
    return out


def median_offdiag(cka: pd.DataFrame) -> float:
    m = cka.to_numpy(dtype=float)
    iu = np.triu_indices_from(m, k=1)
    return float(np.nanmedian(m[iu]))


def functional_corr(delta_long: pd.DataFrame, rho_phi: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = delta_long.merge(rho_phi, left_on=["start", "end"], right_on=["layer_i", "layer_j"], how="inner")

    def corr(column: str) -> float:
        if len(merged) < 3 or merged["delta_score"].std() == 0 or merged[column].std() == 0:
            return float("nan")
        return float(spearmanr(merged["delta_score"], merged[column]).statistic)

    merged = merged.copy()
    merged["Q2"] = merged["Q"] ** 2
    summary = {
        "n_windows": int(len(merged)),
        "spearman_delta_rho": corr("R"),
        "spearman_delta_cka": corr("cka_full"),
        "spearman_delta_Qpsi": corr("one_minus_cka_Qpsi"),
        "spearman_delta_Rphi": corr("one_minus_cka_Rphi"),
        "spearman_delta_Q2": corr("Q2"),
        "best_delta": float(merged["delta_score"].max()) if len(merged) else float("nan"),
    }
    if len(merged):
        best = merged.loc[merged["delta_score"].idxmax()]
        summary["best_window_start"] = int(best["start"])
        summary["best_window_end"] = int(best["end"])
    return merged, summary


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Hugging Face causal LM checkpoint.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "int8", "int4"])
    parser.add_argument("--capture-n", type=int, default=44)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--top-k-dims", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta-long", type=Path, default=None, help="Optional explicit delta_score_long.csv.")
    parser.add_argument("--output-dir", default="results/robust_geometry_controls")
    args = parser.parse_args()

    device = resolve_device()
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int8": None, "int4": None}[
        args.dtype
    ]
    tag = args.model.split("/")[-1]
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_causal_lm(
        args.model,
        device=device,
        dtype=torch_dtype,
        load_in_8bit=args.dtype == "int8",
        load_in_4bit=args.dtype == "int4",
    )
    cka_device = device if device.type == "cuda" else "cpu"

    questions = make_guesstimation_questions(seed=args.seed)[: args.capture_n]
    prompts = pd.DataFrame(
        {"prompt_id": [f"q{i}" for i in range(len(questions))], "prompt": [q["question"] for q in questions]}
    )
    activations = capture_residual_stream(
        model,
        tokenizer,
        prompts,
        batch_size=args.capture_batch_size,
        device=device,
    )

    stats = global_stats(activations)
    stats_public = {key: value for key, value in stats.items() if key not in {"mean", "std", "order"}}
    (out / "massive_activation_stats.json").write_text(json.dumps(stats_public, indent=2))

    delta_path = args.delta_long or latest_delta_long(tag)
    delta_long = pd.read_csv(delta_path) if delta_path is not None and delta_path.exists() else None

    summary: dict[str, object] = {
        "model": args.model,
        "dtype": args.dtype,
        "capture_n": len(questions),
        "top_k_dims": args.top_k_dims,
        "delta_source": str(delta_path) if delta_long is not None else None,
        "massive_activation_stats": stats_public,
        "variants": {},
    }
    all_merged = []

    for variant in variants_for(args.top_k_dims):
        print(f"{tag}: {variant}", flush=True)
        activations_v = transform_activations(activations, variant, stats=stats, top_k_dims=args.top_k_dims)
        rho_phi = residual_force_long(activations_v, upper_only=True, dtype=torch.float64)
        cka = cka_matrix(activations_v, unbiased=False, device=cka_device)
        rho_phi.to_csv(out / f"rho_phi_{variant}.csv", index=False)
        cka.to_csv(out / f"cka_{variant}.csv")

        variant_summary: dict[str, object] = {
            "median_offdiag_cka": median_offdiag(cka),
            "theory_fit": theory_fit(rho_phi),
        }
        if delta_long is not None:
            merged, functional = functional_corr(delta_long, rho_phi)
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
