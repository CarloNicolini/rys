"""Is the Pythia CKA plateau real, or driven by a few strong activations?

Linear CKA is dominated by the top principal components, so a handful of
high-variance *outlier dimensions* (rogue dimensions) or a few high-norm *sink
tokens* can make many layers look near-identical (CKA ~ 1) even when their bulk
representations differ. This script captures the residual stream and:

1. **Quantifies dominance** per layer: the effective rank (participation ratio of
   the activation covariance), the variance fraction in the top principal
   component, the global variance fraction carried by the top few dimensions, and
   the ratio of the largest token norm to the median (the sink-token signature).

2. **Recomputes the connectome and the rho/phi theory fit under controls** that
   *preserve the residual telescoping* (so the rho/phi decomposition stays valid):
   - ``raw``               : baseline (what the experiments used);
   - ``drop_top_dims``     : remove the top-k global outlier dimensions;
   - ``global_standardize``: divide every layer by the same per-dimension std;
   - ``drop_first_token``  : drop the first (sink) token of each prompt.

If the plateau (median off-diagonal CKA) and the rho/phi Spearman survive these
controls, the geometry is genuine residual-stream stationarity. If they collapse,
the connectome was measuring outlier-dimension / sink-token geometry.

Run on muletto for the larger checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix
from rys.eval_core import resolve_device
from rys.gsm8k_mc import load_causal_lm
from rys.guesstimation import make_guesstimation_questions
from rys.residual_force import residual_force_long
from rys.theory_validation import theory_fit

DIVERSE_PROMPTS = [
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War in Europe.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    p = a[0]\n    return quicksort([x for x in a[1:] if x < p]) + [p] + quicksort([x for x in a[1:] if x >= p])",
    "She wandered the rain-soaked streets of Lisbon, searching for a café she half-remembered.",
    "The mitochondrion is often called the powerhouse of the cell.",
    "In 1969, Apollo 11 landed the first humans on the Moon.",
    "A haiku has three lines with five, seven, and five syllables.",
    "Bitcoin uses a proof-of-work consensus mechanism to secure its ledger.",
    "The boiling point of water decreases at higher altitudes.",
    "He whispered that the old lighthouse had not been lit in forty years.",
    "Quantum entanglement links the states of two particles across any distance.",
    "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
    "Shakespeare wrote both tragedies, like Hamlet, and comedies, like Twelfth Night.",
    "The stock market fell sharply after the central bank raised interest rates.",
    "Migrating arctic terns travel from pole to pole each year.",
    "import numpy as np\nx = np.linspace(0, 1, 100)\ny = np.sin(2 * np.pi * x)",
    "The Amazon rainforest produces a significant fraction of the world's oxygen.",
    "Grandmother's garden smelled of jasmine and freshly turned earth.",
    "Tectonic plates drift a few centimetres each year, reshaping continents.",
    "The committee voted seven to two in favour of the new zoning proposal.",
    "Beethoven composed his ninth symphony after losing most of his hearing.",
    "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
    "The detective noticed the muddy footprints leading away from the locked study.",
    "Gravitational waves were first directly detected by LIGO in 2015.",
    "Please remember to water the ferns and feed the cat while we are away.",
    "The Silk Road connected merchants from China to the Mediterranean for centuries.",
    "Neural networks learn by adjusting weights to minimise a loss function.",
    "The volcano erupted at dawn, blanketing the valley in fine grey ash.",
    "Coffee contains caffeine, a stimulant that blocks adenosine receptors.",
    "Her thesis argued that medieval trade routes shaped modern city locations.",
    "The marathon route winds through five historic neighbourhoods of the city.",
    "Light from the nearest star beyond the Sun takes over four years to reach us.",
]


def pooled_by_layer(acts: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for layer, g in acts.groupby("layer"):
        out[int(layer)] = np.concatenate([np.atleast_2d(np.asarray(a)) for a in g["activation"]], axis=0)
    return out


def diagnostics(acts: pd.DataFrame) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    pooled = pooled_by_layer(acts)
    allX = np.concatenate(list(pooled.values()), axis=0).astype(np.float64)
    var = allX.var(axis=0)
    order = np.argsort(var)[::-1]
    total = float(var.sum())

    rows = []
    for layer, X in pooled.items():
        Xc = (X - X.mean(0)).astype(np.float64)
        s = np.linalg.svd(Xc, compute_uv=False)
        lam = s**2
        pr = float((lam.sum() ** 2) / np.square(lam).sum()) if lam.sum() > 0 else 0.0
        tok_norm = np.linalg.norm(X, axis=1)
        rows.append(
            {
                "layer": layer,
                "eff_rank": pr,
                "top1_pc_var_frac": float(lam.max() / lam.sum()) if lam.sum() > 0 else float("nan"),
                "tok_norm_max_over_median": float(tok_norm.max() / np.median(tok_norm)),
            }
        )
    diag = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    glob = {
        "d_model": int(var.size),
        "top1_dim_var_frac": float(var[order[0]] / total),
        "top5_dim_var_frac": float(var[order[:5]].sum() / total),
        "top10_dim_var_frac": float(var[order[:10]].sum() / total),
        "top_dims": order[:10].astype(int).tolist(),
    }
    glob_std = np.sqrt(var + 1e-8)
    return diag, glob, order, glob_std


def transform(acts: pd.DataFrame, kind: str, *, top_dims=None, glob_std=None) -> pd.DataFrame:
    a = acts.copy()
    if kind == "raw":
        return a
    if kind == "drop_top_dims":
        a["activation"] = [np.delete(np.atleast_2d(np.asarray(x)), top_dims, axis=1) for x in a["activation"]]
    elif kind == "global_standardize":
        a["activation"] = [np.atleast_2d(np.asarray(x)) / glob_std for x in a["activation"]]
    elif kind == "drop_first_token":
        a["activation"] = [
            (np.atleast_2d(np.asarray(x))[1:] if np.atleast_2d(np.asarray(x)).shape[0] > 1 else np.atleast_2d(np.asarray(x)))
            for x in a["activation"]
        ]
    else:
        raise ValueError(kind)
    return a


def plateau_score(cka: pd.DataFrame) -> float:
    M = cka.to_numpy()
    iu = np.triu_indices_from(M, k=1)
    return float(np.median(M[iu]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "int8", "int4"])
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--top-k-dims", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/cka_diagnostics")
    args = ap.parse_args()

    device = resolve_device()
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int8": None, "int4": None}[args.dtype]
    tag = args.model.split("/")[-1]
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_causal_lm(
        args.model,
        device=device,
        dtype=torch_dtype,
        load_in_8bit=args.dtype == "int8",
        load_in_4bit=args.dtype == "int4",
    )
    cka_device = device if device.type == "cuda" else "cpu"

    qs = make_guesstimation_questions(seed=args.seed)[: args.n_prompts]
    prompt_sets = {
        "arith": [q["question"] for q in qs],  # homogeneous (what the connectomes used)
        "diverse": DIVERSE_PROMPTS[: args.n_prompts],  # varied topics/lengths/code/prose
    }

    summary = {"model": args.model, "dtype": args.dtype, "prompt_sets": {}}
    for set_name, texts in prompt_sets.items():
        prompts = pd.DataFrame({"prompt_id": [f"p{i}" for i in range(len(texts))], "prompt": texts})
        acts = capture_residual_stream(model, tok, prompts, batch_size=8, device=device)
        diag, glob, order, glob_std = diagnostics(acts)
        top_dims = order[: args.top_k_dims]
        diag.to_csv(out / f"per_layer_diagnostics_{set_name}.csv", index=False)

        variants = {}
        for kind in ["raw", "drop_top_dims", "global_standardize", "drop_first_token"]:
            a = transform(acts, kind, top_dims=top_dims, glob_std=glob_std)
            cka = cka_matrix(a, unbiased=False, device=cka_device)
            cka.to_csv(out / f"cka_{set_name}_{kind}.csv")
            fit = theory_fit(residual_force_long(a, upper_only=True, dtype=torch.float64))
            variants[kind] = {
                "median_offdiag_cka": plateau_score(cka),
                "spearman_plateau_pred": fit.get("spearman_plateau_pred"),
                "median_rho": fit.get("median_R"),
            }
        summary["prompt_sets"][set_name] = {
            "n_prompts": len(texts),
            "global_dim_dominance": glob,
            "median_eff_rank": float(diag["eff_rank"].median()),
            "median_top1_pc_var_frac": float(diag["top1_pc_var_frac"].median()),
            "median_tok_norm_max_over_median": float(diag["tok_norm_max_over_median"].median()),
            "variants": variants,
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
