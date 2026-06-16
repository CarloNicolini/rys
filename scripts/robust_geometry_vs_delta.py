"""Do the RYS structural and functional laws survive massive-activation-robust CKA?

The raw residual stream of Pythia >=410M is effectively rank-1 (a single massive
activation dimension), which makes linear CKA ~1 between all layers and inflates
the connectome plateau / deflates rho. This script recomputes the geometry with a
telescoping-preserving fix -- global per-dimension standardization (divide every
layer by the same per-dim std) -- and checks both laws against the saved per-window
RYS deltas:

- **structural**: Spearman(plateau predictor, 1-CKA), raw vs standardized;
- **functional**: Spearman(Delta, rho) and Spearman(Delta, CKA), raw vs standardized,
  where Delta is the already-computed guesstimation RYS effect per window.

Outputs a per-model comparison plus the standardized rho/phi table and connectome
(for robust figures). Run on muletto for the larger checkpoints.
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


def global_per_dim_std(acts: pd.DataFrame) -> np.ndarray:
    allX = np.concatenate([np.atleast_2d(np.asarray(a)) for a in acts["activation"]], axis=0).astype(np.float64)
    return np.sqrt(allX.var(axis=0) + 1e-8)


def standardize(acts: pd.DataFrame, std: np.ndarray) -> pd.DataFrame:
    a = acts.copy()
    a["activation"] = [np.atleast_2d(np.asarray(x)) / std for x in a["activation"]]
    return a


def latest_delta_long(model_tag: str) -> Path | None:
    runs = sorted(glob.glob(f"results/pythia_guesstimation_rys/pythia-{model_tag}/*/delta_score_long.csv"))
    return Path(runs[-1]) if runs else None


def functional_corr(delta_long: pd.DataFrame, rp: pd.DataFrame) -> dict:
    mg = delta_long.merge(rp, left_on=["start", "end"], right_on=["layer_i", "layer_j"], how="inner")
    return {
        "n_windows": int(len(mg)),
        "spearman_delta_rho": float(spearmanr(mg["delta_score"], mg["R"]).statistic),
        "spearman_delta_cka": float(spearmanr(mg["delta_score"], mg["cka_full"]).statistic),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-410m")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "int4"])
    ap.add_argument("--capture-n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/robust_geometry")
    args = ap.parse_args()

    device = resolve_device()
    load_in_4bit = args.dtype == "int4"
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int4": None}[args.dtype]
    tag = args.model.split("/")[-1]
    short = tag.split("-", 1)[1]
    out = Path(args.output_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load_pythia(args.model, device=device, dtype=torch_dtype, load_in_4bit=load_in_4bit)
    cka_device = device if device.type == "cuda" else "cpu"

    # Same connectome prompts the guesstimation run used.
    qs = make_guesstimation_questions(seed=args.seed)[: args.capture_n]
    prompts = pd.DataFrame({"prompt_id": [f"q{i}" for i in range(len(qs))], "prompt": [q["question"] for q in qs]})
    acts = capture_residual_stream(model, tok, prompts, batch_size=8, device=device)
    std = global_per_dim_std(acts)
    acts_std = standardize(acts, std)

    rp_raw = residual_force_long(acts, upper_only=True, dtype=torch.float64)
    rp_std = residual_force_long(acts_std, upper_only=True, dtype=torch.float64)
    cka_raw = cka_matrix(acts, unbiased=False, device=cka_device)
    cka_std = cka_matrix(acts_std, unbiased=False, device=cka_device)
    rp_std.to_csv(out / "rho_phi_standardized.csv", index=False)
    cka_std.to_csv(out / "cka_standardized.csv")
    cka_raw.to_csv(out / "cka_raw.csv")

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "structural": {
            "spearman_plateau_raw": theory_fit(rp_raw).get("spearman_plateau_pred"),
            "spearman_plateau_std": theory_fit(rp_std).get("spearman_plateau_pred"),
            "median_rho_raw": float(rp_raw["R"].median()),
            "median_rho_std": float(rp_std["R"].median()),
        },
    }

    delta_path = latest_delta_long(short)
    if delta_path is not None:
        delta_long = pd.read_csv(delta_path)
        summary["functional"] = {
            "delta_source": str(delta_path),
            "raw": functional_corr(delta_long, rp_raw),
            "standardized": functional_corr(delta_long, rp_std),
        }
    else:
        summary["functional"] = "no guesstimation delta_score_long found for this model"

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
