"""Unified robust geometry controls and predictive RYS theory, model-agnostic.

This is the model-agnostic companion of ``run_guesstimation_rys.py``. Given any
HuggingFace decoder-only model it captures the residual stream on a fixed
prompt set and computes the full battery of geometry predictors of the RYS
effect, under two activation regimes:

- **raw** activations (massive-activation subspace included);
- **robustified** activations (sink token dropped, top-``k`` highest-variance
  dimensions clamped to their mean, dimensions globally standardised).

The robustified regime tests whether the *bulk* geometry — independent of the
few saturating dimensions that push raw CKA toward 1 — still predicts the
effect. Across the Pythia/Qwen ladder suppressing that subspace sometimes
*strengthens* and sometimes *weakens* the geometry→effect correlation, so both
regimes are reported.

For each regime it computes:

1. The residual-force decomposition (``R``, ``Q``, ``cos_phi``, ``cos_psi``,
   ``cka_full``) via :mod:`rys.residual_force` — the existing theory.
2. The **three-regime decomposition** from :mod:`rys.geometry_predictors`:
   - ``coherent_force``    — the component of the residual force aligned with
     the identity stream (RYS-productive);
   - ``incoherent_force``  — the orthogonal cross-term (RYS-destructive);
   - ``coherence_ratio``   — their ratio, the headline geometric predictor;
   - ``junction_misalignment`` — how far the band-output manifold sits from
     the band-input manifold (boundary condition).
3. The **composite RYS safety score** combining coherence, junction
   misalignment, and (when a matching ``delta_score_long.csv`` with the band
   Jacobian exists) the Jacobian contraction modulus.

If a matching ``delta_score_long.csv`` (and optionally
``jacobian_sigma_max`` via ``delta_score_long.csv``) from a prior
``run_guesstimation_rys`` run is found — or one is supplied via
``--delta-long`` — every predictor is joined to the RYS deltas and a
functional summary (Spearman correlations + best window) is emitted per
regime. No RYS window is reswept here: the deltas are fixed and only the
geometry is recomputed under each control.

Outputs land in ``results/LLM/<family>/robust_geometry_controls/<tag>/`` to
mirror the guesstimation script's layout.

Examples
--------
::

    uv run python scripts/run_robust_geometry_controls.py --model EleutherAI/pythia-70m
    uv run python scripts/run_robust_geometry_controls.py --model Qwen/Qwen3-0.6B --dtype float32
    uv run python scripts/run_robust_geometry_controls.py --model meta-llama/Llama-3.2-1B \
        --delta-long results/LLM/llama/guesstimation_rys/Llama-3.2-1B/<ts>/delta_score_long.csv

Smoke test (tiny subset, any model)::

    uv run python scripts/run_robust_geometry_controls.py --model EleutherAI/pythia-70m \
        --capture-n 4 --top-k-dims 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix
from rys.eval_core import (
    cka_device_for,
    functional_corr,
    infer_family,
    latest_delta_long,
    resolve_device,
    resolve_dtype,
    save_heatmap,
    square_from_long,
)
from rys.geometry_predictors import (
    coherence_table,
    composite_safety_score,
    junction_misalignment_table,
    massive_activation_stats,
    robustify_activations,
)
from rys.gsm8k_mc import load_causal_lm
from rys.guesstimation import make_guesstimation_questions
from rys.residual_force import residual_force_long
from rys.theory_validation import theory_fit


def latest_jacobian_long(family: str, tag: str) -> Path | None:
    """Find the matching delta_score_long.csv (which carries jacobian_sigma_max)."""
    return latest_delta_long(family, tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="HuggingFace checkpoint.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "int8", "int4"])
    parser.add_argument("--capture-n", type=int, default=44, help="Prompts used for the CKA connectome.")
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--top-k-dims", type=int, default=5, help="Number of top-variance dims to clamp in the robustified regime.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta-long", type=Path, default=None, help="Explicit delta_score_long.csv (carries jacobian_sigma_max).")
    parser.add_argument("--family", default="", help="Output family bucket; auto-inferred if empty.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/LLM"))
    args = parser.parse_args()

    device = resolve_device()
    torch_dtype = resolve_dtype(args.dtype)
    tag = args.model.split("/")[-1]
    family = args.family or infer_family(args.model)
    run_dir = args.output_dir / family / "robust_geometry_controls" / tag / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device {device}, dtype {args.dtype}, family {family}, run_dir {run_dir}", flush=True)

    model, tok = load_causal_lm(
        args.model, device=device, dtype=torch_dtype,
        load_in_8bit=args.dtype == "int8", load_in_4bit=args.dtype == "int4",
    )
    L = len(model.model.layers)
    cka_device = cka_device_for(device)

    questions = make_guesstimation_questions(seed=args.seed)[: args.capture_n]
    prompts = pd.DataFrame(
        {"prompt_id": [f"q{i}" for i in range(len(questions))], "prompt": [q["question"] for q in questions]}
    )
    activations = capture_residual_stream(model, tok, prompts, batch_size=args.capture_batch_size, device=device)
    print(f"{tag}: L={L}, captured {len(activations)//L} prompts", flush=True)

    # Massive-activation diagnostics.
    stats = massive_activation_stats(activations)
    stats_public = {k: v for k, v in stats.items() if k not in {"mean", "std", "order"}}
    (run_dir / "massive_activation_stats.json").write_text(json.dumps(stats_public, indent=2))
    print(f"  top1 dim var frac = {stats_public['top1']:.3f}, top5 = {stats_public['top5']:.3f}", flush=True)

    # Locate the matching RYS deltas + Jacobian, if any.
    delta_path = args.delta_long if args.delta_long is not None else latest_delta_long(family, tag)
    if args.delta_long is not None and not args.delta_long.exists():
        raise FileNotFoundError(f"--delta-long does not exist: {args.delta_long}")
    delta_long = pd.read_csv(delta_path) if delta_path is not None else None
    has_jacobian = delta_long is not None and "jacobian_sigma_max" in delta_long.columns
    if delta_long is not None:
        print(f"  joined to deltas: {delta_path} (jacobian={'yes' if has_jacobian else 'no'})", flush=True)

    summary: dict[str, object] = {
        "model": args.model, "family": family, "L": L, "dtype": args.dtype,
        "capture_n": len(questions), "top_k_dims": args.top_k_dims,
        "delta_source": str(delta_path) if delta_long is not None else None,
        "has_jacobian": has_jacobian,
        "massive_activation_stats": stats_public,
        "regimes": {},
    }
    all_merged = []

    for regime in ("raw", "robust"):
        print(f"{tag}: {regime}", flush=True)
        if regime == "raw":
            acts_v = activations
        else:
            acts_v = robustify_activations(
                activations, stats=stats, top_k_dims=args.top_k_dims,
                drop_first_token=True, global_standardize=True,
            )

        # 1. Existing residual-force theory (rho/phi/Q/psi + CKA).
        rho_phi = residual_force_long(acts_v, upper_only=True, dtype=torch.float64)
        cka = cka_matrix(acts_v, unbiased=False, device=cka_device)
        rho_phi.to_csv(run_dir / f"rho_phi_{regime}.csv", index=False)
        cka.to_csv(run_dir / f"cka_{regime}.csv")
        save_heatmap(cka, run_dir / f"cka_{regime}.png",
                      title=f"CKA connectome ({tag}, {regime})", cmap="viridis",
                      cbar_label="CKA", diverging=False)

        # 2. Three-regime decomposition: coherent/incoherent force + junction.
        coh = coherence_table(acts_v, device=cka_device)
        junction = junction_misalignment_table(acts_v)
        coh.to_csv(run_dir / f"coherence_{regime}.csv", index=False)
        junction.to_csv(run_dir / f"junction_misalignment_{regime}.csv", index=False)

        # Heatmaps of the new predictors.
        for col, title, cmap, div in [
            ("coherence_ratio", f"Coherence ratio K ({tag}, {regime})", "RdBu_r", True),
            ("coherent_force", f"Coherent force (RYS-productive) ({tag}, {regime})", "viridis", False),
            ("incoherent_force", f"Incoherent force (RYS-destructive) ({tag}, {regime})", "viridis", False),
        ]:
            save_heatmap(square_from_long(coh, L, col, "layer_i", "layer_j"),
                        run_dir / f"{col}_{regime}.png", title=title, cmap=cmap,
                        cbar_label=col, diverging=div)
        save_heatmap(square_from_long(junction, L, "junction_misalignment", "layer_i", "layer_j"),
                     run_dir / f"junction_misalignment_{regime}.png",
                     title=f"Junction misalignment ({tag}, {regime})", cmap="viridis",
                     cbar_label="m_ij", diverging=False)

        # 3. Composite safety score (with Jacobian if available).
        jacobian_df = None
        if has_jacobian:
            jacobian_df = delta_long[["start", "end", "jacobian_sigma_max"]].copy()
        composite = composite_safety_score(coh, junction, jacobian_df)
        composite.to_csv(run_dir / f"composite_safety_{regime}.csv", index=False)
        save_heatmap(square_from_long(composite, L, "rys_safety_score", "layer_i", "layer_j"),
                     run_dir / f"rys_safety_score_{regime}.png",
                     title=f"Composite RYS safety score ({tag}, {regime})", cmap="RdBu_r",
                     cbar_label="S_ij (higher = safer to duplicate)", diverging=True)

        regime_summary: dict[str, object] = {
            "median_offdiag_cka": float(np.nanmedian(cka.to_numpy(float)[np.triu_indices_from(cka, k=1)])),
            "theory_fit": theory_fit(rho_phi),
        }

        # 4. Functional correlations with the RYS deltas, if available.
        if delta_long is not None:
            predictors = composite if has_jacobian else composite
            merged, functional = functional_corr(delta_long, predictors)
            merged.insert(0, "regime", regime)
            all_merged.append(merged)
            regime_summary["functional"] = functional

        summary["regimes"][regime] = regime_summary

    if all_merged:
        pd.concat(all_merged, ignore_index=True).to_csv(
            run_dir / "delta_geometry_by_regime.csv", index=False
        )

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Wrote {run_dir}", flush=True)


if __name__ == "__main__":
    main()
