"""Numerical verification of the CKA/RYS theory.

Runs:
1. Audit of saved toy + Pythia summaries
2. Synthetic algebra checks (exact CKA vs plateau predictors)
3. RYS doubling ratios from saved SAT theory validation
4. Band-predictor comparison (CKA vs R vs Q2 vs plateau) on Pythia deltas

Outputs: results/cka_theory_verification/summary.json and report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from rys.cka import cka_matrix
from rys.residual_force import residual_force_long, residual_force_matrices
from rys.theory_validation import theory_fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cka_theory_verification"


def _acts_from_layers(layers: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for layer, x in layers.items():
        rows.append({"prompt_id": "synth", "layer": layer, "activation": x, "strategy": "synth"})
    return pd.DataFrame(rows)


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = _center(X.astype(np.float64))
    Yc = _center(Y.astype(np.float64))
    num = np.linalg.norm(Xc.T @ Yc, "fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(num / den) if den > 0 else float("nan")


def synthetic_case(
    *,
    n: int = 64,
    d: int = 32,
    eps: float,
    mode: str,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X = X - X.mean(axis=0)

    if mode == "aligned":
        direction = X.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        S = eps * np.outer(np.ones(n), direction)
    elif mode == "orthogonal":
        # Residual mostly orthogonal to column space of X in feature space
        S = rng.standard_normal((n, d))
        S = S - X @ np.linalg.lstsq(X, S, rcond=None)[0]
        S = eps * S / (np.linalg.norm(S) + 1e-12) * np.linalg.norm(X)
    elif mode == "incoherent":
        S = eps * rng.standard_normal((n, d))
    else:
        raise ValueError(mode)

    Y = X + S
    cka_direct = _linear_cka(X, Y)

    acts = _acts_from_layers({0: X, 1: Y})
    rp = residual_force_long(acts, upper_only=True, dtype=torch.float64)
    row = rp.iloc[0]

    pred_qpsi = float(row["one_minus_cka_Qpsi"])
    pred_rphi = float(row["one_minus_cka_Rphi"])
    pred_q2 = 0.5 * float(row["Q"]) ** 2
    cka_module = float(row["cka_full"])

    return {
        "mode": mode,
        "eps": eps,
        "cka_direct": cka_direct,
        "cka_module": cka_module,
        "one_minus_cka": 1.0 - cka_direct,
        "pred_half_Q2_sin2_Psi": pred_qpsi,
        "pred_half_R2_sin2_Phi": pred_rphi,
        "pred_half_Q2": pred_q2,
        "R": float(row["R"]),
        "Q": float(row["Q"]),
        "cos_phi": float(row["cos_phi"]),
        "cos_psi": float(row["cos_psi"]),
        "abs_cka_err": abs(cka_direct - cka_module),
        "rel_Qpsi_err": abs((1.0 - cka_direct) - pred_qpsi) / max(1.0 - cka_direct, 1e-12),
        "rel_Rphi_err": abs((1.0 - cka_direct) - pred_rphi) / max(1.0 - cka_direct, 1e-12),
        "rel_Q2_err": abs((1.0 - cka_direct) - pred_q2) / max(1.0 - cka_direct, 1e-12),
    }


def run_synthetic_checks() -> dict:
    eps_grid = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 1.0]
    modes = ["aligned", "orthogonal", "incoherent"]
    rows = []
    for mode in modes:
        for eps in eps_grid:
            rows.append(synthetic_case(eps=eps, mode=mode))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_checks.csv", index=False)

    exact_ok = float(df["abs_cka_err"].max()) < 1e-6
    small = df[df["eps"] <= 0.05]

    def best_predictor(sub: pd.DataFrame) -> str:
        mae_qpsi = (sub["one_minus_cka"] - sub["pred_half_Q2_sin2_Psi"]).abs().mean()
        mae_rphi = (sub["one_minus_cka"] - sub["pred_half_R2_sin2_Phi"]).abs().mean()
        mae_q = (sub["one_minus_cka"] - sub["pred_half_Q2"]).abs().mean()
        values = {"Q2_sin2_Psi": mae_qpsi, "R2_sin2_Phi": mae_rphi, "Q2": mae_q}
        return min(values, key=values.get)

    by_mode = {}
    for mode in modes:
        sub = small[small["mode"] == mode]
        by_mode[mode] = {
            "best_small_eps_predictor": best_predictor(sub),
            "mae_Q2_sin2_Psi": float((sub["one_minus_cka"] - sub["pred_half_Q2_sin2_Psi"]).abs().mean()),
            "mae_R2_sin2_Phi": float((sub["one_minus_cka"] - sub["pred_half_R2_sin2_Phi"]).abs().mean()),
            "mae_Q2": float((sub["one_minus_cka"] - sub["pred_half_Q2"]).abs().mean()),
            "median_R": float(sub["R"].median()),
            "median_Q": float(sub["Q"].median()),
            "median_cos_phi": float(sub["cos_phi"].median()),
            "median_cos_psi": float(sub["cos_psi"].median()),
        }

    return {
        "exact_cka_matches_module": exact_ok,
        "max_abs_cka_err": float(df["abs_cka_err"].max()),
        "small_eps_regime_eps_le_0.05": by_mode,
        "note": (
            "Aligned residuals favour R^2 sin^2 Phi at small eps; "
            "incoherent residuals favour Q^2 at leading order."
        ),
    }


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def audit_existing() -> dict:
    toy_paths = {
        "sat": ROOT / "results/sat_theory_validation/20260529_002010/summary.json",
        "nqueens": ROOT / "results/nqueens_theory_validation/20260529_123441/summary.json",
        "coloring": ROOT / "results/coloring_theory_validation/20260530_142014/summary.json",
    }
    toy = {}
    for name, path in toy_paths.items():
        data = load_json(path)
        if data:
            toy[name] = data.get("theory_fit", {})

    pythia_robust = {}
    for summary in sorted((ROOT / "results/robust_geometry_controls").glob("*/summary.json")):
        tag = summary.parent.name
        data = load_json(summary)
        if not data:
            continue
        raw = data["variants"].get("raw", {})
        dft = data["variants"].get("drop_first_token", {})
        pythia_robust[tag] = {
            "median_offdiag_cka_raw": raw.get("median_offdiag_cka"),
            "spearman_plateau_raw": raw.get("theory_fit", {}).get("spearman_plateau_pred"),
            "pearson_plateau_raw": raw.get("theory_fit", {}).get("pearson_plateau_pred"),
            "pearson_Q2_raw": raw.get("theory_fit", {}).get("pearson_Q2"),
            "spearman_delta_cka_raw": raw.get("functional", {}).get("spearman_delta_cka"),
            "spearman_delta_rho_raw": raw.get("functional", {}).get("spearman_delta_rho"),
            "spearman_delta_Q2_raw": raw.get("functional", {}).get("spearman_delta_Q2"),
            "median_offdiag_cka_drop_first_token": dft.get("median_offdiag_cka"),
            "spearman_delta_cka_drop_first_token": dft.get("functional", {}).get("spearman_delta_cka"),
            "spearman_delta_Q2_drop_first_token": dft.get("functional", {}).get("spearman_delta_Q2"),
        }

    gsm8k = {}
    for summary in sorted((ROOT / "results/pythia_gsm8k_rys").glob("*/**/summary.json")):
        tag = summary.parent.parent.name
        data = load_json(summary)
        if data and "theory_fit" in data:
            gsm8k[tag] = data["theory_fit"]

    return {"toy_theory_fit": toy, "pythia_robust_geometry": pythia_robust, "pythia_gsm8k": gsm8k}


def check_rys_doubling() -> dict:
    path = ROOT / "results/sat_theory_validation/20260529_002010/summary.json"
    data = load_json(path)
    if not data:
        return {"error": "SAT theory validation summary not found"}

    windows = data.get("best_windows", [])
    ratios = [w["one_minus_cka_ratio"] for w in windows]
    s_ratios = [w["S_norm_ratio"] for w in windows]
    phi_diff = [abs(w["cos_phi_diff"]) for w in windows]

    return {
        "source": str(path),
        "n_best_windows": len(windows),
        "one_minus_cka_ratio_mean": float(np.mean(ratios)),
        "one_minus_cka_ratio_range": [float(min(ratios)), float(max(ratios))],
        "predicted_ratio": 4.0,
        "S_norm_ratio_mean": float(np.mean(s_ratios)),
        "predicted_S_norm_ratio": 2.0,
        "mean_abs_cos_phi_diff": float(np.mean(phi_diff)),
        "best_ood_window": windows[0] if windows else None,
        "theory_fit": data.get("theory_fit", {}),
    }


def compare_band_predictors() -> dict:
    rows = []
    controls_dir = ROOT / "results/robust_geometry_controls"
    for summary_path in sorted(controls_dir.glob("*/summary.json")):
        tag = summary_path.parent.name
        data = load_json(summary_path)
        if not data or not data.get("delta_source"):
            continue
        delta = pd.read_csv(ROOT / data["delta_source"])
        for variant, vdata in data["variants"].items():
            rp_path = summary_path.parent / f"rho_phi_{variant}.csv"
            if not rp_path.exists():
                continue
            rp = pd.read_csv(rp_path)
            mg = delta.merge(rp, left_on=["start", "end"], right_on=["layer_i", "layer_j"])
            if len(mg) < 3:
                continue
            mg = mg.copy()
            mg["Q2"] = mg["Q"] ** 2
            y = mg["delta_score"]

            def sp(col: str) -> float:
                if mg[col].std() == 0 or y.std() == 0:
                    return float("nan")
                return float(spearmanr(y, mg[col]).statistic)

            rows.append(
                {
                    "model": tag,
                    "variant": variant,
                    "n_windows": len(mg),
                    "spearman_delta_cka": sp("cka_full"),
                    "spearman_delta_R": sp("R"),
                    "spearman_delta_Q2": sp("Q2"),
                    "spearman_delta_plateau": sp("one_minus_cka_plateau"),
                    "median_offdiag_cka": vdata.get("median_offdiag_cka"),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "band_predictor_comparison.csv", index=False)

    summary = {}
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        raw = sub[sub["variant"] == "raw"].iloc[0] if (sub["variant"] == "raw").any() else None
        dft = sub[sub["variant"] == "drop_first_token"].iloc[0] if (sub["variant"] == "drop_first_token").any() else None
        if raw is None:
            continue
        summary[model] = {
            "raw": {
                "spearman_delta_cka": raw["spearman_delta_cka"],
                "spearman_delta_R": raw["spearman_delta_R"],
                "spearman_delta_Q2": raw["spearman_delta_Q2"],
                "spearman_delta_plateau": raw["spearman_delta_plateau"],
            },
            "drop_first_token": None if dft is None else {
                "spearman_delta_cka": dft["spearman_delta_cka"],
                "spearman_delta_R": dft["spearman_delta_R"],
                "spearman_delta_Q2": dft["spearman_delta_Q2"],
            },
            "best_functional_variant": sub.loc[
                sub[["spearman_delta_cka", "spearman_delta_Q2"]].abs().max(axis=1).idxmax(), "variant"
            ]
            if len(sub)
            else None,
        }

    return {"by_model": summary, "n_rows": len(df)}


def write_report(payload: dict) -> None:
    lines = [
        "# CKA Theory Numerical Verification",
        "",
        "## Verdict",
        f"**{payload['verdict']}**",
        "",
        payload["verdict_detail"],
        "",
        "## 1. Exact CKA",
        f"- Module matches direct linear CKA: `{payload['synthetic']['exact_cka_matches_module']}`",
        f"- Max absolute error: `{payload['synthetic']['max_abs_cka_err']:.2e}`",
        "",
        "## 2. Plateau predictor at small eps (<= 0.05)",
    ]
    for mode, stats in payload["synthetic"]["small_eps_regime_eps_le_0.05"].items():
        lines.append(
            f"- **{mode}**: best predictor = `{stats['best_small_eps_predictor']}`; "
            f"MAE(½Q²sin²Ψ)={stats['mae_Q2_sin2_Psi']:.2e}, "
            f"MAE(½R²sin²Φ)={stats['mae_R2_sin2_Phi']:.2e}, "
            f"MAE(½Q²)={stats['mae_Q2']:.2e}"
        )

    lines.extend(
        [
            "",
            "## 3. Saved toy theory fits (Spearman plateau pred)",
        ]
    )
    for name, fit in payload["audit"]["toy_theory_fit"].items():
        lines.append(
            f"- **{name}**: Spearman={fit.get('spearman_plateau_pred', 'n/a'):.3f}, "
            f"Pearson plateau={fit.get('pearson_plateau_pred', 'n/a'):.3f}, "
            f"Pearson Q²={fit.get('pearson_Q2', 'n/a'):.3f}"
        )

    lines.extend(["", "## 4. RYS doubling (SAT best windows)"])
    dbl = payload["rys_doubling"]
    lines.append(
        f"- Mean (1-CKA) ratio = {dbl['one_minus_cka_ratio_mean']:.2f} "
        f"(predicted 4; range {dbl['one_minus_cka_ratio_range']})"
    )
    lines.append(f"- Mean S_norm ratio = {dbl['S_norm_ratio_mean']:.2f} (predicted 2)")

    lines.extend(["", "## 5. Band predictors vs RYS delta (Pythia guesstimation)"])
    for model, stats in payload["band_predictors"]["by_model"].items():
        raw = stats["raw"]
        lines.append(
            f"- **{model}** raw: Δ~CKA={raw['spearman_delta_cka']:.3f}, "
            f"Δ~R={raw['spearman_delta_R']:.3f}, Δ~Q²={raw['spearman_delta_Q2']:.3f}, "
            f"Δ~plateau={raw['spearman_delta_plateau']:.3f}"
        )

    lines.extend(["", "## 6. Paper wording recommendation", payload["paper_recommendation"]])
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def decide_verdict(payload: dict) -> tuple[str, str, str]:
    synthetic = payload["synthetic"]
    toy = payload["audit"]["toy_theory_fit"]

    spearmans = [v.get("spearman_plateau_pred", 0) for v in toy.values()]
    pearsons_plateau = [v.get("pearson_plateau_pred", 0) for v in toy.values()]
    pearsons_q2 = [v.get("pearson_Q2", 0) for v in toy.values()]

    rank_good = all(s > 0.9 for s in spearmans) if spearmans else False
    magnitude_mixed = any(p < 0.75 for p in pearsons_plateau) and any(p > 0.85 for p in pearsons_q2)

    incoherent_best = synthetic["small_eps_regime_eps_le_0.05"].get("incoherent", {}).get(
        "best_small_eps_predictor"
    ) in {"Q2_sin2_Psi", "Q2"}

    if not synthetic["exact_cka_matches_module"]:
        return (
            "Needs correction",
            "Exact CKA implementation does not match direct computation.",
            "Fix residual_force closed form before claiming theory validity.",
        )

    if rank_good and (magnitude_mixed or incoherent_best):
        detail = (
            "Exact CKA is numerically correct. The current plateau column now uses the generic "
            "`½ Q² sin² Ψ` predictor, while the legacy `½ R² sin² Φ` quantity is retained under "
            "an explicit name for comparison."
        )
        rec = (
            "Regenerate rho_phi tables before using notebook plots: old CSVs used "
            "`one_minus_cka_plateau` for the legacy R/Phi predictor and do not contain Ψ."
        )
        return "Correct but qualified", detail, rec

    if rank_good:
        return (
            "Correct as written",
            "Theory matches numerically in tested regimes.",
            "Minor clarifications on regime of validity still recommended.",
        )

    return (
        "Needs correction",
        "Rank correlations weaker than expected on saved runs.",
        "Revisit derivation and plateau threshold.",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    payload = {
        "synthetic": run_synthetic_checks(),
        "audit": audit_existing(),
        "rys_doubling": check_rys_doubling(),
        "band_predictors": compare_band_predictors(),
    }
    verdict, detail, rec = decide_verdict(payload)
    payload["verdict"] = verdict
    payload["verdict_detail"] = detail
    payload["paper_recommendation"] = rec

    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    write_report(payload)
    print(json.dumps({"verdict": verdict, "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
