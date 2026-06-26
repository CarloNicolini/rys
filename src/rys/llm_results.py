"""Load RYS experiment artifacts from ``results/LLM/<family>/``.

Unified path helpers for outputs of :mod:`scripts.run_guesstimation_rys` and
:mod:`scripts.run_robust_geometry_controls`. Supports Pythia, Qwen, Llama, and
any future family that follows the same layout:

::

    results/LLM/<family>/guesstimation_rys/<model_tag>/<timestamp>/
    results/LLM/<family>/robust_geometry_controls/<model_tag>/          # legacy flat
    results/LLM/<family>/robust_geometry_controls/<model_tag>/<timestamp>/  # new runs

The legacy robust-geometry script wrote many per-variant CSVs directly under
``<model_tag>/`` (``rho_phi_global_standardize_drop_first_token.csv``, …).
The unified script writes timestamped runs with two regimes (``raw``,
``robust``) plus optional coherence / composite-safety tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GUESSTIMATION = "guesstimation_rys"
ROBUST_GEOMETRY = "robust_geometry_controls"
GSM8K_MC = "gsm8k_rys"

FAMILIES = ("pythia", "qwen", "llama")

# Preferred geometry regime when joining to guesstimation deltas.
CANONICAL_GEOMETRY_REGIMES = (
    "robust",
    "global_standardize_drop_first_token",
    "global_standardize_drop_top_dims_drop_first_token",
    "raw",
)

PARAMS: dict[str, float] = {
    "pythia-70m": 70e6,
    "pythia-160m": 160e6,
    "pythia-410m": 410e6,
    "pythia-1b": 1.0e9,
    "pythia-1.4b": 1.4e9,
    "pythia-2.8b": 2.8e9,
    "pythia-6.9b": 6.9e9,
    "pythia-12b": 12.0e9,
    "Qwen3-0.6B": 0.6e9,
    "Qwen3-1.7B": 1.7e9,
    "Qwen3-4B": 4.0e9,
    "Qwen3-8B": 8.0e9,
    "Qwen3-14B": 14.0e9,
    "Llama-3.2-1B": 1.0e9,
    "Llama-3.2-3B": 3.0e9,
    "Llama-3-8B": 8.0e9,
}


def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "results").exists():
            return cand
    raise FileNotFoundError("Run from inside the rys repo (need pyproject.toml and results/).")


def llm_root(repo: Path | None = None) -> Path:
    return find_repo_root(repo) / "results" / "LLM"


def experiment_root(family: str, experiment: str, repo: Path | None = None) -> Path:
    return llm_root(repo) / family / experiment


def list_models(family: str, experiment: str = GUESSTIMATION, repo: Path | None = None) -> list[str]:
    """Checkpoint tags under ``results/LLM/<family>/<experiment>/``."""
    root = experiment_root(family, experiment, repo)
    if not root.is_dir():
        return []
    skip = {"logs", "MASTER.log", "OLD"}
    tags = [p.name for p in root.iterdir() if p.is_dir() and p.name not in skip]
    return sorted(tags, key=lambda t: (param_count(t), t))


def short_tag(model_tag: str, family: str | None = None) -> str:
    """Human-readable label: strip ``<family>-`` prefix when present."""
    if family:
        prefix = f"{family}-"
        if model_tag.lower().startswith(prefix):
            return model_tag[len(prefix) :]
    return model_tag


def param_count(model_tag: str) -> float:
    if model_tag in PARAMS:
        return PARAMS[model_tag]
    m = re.search(r"(\d+(?:\.\d+)?)\s*([mb])", model_tag, re.I)
    if not m:
        return 0.0
    val = float(m.group(1))
    return val * (1e6 if m.group(2).lower() == "m" else 1e9)


def latest_timestamped_run(
    family: str,
    experiment: str,
    model_tag: str,
    *,
    must_contain: str = "summary.json",
    repo: Path | None = None,
) -> Path | None:
    """Latest ``<timestamp>/`` subfolder that contains ``must_contain``."""
    base = experiment_root(family, experiment, repo) / model_tag
    if not base.is_dir():
        return None
    runs = sorted(
        d for d in base.iterdir()
        if d.is_dir() and (d / must_contain).exists()
    )
    return runs[-1] if runs else None


def guesstimation_run_dir(
    family: str,
    model_tag: str,
    repo: Path | None = None,
) -> Path | None:
    return latest_timestamped_run(family, GUESSTIMATION, model_tag, repo=repo)


def robust_run_dir(family: str, model_tag: str, repo: Path | None = None) -> Path | None:
    """Resolve robust-geometry root: flat legacy layout or latest timestamped run."""
    base = experiment_root(family, ROBUST_GEOMETRY, repo) / model_tag
    if not base.is_dir():
        return None
    legacy_markers = ("summary.json", "rho_phi_raw.csv", "rho_phi_robust.csv")
    if any((base / name).exists() for name in legacy_markers):
        return base
    # Flat layout with only variant CSVs (no summary at top).
    if any(base.glob("rho_phi_*.csv")):
        return base
    return latest_timestamped_run(family, ROBUST_GEOMETRY, model_tag, repo=repo)


def list_geometry_regimes(family: str, model_tag: str, repo: Path | None = None) -> list[str]:
    """Regime / variant names available under robust_geometry_controls."""
    root = robust_run_dir(family, model_tag, repo)
    if root is None:
        return []
    names = {p.stem.removeprefix("rho_phi_") for p in root.glob("rho_phi_*.csv")}
    order = {name: i for i, name in enumerate(CANONICAL_GEOMETRY_REGIMES)}
    return sorted(names, key=lambda n: (order.get(n, 99), n))


def canonical_geometry_regime(family: str, model_tag: str, repo: Path | None = None) -> str:
    available = list_geometry_regimes(family, model_tag, repo)
    for candidate in CANONICAL_GEOMETRY_REGIMES:
        if candidate in available:
            return candidate
    return available[0] if available else "raw"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_guesstimation_summary(family: str, model_tag: str, repo: Path | None = None) -> dict | None:
    run = guesstimation_run_dir(family, model_tag, repo)
    if run is None:
        return None
    return load_json(run / "summary.json")


def load_guesstimation_long(family: str, model_tag: str, repo: Path | None = None) -> pd.DataFrame | None:
    run = latest_timestamped_run(
        family, GUESSTIMATION, model_tag, must_contain="delta_score_long.csv", repo=repo
    )
    if run is None:
        return None
    return pd.read_csv(run / "delta_score_long.csv")


def load_guesstimation_matrix(family: str, model_tag: str, repo: Path | None = None) -> pd.DataFrame | None:
    run = latest_timestamped_run(
        family, GUESSTIMATION, model_tag, must_contain="delta_score.csv", repo=repo
    )
    if run is None:
        return None
    mat = pd.read_csv(run / "delta_score.csv", index_col=0)
    mat.columns = mat.columns.astype(int)
    mat.index = mat.index.astype(int)
    return mat


def load_run_cka(family: str, model_tag: str, repo: Path | None = None) -> pd.DataFrame | None:
    run = latest_timestamped_run(family, GUESSTIMATION, model_tag, must_contain="cka.csv", repo=repo)
    if run is None:
        return None
    cka = pd.read_csv(run / "cka.csv", index_col=0)
    cka.columns = cka.columns.astype(int)
    cka.index = cka.index.astype(int)
    return cka


def ensure_geometry_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize rho/phi tables across legacy and unified script outputs."""
    out = df.copy()
    if "one_minus_cka_Rphi" not in out and "one_minus_cka_plateau" in out:
        out["one_minus_cka_Rphi"] = out["one_minus_cka_plateau"]
    if "one_minus_cka_Qpsi" not in out:
        out["one_minus_cka_Qpsi"] = np.nan
    if "one_minus_cka_plateau" not in out and "one_minus_cka_Qpsi" in out:
        out["one_minus_cka_plateau"] = out["one_minus_cka_Qpsi"]
    if "one_minus_cka_full" not in out and "cka_full" in out:
        out["one_minus_cka_full"] = 1.0 - out["cka_full"]
    if "cos_psi" not in out:
        out["cos_psi"] = np.nan
        out["sin2_psi"] = np.nan
    out["plateau_predictor"] = out["one_minus_cka_Qpsi"]
    return out


def load_robust_rho_phi(
    family: str,
    model_tag: str,
    regime: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame | None:
    root = robust_run_dir(family, model_tag, repo)
    if root is None:
        return None
    regime = regime or canonical_geometry_regime(family, model_tag, repo)
    path = root / f"rho_phi_{regime}.csv"
    if not path.exists():
        return None
    df = ensure_geometry_schema(pd.read_csv(path))
    df["geometry_regime"] = regime
    df["model_tag"] = model_tag
    df["family"] = family
    return df


def load_robust_cka(
    family: str,
    model_tag: str,
    regime: str | None = None,
    repo: Path | None = None,
) -> pd.DataFrame | None:
    root = robust_run_dir(family, model_tag, repo)
    if root is None:
        return None
    regime = regime or canonical_geometry_regime(family, model_tag, repo)
    path = root / f"cka_{regime}.csv"
    if not path.exists():
        return None
    cka = pd.read_csv(path, index_col=0)
    cka.columns = cka.columns.astype(int)
    cka.index = cka.index.astype(int)
    return cka


def load_guesstimation_windows(
    family: str,
    model_tag: str,
    *,
    geometry_regime: str | None = None,
    use_run_local_geometry: bool = False,
    repo: Path | None = None,
) -> pd.DataFrame | None:
    """Per-window behavioural deltas merged with geometry (one row per band)."""
    run = latest_timestamped_run(
        family, GUESSTIMATION, model_tag, must_contain="delta_score_long.csv", repo=repo
    )
    if run is None:
        return None
    long = pd.read_csv(run / "delta_score_long.csv")

    if use_run_local_geometry:
        rp_path = run / "rho_phi_table.csv"
        rp = ensure_geometry_schema(pd.read_csv(rp_path)) if rp_path.exists() else None
        geom_label = "run-local"
    else:
        regime = geometry_regime or canonical_geometry_regime(family, model_tag, repo)
        rp = load_robust_rho_phi(family, model_tag, regime, repo)
        geom_label = regime
        if rp is None:
            rp_path = run / "rho_phi_table.csv"
            if rp_path.exists():
                rp = ensure_geometry_schema(pd.read_csv(rp_path))
                geom_label = "run-local"

    if rp is None:
        return None

    merged = long.merge(rp, left_on=["start", "end"], right_on=["layer_i", "layer_j"], how="left")
    merged["family"] = family
    merged["model_tag"] = model_tag
    merged["model"] = short_tag(model_tag, family)
    merged["params"] = param_count(model_tag)
    merged["geometry"] = geom_label
    merged["run_dir"] = str(run)
    if "jacobian_sigma_max" not in merged.columns and "jacobian_sigma_max" in long.columns:
        pass  # already in long from unified guesstimation script
    return merged


def load_robust_summaries(family: str, repo: Path | None = None) -> pd.DataFrame:
    """One row per (model, regime/variant) from each robust ``summary.json``."""
    rows: list[dict] = []
    for model_tag in list_models(family, ROBUST_GEOMETRY, repo):
        root = robust_run_dir(family, model_tag, repo)
        if root is None or not (root / "summary.json").exists():
            continue
        summary = load_json(root / "summary.json")
        stats = summary.get("massive_activation_stats", {})
        # Unified script: ``regimes``; legacy script: ``variants``.
        payloads = summary.get("regimes") or summary.get("variants") or {}
        for regime, payload in payloads.items():
            fit = payload.get("theory_fit", {})
            functional = payload.get("functional", {})
            rows.append(
                {
                    "family": family,
                    "model_tag": model_tag,
                    "model": short_tag(model_tag, family),
                    "regime": regime,
                    "d_model": stats.get("d_model", np.nan),
                    "top1_dim_var_frac": stats.get("top1_dim_var_frac", stats.get("top1", np.nan)),
                    "top5_dim_var_frac": stats.get("top5_dim_var_frac", stats.get("top5", np.nan)),
                    "median_offdiag_cka": payload.get("median_offdiag_cka", np.nan),
                    "spearman_plateau_pred": fit.get("spearman_plateau_pred", np.nan),
                    "spearman_Qpsi": fit.get("spearman_Qpsi", fit.get("spearman_plateau_pred", np.nan)),
                    "spearman_delta_rho": functional.get("spearman_delta_rho", np.nan),
                    "spearman_delta_cka": functional.get("spearman_delta_cka", np.nan),
                    "spearman_delta_Qpsi": functional.get(
                        "spearman_delta_Qpsi",
                        functional.get("spearman_delta_plateau_pred", np.nan),
                    ),
                    "spearman_delta_jac_sigma": functional.get("spearman_delta_jac_sigma", np.nan),
                    "best_delta": functional.get("best_delta", np.nan),
                    "best_window_start": functional.get("best_window_start", np.nan),
                    "best_window_end": functional.get("best_window_end", np.nan),
                }
            )
    return pd.DataFrame(rows)


def load_guesstimation_summaries(family: str, repo: Path | None = None) -> pd.DataFrame:
    """Headline numbers from each model's latest guesstimation run."""
    rows: list[dict] = []
    for model_tag in list_models(family, GUESSTIMATION, repo):
        summary = load_guesstimation_summary(family, model_tag, repo)
        if summary is None:
            continue
        row = {
            "family": family,
            "model_tag": model_tag,
            "model": short_tag(model_tag, family),
            "params": param_count(model_tag),
            **{k: summary[k] for k in summary if k != "theory_fit"},
        }
        if "theory_fit" in summary:
            for k, v in summary["theory_fit"].items():
                row[f"theory_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def inventory(families: list[str] | None = None, repo: Path | None = None) -> pd.DataFrame:
    """Quick table of what is available per family and experiment."""
    families = families or list(FAMILIES)
    rows: list[dict] = []
    for family in families:
        for experiment in (GUESSTIMATION, ROBUST_GEOMETRY, GSM8K_MC):
            for model_tag in list_models(family, experiment, repo):
                if experiment == GUESSTIMATION:
                    run = guesstimation_run_dir(family, model_tag, repo)
                    n_windows = None
                    if run and (run / "delta_score_long.csv").exists():
                        n_windows = len(pd.read_csv(run / "delta_score_long.csv"))
                    rows.append(
                        {
                            "family": family,
                            "experiment": experiment,
                            "model_tag": model_tag,
                            "model": short_tag(model_tag, family),
                            "has_run": run is not None,
                            "n_windows": n_windows,
                            "robust_regimes": list_geometry_regimes(family, model_tag, repo)
                            if list_geometry_regimes(family, model_tag, repo)
                            else None,
                        }
                    )
                elif experiment == ROBUST_GEOMETRY:
                    root = robust_run_dir(family, model_tag, repo)
                    rows.append(
                        {
                            "family": family,
                            "experiment": experiment,
                            "model_tag": model_tag,
                            "model": short_tag(model_tag, family),
                            "has_run": root is not None,
                            "n_windows": None,
                            "robust_regimes": list_geometry_regimes(family, model_tag, repo),
                        }
                    )
                else:
                    run = latest_timestamped_run(family, experiment, model_tag, repo=repo)
                    rows.append(
                        {
                            "family": family,
                            "experiment": experiment,
                            "model_tag": model_tag,
                            "model": short_tag(model_tag, family),
                            "has_run": run is not None,
                            "n_windows": None,
                            "robust_regimes": None,
                        }
                    )
    return pd.DataFrame(rows)
