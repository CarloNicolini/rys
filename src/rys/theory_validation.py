"""Validate the rho/phi CKA theory on a trained residual model.

The companion analysis derives a closed form for the linear-CKA edge between
two residual-stream layers.  The generic small-residual law is governed by the
cross-kernel perturbation,

    1 - CKA_ij ~ 1/2 Q_ij^2 sin^2 Psi_ij,

with the legacy self-Gram R/Phi predictor retained as a diagnostic for coherent
residuals.  This module turns those statements into measurements on captured
activations, reusing :mod:`rys.residual_force` for the heavy lifting and adding
two dynamical-systems diagnostics (junction mismatch and block-Jacobian
spectral radius) so the predictions can be linked to RYS behavioural windows.

All activation inputs use the long-format schema of :mod:`rys.cka`
(``prompt_id, layer, activation, strategy``); the message-passing experiment
captures the raw pre-norm variable stream in exactly this layout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from rys.residual_force import amplification_long, residual_force_long, residual_force_matrices


def rho_phi_table(activations: pd.DataFrame) -> pd.DataFrame:
    """Per-pair rho/phi table with measured and predicted ``1 - CKA``."""
    return residual_force_long(activations, upper_only=True, dtype=torch.float64)


def theory_fit(table: pd.DataFrame, *, plateau_threshold: float = 0.5) -> dict[str, float]:
    """Quantify how well the plateau predictors explain measured ``1 - CKA``.

    Returns Pearson and Spearman correlations between the measured
    ``one_minus_cka_full`` and the corrected ``1/2 Q^2 sin^2 Psi`` predictor,
    plus legacy R/Phi and orthogonal ``1/2 Q^2`` diagnostics.  Correlations are
    computed on the plateau subset (measured ``1 - CKA <= plateau_threshold``)
    where the expansion is at least plausible.
    """
    plateau = table[table["one_minus_cka_full"] <= plateau_threshold].copy()
    plateau["Q2"] = plateau["Q"] ** 2
    plateau["half_Q2"] = 0.5 * plateau["Q2"]
    measured = plateau["one_minus_cka_full"]

    def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(a.corr(b, method=method))

    return {
        "n_pairs_total": int(len(table)),
        "n_pairs_plateau": int(len(plateau)),
        "pearson_plateau_pred": _safe_corr(measured, plateau["one_minus_cka_plateau"], "pearson"),
        "spearman_plateau_pred": _safe_corr(measured, plateau["one_minus_cka_plateau"], "spearman"),
        "pearson_Qpsi": _safe_corr(measured, plateau["one_minus_cka_Qpsi"], "pearson"),
        "spearman_Qpsi": _safe_corr(measured, plateau["one_minus_cka_Qpsi"], "spearman"),
        "pearson_Rphi": _safe_corr(measured, plateau["one_minus_cka_Rphi"], "pearson"),
        "spearman_Rphi": _safe_corr(measured, plateau["one_minus_cka_Rphi"], "spearman"),
        "pearson_Q2": _safe_corr(measured, plateau["Q2"], "pearson"),
        "spearman_Q2": _safe_corr(measured, plateau["Q2"], "spearman"),
        "pearson_half_Q2": _safe_corr(measured, plateau["half_Q2"], "pearson"),
        "spearman_half_Q2": _safe_corr(measured, plateau["half_Q2"], "spearman"),
        "median_R": float(table["R"].median()),
        "median_cos_phi": float(table["cos_phi"].median()),
        "median_Q": float(table["Q"].median()),
        "median_cos_psi": float(table["cos_psi"].median()),
    }


def rys_amplification_summary(
    base_activations: pd.DataFrame,
    rys_activations: pd.DataFrame,
    window: tuple[int, int],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Test the RYS doubling/quadrupling prediction region by region.

    ``window`` is the half-open ``(start, end)`` interval used by
    :func:`rys.surgery.apply_rys`.  The summary reports the mean amplification
    ratios inside the duplicated window, where theory predicts
    ``S_norm_ratio ~ 2``, ``R_ratio ~ 4``, ``one_minus_cka_ratio ~ 4`` and
    ``cos_phi_diff ~ 0``.
    """
    base_bundle = residual_force_matrices(base_activations, dtype=torch.float64)
    rys_bundle = residual_force_matrices(rys_activations, dtype=torch.float64)
    long = amplification_long(base_bundle, rys_bundle, window=window)
    in_window = long[long["region"] == "in_window"]
    summary = {
        "window_start": window[0],
        "window_end": window[1],
        "n_in_window": int(len(in_window)),
        "S_norm_ratio_mean": float(in_window["S_norm_ratio"].mean()),
        "R_ratio_mean": float(in_window["R_ratio"].mean()),
        "one_minus_cka_ratio_mean": float(in_window["one_minus_cka_ratio"].mean()),
        "cos_phi_diff_mean": float(in_window["cos_phi_diff"].mean()),
    }
    return long, summary


def junction_mismatch(activations: pd.DataFrame, window: tuple[int, int]) -> float:
    """Standardised distance between the block-output and block-input manifolds.

    For a window ``(start, end)`` the RYS loop feeds the layer-``end`` state back
    into layer ``start``.  We measure how far the layer-``end`` activation cloud
    sits from the layer-``start`` cloud, in units of the start cloud's spread:

        ``|| mean(x_end) - mean(x_start) || / RMS_token-spread(x_start)``.

    Low values mean the loop junction stays on the start layer's input manifold,
    the condition the theory predicts for safe RYS windows.
    """
    start, end = window
    pivot = activations.pivot_table(index="prompt_id", columns="layer", values="activation", aggfunc="first")
    x_start = np.stack([np.atleast_2d(v).mean(axis=0) for v in pivot[start].to_numpy()])
    x_end = np.stack([np.atleast_2d(v).mean(axis=0) for v in pivot[end].to_numpy()])
    mean_gap = np.linalg.norm(x_end.mean(axis=0) - x_start.mean(axis=0))
    spread = np.sqrt(((x_start - x_start.mean(axis=0)) ** 2).sum(axis=1).mean())
    return float(mean_gap / spread) if spread > 0 else float("nan")


@torch.no_grad()
def _noop() -> None:  # pragma: no cover - placeholder for symmetry
    return None


def block_jacobian_sigma_max(
    block_map,
    hidden_in: torch.Tensor,
    *,
    n_iter: int = 12,
    seed: int = 0,
) -> float:
    """Estimate the largest singular value of a block map's Jacobian.

    ``block_map`` maps a packed hidden state to the post-block hidden state.
    The dominant singular value of its Jacobian is found by power iteration on
    ``J^T J`` using forward- and reverse-mode automatic differentiation, so no
    dense Jacobian is ever formed.  A value near 1 means the block is a stable
    refinement operator (neither a dead identity nor a chaotic expander), the
    regime in which iterating it (RYS) is safe.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(hidden_in.shape, generator=generator).to(hidden_in.device, hidden_in.dtype)
    v = v / v.norm()
    sigma = 0.0
    for _ in range(n_iter):
        _, jvp = torch.autograd.functional.jvp(block_map, hidden_in, v)
        _, vjp = torch.autograd.functional.vjp(block_map, hidden_in, jvp)
        norm = vjp.norm()
        if norm == 0:
            return 0.0
        v = vjp / norm
        sigma = float(norm.sqrt())
    return sigma
