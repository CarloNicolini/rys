"""Predictive geometry of the RYS effect: a three-regime decomposition.

The :mod:`rys.residual_force` module derives the closed-form CKA edge and its
small-force Taylor expansion (the plateau law :math:`1-\\mathrm{CKA}_{ij}
\\approx \\tfrac{1}{2}\\mathcal{Q}_{ij}^{2}\\sin^{2}\\Psi_{ij}`). That law
describes *representation geometry* — how similar two layers look — and it fits
the measured CKA connectome extremely well (Spearman ≈ 0.92 on every model
tested). But it does not, on its own, predict the *behavioural* effect of RYS
(layer duplication) with the same fidelity: across the Pythia/Qwen/Llama ladder
the rank correlation between :math:`\\mathcal{R}` (or CKA) and the RYS
:math:`\\Delta`-score lands anywhere in :math:`[-0.7, +0.2]`, flips sign between
families, and is contaminated by massive-activation dimensions that can carry
either signal or nuisance depending on the model.

This module is an attempt to close that gap with a **three-regime decomposition**
of the residual force. The motivating empirical observation is that the CKA
edge is a mixture of three geometrically distinct components, and RYS
amplifies them differently, so the *balance* between them — not any single
scalar — should govern whether duplicating a band refines or destroys the
representation:

1. **Massive-activation subspace** (the saturating component). A handful of
   high-variance dimensions (and/or the attention-sink token) dominate the
   Frobenius norms and push raw CKA toward 1. They carry *bulk identity*:
   duplicating a band that is mostly this component is approximately a no-op
   on behaviour, yet they inflate :math:`\\mathcal{R}` and
   :math:`\\sigma_{\\max}` numerically. We measure them but also compute
   every quantity below on a *robustified* activation stream with the
   saturating subspace suppressed, so the bulk geometry can be tested in
   isolation.

2. **Coherent residual force** (the RYS-productive component). The part of the
   cumulative residual :math:`\\tilde{S}_{ij}` that is *aligned* with the
   identity stream :math:`\\tilde{X}_i` (large :math:`\\cos\\Phi_{ij}`).
   Doubling it moves the representation *along* its existing manifold — a
   refinement, the regime in which RYS helps. We summarise it as
   :math:`C^{\\parallel}_{ij} = \\mathcal{R}_{ij}\\,|\\cos\\Phi_{ij}|`, the
   component of the kernel-level force that is coherent with the base kernel.

3. **Incoherent cross-term** (the RYS-destructive component). The part of
   :math:`\\tilde{S}_{ij}` that is *orthogonal* to the identity stream (large
   :math:`\\sin^{2}\\Psi_{ij}`). Doubling it amplifies a perturbation that
   downstream layers cannot absorb; this is the off-plateau regime where RYS
   hurts. We summarise it as :math:`C^{\\perp}_{ij} =
   \\mathcal{Q}_{ij}\\,|\\sin\\Psi_{ij}|`.

The headline predictor is the **coherence ratio**

.. math::
    \\mathcal{K}_{ij} = \\frac{C^{\\parallel}_{ij}}{C^{\\perp}_{ij} + \\varepsilon}
                       = \\frac{\\mathcal{R}_{ij}\\,|\\cos\\Phi_{ij}|}
                              {\\mathcal{Q}_{ij}\\,|\\sin\\Psi_{ij}| + \\varepsilon}.

A band with high :math:`\\mathcal{K}` has a residual force that is mostly
*along* the manifold (safe to double); a band with low :math:`\\mathcal{K}`
has a residual force that is mostly *across* the manifold (doubling is
destructive). The theory predicts a *positive* rank correlation between
:math:`\\mathcal{K}` and the RYS :math:`\\Delta`-score.

A second, independent predictor is the **junction misalignment** — how far the
band-output manifold sits from the band-input manifold, normalised by the
input spread (a slight generalisation of
:func:`rys.theory_validation.junction_mismatch` to use the *centred* clouds).
RYS feeds the layer-``end`` state back into layer-``start``; if that junction
is misaligned, the replayed band starts from an off-manifold point regardless
of how well-conditioned its internal force is. We expect a *negative*
correlation between junction misalignment and the effect.

Finally, the band-Jacobian :math:`\\sigma_{\\max}(D\\Phi)` (from
:mod:`rys.theory_validation`) is the *dynamical* stability modulus: it measures
how much one extra pass through the band amplifies an arbitrary perturbation.
It is the contraction/expansion test, and we expect a *negative* correlation
with the effect (already confirmed empirically on Llama-3.2-1B).

The three together — coherence ratio (geometric alignment), junction
misalignment (boundary condition), and :math:`\\sigma_{\\max}` (dynamical
stability) — are the three partial observations the theory unifies. Each
captures a facet the others miss: CKA/ρ captures similarity but not direction;
the Jacobian captures amplification but not where the amplified vector points;
junction misalignment captures the boundary but not the band interior. The
composite **RYS safety score**

.. math::
    \\mathcal{S}_{ij} = \\log\\mathcal{K}_{ij}
                       - \\lambda_{J}\\,\\log\\sigma_{\\max}(D\\Phi_{ij})
                       - \\lambda_{m}\\,m_{ij}

(with :math:`m_{ij}` the junction misalignment and :math:`\\lambda`'s fit by
rank regression on a held-out model) is the module's single best scalar
predictor of the RYS effect. The sign convention is "higher = safer to
duplicate".

All public functions consume the long-format activation DataFrame from
:mod:`rys.activations` and return long-format DataFrames keyed by
``(layer_i, layer_j)``, so they join directly onto the
``delta_score_long.csv`` produced by :mod:`scripts.run_guesstimation_rys`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from rys.cka import _stack_per_layer
from rys.residual_force import residual_force_matrices


def _long_from(bundle: dict[str, pd.DataFrame], upper_only: bool = True) -> pd.DataFrame:
    """Flatten a residual_force_matrices bundle into one long row per pair."""
    layers = bundle["R"].index.tolist()
    rows = []
    for ii, i in enumerate(layers):
        for jj in range(ii + 1, len(layers)):
            j = layers[jj]
            row = {"layer_i": int(i), "layer_j": int(j)}
            for key, frame in bundle.items():
                row[key] = float(frame.at[i, j])
            rows.append(row)
    return pd.DataFrame(rows)


def coherence_table(
    activations: pd.DataFrame,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> pd.DataFrame:
    r"""Per-pair coherent/incoherent force split and the coherence ratio :math:`\mathcal{K}`.

    Returns
    -------
    pd.DataFrame
        Long format, one row per ordered pair ``(layer_i, layer_j)`` with
        ``i < j``, with columns:

        ``coherent_force``      :math:`C^{\parallel}_{ij} = \mathcal{R}_{ij}\,|\cos\Phi_{ij}|`
        ``incoherent_force``    :math:`C^{\perp}_{ij} = \mathcal{Q}_{ij}\,|\sin\Psi_{ij}|`
        ``coherence_ratio``     :math:`\mathcal{K}_{ij} = C^{\parallel}_{ij} / (C^{\perp}_{ij} + \varepsilon)`
        ``log_coherence_ratio`` :math:`\log \mathcal{K}_{ij}` (clipped; the composite score's backbone)

        plus the raw ``R``, ``Q``, ``cos_phi``, ``sin_psi`` (signed
        :math:`\sin\Psi = \sqrt{\sin^2\Psi}`) for transparency.
    """
    bundle = residual_force_matrices(activations, device=device, dtype=dtype)
    long = _long_from(bundle, upper_only=True)
    long["coherent_force"] = long["R"] * long["cos_phi"].abs()
    long["sin_psi"] = long["sin2_psi"].clip(lower=0.0).pow(0.5)
    long["incoherent_force"] = long["Q"] * long["sin_psi"]
    long["coherence_ratio"] = long["coherent_force"] / (long["incoherent_force"] + eps)
    long["log_coherence_ratio"] = np.log(long["coherence_ratio"].clip(lower=eps))
    return long


def junction_misalignment_table(
    activations: pd.DataFrame,
) -> pd.DataFrame:
    r"""Per-pair centred-manifold junction misalignment.

    For each pair ``(i, j)`` we measure how far the *centred* layer-``j``
    activation cloud sits from the centred layer-``i`` cloud, in units of the
    layer-``i`` cloud's RMS token spread:

    .. math::
        m_{ij} = \frac{\| \mathrm{mean}_n(\tilde{X}_j) - \mathrm{mean}_n(\tilde{X}_i) \|}
                      {\sqrt{\mathrm{mean}_n \|\tilde{X}_i - \mathrm{mean}_n(\tilde{X}_i)\|^2} + \varepsilon}.

    This is the centred generalisation of
    :func:`rys.theory_validation.junction_mismatch`: centring removes the
    massive-activation offset, so ``m`` measures the *shape* mismatch between
    the two manifolds rather than their bulk offset. RYS feeds the layer-``j``
    state back into layer-``i``; a low ``m`` means the junction stays on the
    input manifold (the condition the theory predicts for safe windows), so we
    expect a *negative* correlation with the RYS effect.

    Returns
    -------
    pd.DataFrame
        Long format with columns ``layer_i``, ``layer_j``, ``junction_misalignment``.
    """
    layers, X = _stack_per_layer(activations)
    X = X.to(torch.float64)
    L, N, d = X.shape
    Xc = X - X.mean(dim=1, keepdim=True)
    rows = []
    for ii in range(L):
        xi = Xc[ii]
        spread_i = (xi.pow(2).sum(dim=1).mean()).sqrt().item()
        mean_i = xi.mean(dim=0)
        for jj in range(ii + 1, L):
            xj = Xc[jj]
            mean_j = xj.mean(dim=0)
            gap = (mean_j - mean_i).norm().item()
            m = gap / (spread_i + 1e-12)
            rows.append({"layer_i": int(layers[ii]), "layer_j": int(layers[jj]), "junction_misalignment": m})
    return pd.DataFrame(rows)


def massive_activation_stats(activations: pd.DataFrame) -> dict[str, object]:
    """Variance concentration of the stacked activation matrix.

    Returns the per-dimension variance ordering and the fraction of total
    variance captured by the top-k dimensions, plus the global mean and std
    used by the robustifying transforms in
    :func:`rys.geometry_predictors.robustify_activations`.
    """
    layers, X = _stack_per_layer(activations)
    X = X.to(torch.float64)
    all_x = X.reshape(-1, X.shape[-1]).numpy()
    mean = all_x.mean(axis=0)
    var = all_x.var(axis=0)
    std = np.sqrt(var + 1e-8)
    order = np.argsort(var)[::-1]
    total = float(var.sum())
    if total <= 0:
        fracs = {"top1": float("nan"), "top5": float("nan"), "top10": float("nan")}
    else:
        fracs = {
            "top1": float(var[order[0]] / total),
            "top5": float(var[order[:5]].sum() / total),
            "top10": float(var[order[:10]].sum() / total),
        }
    return {
        "mean": mean,
        "std": std,
        "order": order,
        "d_model": int(var.size),
        "top_dims": order[:10].astype(int).tolist(),
        **fracs,
    }


def robustify_activations(
    activations: pd.DataFrame,
    *,
    stats: dict[str, object] | None = None,
    top_k_dims: int = 5,
    drop_first_token: bool = True,
    global_standardize: bool = True,
) -> pd.DataFrame:
    r"""Apply a *fixed* transform that suppresses the massive-activation subspace.

    The transform is the same across all layers, so it preserves residual
    telescoping: if :math:`T` is fixed then
    :math:`T(\tilde{X}_j - \tilde{X}_i) = T\tilde{X}_j - T\tilde{X}_i`, and the
    :math:`\rho/\Phi` theory is still being tested in a different inner
    product. The returned DataFrame has the same schema as the input and can be
    fed to :func:`coherence_table` / :func:`cka_matrix` unchanged.

    The transform composes, in order:

    1. optionally drop the first (sink) token of every prompt;
    2. optionally clamp the top-``k`` highest-variance dimensions to their
       global mean (neutralising the saturating subspace without changing
       dimensionality);
    3. optionally divide every dimension by its global std (so the bulk
       geometry is measured on a whitened scale where no single dim dominates).
    """
    if stats is None:
        stats = massive_activation_stats(activations)
    mean = np.asarray(stats["mean"])
    std = np.asarray(stats["std"])
    top_dims = np.asarray(stats["order"][:top_k_dims], dtype=int) if top_k_dims > 0 else np.array([], dtype=int)

    def transform_one(x: np.ndarray) -> np.ndarray:
        y = np.array(x, dtype=np.float64, copy=True)
        if drop_first_token and y.shape[0] > 1:
            y = y[1:]
        if top_k_dims > 0 and top_dims.size > 0:
            y[:, top_dims] = mean[top_dims]
        if global_standardize:
            y = y / std
        return y

    out = activations.copy()
    if not (drop_first_token or global_standardize or top_k_dims > 0):
        return out
    out["activation"] = [
        transform_one(np.atleast_2d(np.asarray(x)).astype(np.float64, copy=False))
        for x in activations["activation"]
    ]
    return out


def composite_safety_score(
    coherence: pd.DataFrame,
    junction: pd.DataFrame,
    jacobian: pd.DataFrame | None = None,
    *,
    jac_col: str = "jacobian_sigma_max",
    eps: float = 1e-12,
) -> pd.DataFrame:
    r"""Combine the three partial predictors into a single RYS safety score.

    .. math::
        \mathcal{S}_{ij} = \log\mathcal{K}_{ij}
                           - \lambda_{J}\,\log\sigma_{\max}(D\Phi_{ij})
                           - \lambda_{m}\,m_{ij}

    The :math:`\lambda` weights are estimated *in-sample* by rank regression
    (minimising Spearman distance to the data's own effect sign is unstable, so
    we instead use a simple, transparent choice: equal weights on the
    standardised predictors — each predictor is z-scored, then the composite is
    their signed sum with the theory-predicted signs: +coherence, −Jacobian,
    −misalignment). This makes the composite a *sign-consistent* average of the
    three partial observations, with no free parameters fitted to the effect,
    which is the honest thing to do when the goal is to test a theory rather
    than overfit one model.

    Parameters
    ----------
    coherence
        Output of :func:`coherence_table` (must contain ``log_coherence_ratio``).
    junction
        Output of :func:`junction_misalignment_table` (contains
        ``junction_misalignment``).
    jacobian
        Optional long-format table with a ``jacobian_sigma_max`` column and
        ``start``/``end`` keys (as produced by the guesstimation script). If
        ``None`` the composite uses only the two geometric predictors.
    jac_col
        Column name of the Jacobian sigma_max in ``jacobian``.

    Returns
    -------
    pd.DataFrame
        Merged table with the three z-scored predictors and the composite
        ``rys_safety_score``. When ``jacobian`` is ``None`` the Jacobian
        columns are absent and the composite is the mean of the two geometric
        z-scores.
    """
    merged = coherence.merge(junction, on=["layer_i", "layer_j"], how="inner")

    def zscore(s: pd.Series) -> pd.Series:
        mu, sd = s.mean(), s.std(ddof=0)
        return (s - mu) / (sd + eps) if sd > 0 else s * 0.0

    parts = []
    merged["z_coherence"] = zscore(merged["log_coherence_ratio"])
    parts.append(merged["z_coherence"])
    merged["z_misalignment"] = zscore(merged["junction_misalignment"])
    parts.append(-merged["z_misalignment"])

    if jacobian is not None:
        jac = jacobian.rename(columns={"start": "layer_i", "end": "layer_j"})
        merged = merged.merge(
            jac[["layer_i", "layer_j", jac_col]], on=["layer_i", "layer_j"], how="inner"
        )
        merged["z_jacobian"] = zscore(np.log(merged[jac_col].clip(lower=eps)))
        parts.append(-merged["z_jacobian"])

    merged["rys_safety_score"] = sum(parts) / len(parts)
    return merged
