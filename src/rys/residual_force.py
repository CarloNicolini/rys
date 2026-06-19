"""Residual-force decomposition of the CKA connectome.

The blog post

  https://carlonicolini.github.io/sections/science/_posts/2026-04-29-Similarity-of-neural-networks-represetations.md

derives a closed-form decomposition of the linear-CKA edge between two layers
in a residual stream.  Defining the centred per-layer activations
:math:`\\tilde{X}_l = X_l - \\bar{X}_l`, the cumulative residual
:math:`\\tilde{S}_{ij} = \\tilde{X}_j - \\tilde{X}_i`, and the feature-space
Grams

  - :math:`A_i  = \\tilde{X}_i^\\top \\tilde{X}_i`
  - :math:`B_{ij} = \\tilde{X}_i^\\top \\tilde{S}_{ij}`
  - :math:`C_{ij} = \\tilde{S}_{ij}^\\top \\tilde{S}_{ij}`

the post introduces two scalar summary statistics (Eq. 9 of the post)

  - :math:`\\mathcal{R}_{ij} = \\| C_{ij} \\|_F / \\| A_i \\|_F`
  - :math:`\\cos\\Phi_{ij} = \\langle A_i, C_{ij} \\rangle_F /
    (\\| A_i \\|_F \\, \\| C_{ij} \\|_F)`

and originally predicted the self-Gram small-force approximation

  :math:`1 - \\mathrm{CKA}_{ij} \\approx \\tfrac{1}{2}\\, \\mathcal{R}_{ij}^{2}\\,
  \\sin^{2}\\Phi_{ij}` .

This module computes both that legacy :math:`\\mathcal{R}/\\Phi` predictor and
the corrected generic cross-Gram :math:`\\mathcal{Q}/\\Psi` predictor, and it
also forecasts the post-RYS CKA via the doubling substitution
:math:`\\tilde{S}^{\\mathrm{rys}} \\approx 2\\,\\tilde{S}^{(0)}` of Eq. 13
(without re-running the model).

Important caveat about Eq. 10
-----------------------------
The plateau formula folds only the :math:`C_{ij}` (residual self-Gram) term
of the perturbation :math:`K_Y - K_X = M_{ij} + M_{ij}^\\top + C_{ij}`.  When
the residual :math:`\\tilde{S}_{ij}` is approximately *incoherent* with the
identity stream :math:`\\tilde{X}_i` (random-looking perturbation), the
cross-term :math:`M_{ij} = \\tilde{X}_i^\\top \\tilde{S}_{ij}` actually
dominates :math:`1 - \\mathrm{CKA}` at order :math:`\\rho_{ij}^{2}` while
:math:`\\mathcal{R}^{2}\\sin^{2}\\Phi` only contributes at order
:math:`\\rho_{ij}^{4}`.  We therefore expose a third scalar

  :math:`\\mathcal{Q}_{ij} = \\| M_{ij} + M_{ij}^\\top \\|_F / \\| A_i \\|_F`

with phase :math:`\\Psi_{ij}` against the base kernel.  The generic plateau
law is therefore

  :math:`1 - \\mathrm{CKA}_{ij} \\approx \\tfrac{1}{2}\\, \\mathcal{Q}_{ij}^{2}\\,
  \\sin^{2}\\Psi_{ij}` .

so the user can check empirically which of :math:`\\mathcal{R}` and
:math:`\\mathcal{Q}` carries the leading-order signal in their connectome,
and rebuild a more accurate small-force prediction if needed.  The closed-
form ``cka_full`` is the only quantity that is exact at all orders.

All public functions take and return pandas DataFrames with integer layer-id
indices, matching the convention used by :mod:`rys.cka`.

Implementation notes
--------------------
- Centering is per-layer: the column means of :math:`X_l` along the *sample*
  (token) axis are subtracted.  This matches the centring convention required
  by linear CKA and by the closed-form derivation in the post.
- Computations are carried out on sample-space Grams
  :math:`G_l = \\tilde{X}_l \\tilde{X}_l^\\top \\in \\mathbb{R}^{N \\times N}`.
  Frobenius norms and inner products of feature-space Grams equal those of the
  sample-space Grams via the trace cycle, so we never materialise the
  :math:`d \\times d` blocks explicitly.  This keeps memory bounded by
  :math:`O(N^{2})` rather than :math:`O(d^{2})`, an order of magnitude saving
  for typical LLM extractions where :math:`d > N`.
- For numerical stability we promote everything to ``torch.float64`` inside
  the inner loop.  CKA values themselves are :math:`O(1)`; the small-residual
  numerator :math:`1 - \\mathrm{CKA}` is what we care about and is the most
  sensitive to round-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from rys.cka import _stack_per_layer

# ---------------------------------------------------------------------------
# Internal: shared sample-space Gram helpers
# ---------------------------------------------------------------------------


def _center(X: torch.Tensor) -> torch.Tensor:
    """Column-mean centre activations along the sample axis.

    Accepts both ``(N, d)`` (single layer) and ``(L, N, d)`` (stacked layers).
    In the 3-D case the mean is taken per-layer over the sample axis only,
    not across layers.  This matches the centring used by linear CKA and
    the post's derivation of :math:`A_i`, :math:`B_{ij}`, :math:`C_{ij}`.
    """
    if X.dim() == 2:
        return X - X.mean(dim=0, keepdim=True)
    if X.dim() == 3:
        return X - X.mean(dim=1, keepdim=True)
    raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(X.shape)}.")


def _gram_pack(X_centred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-layer sample-space Grams ``G_l`` and their Frobenius norms.

    ``X_centred`` is shape ``(L, N, d)``.  Returns ``(G, gram_norm)`` with
    ``G`` shape ``(L, N, N)`` and ``gram_norm`` shape ``(L,)`` containing
    :math:`\\| G_l \\|_F`.
    """
    G = X_centred @ X_centred.transpose(1, 2)  # (L, N, N)
    gram_norm = torch.linalg.norm(G.reshape(G.shape[0], -1), dim=1)
    return G, gram_norm


def _frobenius_inner(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product :math:`\\langle A, B\\rangle_F` for matrices."""
    return (A * B).sum()


# ---------------------------------------------------------------------------
# Public: residual-force decomposition of the CKA connectome
# ---------------------------------------------------------------------------


def residual_force_matrices(
    activations: pd.DataFrame,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, pd.DataFrame]:
    """Return per-pair scalars of the residual-force decomposition.

    Parameters
    ----------
    activations
        Long-format activations DataFrame produced by
        :func:`rys.activations.capture_residual_stream` or
        :func:`rys.activations.capture_generated_residual_stream`.
    device
        Torch device on which the matmuls are performed.  CPU is fine for
        ``L * N <= ~5000``; GPU is recommended above.
    dtype
        Working precision.  Default ``float64`` because :math:`1 -
        \\mathrm{CKA}` is the small quantity we care about.

    Returns
    -------
    dict[str, pd.DataFrame]
        Each value is a square ``L x L`` DataFrame indexed by layer id.

        ``S_norm``           :math:`\\| \\tilde{S}_{ij} \\|_F` — feature-space
                              cumulative residual norm; doubles under RYS to
                              first order.
        ``R``                :math:`\\mathcal{R}_{ij}` — kernel-level relative
                              force; quadruples under RYS to first order.
        ``R2``               :math:`\\mathcal{R}_{ij}^{2}`.
        ``cos_phi``          :math:`\\cos\\Phi_{ij}` — alignment between
                              :math:`A_i` and :math:`C_{ij}`; invariant under
                              RYS to first order.
        ``sin2_phi``         :math:`1 - \\cos^{2}\\Phi_{ij}`.
        ``cos_psi``          :math:`\\cos\\Psi_{ij}` — alignment between the
                              base sample Gram and the cross-kernel
                              perturbation.
        ``sin2_psi``         :math:`1 - \\cos^{2}\\Psi_{ij}`.
        ``Q``                :math:`\\mathcal{Q}_{ij} = \\| M_{ij} + M_{ij}^\\top
                              \\|_F / \\| A_i \\|_F` where :math:`M_{ij} =
                              \\tilde{X}_i^\\top \\tilde{S}_{ij}` — the
                              cross-term that dominates Eq. 10 when the
                              residual is incoherent with the identity stream.
        ``one_minus_cka_Rphi`` Legacy self-Gram prediction of :math:`1 -
                              \\mathrm{CKA}_{ij}`:
                              :math:`\\tfrac{1}{2} \\mathcal{R}_{ij}^{2}
                              \\sin^{2}\\Phi_{ij}`.  Quantitatively correct
                              only when the residual is highly *coherent*
                              with the identity stream.
        ``one_minus_cka_Qpsi`` Generic cross-Gram plateau-regime prediction:
                              :math:`\\tfrac{1}{2} \\mathcal{Q}_{ij}^{2}
                              \\sin^{2}\\Psi_{ij}`.
        ``one_minus_cka_plateau`` Alias of ``one_minus_cka_Qpsi`` for the
                              current paper convention.
        ``cka_full``          Closed-form linear CKA from sample Grams
                              :math:`\\langle G_i, G_j\\rangle_F /
                              (\\|G_i\\|_F \\|G_j\\|_F)`.  Biased linear CKA;
                              agrees with :func:`rys.cka.cka_matrix` up to
                              the biased-vs-unbiased HSIC choice and is the
                              exact reference against which
                              ``one_minus_cka_plateau`` is compared.

    Notes
    -----
    The matrices are symmetric except for ``S_norm``, ``R``, ``R2``,
    ``cos_phi``, ``sin2_phi``, ``Q``, ``cos_psi`` and ``sin2_psi`` which are *not* symmetric: their value at
    :math:`(i, j)` is normalised by :math:`\\| A_i \\|_F` and so depends on
    which layer plays the role of the *identity stream*.  The post's
    convention is :math:`i < j`; for completeness we fill the strictly lower
    triangle with the analogous quantities normalised by :math:`\\| A_j
    \\|_F` so every cell is meaningful.  Diagonal entries are zero by
    construction (:math:`\\tilde{S}_{ii} = 0`).
    """
    layers, X = _stack_per_layer(activations)
    X = X.to(device=device, dtype=dtype)
    L = X.shape[0]
    Xc = _center(X)
    G, g_norm = _gram_pack(Xc)

    s_norm = torch.zeros((L, L), dtype=dtype, device=device)
    R = torch.zeros((L, L), dtype=dtype, device=device)
    cos_phi = torch.full((L, L), float("nan"), dtype=dtype, device=device)
    cos_psi = torch.full((L, L), float("nan"), dtype=dtype, device=device)
    Q = torch.zeros((L, L), dtype=dtype, device=device)
    cka_full = torch.zeros((L, L), dtype=dtype, device=device)
    one_minus_cka_full_pred = torch.zeros((L, L), dtype=dtype, device=device)

    for i in range(L):
        cka_full[i, i] = 1.0
        for j in range(L):
            if i == j:
                continue
            S = Xc[j] - Xc[i]                       # (N, d)
            G_S = S @ S.transpose(0, 1)              # (N, N) — sample-Gram of S
            G_M = Xc[i] @ S.transpose(0, 1)          # (N, N) — sample form of M = X_i^T S
            S_F = torch.linalg.norm(S)
            G_S_norm = torch.linalg.norm(G_S)
            G_M_norm = torch.linalg.norm(G_M)        # = ||M||_F via trace cycle
            denom_phi = g_norm[i] * G_S_norm
            cos = (
                _frobenius_inner(G[i], G_S) / denom_phi
                if denom_phi > 0
                else torch.tensor(float("nan"))
            )
            s_norm[i, j] = S_F
            R[i, j] = G_S_norm / g_norm[i]
            cos_phi[i, j] = cos
            # Q = ||M + M^T||_F / ||A_i||_F. Using ||M+M^T||^2 = 2||M||^2 + 2 tr(M^2)
            # and the trace cycle tr(M^2) = tr((X S^T)(X S^T)) = ||S X^T X S^T... we can
            # compute it directly from G_M.
            T = G_M + G_M.transpose(0, 1)
            T_norm = torch.linalg.norm(T)
            Q[i, j] = T_norm / g_norm[i]
            cos_psi[i, j] = (
                _frobenius_inner(G[i], T) / (g_norm[i] * T_norm)
                if T_norm > 0
                else torch.tensor(float("nan"))
            )
            # Closed-form CKA (biased; sample-Gram form).
            cka_ij = _frobenius_inner(G[i], G[j]) / (g_norm[i] * g_norm[j])
            cka_full[i, j] = cka_ij
            one_minus_cka_full_pred[i, j] = 1.0 - cka_ij

    R2 = R * R
    sin2_phi = 1.0 - cos_phi * cos_phi
    sin2_psi = 1.0 - cos_psi * cos_psi
    one_minus_cka_Rphi = 0.5 * R2 * sin2_phi
    one_minus_cka_Qpsi = 0.5 * Q * Q * sin2_psi

    def _frame(t: torch.Tensor) -> pd.DataFrame:
        return pd.DataFrame(
            t.detach().cpu().numpy().astype(np.float64),
            index=layers,
            columns=layers,
        )

    return {
        "S_norm": _frame(s_norm),
        "R": _frame(R),
        "R2": _frame(R2),
        "cos_phi": _frame(cos_phi),
        "sin2_phi": _frame(sin2_phi),
        "cos_psi": _frame(cos_psi),
        "sin2_psi": _frame(sin2_psi),
        "Q": _frame(Q),
        "one_minus_cka_Rphi": _frame(one_minus_cka_Rphi),
        "one_minus_cka_Qpsi": _frame(one_minus_cka_Qpsi),
        "one_minus_cka_plateau": _frame(one_minus_cka_Qpsi),
        "cka_full": _frame(cka_full),
    }


def residual_force_long(
    activations: pd.DataFrame,
    *,
    upper_only: bool = True,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    """Long-format companion of :func:`residual_force_matrices`.

    One row per ordered pair ``(layer_i, layer_j)`` with all scalars in
    columns.  This format is convenient for scatter plots, log-log regressions
    of empirical vs predicted :math:`1 - \\mathrm{CKA}`, and groupby-based
    comparisons across blocks (encoder / plateau / decoder).

    Parameters
    ----------
    upper_only
        If True (default), only emit pairs with ``layer_i < layer_j``.  Set
        False to keep the full :math:`L^2 - L` table.
    """
    bundle = residual_force_matrices(activations, device=device, dtype=dtype)
    layers = bundle["R"].index
    rows = []
    for i in layers:
        for j in layers:
            if i == j:
                continue
            if upper_only and i >= j:
                continue
            rows.append(
                {
                    "layer_i": int(i),
                    "layer_j": int(j),
                    "S_norm": float(bundle["S_norm"].at[i, j]),
                    "R": float(bundle["R"].at[i, j]),
                    "R2": float(bundle["R2"].at[i, j]),
                    "cos_phi": float(bundle["cos_phi"].at[i, j]),
                    "sin2_phi": float(bundle["sin2_phi"].at[i, j]),
                    "cos_psi": float(bundle["cos_psi"].at[i, j]),
                    "sin2_psi": float(bundle["sin2_psi"].at[i, j]),
                    "Q": float(bundle["Q"].at[i, j]),
                    "one_minus_cka_Rphi": float(bundle["one_minus_cka_Rphi"].at[i, j]),
                    "one_minus_cka_Qpsi": float(bundle["one_minus_cka_Qpsi"].at[i, j]),
                    "one_minus_cka_plateau": float(bundle["one_minus_cka_plateau"].at[i, j]),
                    "cka_full": float(bundle["cka_full"].at[i, j]),
                    "one_minus_cka_full": float(1.0 - bundle["cka_full"].at[i, j]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public: forecast the post-RYS CKA from base activations alone
# ---------------------------------------------------------------------------


def predict_cka_under_rys(
    activations: pd.DataFrame,
    window: tuple[int, int],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    """Predict the post-RYS CKA matrix from base activations only.

    Uses the first-order linearisation of the post (Sec. 7):
    :math:`\\tilde{S}^{\\mathrm{rys}}_{i_w, j_w} \\approx
    2\\,\\tilde{S}^{(0)}_{i_w, j_w}` and the propagation rule

      :math:`\\tilde{X}^{\\mathrm{rys}}_l = \\tilde{X}^{(0)}_l + (\\tilde{X}^{(0)}_{j_w}
      - \\tilde{X}^{(0)}_{i_w})` for layers downstream of the window
      (and for second-pass states inside the window),

      :math:`\\tilde{X}^{\\mathrm{rys}}_l = \\tilde{X}^{(0)}_l` otherwise.

    The closed-form linear CKA is then computed on these synthesised
    activations.  The returned matrix is what :func:`rys.cka.cka_matrix`
    *should* produce when called inside ``apply_rys`` with the same window,
    to first order in the residual force.  Comparing the two (predicted vs
    empirical) is the core test of the RYS amplification claim.

    Parameters
    ----------
    activations
        Base-model activations (unmodified forward pass).
    window
        ``(i_w, j_w)`` half-open layer interval matching the convention of
        :func:`rys.surgery.apply_rys`.  Layers ``i_w .. j_w - 1`` are the
        duplicated set; layer ``j_w`` is the layer at whose input the
        doubled stream is injected.

    Returns
    -------
    pd.DataFrame
        Square symmetric ``L x L`` predicted-CKA matrix, indexed by layer id.
    """
    i_w, j_w = window
    if i_w >= j_w:
        raise ValueError(f"Bad window {window}: need start < end.")

    layers, X = _stack_per_layer(activations)
    if i_w < layers[0] or j_w > layers[-1] + 1:
        raise IndexError(
            f"Window {window} is outside captured layer range "
            f"[{layers[0]}, {layers[-1] + 1}]."
        )
    X = X.to(device=device, dtype=dtype)
    Xc = _center(X)
    L = Xc.shape[0]

    layer_to_pos = {layer: pos for pos, layer in enumerate(layers)}
    pos_iw = layer_to_pos[i_w]
    # In the half-open convention the duplicated layers are ``i_w .. j_w-1``
    # and the affected post-RYS state at layer ``j_w`` itself is the doubled
    # one.  We treat any layer >= j_w as carrying the +Delta offset.
    delta = Xc[layer_to_pos[j_w]] - Xc[pos_iw] if j_w in layer_to_pos else (
        Xc[layer_to_pos[layers[-1]]] - Xc[pos_iw]
    )

    Xc_rys = Xc.clone()
    for pos, layer in enumerate(layers):
        if layer >= j_w or i_w < layer < j_w:
            Xc_rys[pos] = Xc[pos] + delta
    # Re-centre, since the additive delta has zero column-mean only if the
    # base centring was exact in float64 — be defensive.
    Xc_rys = _center(Xc_rys)

    G_rys, g_norm_rys = _gram_pack(Xc_rys)
    M = torch.empty((L, L), dtype=dtype, device=device)
    for i in range(L):
        M[i, i] = 1.0
        for j in range(i + 1, L):
            denom = g_norm_rys[i] * g_norm_rys[j]
            val = _frobenius_inner(G_rys[i], G_rys[j]) / denom if denom > 0 else torch.tensor(float("nan"))
            M[i, j] = val
            M[j, i] = val
    return pd.DataFrame(M.detach().cpu().numpy().astype(np.float64), index=layers, columns=layers)


# ---------------------------------------------------------------------------
# Public: amplification scalars to plot pair-by-pair against base
# ---------------------------------------------------------------------------


def amplification_long(
    base_bundle: dict[str, pd.DataFrame],
    rys_bundle: dict[str, pd.DataFrame],
    *,
    window: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Build a long-format DataFrame of base / RYS / ratio scalars per pair.

    Targets predicted by the post (Sec. *RYS doubling on the CKA edge*).
    The leading-order $$1-\\mathrm{CKA}$$ in the typical incoherent regime
    is set by the cross-term :math:`\\mathcal{Q} = \\| M + M^\\top \\|_F /
    \\| A_i \\|_F`, which doubles under RYS (because :math:`\\tilde S` doubles
    and :math:`M = \\tilde X^\\top \\tilde S` is linear in :math:`\\tilde S`).
    Squaring the cross-term gives a factor four, which is the expected
    off-plateau amplification:

    =====================  ===============================
    column                 theoretical RYS / base ratio
    =====================  ===============================
    ``S_norm_ratio``       :math:`\\approx 2`
    ``Q_ratio``            :math:`\\approx 2` (cross-term doubling)
    ``R_ratio``            :math:`\\approx 4` (kernel-self-Gram quadrupling)
    ``cos_phi_diff``       :math:`\\approx 0`
    ``one_minus_cka_ratio``  :math:`\\approx 4`
                           (saturates as
                           :math:`1 - \\mathrm{CKA}^{(0)} \\to 1/4`)
    =====================  ===============================

    The intended workflow is::

        base = residual_force_matrices(act_base)
        rys  = residual_force_matrices(act_rys)   # captured under apply_rys
        df   = amplification_long(base, rys, window=(34, 50))

    and then ``df.plot.scatter("S_norm_base", "S_norm_rys", logx=True, logy=True)``
    or ``df.groupby("region")["one_minus_cka_ratio"].describe()``.

    Parameters
    ----------
    base_bundle, rys_bundle
        Outputs of :func:`residual_force_matrices` for the base model and
        the RYS-modified model respectively.
    window
        Optional ``(i_w, j_w)`` to tag pairs with a categorical ``region``
        column: ``"in_window"`` if both endpoints lie inside ``[i_w, j_w]``,
        ``"crossing"`` if exactly one does, ``"outside"`` if neither.

    Returns
    -------
    pd.DataFrame
        One row per ordered pair ``(layer_i, layer_j)`` with ``i < j``.
    """
    layers = base_bundle["R"].index
    if not layers.equals(rys_bundle["R"].index):
        raise ValueError("Base and RYS bundles must share the same layer index.")

    rows = []
    for i in layers:
        for j in layers:
            if i >= j:
                continue
            base_S = float(base_bundle["S_norm"].at[i, j])
            rys_S = float(rys_bundle["S_norm"].at[i, j])
            base_R = float(base_bundle["R"].at[i, j])
            rys_R = float(rys_bundle["R"].at[i, j])
            base_omc = float(1.0 - base_bundle["cka_full"].at[i, j])
            rys_omc = float(1.0 - rys_bundle["cka_full"].at[i, j])
            row = {
                "layer_i": int(i),
                "layer_j": int(j),
                "S_norm_base": base_S,
                "S_norm_rys": rys_S,
                "S_norm_ratio": rys_S / base_S if base_S > 0 else float("nan"),
                "R_base": base_R,
                "R_rys": rys_R,
                "R_ratio": rys_R / base_R if base_R > 0 else float("nan"),
                "cos_phi_base": float(base_bundle["cos_phi"].at[i, j]),
                "cos_phi_rys": float(rys_bundle["cos_phi"].at[i, j]),
                "cos_phi_diff": float(rys_bundle["cos_phi"].at[i, j]
                                      - base_bundle["cos_phi"].at[i, j]),
                "one_minus_cka_base": base_omc,
                "one_minus_cka_rys": rys_omc,
                "one_minus_cka_ratio": rys_omc / base_omc if base_omc > 0 else float("nan"),
            }
            if window is not None:
                i_w, j_w = window
                in_i = i_w <= i <= j_w
                in_j = i_w <= j <= j_w
                if in_i and in_j:
                    row["region"] = "in_window"
                elif in_i ^ in_j:
                    row["region"] = "crossing"
                else:
                    row["region"] = "outside"
            rows.append(row)
    return pd.DataFrame(rows)
