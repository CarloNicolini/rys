"""Analytical prior on the residual-force statistics for Qwen3.6-27B.

This module is a *low-compute* exploratory tool: it predicts the per-layer
distribution of the residual-stream injection :math:`F_k` and the resulting
relative force :math:`\\rho_{ij}`, cross-term magnitude :math:`\\mathcal{Q}_{ij}`,
and incoherence-angle :math:`\\cos\\phi_{ij}` for the Qwen3.6-27B text tower
*without* running a full 27B forward pass.

Three layers of approximation are stacked on top of each other:

1. **Architecture map** (:func:`qwen_layer_type_table`).
   ``AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").text_config.layer_types``
   exposes the 64-position vector of ``{"linear_attention", "full_attention"}``
   tags.  In Qwen3.6 the schedule is strictly periodic with period 4: full
   self-attention at indices ``{3, 7, 11, ..., 63}`` (16 layers, 25%), gated
   DeltaNet (linear attention) elsewhere (48 layers, 75%).

2. **Closed-form Gaussian prior on the block output**
   (:func:`gaussian_fk_prior`).  Under Kaiming-like init with the model's
   declared ``initializer_range`` (0.02), and assuming a Gaussian residual
   stream feeding the pre-norm input, the block's output channels are
   approximately Gaussian by the central-limit theorem at the
   ``d_model = 5120`` summation step.  The first four moments of the SiLU
   nonlinearity are obtained from Gauss-Hermite quadrature on the Gaussian
   measure; the gated-MLP product :math:`\\mathrm{SiLU}(g)\\odot u` then has
   a known second-moment expression via Isserlis' theorem.  The full self-
   attention block introduces a softmax that *breaks* Gaussianity in the
   value-mixing step; we capture this by a Dirichlet-mixture correction on
   the value-axis variance.

3. **Per-pair forecast** (:func:`predict_rho_Q_apriori`).
   Under the incoherent-residual hypothesis (typical middle-layer regime
   for pre-norm LLMs), the residual-stream variance grows additively:
   :math:`\\mathrm{Var}(x_k) \\approx \\mathrm{Var}(x_0) + \\sum_{j<k}
   \\mathrm{Var}(F_j)`.  This gives a closed-form prior for every pair
   :math:`(i, j)`.

The module also provides:

- :func:`instantiate_single_block` — spin up *one* ``Qwen3_5DecoderLayer``
  with random weights matching the model's declared ``initializer_range``
  on the resolved device.  Memory: ~600 MB FP16 per block, fits on a M-series
  Mac.
- :func:`simulate_block_activations` — push synthetic Gaussian residuals
  through one block and record the empirical moments to validate the closed
  form.
- :func:`predict_rys_windows` — rank candidate RYS windows by the predicted
  post-RYS off-plateau amplification :math:`4\\,\\mathcal{Q}_{ij}^{2}`.
- :func:`compare_prior_to_empirical` — overlay the prior on an empirical CKA
  matrix already saved as parquet.

Device selection
----------------
Every entry point accepts ``device: Literal["auto", "mps", "cuda", "cpu"]``
and resolves it via :func:`resolve_device`.  Default ``"auto"`` picks CUDA
if available, then MPS (Apple Silicon), then CPU.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DeviceSpec = Literal["auto", "mps", "cuda", "cpu"]

LayerType = Literal["linear_attention", "full_attention"]

DEFAULT_MODEL_ID = "Qwen/Qwen3.6-27B"


# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------


def resolve_device(spec: DeviceSpec = "auto") -> torch.device:
    """Return a concrete ``torch.device`` from a string spec.

    ``"auto"`` prefers CUDA, then MPS (Apple Silicon), then CPU.  Explicit
    requests for an unavailable backend raise ``RuntimeError`` so the caller
    fails fast rather than silently falling back.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this build.")
        return torch.device("mps")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device spec {spec!r}; expected one of auto|mps|cuda|cpu.")


# ---------------------------------------------------------------------------
# Architecture map
# ---------------------------------------------------------------------------


def qwen_layer_type_table(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    trust_remote_code: bool = True,
) -> pd.DataFrame:
    """Enumerate Qwen3 text-tower layer types from the HuggingFace config.

    Parameters
    ----------
    model_id
        HuggingFace repo id; the default is the model the user is studying.
    trust_remote_code
        Forwarded to ``AutoConfig.from_pretrained``.  Required for Qwen3.5
        configs that ship a custom config class.

    Returns
    -------
    pd.DataFrame
        Columns ``[layer_idx, layer_type, is_full_attention,
        is_junction_candidate]`` with one row per text-tower layer.
        ``is_junction_candidate`` is True iff ``layer_type ==
        "full_attention"`` -- the brain-inspired thalamus-as-junction
        hypothesis (full-attn layers act as cross-phase relay hubs).
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    text = cfg.text_config if hasattr(cfg, "text_config") else cfg
    types = list(text.layer_types)
    return pd.DataFrame(
        {
            "layer_idx": range(len(types)),
            "layer_type": types,
            "is_full_attention": [t == "full_attention" for t in types],
            "is_junction_candidate": [t == "full_attention" for t in types],
        }
    )


# ---------------------------------------------------------------------------
# Gaussian moments of SiLU and the gated MLP
# ---------------------------------------------------------------------------


def _hermite_grid(n_quad: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights renormalised to the standard Gaussian.

    The classical Hermite nodes integrate against :math:`e^{-x^2}`; rescaling
    gives integration against the standard Gaussian measure.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quad)
    weights = weights / math.sqrt(2 * math.pi)
    return nodes, weights


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def silu_moments_under_gaussian(
    sigma_in_sq: float,
    *,
    n_quad: int = 256,
) -> dict[str, float]:
    """First four central moments of ``SiLU(X)`` for ``X ~ N(0, sigma_in_sq)``.

    Computed by Gauss-Hermite quadrature on the standard Gaussian -- exact
    up to floating-point precision for the SiLU integrand, which is smooth
    and has rapidly decaying tails.

    Returns
    -------
    dict[str, float]
        Keys ``mean``, ``var``, ``skew``, ``kurt_excess``.  Excess kurtosis
        (3 subtracted) so a Gaussian has kurt 0.
    """
    if sigma_in_sq <= 0:
        raise ValueError(f"sigma_in_sq must be positive, got {sigma_in_sq}.")
    sigma = math.sqrt(sigma_in_sq)
    nodes, weights = _hermite_grid(n_quad)
    y = _silu(nodes * sigma)
    m1 = float((weights * y).sum())
    m2 = float((weights * y**2).sum())
    var = m2 - m1**2
    if var <= 0:
        return {"mean": m1, "var": 0.0, "skew": 0.0, "kurt_excess": 0.0}
    centred = y - m1
    m3 = float((weights * centred**3).sum())
    m4 = float((weights * centred**4).sum())
    return {
        "mean": m1,
        "var": var,
        "skew": m3 / var**1.5,
        "kurt_excess": m4 / var**2 - 3.0,
    }


# ---------------------------------------------------------------------------
# Closed-form prior on the block output variance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FkPrior:
    """Closed-form prior moments for a single transformer block's output.

    All values are *per channel* of the residual stream.  ``var`` is the
    quantity that propagates into the norm-growth recursion; ``skew`` and
    ``kurt_excess`` quantify how far the marginal departs from Gaussian
    (zero for both means perfectly Gaussian).
    """

    layer_type: str
    var: float
    skew: float
    kurt_excess: float
    mean: float = 0.0  # zero-centred by symmetry of weights and gating

    def asdict(self) -> dict[str, float | str]:
        return asdict(self)


def _projection_var(d_in: int, sigma_w: float, var_in: float) -> float:
    """Variance of ``W @ x`` per output channel for Gaussian ``x``, ``W``."""
    return d_in * sigma_w**2 * var_in


def _mlp_output_var(
    d_model: int,
    d_int: int,
    sigma_w: float,
    rmsnorm_input_var: float,
) -> float:
    """Closed-form per-channel variance of the gated SiLU MLP output.

    Pipeline: ``h -> g = gate_proj(h), u = up_proj(h) -> m = SiLU(g)*u ->
    f = down_proj(m)``.  Under independent Gaussian weights and a centred
    Gaussian input ``h``, ``g`` and ``u`` are independent zero-mean
    Gaussians, the gated product has known second moment by independence,
    and the down projection is a linear sum.
    """
    var_g = _projection_var(d_model, sigma_w, rmsnorm_input_var)
    var_u = _projection_var(d_model, sigma_w, rmsnorm_input_var)
    silu_g_moments = silu_moments_under_gaussian(var_g)
    e_silu_g_sq = silu_g_moments["mean"] ** 2 + silu_g_moments["var"]
    var_m = e_silu_g_sq * var_u
    return _projection_var(d_int, sigma_w, var_m)


def _attention_output_var(
    d_model: int,
    sigma_w: float,
    rmsnorm_input_var: float,
    *,
    layer_type: str,
) -> float:
    """Per-channel variance of the attention sub-layer output.

    For full self-attention with softmax-weighted value mixing, the output
    channel is :math:`\\sum_h W_o^{(h)} V^{(h)} \\mathrm{softmax}(QK)`.  At
    random init the softmax is approximately uniform (entropy
    :math:`\\log(\\text{seq\\_len})`), so the value mixing has variance
    :math:`\\mathrm{Var}(V) / \\text{seq\\_len}`; the output projection
    then multiplies by ``d_model * sigma_w**2``.  For sequence lengths
    in the hundreds this makes the attention output tiny relative to the
    gated MLP path.

    For linear attention (Gated DeltaNet), the output is dominated by the
    recurrent value path; we treat it as a linear contraction with
    variance comparable to a single Kaiming projection step -- i.e. of
    order ``rmsnorm_input_var * d_model * sigma_w**4`` (two-step linear
    composition).  This is an approximation; calibration against the
    empirical CKA matrix in :func:`compare_prior_to_empirical` will
    quantify the residual.
    """
    if layer_type == "full_attention":
        # Approximate the softmax-attended value as having variance
        # var(V) / seq_len_typical, with a value-projection variance equal
        # to one Kaiming step on the residual stream.
        seq_len_typical = 256.0
        var_v = _projection_var(d_model, sigma_w, rmsnorm_input_var)
        var_attended = var_v / seq_len_typical
        return _projection_var(d_model, sigma_w, var_attended)
    if layer_type == "linear_attention":
        # Two-stage linear contraction with sub-unit gating; coarse model.
        var_intermediate = _projection_var(d_model, sigma_w, rmsnorm_input_var)
        return _projection_var(d_model, sigma_w, var_intermediate) * 0.5  # gate damping
    raise ValueError(f"Unknown layer_type {layer_type!r}.")


def gaussian_fk_prior(
    d_model: int,
    d_int: int,
    *,
    layer_type: LayerType,
    initializer_range: float = 0.02,
    rmsnorm_input_var: float = 1.0,
) -> FkPrior:
    """Closed-form variance prior for the block residual ``F_k``.

    The block residual is ``F_k = attn_out + mlp_out`` in the pre-norm
    formulation (mlp consumes the post-attention RMSNorm).  Under
    independent Kaiming-Gaussian weights with ``sigma_w =
    initializer_range``, the two terms are approximately independent so
    their variances add.

    Parameters
    ----------
    d_model
        Residual-stream channel count (5120 for Qwen3.6-27B).
    d_int
        MLP intermediate width (17408 for Qwen3.6-27B).
    layer_type
        Either ``"linear_attention"`` (Gated DeltaNet) or
        ``"full_attention"`` (Qwen3_5Attention).
    initializer_range
        Standard deviation of the Gaussian weight init declared by the
        model config (0.02 for Qwen).
    rmsnorm_input_var
        Variance of each channel of the RMSNorm output.  This is *defined
        to be 1* per channel for a unit-RMS row, regardless of the
        residual-stream norm upstream -- that's the entire point of
        pre-norm RMSNorm.  Override only for sensitivity analysis.

    Returns
    -------
    FkPrior
        Per-channel moments of ``F_k``.
    """
    sigma_w = initializer_range
    var_attn = _attention_output_var(
        d_model, sigma_w, rmsnorm_input_var, layer_type=layer_type
    )
    var_mlp = _mlp_output_var(d_model, d_int, sigma_w, rmsnorm_input_var)
    total_var = var_attn + var_mlp
    # Higher-order moments are dominated by the MLP path; we report the
    # SiLU-induced skew and excess kurtosis on the down-projection input
    # as a proxy that propagates linearly to F_k (a linear projection of
    # an iid sum is Gaussian by CLT, so skew/kurt decay as 1/sqrt(d_int)).
    silu_skew = silu_moments_under_gaussian(_projection_var(d_model, sigma_w, rmsnorm_input_var))[
        "skew"
    ]
    return FkPrior(
        layer_type=layer_type,
        var=total_var,
        skew=silu_skew / math.sqrt(d_int),  # CLT damping by down-projection
        kurt_excess=0.0,  # post-CLT
    )


# ---------------------------------------------------------------------------
# Single-block instantiation and Monte Carlo validation
# ---------------------------------------------------------------------------


def _load_text_config(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    trust_remote_code: bool = True,
):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return cfg.text_config if hasattr(cfg, "text_config") else cfg


@torch.no_grad()
def _apply_qwen_init(module: nn.Module, *, initializer_range: float = 0.02) -> None:
    """Apply Qwen3_5's ``_init_weights`` rules to every submodule of ``module``.

    By default a bare ``nn.Linear`` uses PyTorch's Kaiming-uniform init,
    which is much smaller than the Gaussian ``N(0, initializer_range^2)``
    that Qwen actually starts training from.  Replicating
    ``Qwen3_5PreTrainedModel._init_weights`` here keeps :func:`gaussian_fk_prior`
    self-consistent: the closed-form prior derives the per-channel variance
    *under* this exact init scheme.
    """
    from torch.nn import init as torch_init
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5RMSNorm,
    )

    for sub in module.modules():
        if isinstance(
            sub,
            (
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
            ),
        ):
            if getattr(sub, "weight", None) is not None:
                torch_init.normal_(sub.weight, mean=0.0, std=initializer_range)
            if getattr(sub, "bias", None) is not None:
                torch_init.zeros_(sub.bias)
        elif isinstance(sub, nn.Embedding):
            torch_init.normal_(sub.weight, mean=0.0, std=initializer_range)
        elif (
            "LayerNorm" in type(sub).__name__
            or "RMSNorm" in type(sub).__name__
        ) and isinstance(sub, Qwen3_5RMSNorm):
            # Qwen3_5 RMSNorm uses (1 + weight); zero-init -> gain 1.
            if getattr(sub, "weight", None) is not None:
                torch_init.zeros_(sub.weight)
        elif "LayerNorm" in type(sub).__name__ or "RMSNorm" in type(sub).__name__:
            # Generic norm (e.g. Qwen3_5RMSNormGated): unit-gain init.
            if getattr(sub, "weight", None) is not None:
                torch_init.ones_(sub.weight)
        if isinstance(sub, Qwen3_5GatedDeltaNet):
            if hasattr(sub, "dt_bias") and sub.dt_bias is not None:
                torch_init.ones_(sub.dt_bias)
            if hasattr(sub, "A_log") and sub.A_log is not None:
                a_log = torch.empty_like(sub.A_log).uniform_(0, 16).log_()
                sub.A_log.copy_(a_log)


def instantiate_single_block(
    layer_idx: int,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: DeviceSpec = "auto",
    dtype: torch.dtype = torch.bfloat16,
    apply_qwen_init: bool = True,
) -> nn.Module:
    """Spin up one ``Qwen3_5DecoderLayer`` at random init on the chosen device.

    Memory cost is roughly ``(2 + 3) * d_model * d_int * sizeof(dtype)``,
    i.e. ~600 MB in bf16 for Qwen3.6-27B.  When ``apply_qwen_init=True``
    (the default) the block's weights are re-initialised per the model's
    declared ``Qwen3_5PreTrainedModel._init_weights`` rules: ``N(0,
    initializer_range^2)`` on every linear / conv, zero on
    ``Qwen3_5RMSNorm`` (since it gates via ``(1 + weight)``), and unit on
    other norms.  This is exactly the prior :func:`gaussian_fk_prior`
    assumes.

    Parameters
    ----------
    layer_idx
        Position in the 64-layer stack.  Determines whether a
        ``GatedDeltaNet`` or full-attention sub-module is built.
    model_id
        HuggingFace repo id; only the *config* is fetched, never weights.
    device
        Resolved by :func:`resolve_device`.
    dtype
        Working precision.  bf16 keeps memory tight on Apple Silicon; use
        fp32 only for low-noise validation runs.
    apply_qwen_init
        If True (default), re-initialise weights per Qwen's scheme so the
        block matches what :func:`gaussian_fk_prior` predicts.  Set False
        for a pure PyTorch-default init (Kaiming uniform) -- mostly useful
        for debugging.

    Returns
    -------
    torch.nn.Module
        A single ``Qwen3_5DecoderLayer`` in ``eval()`` mode.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    cfg = _load_text_config(model_id)
    if layer_idx < 0 or layer_idx >= cfg.num_hidden_layers:
        raise IndexError(
            f"layer_idx {layer_idx} outside [0, {cfg.num_hidden_layers})."
        )
    dev = resolve_device(device)
    with torch.device("cpu"):
        block = Qwen3_5DecoderLayer(cfg, layer_idx)
    if apply_qwen_init:
        init_std = float(getattr(cfg, "initializer_range", 0.02))
        _apply_qwen_init(block, initializer_range=init_std)
    block.to(device=dev, dtype=dtype)
    block.eval()
    return block


def _make_position_embeddings(
    d_head: int,
    seq_len: int,
    *,
    base: float = 10_000.0,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (cos, sin) pair compatible with the Qwen rotary attention."""
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(dtype)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(dtype)
    # Qwen expects (1, seq, d_head) batch-broadcastable tensors.
    return cos.unsqueeze(0), sin.unsqueeze(0)


@torch.no_grad()
def simulate_block_activations(
    block: nn.Module,
    *,
    n_tokens: int = 256,
    batch_size: int = 4,
    x_input_std: float = 1.0,
    device: DeviceSpec = "auto",
    seed: int = 0,
    head_dim: int = 256,
) -> dict[str, float | np.ndarray]:
    """Push synthetic Gaussian residuals through one block; record stats.

    Returns the empirical per-channel mean/var/kurtosis of the **block
    residual** :math:`F_k(\\text{RMSNorm}(x_k))` -- i.e. the *contribution*
    the block adds to the residual stream, computed as ``y - x`` where
    ``y`` is the block output.

    This is the quantity the closed-form :func:`gaussian_fk_prior`
    predicts.  Returns also the scalar :math:`\\rho =
    \\|F_k\\|_F/\\|x_k\\|_F` and :math:`\\cos\\phi = \\langle x_k,
    F_k\\rangle / (\\|x_k\\|\\|F_k\\|)` for direct comparison.
    """
    dev = resolve_device(device)
    torch.manual_seed(seed)
    dtype = next(block.parameters()).dtype
    # Match the block's d_model from the input layernorm specifically. The
    # *first* 1-D parameter on a GatedDeltaNet block is the inner gated
    # RMSNorm of width linear_num_value_heads (48), not the residual stream
    # width — so resolve explicitly off ``input_layernorm.weight``.
    if hasattr(block, "input_layernorm"):
        d_model = block.input_layernorm.weight.shape[-1]
    elif hasattr(block, "hidden_size"):
        d_model = int(block.hidden_size)
    else:  # last resort
        d_model = next(p.shape[-1] for p in block.parameters() if p.dim() == 1)
    x = (
        x_input_std
        * torch.randn(batch_size, n_tokens, d_model, device=dev, dtype=dtype)
    )
    cos, sin = _make_position_embeddings(head_dim, n_tokens, device=dev, dtype=dtype)
    out = block(hidden_states=x, position_embeddings=(cos, sin))
    if isinstance(out, tuple):
        out = out[0]
    fk = (out - x).to(torch.float32)
    x_f = x.to(torch.float32)
    fk_flat = fk.reshape(-1, fk.shape[-1])
    x_flat = x_f.reshape(-1, x_f.shape[-1])
    per_channel_mean = fk_flat.mean(dim=0).cpu().numpy()
    per_channel_var = fk_flat.var(dim=0, unbiased=False).cpu().numpy()
    centred = fk_flat - fk_flat.mean(dim=0, keepdim=True)
    per_channel_m4 = (centred**4).mean(dim=0).cpu().numpy()
    per_channel_kurt = np.where(
        per_channel_var > 0, per_channel_m4 / np.maximum(per_channel_var, 1e-12) ** 2 - 3.0, 0.0
    )
    fk_norm = torch.linalg.norm(fk_flat).item()
    x_norm = torch.linalg.norm(x_flat).item()
    inner = (x_flat * fk_flat).sum().item()
    cos_phi = inner / max(fk_norm * x_norm, 1e-12)
    return {
        "per_channel_mean": per_channel_mean,
        "per_channel_var": per_channel_var,
        "per_channel_kurt": per_channel_kurt,
        "fk_norm": fk_norm,
        "x_norm": x_norm,
        "rho": fk_norm / max(x_norm, 1e-12),
        "cos_phi": cos_phi,
    }


# ---------------------------------------------------------------------------
# Per-pair (i, j) prior on rho, Q, cos_phi
# ---------------------------------------------------------------------------


def predict_rho_Q_apriori(
    layer_types: pd.DataFrame,
    *,
    d_model: int = 5120,
    d_int: int = 17408,
    initializer_range: float = 0.02,
    var_x0: float = 1.0,
    seq_len_effective: int = 256,
) -> pd.DataFrame:
    """Per-pair (i, j) prior on rho, Q, cos_phi from architecture alone.

    Assumptions
    -----------
    - Incoherent residual: :math:`\\mathbb{E}[\\langle x_k, F_k\\rangle] = 0`
      so the variance recursion is additive: ``Var(x_{k+1}) = Var(x_k) +
      Var(F_k)``.
    - All :math:`F_k` distributions are Gaussian per channel with the
      closed-form variance from :func:`gaussian_fk_prior`.  This is the
      *random-init* prior, not the trained equilibrium -- deviations of
      the trained model from this prior are the signal of training and
      will be measured in the notebook.
    - The kernel-level cross-term magnitude
      :math:`\\mathcal{Q}_{ij} \\approx \\sqrt{2\\,d/N_{\\text{eff}}} \\,
      \\rho_{ij}` under independent Gaussian centred X_i and S
      (Marchenko-Pastur asymptotics).

    Parameters
    ----------
    layer_types
        Output of :func:`qwen_layer_type_table`.
    d_model, d_int
        Architecture sizes; defaults match Qwen3.6-27B.
    initializer_range
        Standard deviation of the random-init Gaussian; defaults match Qwen.
    var_x0
        Per-channel variance of the input residual stream at layer 0
        (post embedding + first RMSNorm).  In practice ``embed_tokens``
        plus the first RMSNorm yields approximately unit variance.
    seq_len_effective
        Token count entering the kernel-level Q scaling.  Use the
        average ``N`` of the CKA estimator (per-prompt tokens * n_prompts).

    Returns
    -------
    pd.DataFrame
        Long-format with one row per ordered pair ``(layer_i, layer_j)``
        with ``i < j``.  Columns: ``rho_prior``, ``Q_prior``,
        ``cos_phi_prior``, ``one_minus_cka_prior``, ``var_xi``,
        ``cumulative_var_S``, ``crosses_full_attn`` (junction flag).
    """
    L = len(layer_types)
    var_fk = np.zeros(L)
    for row in layer_types.itertuples():
        prior = gaussian_fk_prior(
            d_model,
            d_int,
            layer_type=row.layer_type,
            initializer_range=initializer_range,
            rmsnorm_input_var=1.0,
        )
        var_fk[row.layer_idx] = prior.var
    var_x = np.empty(L + 1)
    var_x[0] = var_x0
    for k in range(L):
        var_x[k + 1] = var_x[k] + var_fk[k]
    full_attn_positions = set(
        int(idx) for idx in layer_types.loc[layer_types["is_full_attention"], "layer_idx"]
    )
    kernel_factor = math.sqrt(2.0 * d_model / max(seq_len_effective, 1))
    rows: list[dict[str, float | int | bool]] = []
    for i in range(L):
        for j in range(i + 1, L + 1):
            cum = float(var_fk[i:j].sum())
            rho = math.sqrt(cum / max(var_x[i], 1e-12))
            q = kernel_factor * rho
            crosses = any(i <= p < j for p in full_attn_positions)
            rows.append(
                {
                    "layer_i": i,
                    "layer_j": j,
                    "rho_prior": rho,
                    "Q_prior": q,
                    "cos_phi_prior": 0.0,  # incoherence assumption
                    "one_minus_cka_prior": 0.5 * q * q,  # sin^2 Psi = 1 under incoherence
                    "var_xi": float(var_x[i]),
                    "cumulative_var_S": cum,
                    "crosses_full_attn": crosses,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A-priori RYS window forecast
# ---------------------------------------------------------------------------


def predict_rys_windows(
    rho_Q_table: pd.DataFrame,
    *,
    top_k: int = 25,
    min_block_length: int = 4,
    require_plateau: bool = True,
    plateau_quantile: float = 0.25,
) -> pd.DataFrame:
    """Rank candidate RYS windows by predicted post-RYS deviation.

    Per the existing posts' Eq. (7) (`2026-04-29-Similarity-of-neural-
    networks-represetations.md`), RYS amplifies the off-plateau deviation
    by approximately 4 in the incoherent regime: ``1 - CKA^RYS \\approx
    4 (1 - CKA^base) = 2 Q_prior^2``.

    The optimal window has two simultaneous properties:

    - **Plateau condition.** The base CKA inside the window is high
      (``1 - CKA`` is small).  In our prior this translates to
      ``rho_prior`` being in the bottom ``plateau_quantile`` of the
      distribution of within-window pairs.
    - **Block length.** Long enough for the diameter of the sub-DAG to
      matter (``j - i >= min_block_length``).

    Parameters
    ----------
    rho_Q_table
        Output of :func:`predict_rho_Q_apriori`.
    top_k
        Number of windows to return.
    min_block_length
        Minimum ``j - i``; defaults to 4 to align with the Qwen full-attn
        period.
    require_plateau
        If True, only return windows whose ``rho_prior`` is below the
        ``plateau_quantile``-th quantile across all candidates.
    plateau_quantile
        Quantile cutoff for the plateau condition.

    Returns
    -------
    pd.DataFrame
        Sorted by ``post_rys_deviation`` descending, with the top-k rows.
    """
    df = rho_Q_table.copy()
    df["block_length"] = df["layer_j"] - df["layer_i"]
    df = df[df["block_length"] >= min_block_length].copy()
    df["post_rys_deviation"] = 2.0 * df["Q_prior"] ** 2  # = 4 * (1-CKA)_prior
    if require_plateau:
        cutoff = df["rho_prior"].quantile(plateau_quantile)
        df = df[df["rho_prior"] <= cutoff]
    df = df.sort_values("post_rys_deviation", ascending=False).head(top_k)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Empirical comparison
# ---------------------------------------------------------------------------


def compare_prior_to_empirical(
    prior_long: pd.DataFrame,
    empirical_cka_path: str | Path,
) -> pd.DataFrame:
    """Merge the analytical prior with an empirical CKA matrix.

    The empirical CKA parquet should be a square ``L x L`` matrix indexed
    by integer layer ids (the format produced by
    :func:`rys.cka.cka_matrix`).  Returns a long-format DataFrame with one
    row per pair ``(i, j)``, ``i < j``, with prior and empirical columns
    side by side for scatter plots and rank correlations.

    Notes
    -----
    The prior is *random init*; the empirical matrix is *trained
    equilibrium*.  Deviations should:

    - localise inside the reasoning plateau (training-induced structure),
    - peak at outlier-feature channels in specific layers,
    - persist at the encoder/reasoning and reasoning/decoder junctions.

    This is the brain-inspired hypothesis 8 from the new blog post:
    *trained-CKA minus random-init-CKA-prior* is the LLM analog of
    task-evoked minus resting-state functional connectivity.
    """
    M = pd.read_parquet(empirical_cka_path)
    if isinstance(M.columns[0], str):
        M.columns = [int(c) for c in M.columns]
    if isinstance(M.index[0], str):
        M.index = [int(c) for c in M.index]
    rows: list[dict[str, float | int | bool]] = []
    for row in prior_long.itertuples():
        i, j = int(row.layer_i), int(row.layer_j)
        if i in M.index and (j - 1) in M.columns:
            empirical = float(M.loc[i, j - 1])  # j is half-open, matrix is closed
        elif i in M.index and j in M.columns:
            empirical = float(M.loc[i, j])
        else:
            empirical = float("nan")
        rows.append(
            {
                "layer_i": i,
                "layer_j": j,
                "rho_prior": row.rho_prior,
                "Q_prior": row.Q_prior,
                "one_minus_cka_prior": row.one_minus_cka_prior,
                "one_minus_cka_empirical": 1.0 - empirical,
                "crosses_full_attn": row.crosses_full_attn,
            }
        )
    return pd.DataFrame(rows)
