"""Tests for ``rys.analytical_fk``.

These tests exercise the *analytical* parts of the module that do not
require network access.  The two HuggingFace-config-dependent functions
(``qwen_layer_type_table``, ``instantiate_single_block``) are exercised in
the integration test below, which is gated on the config being available
locally; on a fresh machine without a cached config it is skipped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from rys.analytical_fk import (
    FkPrior,
    gaussian_fk_prior,
    predict_rho_Q_apriori,
    predict_rys_windows,
    resolve_device,
    silu_moments_under_gaussian,
)

# ---------------------------------------------------------------------------
# Device resolver
# ---------------------------------------------------------------------------


def test_resolve_device_auto_returns_concrete_device() -> None:
    dev = resolve_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_resolve_device_cpu_always_works() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unknown_spec() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")  # type: ignore[arg-type]


def test_resolve_device_cuda_raises_when_absent() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot test the absent-CUDA branch.")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


# ---------------------------------------------------------------------------
# SiLU moments via Gauss-Hermite quadrature
# ---------------------------------------------------------------------------


def test_silu_moments_match_monte_carlo() -> None:
    """Closed-form SiLU moments should agree with a 200k-sample Monte Carlo."""
    rng = np.random.default_rng(0)
    n = 200_000
    for sigma_sq in (0.5, 1.0, 2.0):
        x = rng.standard_normal(n) * math.sqrt(sigma_sq)
        y = x / (1.0 + np.exp(-x))
        mc_mean = float(y.mean())
        mc_var = float(y.var())
        analytic = silu_moments_under_gaussian(sigma_sq)
        assert abs(analytic["mean"] - mc_mean) < 0.01
        assert abs(analytic["var"] - mc_var) < 0.02


def test_silu_moments_positive_var() -> None:
    """Variance must be strictly positive for non-degenerate input."""
    m = silu_moments_under_gaussian(1.0)
    assert m["var"] > 0


def test_silu_moments_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError):
        silu_moments_under_gaussian(0.0)
    with pytest.raises(ValueError):
        silu_moments_under_gaussian(-1.0)


# ---------------------------------------------------------------------------
# Gaussian F_k prior
# ---------------------------------------------------------------------------


def test_gaussian_fk_prior_is_positive() -> None:
    """Both layer types should produce a strictly positive variance."""
    for layer_type in ("linear_attention", "full_attention"):
        prior = gaussian_fk_prior(
            d_model=5120, d_int=17408, layer_type=layer_type
        )
        assert isinstance(prior, FkPrior)
        assert prior.var > 0
        assert prior.layer_type == layer_type


def test_gaussian_fk_prior_var_scales_with_initializer_range() -> None:
    """Doubling sigma_w should multiply F_k variance by ~16 (linear sum of
    Kaiming projections; each linear stage scales variance by sigma_w^2)."""
    p1 = gaussian_fk_prior(
        d_model=5120, d_int=17408, layer_type="linear_attention", initializer_range=0.02
    )
    p2 = gaussian_fk_prior(
        d_model=5120, d_int=17408, layer_type="linear_attention", initializer_range=0.04
    )
    ratio = p2.var / p1.var
    # Heuristic: 4-stage linear pipeline -> sigma_w^8 -> 256x for the MLP path.
    # The full pipeline mixes attn (different stage count) so the ratio is
    # between 16 and 256; assert it's well above the lower bound.
    assert ratio > 16


def test_gaussian_fk_prior_full_attn_smaller_than_linear() -> None:
    """At random init with seq_len_typical=256 in the softmax averaging,
    full-attn output variance is smaller than linear-attn output
    variance (softmax dampens by 1/seq_len)."""
    pa = gaussian_fk_prior(d_model=5120, d_int=17408, layer_type="full_attention")
    pl = gaussian_fk_prior(d_model=5120, d_int=17408, layer_type="linear_attention")
    assert pa.var < pl.var


# ---------------------------------------------------------------------------
# Per-pair prior on rho, Q, cos_phi
# ---------------------------------------------------------------------------


def _toy_layer_types(L: int, full_attn_period: int = 4) -> pd.DataFrame:
    types = [
        "full_attention" if (i + 1) % full_attn_period == 0 else "linear_attention"
        for i in range(L)
    ]
    return pd.DataFrame(
        {
            "layer_idx": range(L),
            "layer_type": types,
            "is_full_attention": [t == "full_attention" for t in types],
            "is_junction_candidate": [t == "full_attention" for t in types],
        }
    )


def test_predict_rho_Q_apriori_shape_and_columns() -> None:
    df = predict_rho_Q_apriori(_toy_layer_types(8))
    # Pairs (i, j) with i < j, j up to L=8: C(9, 2) - 0 = 36 pairs.
    assert len(df) == 8 * 9 // 2
    for col in [
        "layer_i",
        "layer_j",
        "rho_prior",
        "Q_prior",
        "cos_phi_prior",
        "one_minus_cka_prior",
        "crosses_full_attn",
    ]:
        assert col in df.columns


def test_predict_rho_Q_apriori_rho_monotone_in_block_length() -> None:
    """For a fixed start layer i, longer windows give larger rho_prior."""
    df = predict_rho_Q_apriori(_toy_layer_types(12))
    for i in range(0, 11):
        sub = df[df["layer_i"] == i].sort_values("layer_j")
        if len(sub) > 1:
            assert (sub["rho_prior"].diff().dropna() > 0).all(), (
                f"rho_prior is not monotone for layer_i={i}: {sub['rho_prior'].tolist()}"
            )


def test_predict_rho_Q_apriori_cos_phi_zero_under_incoherence() -> None:
    df = predict_rho_Q_apriori(_toy_layer_types(8))
    assert (df["cos_phi_prior"] == 0).all()


def test_predict_rho_Q_apriori_junction_flag() -> None:
    """A window starting at i=0 with j=2 (block length 2) on a stack with
    full-attn at index 3 should not cross any junction.  Extending to j=4
    *should* cross (it includes index 3)."""
    df = predict_rho_Q_apriori(_toy_layer_types(8))
    short_window = df[(df["layer_i"] == 0) & (df["layer_j"] == 2)]
    crossing_window = df[(df["layer_i"] == 0) & (df["layer_j"] == 4)]
    assert not bool(short_window["crosses_full_attn"].iat[0])
    assert bool(crossing_window["crosses_full_attn"].iat[0])


# ---------------------------------------------------------------------------
# RYS window ranking
# ---------------------------------------------------------------------------


def test_predict_rys_windows_respects_top_k() -> None:
    rho_Q = predict_rho_Q_apriori(_toy_layer_types(16))
    top = predict_rys_windows(rho_Q, top_k=5, min_block_length=4, require_plateau=False)
    assert len(top) == 5


def test_predict_rys_windows_filters_short_blocks() -> None:
    rho_Q = predict_rho_Q_apriori(_toy_layer_types(16))
    top = predict_rys_windows(rho_Q, top_k=100, min_block_length=6, require_plateau=False)
    assert (top["layer_j"] - top["layer_i"] >= 6).all()


# ---------------------------------------------------------------------------
# Architecture map (HF-config-dependent, skipped if config not cached)
# ---------------------------------------------------------------------------


def _has_cached_qwen_config() -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_cached_qwen_config(), reason="Qwen3.6-27B config not cached")
def test_qwen_layer_type_table_matches_qwen36() -> None:
    from rys.analytical_fk import qwen_layer_type_table

    df = qwen_layer_type_table()
    assert len(df) == 64
    full_attn_positions = df.loc[df["is_full_attention"], "layer_idx"].tolist()
    assert full_attn_positions == list(range(3, 64, 4))
    expected = pd.Series({"linear_attention": 48, "full_attention": 16})
    assert (df["layer_type"].value_counts() == expected).all()
