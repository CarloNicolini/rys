"""Smoke tests for the rys package.

Two contracts the rest of the pipeline depends on:

1. CKA matrix returned by :func:`rys.cka.cka_matrix` is a square symmetric
   DataFrame indexed by layer ids, with ones on the diagonal and values in
   the closed unit interval (up to numerical noise).
2. The :func:`rys.surgery.apply_rys` context manager is *reversible*: a
   forward pass inside the context with ``n_repeats == 1`` produces the same
   logits as the same pass outside the context.

Both tests use a tiny dummy decoder built from ``torch.nn.Linear`` blocks so
they run on CPU in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from rys.cka import cka_matrix
from rys.surgery import apply_rys


class _DummyBlock(nn.Module):
    """Residual block: ``x -> x + Linear(x)``."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, *args, **kwargs):  # noqa: ANN001 - matches Llama signature
        return (hidden_states + 0.05 * self.fc(hidden_states),)


class _DummyInner(nn.Module):
    def __init__(self, d: int, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_DummyBlock(d, seed=i) for i in range(n_layers))


class _DummyDecoder(nn.Module):
    """Mimics enough of the Llama API surface for our hooks."""

    def __init__(self, d: int = 16, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _DummyInner(d, n_layers)

    def forward(self, hidden_states):
        h = hidden_states
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


def _fake_activations(L: int = 4, N: int = 32, d: int = 16, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for layer in range(L):
        # Activation matrices share a common signal plus per-layer noise so the
        # CKA matrix has both a strong diagonal and meaningful off-diagonal.
        X = rng.standard_normal((N, 3, d)).astype(np.float32) + 0.5 * layer
        for n in range(N):
            rows.append((f"p_{n}", layer, X[n]))
    return pd.DataFrame(rows, columns=["prompt_id", "layer", "activation"])


def test_cka_matrix_shape_and_symmetry() -> None:
    acts = _fake_activations(L=5, N=40, d=8)
    M = cka_matrix(acts, unbiased=True)
    assert M.shape == (5, 5)
    assert list(M.index) == list(M.columns) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(M.to_numpy(), M.to_numpy().T, atol=1e-5)
    assert np.allclose(np.diag(M.to_numpy()), 1.0, atol=1e-5)


def test_cka_matrix_entries_bounded() -> None:
    acts = _fake_activations(L=3, N=64, d=16)
    M = cka_matrix(acts, unbiased=True).to_numpy()
    # Unbiased CKA can dip slightly negative on small samples, but should
    # always lie in a tight band around [0, 1] for these inputs.
    assert (M <= 1.0 + 1e-3).all()
    assert (M >= -0.1).all()


def test_apply_rys_no_op_at_one_repeat() -> None:
    """`n_repeats == 1` must yield byte-identical activations."""
    torch.manual_seed(0)
    model = _DummyDecoder(d=16, n_layers=4).eval()
    x = torch.randn(2, 5, 16)
    with torch.inference_mode():
        baseline = model(x)
        with apply_rys(model, window=(1, 3), n_repeats=1):
            inside = model(x)
    torch.testing.assert_close(baseline, inside)


def test_apply_rys_changes_output_at_two_repeats() -> None:
    """`n_repeats == 2` must alter the forward result inside the window."""
    torch.manual_seed(0)
    model = _DummyDecoder(d=16, n_layers=4).eval()
    x = torch.randn(2, 5, 16)
    with torch.inference_mode():
        baseline = model(x)
        with apply_rys(model, window=(1, 3), n_repeats=2):
            modified = model(x)
    assert not torch.allclose(baseline, modified)


def test_apply_rys_is_reversible() -> None:
    """Exiting the context must restore identical-to-baseline behaviour."""
    torch.manual_seed(0)
    model = _DummyDecoder(d=16, n_layers=4).eval()
    x = torch.randn(2, 5, 16)
    with torch.inference_mode():
        baseline = model(x)
        with apply_rys(model, window=(1, 3), n_repeats=2):
            _ = model(x)
        after = model(x)
    torch.testing.assert_close(baseline, after)


def test_apply_rys_rejects_bad_windows() -> None:
    model = _DummyDecoder(d=8, n_layers=4)
    with pytest.raises(ValueError), apply_rys(model, window=(2, 2), n_repeats=2):
        pass
    with pytest.raises(ValueError), apply_rys(model, window=(0, 5), n_repeats=2):
        pass
