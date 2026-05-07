"""Linear CKA on per-layer activations.

We rely on `ckatorch` for the unbiased HSIC1 estimator (Nguyen, Raghu &
Kornblith 2020) instead of reimplementing the trace formula. This module
exposes pandas-friendly wrappers that:

1. Stack the long-format activation table from `rys.activations` into one
   ``(n_tokens, d)`` tensor per layer.
2. Compute the full ``L x L`` symmetric CKA matrix in vectorised form.
3. Optionally produce bootstrap 95% confidence intervals.

Notation matches the blog-post equations: the matrix entry M_ij is the linear
CKA between the centred activation matrices of layers i and j, equivalent to
the cosine of the angle between the vectorised centred Gram matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from ckatorch import cka_base


def _stack_per_layer(activations: pd.DataFrame) -> tuple[list[int], torch.Tensor]:
    """Pivot the long-format activations table into a (L, n_tokens, d) tensor.

    Parameters
    ----------
    activations
        DataFrame with columns ``[prompt_id, layer, activation]`` as produced by
        :func:`rys.activations.capture_residual_stream`. Each activation cell is
        an ``(n_tokens_for_prompt, d)`` matrix. Older one-vector-per-prompt
        inputs are also accepted and treated as a single-token sequence.

    Returns
    -------
    layers, tensor
        Sorted list of layer indices and a ``torch.float32`` tensor of shape
        ``(L, n_tokens, d)``.
    """
    if "strategy" in activations.columns and activations["strategy"].nunique() != 1:
        raise ValueError(
            "Mixed token strategies in the activations DataFrame. "
            "Filter to a single strategy before computing CKA."
        )
    layers = sorted(activations["layer"].unique().tolist())
    pivot = activations.pivot_table(
        index="prompt_id",
        columns="layer",
        values="activation",
        aggfunc="first",
    )
    pivot = pivot[layers]  # enforce column order
    # Each cell holds an activation matrix for one prompt. Concatenating over
    # prompts gives the blog-post object X_l in R^{n_tokens x d}.
    arrs = np.stack(
        [_concat_prompt_matrices(pivot[col].to_numpy()) for col in pivot.columns],
        axis=0,
    )
    return layers, torch.from_numpy(arrs.astype(np.float32))


def _concat_prompt_matrices(values: np.ndarray) -> np.ndarray:
    matrices = [np.atleast_2d(np.asarray(value)) for value in values]
    return np.concatenate(matrices, axis=0)


def cka_matrix(
    activations: pd.DataFrame,
    *,
    unbiased: bool = True,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Return the symmetric ``L x L`` linear CKA matrix as a DataFrame.

    Uses :func:`ckatorch.cka_base` with the unbiased HSIC1 estimator. The sample
    axis is the concatenated valid-token axis, not the prompt axis.

    The returned DataFrame is indexed and columned by integer layer id so the
    rest of the pipeline can use ``.loc`` slicing directly.
    """
    layers, X = _stack_per_layer(activations)
    L = len(layers)
    X = X.to(device)

    M = torch.empty((L, L), dtype=torch.float32, device=device)
    # cka_base is symmetric in (x, y); fill the upper triangle and mirror.
    for i in range(L):
        M[i, i] = 1.0
        for j in range(i + 1, L):
            val = cka_base(X[i], X[j], kernel="linear", unbiased=unbiased)
            M[i, j] = val
            M[j, i] = val
    return pd.DataFrame(M.cpu().numpy(), index=layers, columns=layers)


def cka_matrix_bootstrap(
    activations: pd.DataFrame,
    *,
    n_resamples: int = 200,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap CIs on every CKA entry by resampling token rows with replacement.

    Returns
    -------
    median, lower, upper
        Three DataFrames with the same shape and index/columns as
        :func:`cka_matrix`, holding the 50th, 2.5th and 97.5th percentile of
        the CKA value over ``n_resamples`` bootstrap draws.
    """
    layers, X = _stack_per_layer(activations)
    L, N, _ = X.shape
    X = X.to(device)
    rng = np.random.default_rng(seed)

    samples = np.empty((n_resamples, L, L), dtype=np.float32)
    for b in range(n_resamples):
        idx = torch.from_numpy(rng.choice(N, size=N, replace=True)).to(device)
        Xb = X.index_select(1, idx)
        for i in range(L):
            samples[b, i, i] = 1.0
            for j in range(i + 1, L):
                v = float(cka_base(Xb[i], Xb[j], kernel="linear", unbiased=True))
                samples[b, i, j] = v
                samples[b, j, i] = v

    median = np.median(samples, axis=0)
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)

    def fmt(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=layers, columns=layers)

    return fmt(median), fmt(lower), fmt(upper)
