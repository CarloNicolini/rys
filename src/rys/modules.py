"""Graph analytics on the CKA connectome.

Three primitives:

- :func:`leiden_communities` — Leiden community detection on a CKA matrix
  treated as a weighted undirected graph (off-diagonal entries are edge
  weights, threshold-clipped to keep the graph sparse and meaningful).
- :func:`change_points` — PELT change-point detection on the lag-1
  sub-diagonal of the matrix, which is the per-layer transition strength;
  this localises encoder/reasoning/decoder boundaries.
- :func:`plateau_metric` — the off-plateau scalar
  ``1 - mean(M[i, j])`` over the rectangle ``window^2``. By the blog post's
  Eq. (7) this should be amplified by ~4x under RYS in the typical
  incoherent regime that empirical LLM residual streams occupy.
"""

from __future__ import annotations

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
import ruptures as rpt


def _matrix_to_igraph(M: pd.DataFrame, *, weight_threshold: float) -> ig.Graph:
    """Build an undirected weighted igraph from a CKA matrix.

    Edges with CKA below ``weight_threshold`` are dropped. Self-loops are
    excluded. We use the matrix index as the vertex ``name`` attribute so
    callers can map back to layer ids.
    """
    layers = list(M.index)
    L = len(layers)
    M_arr = M.to_numpy()
    edges, weights = [], []
    for i in range(L):
        for j in range(i + 1, L):
            w = float(M_arr[i, j])
            if w >= weight_threshold:
                edges.append((i, j))
                weights.append(w)
    g = ig.Graph(n=L, edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["name"] = [str(layer) for layer in layers]
    g.vs["layer"] = layers
    return g


def leiden_communities(
    M: pd.DataFrame,
    *,
    resolution: float = 1.0,
    weight_threshold: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Run Leiden modularity-maximisation on the CKA graph.

    Parameters
    ----------
    M
        Symmetric ``L x L`` CKA matrix as a DataFrame.
    resolution
        Resolution parameter of the Reichardt-Bornholdt modularity (CPM-like
        when high). Larger values produce more, smaller communities.
    weight_threshold
        Edge weight threshold; CKA below this is treated as no edge.
    seed
        RNG seed for igraph's Leiden implementation.

    Returns
    -------
    pd.DataFrame
        Columns ``[layer, community_id, modularity_Q]``. ``modularity_Q`` is
        the same value for every row — it's a property of the partition, not
        of the layer; we duplicate it for ease of joining downstream.
    """
    g = _matrix_to_igraph(M, weight_threshold=weight_threshold)
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    Q = partition.modularity
    rows = [
        (g.vs[i]["layer"], int(partition.membership[i]), float(Q))
        for i in range(g.vcount())
    ]
    return pd.DataFrame(rows, columns=["layer", "community_id", "modularity_Q"])


def change_points(
    M: pd.DataFrame,
    *,
    n_breaks: int = 2,
    model: str = "rbf",
) -> list[int]:
    """Detect transition points along the lag-1 sub-diagonal of M.

    The lag-1 sub-diagonal ``M[l, l+1]`` is the layer-to-next-layer CKA. A
    drop in this sequence flags a regime change in the residual stream. We
    run PELT (Killick et al. 2012) to localise ``n_breaks`` change points,
    which delineate the encode / reason / decode segments.
    """
    M_arr = M.to_numpy()
    diag = np.array([M_arr[k, k + 1] for k in range(M_arr.shape[0] - 1)])
    algo = rpt.Binseg(model=model).fit(diag.reshape(-1, 1))
    bkps = algo.predict(n_bkps=n_breaks)
    # `bkps` is 1-indexed and ends at len(diag); strip the trailing endpoint
    # so the returned breaks are interior layer indices.
    return [int(b) for b in bkps[:-1]]


def plateau_metric(M: pd.DataFrame, window: tuple[int, int]) -> float:
    """Return ``1 - mean(M[i, j])`` over the closed rectangle ``window^2``.

    By the post's Eq. (7) this value should be amplified by ~4x when RYS
    duplicates the same window in the typical incoherent regime. The metric
    is invariant to which window boundary convention the caller uses
    (closed or half-open) because both reduce to the same averaged sub-block
    in the limit of large windows.
    """
    start, end = window
    block = M.loc[start:end, start:end].to_numpy()
    return float(1.0 - block.mean())
