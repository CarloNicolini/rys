"""Synthetic graph k-colouring data for controlled RYS experiments.

Parallel to :mod:`rys.sat_data` and :mod:`rys.nqueens_data`.  The task is
*conditional graph colouring completion*: each instance is a sparse graph that
is guaranteed ``k``-colourable (a colouring is planted, then only edges between
differently-coloured vertices are sampled), a few vertices are revealed with
their colour (``givens``), and the model must assign one colour per remaining
vertex so that no edge is monochromatic.

Why colouring complements SAT and N-Queens
------------------------------------------
- It is a second multi-solution CSP: every proper colouring stays proper under
  any permutation of the ``k`` colours, so the deterministic-averaging mode
  collapse that caps SAT validity is *even stronger* here.  This makes it the
  natural testbed for the width axis (stochastic transitions, ``valid@K``).
- The constraint graph is **sparse**, unlike the complete N-Queens graph, so a
  message must traverse several hops: useful reasoning depth is tied to the
  graph diameter rather than to a single round.
- The solver is **permutation-equivariant over vertices** and the colour
  vocabulary is fixed, so larger graphs out of distribution do *not* hit the
  untrained-vocabulary collapse that forced the N-Queens OOD split to reduce
  givens instead of growing the board.

Representation
--------------
One discrete variable per vertex taking a colour in ``{0, .., k-1}``.  A graph
is encoded by per-vertex fields plus a dense adjacency matrix:

- ``given_color``  colour of a revealed vertex, or ``k`` (the "unknown" index);
- ``is_given``     1 for revealed vertices, 0 otherwise;
- ``degree``       vertex degree (a permutation-invariant symmetry-breaking
                   feature, clamped to ``max_degree``);
- ``vertex_mask``  1 for real vertices (``v < n``);
- ``adjacency``    dense ``(max_v, max_v)`` boolean, symmetric, zero diagonal;
- ``labels``       the planted colour per vertex (diagnostic only).

The **primary metric** is ``valid_coloring_rate`` from
:func:`verify_coloring_tensor` (no monochromatic edge, givens respected), never
exact match to the planted colouring — many proper colourings exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Edge = tuple[int, int]


@dataclass(frozen=True)
class ColoringExample:
    """One graph k-colouring completion instance."""

    n: int  # number of vertices
    k: int  # number of colours
    edges: tuple[Edge, ...]
    solution: tuple[int, ...]  # planted proper colouring, colour per vertex
    givens: tuple[int, ...]  # length n; colour for revealed vertices, -1 otherwise
    prompt_id: str


def coloring_is_valid(
    colors: Sequence[int],
    n: int,
    k: int,
    edges: Sequence[Edge],
    givens: Sequence[int] | None = None,
) -> bool:
    """Exact validity check: in-range colours, no monochromatic edge, givens kept."""
    colors = list(colors)
    if len(colors) != n:
        return False
    if any(not (0 <= c < k) for c in colors):
        return False
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    if givens is not None:
        for v, g in enumerate(givens):
            if g >= 0 and colors[v] != g:
                return False
    return True


def _planted_coloring_graph(
    n: int,
    k: int,
    n_edges: int,
    rng: np.random.Generator,
) -> tuple[tuple[Edge, ...], tuple[int, ...]]:
    """Plant a random colouring, then sample edges between differing colours.

    This guarantees the planted colouring is proper, so the graph is
    ``k``-colourable by construction (the colouring analogue of planting a
    satisfying assignment in :mod:`rys.sat_data`).
    """
    if n < 2:
        raise ValueError("Graph colouring needs at least two vertices.")
    if k < 2:
        raise ValueError("Need at least two colours.")
    colors = rng.integers(0, k, size=n)
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if colors[u] != colors[v]]
    if not candidates:  # pragma: no cover - only if all vertices share a colour
        raise RuntimeError("No bichromatic candidate edges; increase n or k.")
    n_edges = min(n_edges, len(candidates))
    chosen = rng.choice(len(candidates), size=n_edges, replace=False)
    edges = tuple(sorted(candidates[i] for i in chosen))
    return edges, tuple(int(c) for c in colors)


def make_coloring_examples(
    n_examples: int,
    *,
    n: int = 12,
    k: int = 3,
    n_edges: int = 24,
    n_givens: int = 3,
    seed: int = 0,
    prefix: str = "coloring",
) -> list[ColoringExample]:
    """Create deterministic graph-colouring completion instances.

    Each instance plants a proper colouring, samples ``n_edges`` bichromatic
    edges, then reveals ``n_givens`` vertices with their planted colour.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be positive.")
    if not 0 <= n_givens < n:
        raise ValueError("n_givens must satisfy 0 <= n_givens < n.")
    rng = np.random.default_rng(seed)
    examples: list[ColoringExample] = []
    for idx in range(n_examples):
        edges, colors = _planted_coloring_graph(n, k, n_edges, rng)
        given_vertices = (
            sorted(rng.choice(n, size=n_givens, replace=False).tolist()) if n_givens else []
        )
        givens = tuple(colors[v] if v in set(given_vertices) else -1 for v in range(n))
        examples.append(
            ColoringExample(
                n=n,
                k=k,
                edges=edges,
                solution=colors,
                givens=givens,
                prompt_id=f"{prefix}_{idx:05d}",
            )
        )
    return examples


def make_coloring_splits(
    *,
    n_train: int = 4096,
    n_val: int = 1024,
    n_test: int = 1024,
    n: int = 12,
    k: int = 3,
    n_edges: int = 24,
    n_givens: int = 3,
    seed: int = 0,
) -> dict[str, list[ColoringExample]]:
    """Deterministic train/val/test splits (same graph size and given count)."""
    return {
        "train": make_coloring_examples(
            n_train, n=n, k=k, n_edges=n_edges, n_givens=n_givens, seed=seed, prefix="train"
        ),
        "val": make_coloring_examples(
            n_val, n=n, k=k, n_edges=n_edges, n_givens=n_givens, seed=seed + 1, prefix="val"
        ),
        "test": make_coloring_examples(
            n_test, n=n, k=k, n_edges=n_edges, n_givens=n_givens, seed=seed + 2, prefix="test"
        ),
    }


class ColoringDataset(Dataset):
    """PyTorch dataset over graph-colouring instances, padded to ``max_v``."""

    def __init__(
        self,
        examples: Sequence[ColoringExample],
        *,
        max_v: int | None = None,
        n_colors: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("ColoringDataset requires at least one example.")
        self.examples = list(examples)
        self.max_v = max_v or max(ex.n for ex in self.examples)
        self.n_colors = n_colors or max(ex.k for ex in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        m = self.max_v
        unknown = self.n_colors  # index into a (n_colors + 1) embedding table
        given_color = torch.full((m,), unknown, dtype=torch.long)
        is_given = torch.zeros(m, dtype=torch.bool)
        vertex_mask = torch.zeros(m, dtype=torch.bool)
        labels = torch.full((m,), -100, dtype=torch.long)
        adjacency = torch.zeros((m, m), dtype=torch.bool)
        for v in range(ex.n):
            vertex_mask[v] = True
            labels[v] = ex.solution[v]
            if ex.givens[v] >= 0:
                given_color[v] = ex.givens[v]
                is_given[v] = True
        for u, v in ex.edges:
            adjacency[u, v] = True
            adjacency[v, u] = True
        degree = adjacency.sum(dim=1).long()
        return {
            "given_color": given_color,
            "is_given": is_given,
            "degree": degree,
            "vertex_mask": vertex_mask,
            "adjacency": adjacency,
            "labels": labels,
            "n": torch.tensor(ex.n, dtype=torch.long),
            "prompt_id": ex.prompt_id,
        }


def verify_coloring_tensor(
    colors: torch.Tensor,
    vertex_mask: torch.Tensor,
    adjacency: torch.Tensor,
    given_color: torch.Tensor,
    is_given: torch.Tensor,
) -> torch.Tensor:
    """Boolean vector marking colourings that satisfy every constraint.

    ``colors`` is ``(batch, max_v)`` predicted colour per vertex (ints).
    Inactive vertices (``~vertex_mask``) are ignored.  Givens must be respected.
    """
    same = colors.unsqueeze(2) == colors.unsqueeze(1)  # (B, V, V)
    pair = adjacency & vertex_mask.unsqueeze(2) & vertex_mask.unsqueeze(1)
    conflict = (same & pair).reshape(colors.shape[0], -1).any(dim=1)
    givens_ok = ((colors == given_color) | ~is_given | ~vertex_mask).all(dim=1)
    return (~conflict) & givens_ok


def soft_coloring_loss(
    logits: torch.Tensor,
    adjacency: torch.Tensor,
    vertex_mask: torch.Tensor,
    *,
    given_color: torch.Tensor | None = None,
    is_given: torch.Tensor | None = None,
    given_ce_weight: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable relaxation of the colouring verifier.

    ``logits`` is ``(batch, max_v, n_colors)``.  For every active edge ``(u, v)``
    the monochromatic probability under independent per-vertex colour
    distributions is ``sum_c p_u[c] p_v[c]``; the loss is the mean negative
    log-probability that each edge is bichromatic, plus an optional
    cross-entropy anchoring the given vertices to their colour.  It is invariant
    to permuting the colours, so it rewards *any* proper colouring.
    """
    p = logits.softmax(dim=-1)  # (B, V, C)
    conflict = torch.einsum("bvc,bwc->bvw", p, p)  # (B, V, V) monochromatic prob
    pair = adjacency & vertex_mask.unsqueeze(2) & vertex_mask.unsqueeze(1)
    triu = torch.triu(torch.ones_like(pair[0]), diagonal=1).bool()
    edge_mask = (pair & triu).float()  # count each edge once
    per_edge = -torch.log((1.0 - conflict).clamp(min=eps)) * edge_mask
    denom = edge_mask.reshape(edge_mask.shape[0], -1).sum(dim=1).clamp(min=1.0)
    loss = (per_edge.reshape(per_edge.shape[0], -1).sum(dim=1) / denom).mean()
    if given_ce_weight > 0 and given_color is not None and is_given is not None:
        b, v, c = logits.shape
        target = torch.where(is_given & vertex_mask, given_color, torch.full_like(given_color, -100))
        ce = F.cross_entropy(logits.reshape(b * v, c), target.reshape(b * v), ignore_index=-100)
        if torch.isfinite(ce):
            loss = loss + given_ce_weight * ce
    return loss
