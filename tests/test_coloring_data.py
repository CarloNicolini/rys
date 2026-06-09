"""Tests for synthetic graph k-colouring generation, verifier, and soft loss."""

from __future__ import annotations

import torch

from rys.coloring_data import (
    ColoringDataset,
    coloring_is_valid,
    make_coloring_examples,
    make_coloring_splits,
    soft_coloring_loss,
    verify_coloring_tensor,
)


def test_coloring_validity_known_cases() -> None:
    # Triangle needs three colours; a proper 3-colouring passes.
    edges = [(0, 1), (1, 2), (0, 2)]
    assert coloring_is_valid([0, 1, 2], 3, 3, edges)
    # Monochromatic edge (0, 1) fails.
    assert not coloring_is_valid([0, 0, 2], 3, 3, edges)
    # Out-of-range colour fails.
    assert not coloring_is_valid([0, 1, 3], 3, 3, edges)


def test_generator_is_k_colorable_and_respects_givens() -> None:
    examples = make_coloring_examples(16, n=12, k=3, n_edges=24, n_givens=3, seed=7)
    for ex in examples:
        # The planted colouring is proper by construction.
        assert coloring_is_valid(ex.solution, ex.n, ex.k, ex.edges)
        # Edges only connect differently-coloured vertices.
        for u, v in ex.edges:
            assert ex.solution[u] != ex.solution[v]
        # Givens equal the planted colour and there are exactly n_givens of them.
        assert sum(g >= 0 for g in ex.givens) == 3
        for v, g in enumerate(ex.givens):
            if g >= 0:
                assert ex.solution[v] == g


def test_splits_are_deterministic() -> None:
    a = make_coloring_splits(n_train=8, n_val=4, n_test=4, n=12, k=3, n_edges=20, n_givens=3, seed=123)
    b = make_coloring_splits(n_train=8, n_val=4, n_test=4, n=12, k=3, n_edges=20, n_givens=3, seed=123)
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]
    assert a["test"] == b["test"]


def test_dataset_shapes_and_masks() -> None:
    examples = make_coloring_examples(4, n=12, k=3, n_edges=20, n_givens=3, seed=1)
    dataset = ColoringDataset(examples, max_v=16, n_colors=3)
    item = dataset[0]
    assert item["given_color"].shape == (16,)
    assert item["adjacency"].shape == (16, 16)
    assert item["vertex_mask"].sum() == 12
    # Adjacency is symmetric, has a zero diagonal, and respects the vertex mask.
    adj = item["adjacency"]
    assert bool((adj == adj.t()).all())
    assert int(adj.diagonal().sum()) == 0
    assert not bool(adj[12:].any())
    # Padded vertices are masked, unknown-coloured, and label-ignored.
    assert not bool(item["vertex_mask"][12])
    assert int(item["labels"][12]) == -100
    assert int(item["given_color"][12]) == 3  # unknown index = n_colors
    # Degree matches the adjacency row sums.
    assert bool((item["degree"] == adj.sum(dim=1)).all())


def test_verifier_accepts_valid_and_rejects_monochromatic_edge() -> None:
    adjacency = torch.zeros(1, 3, 3, dtype=torch.bool)
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        adjacency[0, u, v] = adjacency[0, v, u] = True
    vertex_mask = torch.ones(1, 3, dtype=torch.bool)
    given_color = torch.full((1, 3), 3, dtype=torch.long)
    is_given = torch.zeros(1, 3, dtype=torch.bool)

    good = torch.tensor([[0, 1, 2]])
    bad = torch.tensor([[0, 0, 2]])
    assert bool(verify_coloring_tensor(good, vertex_mask, adjacency, given_color, is_given)[0])
    assert not bool(verify_coloring_tensor(bad, vertex_mask, adjacency, given_color, is_given)[0])


def test_verifier_enforces_givens() -> None:
    adjacency = torch.zeros(1, 3, 3, dtype=torch.bool)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
    vertex_mask = torch.ones(1, 3, dtype=torch.bool)
    given_color = torch.full((1, 3), 3, dtype=torch.long)
    is_given = torch.zeros(1, 3, dtype=torch.bool)
    given_color[0, 0] = 2  # require vertex 0 -> colour 2
    is_given[0, 0] = True
    coloring = torch.tensor([[0, 1, 2]])  # proper, but vertex 0 = 0, not 2
    assert not bool(verify_coloring_tensor(coloring, vertex_mask, adjacency, given_color, is_given)[0])


def test_verifier_accepts_permuted_colouring() -> None:
    # Colour-permutation symmetry: two proper colourings of the same graph pass.
    adjacency = torch.zeros(2, 3, 3, dtype=torch.bool)
    for b in range(2):
        for u, v in [(0, 1), (1, 2), (0, 2)]:
            adjacency[b, u, v] = adjacency[b, v, u] = True
    vertex_mask = torch.ones(2, 3, dtype=torch.bool)
    given_color = torch.full((2, 3), 3, dtype=torch.long)
    is_given = torch.zeros(2, 3, dtype=torch.bool)
    colorings = torch.tensor([[0, 1, 2], [2, 0, 1]])
    valid = verify_coloring_tensor(colorings, vertex_mask, adjacency, given_color, is_given)
    assert bool(valid.all())


def test_soft_loss_low_for_confident_proper_colouring() -> None:
    adjacency = torch.zeros(1, 3, 3, dtype=torch.bool)
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        adjacency[0, u, v] = adjacency[0, v, u] = True
    vertex_mask = torch.ones(1, 3, dtype=torch.bool)
    logits = torch.full((1, 3, 3), -6.0)
    for vtx, col in enumerate([0, 1, 2]):
        logits[0, vtx, col] = 6.0
    loss = soft_coloring_loss(logits, adjacency, vertex_mask, given_ce_weight=0.0)
    assert float(loss) < 0.05


def test_soft_loss_high_for_monochromatic_colouring() -> None:
    adjacency = torch.zeros(1, 3, 3, dtype=torch.bool)
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        adjacency[0, u, v] = adjacency[0, v, u] = True
    vertex_mask = torch.ones(1, 3, dtype=torch.bool)
    logits = torch.full((1, 3, 3), -6.0)
    logits[0, :, 0] = 6.0  # every vertex insists on colour 0 -> all edges conflict
    loss = soft_coloring_loss(logits, adjacency, vertex_mask, given_ce_weight=0.0)
    assert float(loss) > 1.0
