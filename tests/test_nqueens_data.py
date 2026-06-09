"""Tests for synthetic N-Queens generation, encoding, verifier, and soft loss."""

from __future__ import annotations

import numpy as np
import torch

from rys.nqueens_data import (
    QueensDataset,
    board_is_valid,
    complete_with_givens,
    find_solution,
    make_queens_examples,
    make_queens_splits,
    soft_nqueens_loss,
    verify_boards_tensor,
)


def test_board_validity_known_cases() -> None:
    # Canonical 8-queens solution.
    assert board_is_valid((0, 4, 7, 5, 2, 6, 1, 3), 8)
    # Column clash.
    assert not board_is_valid((0, 4, 7, 5, 2, 6, 1, 0), 8)
    # Diagonal clash (rows 0,1 columns 0,1).
    assert not board_is_valid((0, 1, 3, 5, 7, 2, 4, 6), 8)


def test_generator_yields_valid_solutions_and_respects_givens() -> None:
    examples = make_queens_examples(16, n=8, n_givens=3, seed=7)
    for ex in examples:
        assert board_is_valid(ex.solution, ex.n)
        for r, g in enumerate(ex.givens):
            if g >= 0:
                assert ex.solution[r] == g
        assert sum(g >= 0 for g in ex.givens) == 3


def test_completion_consistent_with_givens() -> None:
    rng = np.random.default_rng(0)
    board = find_solution(8, rng)
    givens = {2: board[2], 5: board[5]}
    completed = complete_with_givens(8, givens, rng)
    assert completed is not None
    assert board_is_valid(completed, 8)
    assert completed[2] == board[2] and completed[5] == board[5]


def test_splits_are_deterministic() -> None:
    a = make_queens_splits(n_train=8, n_val=4, n_test=4, n=8, n_givens=2, seed=123)
    b = make_queens_splits(n_train=8, n_val=4, n_test=4, n=8, n_givens=2, seed=123)
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]
    assert a["test"] == b["test"]


def test_dataset_shapes_and_masks() -> None:
    examples = make_queens_examples(4, n=8, n_givens=2, seed=1)
    dataset = QueensDataset(examples, max_n=10)
    item = dataset[0]
    assert item["given_col"].shape == (10,)
    assert item["row_mask"].sum() == 8
    assert item["col_mask"].sum() == 8
    # Padded rows are masked and labelled with the ignore index.
    assert not bool(item["row_mask"][8])
    assert int(item["labels"][8]) == -100
    # Given rows carry a real column and the is_given flag.
    given_rows = [r for r in range(8) if bool(item["is_given"][r])]
    assert len(given_rows) == 2
    for r in given_rows:
        assert 0 <= int(item["given_col"][r]) < 8


def test_verifier_accepts_valid_and_rejects_violations() -> None:
    row_mask = torch.ones(1, 8, dtype=torch.bool)
    given_col = torch.full((1, 8), 8, dtype=torch.long)
    is_given = torch.zeros(1, 8, dtype=torch.bool)

    good = torch.tensor([[0, 4, 7, 5, 2, 6, 1, 3]])
    col_clash = torch.tensor([[0, 4, 7, 5, 2, 6, 1, 0]])
    diag_clash = torch.tensor([[0, 1, 3, 5, 7, 2, 4, 6]])

    assert bool(verify_boards_tensor(good, row_mask, given_col, is_given)[0])
    assert not bool(verify_boards_tensor(col_clash, row_mask, given_col, is_given)[0])
    assert not bool(verify_boards_tensor(diag_clash, row_mask, given_col, is_given)[0])


def test_verifier_enforces_givens() -> None:
    row_mask = torch.ones(1, 8, dtype=torch.bool)
    given_col = torch.full((1, 8), 8, dtype=torch.long)
    is_given = torch.zeros(1, 8, dtype=torch.bool)
    given_col[0, 0] = 1  # require row 0 -> column 1
    is_given[0, 0] = True
    board = torch.tensor([[0, 4, 7, 5, 2, 6, 1, 3]])  # valid but row 0 = 0, not 1
    assert not bool(verify_boards_tensor(board, row_mask, given_col, is_given)[0])


def test_verifier_accepts_non_canonical_valid_completion() -> None:
    # Two different valid 8-queens boards sharing a given must both pass.
    row_mask = torch.ones(2, 8, dtype=torch.bool)
    given_col = torch.full((2, 8), 8, dtype=torch.long)
    is_given = torch.zeros(2, 8, dtype=torch.bool)
    boards = torch.tensor([[0, 4, 7, 5, 2, 6, 1, 3], [0, 5, 7, 2, 6, 3, 1, 4]])
    valid = verify_boards_tensor(boards, row_mask, given_col, is_given)
    assert bool(valid.all())


def test_soft_loss_low_for_confident_valid_board() -> None:
    board = [0, 4, 7, 5, 2, 6, 1, 3]
    logits = torch.full((1, 8, 8), -6.0)
    for r, c in enumerate(board):
        logits[0, r, c] = 6.0
    row_mask = torch.ones(1, 8, dtype=torch.bool)
    col_mask = torch.ones(1, 8, dtype=torch.bool)
    loss = soft_nqueens_loss(logits, row_mask, col_mask, given_ce_weight=0.0)
    assert float(loss) < 0.05


def test_soft_loss_high_for_conflicting_board() -> None:
    logits = torch.full((1, 8, 8), -6.0)
    logits[0, :, 0] = 6.0  # every row insists on column 0 -> all-pairs conflict
    row_mask = torch.ones(1, 8, dtype=torch.bool)
    col_mask = torch.ones(1, 8, dtype=torch.bool)
    loss = soft_nqueens_loss(logits, row_mask, col_mask, given_ce_weight=0.0)
    assert float(loss) > 1.0
