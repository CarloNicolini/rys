"""Tests for synthetic 3-SAT generation and encoding."""

from __future__ import annotations

import numpy as np
import torch

from rys.sat_data import (
    CLS,
    PAD,
    FACTOR_QUERY,
    SatAssignmentDataset,
    SatDataset,
    assignment_satisfies_formula,
    encode_clause_tensor,
    encode_formula_assignment_factorized,
    encode_formula,
    encode_formula_factorized,
    factorized_assignment_sequence_length,
    factorized_sequence_length,
    first_satisfying_assignment,
    formula_is_satisfied,
    is_satisfiable,
    make_sat_assignment_examples,
    make_sat_examples,
    sequence_length,
    verify_assignment_tensor,
    vocab_size,
)


def test_formula_satisfaction_and_unsat_bruteforce() -> None:
    assignment = np.array([True, False, True])
    assert formula_is_satisfied(((1, -2, 3),), assignment)
    assert assignment_satisfies_formula(((1, -2, 3),), [1, 0, 1])
    assert not assignment_satisfies_formula(((1, 2, 3),), [0, 0, 0])

    all_sign_clauses = tuple(
        (s1 * 1, s2 * 2, s3 * 3)
        for s1 in (-1, 1)
        for s2 in (-1, 1)
        for s3 in (-1, 1)
    )
    assert not is_satisfiable(all_sign_clauses, n_vars=3)
    assert first_satisfying_assignment(((1, -2, 3),), n_vars=3) == (0, 0, 0)


def test_assignment_verifier_accepts_any_valid_solution() -> None:
    formula = ((1, 2, 3), (-1, -2, -3))
    encoded = encode_clause_tensor(formula, n_vars=3, max_clauses=3)
    assignments = torch.tensor(
        [
            [1, 0, 0],  # valid: first clause by x1, second by not x2
            [0, 1, 1],  # also valid: different satisfying assignment
            [1, 1, 1],  # invalid: second clause false
            [0, 0, 0],  # invalid: first clause false
        ],
        dtype=torch.long,
    )
    valid = verify_assignment_tensor(
        assignments,
        encoded["clause_variable_ids"].unsqueeze(0).expand(assignments.shape[0], -1, -1),
        encoded["clause_sign_ids"].unsqueeze(0).expand(assignments.shape[0], -1, -1),
        encoded["clause_mask"].unsqueeze(0).expand(assignments.shape[0], -1),
    )
    assert valid.tolist() == [True, True, False, False]


def test_encode_formula_shape_and_tokens() -> None:
    formula = ((1, -2, 3), (-1, 2, -3))
    tokens, mask = encode_formula(formula, n_vars=3, max_clauses=4)
    assert tokens.shape == mask.shape == (sequence_length(4),)
    assert int(tokens[0]) == CLS
    assert int(mask.sum()) == 1 + 2 * 7
    assert int(tokens[-1]) == PAD
    assert vocab_size(3) > int(tokens.max())


def test_encode_formula_factorized_fields() -> None:
    formula = ((1, -2, 3), (-1, 2, -3))
    encoded = encode_formula_factorized(formula, n_vars=3, max_clauses=4)
    assert encoded["variable_ids"].shape == (factorized_sequence_length(4),)
    assert encoded["factor_attention_mask"].sum() == 1 + 2 * 3
    assert int(encoded["variable_ids"][1]) == 1
    assert int(encoded["sign_ids"][1]) == 1
    assert int(encoded["sign_ids"][2]) == 2
    assert int(encoded["clause_ids"][1]) == 1
    assert int(encoded["slot_ids"][3]) == 3


def test_encode_assignment_query_slots() -> None:
    formula = ((1, -2, 3), (-1, 2, -3))
    encoded = encode_formula_assignment_factorized(
        formula,
        n_vars=3,
        max_vars=5,
        max_clauses=4,
    )
    formula_len = factorized_sequence_length(4)
    assert encoded["variable_ids"].shape == (factorized_assignment_sequence_length(5, 4),)
    assert int(encoded["token_type_ids"][formula_len]) == FACTOR_QUERY
    assert int(encoded["variable_ids"][formula_len + 2]) == 3
    assert bool(encoded["factor_attention_mask"][formula_len + 2])
    assert not bool(encoded["factor_attention_mask"][formula_len + 3])


def test_sat_dataset_is_deterministic() -> None:
    examples_a = make_sat_examples(
        4,
        n_vars=4,
        n_clauses=8,
        seed=123,
        balanced=False,
    )
    examples_b = make_sat_examples(
        4,
        n_vars=4,
        n_clauses=8,
        seed=123,
        balanced=False,
    )
    assert examples_a == examples_b

    dataset = SatDataset(examples_a, max_vars=5, max_clauses=10)
    item = dataset[0]
    assert item["input_ids"].shape == (sequence_length(10),)
    assert item["variable_ids"].shape == (factorized_sequence_length(10),)
    assert item["attention_mask"].dtype == torch.bool
    assert item["factor_attention_mask"].dtype == torch.bool


def test_sat_assignment_dataset_labels_and_shapes() -> None:
    examples = make_sat_assignment_examples(
        4,
        n_vars=4,
        n_clauses=12,
        seed=321,
    )
    assert examples == make_sat_assignment_examples(
        4,
        n_vars=4,
        n_clauses=12,
        seed=321,
    )
    for ex in examples:
        assert formula_is_satisfied(ex.formula, np.array(ex.assignment, dtype=bool))

    dataset = SatAssignmentDataset(examples, max_vars=6, max_clauses=12)
    item = dataset[0]
    assert item["variable_ids"].shape == (factorized_assignment_sequence_length(6, 12),)
    assert item["assignment_labels"].shape == (6,)
    assert item["assignment_mask"].sum() == 4
    assert int(item["assignment_labels"][4]) == -100
    assert item["clause_variable_ids"].shape == (12, 3)
    assert item["clause_sign_ids"].shape == (12, 3)
    assert item["clause_mask"].sum() == 12
