"""Tests for the generation-free GSM8K multiple-choice scorer.

Two layers of testing:

1. Fast logic tests (no model, no network): the distractor builder and the
   accuracy reducer.
2. Hand-made end-to-end checks on ``pythia-70m`` that validate the scorer's
   *goodness*: a deterministic copy probe any LM must pass, and a small set of
   easy one-step word problems where even the 70M base model beats chance. These
   skip cleanly when the model cannot be loaded (offline CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rys.gsm8k_mc import build_distractors, mc_accuracy

# Hand-made arithmetic problems with obvious gold answers, used to confirm the
# scorer detects a genuine model preference rather than noise.
HANDMADE_ARITHMETIC = [
    ("Sam has 4 apples and buys 4 more. How many apples does he have?", 8),
    ("A box has 10 balls and 6 are removed. How many balls are left?", 4),
    ("Tom reads 3 pages then 2 more pages. How many pages did he read?", 5),
    ("There are 6 cats and 6 dogs. How many animals in total?", 12),
    ("Jane had 7 sweets and ate 3. How many sweets are left?", 4),
    ("A shelf holds 5 books and 5 books are added. How many books now?", 10),
    ("Two teams have 3 players each. How many players in total?", 6),
    ("A jar has 8 coins and 1 is taken out. How many coins remain?", 7),
]

ARITHMETIC_FEWSHOT = (
    "Q: Anna has 2 pens and gets 3 more. How many pens does she have?\nA: 5\n\n"
    "Q: There are 9 birds and 4 fly away. How many birds remain?\nA: 5\n\n"
)


@pytest.mark.parametrize("gold", [0, 1, 2, 7, 18, 100, 1080])
def test_build_distractors_properties(gold: int) -> None:
    rng = np.random.default_rng(0)
    distractors = build_distractors(gold, k=4, rng=rng)
    assert len(distractors) == 4
    assert len(set(distractors)) == 4
    assert gold not in distractors
    assert all(d >= 0 for d in distractors)


def test_build_distractors_is_deterministic() -> None:
    a = build_distractors(42, 4, np.random.default_rng(3))
    b = build_distractors(42, 4, np.random.default_rng(3))
    assert a == b


def test_mc_accuracy_ranks_by_loglik() -> None:
    # Problem "a": gold has the highest sum but a lower per-token mean than the
    # short distractor, so acc (sum) and acc_norm (mean) disagree. Problem "b":
    # gold loses outright.
    scores = pd.DataFrame(
        [
            {"prompt_id": "a", "cand_idx": 0, "is_gold": True, "n_tokens": 3, "loglik_sum": -3.0, "loglik_mean": -1.0},
            {"prompt_id": "a", "cand_idx": 1, "is_gold": False, "n_tokens": 1, "loglik_sum": -2.0, "loglik_mean": -2.0},
            {"prompt_id": "b", "cand_idx": 0, "is_gold": False, "n_tokens": 1, "loglik_sum": -0.5, "loglik_mean": -0.5},
            {"prompt_id": "b", "cand_idx": 1, "is_gold": True, "n_tokens": 1, "loglik_sum": -3.0, "loglik_mean": -3.0},
        ]
    )
    result = mc_accuracy(scores)
    assert result["n"] == 2
    assert result["acc"] == pytest.approx(0.0)  # both gold answers lose on raw sum
    assert result["acc_norm"] == pytest.approx(0.5)  # "a" gold wins once length is normalised


def test_mc_accuracy_empty() -> None:
    empty = pd.DataFrame(columns=["prompt_id", "is_gold", "loglik_sum", "loglik_mean"])
    assert mc_accuracy(empty)["n"] == 0


@pytest.fixture(scope="module")
def pythia70m():
    import torch

    from rys.gsm8k_mc import load_pythia

    try:
        return load_pythia("EleutherAI/pythia-70m", device="cpu", dtype=torch.float32)
    except Exception as exc:  # offline / no cached weights
        pytest.skip(f"pythia-70m unavailable: {exc}")


def _mc_row(prompt_id: str, prompt: str, gold: int, distractors: list[int]) -> dict:
    candidates = [gold, *distractors]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "gold": gold,
        "candidates": candidates,
        "candidate_strs": [f" {c}" for c in candidates],
        "gold_idx": 0,
    }


def test_scorer_passes_copy_probe(pythia70m) -> None:
    """A repeated-number context: every candidate is plausible, but the gold is
    the one the context literally repeats. Any working LM + scorer ranks it #1."""
    from rys.gsm8k_mc import score_mc

    model, tok = pythia70m
    rows = []
    for i, num in enumerate([8, 3, 5, 12]):
        prompt = (
            f"Q: Repeat the number {num}.\nA: {num}\n\n"
            f"Q: Repeat the number {num}.\nA: {num}\n\n"
            f"Q: Repeat the number {num}.\nA:"
        )
        rows.append(_mc_row(f"copy_{i}", prompt, num, [num + 1, num + 2, num - 1]))
    acc = mc_accuracy(score_mc(model, tok, pd.DataFrame(rows), device="cpu"))
    assert acc["acc_norm"] == pytest.approx(1.0)


def test_scorer_beats_chance_on_easy_arithmetic(pythia70m) -> None:
    """On easy one-step problems the 70M base model should clearly exceed the
    1/(K+1) chance rate, confirming the scorer reflects real model preference."""
    from rys.gsm8k_mc import score_mc

    model, tok = pythia70m
    rng = np.random.default_rng(0)
    rows = []
    for i, (question, gold) in enumerate(HANDMADE_ARITHMETIC):
        distractors = build_distractors(gold, 3, rng)
        prompt = ARITHMETIC_FEWSHOT + f"Q: {question}\nA:"
        rows.append(_mc_row(f"ar_{i}", prompt, gold, distractors))
    acc = mc_accuracy(score_mc(model, tok, pd.DataFrame(rows), device="cpu"))
    assert acc["acc_norm"] > 0.25  # strictly above 4-way chance
