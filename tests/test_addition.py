"""Tests for the N-ary addition data and RoPE Transformer."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from rys.addition_data import (
    AdditionDataset,
    digits_to_int,
    make_addition_examples,
    make_addition_splits,
    verify_addition_tensor,
)
from rys.addition_transformer import AdditionConfig, AdditionTransformer
from rys.surgery import apply_rys


def _loader(n_operands: int = 4, n: int = 8, batch: int = 4, answer_width: int = 6) -> DataLoader:
    examples = make_addition_examples(n, n_operands, seed=0, answer_width=answer_width)
    dataset = AdditionDataset(examples)
    return DataLoader(dataset, batch_size=batch, shuffle=False)


def _model() -> AdditionTransformer:
    torch.manual_seed(0)
    config = AdditionConfig(
        d_model=32,
        n_layers=3,
        n_heads=4,
        d_mlp=64,
    )
    return AdditionTransformer(config).eval()


def test_targets_are_true_sum_and_deterministic() -> None:
    a = make_addition_examples(4, 4, seed=123, answer_width=6)
    b = make_addition_examples(4, 4, seed=123, answer_width=6)
    assert a == b
    for ex in a:
        assert ex.answer == sum(ex.operands)
        assert len(ex.input_tokens) == 4 * 4 + 6  # (digits+1)*n + answer_width
        assert len(ex.target_digits) == 6
        # target digits are reversed (LSB-first); decoding recovers the answer.
        decoded = digits_to_int(torch.tensor([ex.target_digits]), reverse=True)
        assert int(decoded.item()) == ex.answer


def test_splits_have_expected_counts() -> None:
    splits = make_addition_splits(
        train_ns=(2, 4),
        ood_ns=(8, 16),
        val_n=4,
        n_train=10,
        n_val=5,
        n_test=5,
        seed=0,
    )
    assert {ex.n_operands for ex in splits["val"]} == {4}
    assert {ex.n_operands for ex in splits["test"]} == {4}
    assert {ex.n_operands for ex in splits["ood_8"]} == {8}
    assert {ex.n_operands for ex in splits["ood_16"]} == {16}
    # Train mixes the requested operand counts.
    assert {ex.n_operands for ex in splits["train"]} == {2, 4}


def test_verifier_and_digits_to_int() -> None:
    # answer 1304 reversed to width 6 is [4, 0, 3, 1, 0, 0].
    reversed_digits = torch.tensor([[4, 0, 3, 1, 0, 0], [9, 0, 0, 0, 0, 0]])
    answer = torch.tensor([1304, 9])
    assert digits_to_int(reversed_digits, reverse=True).tolist() == [1304, 9]
    assert verify_addition_tensor(reversed_digits, answer).tolist() == [True, True]
    wrong = torch.tensor([[3, 0, 3, 1, 0, 0], [9, 0, 0, 0, 0, 0]])
    assert verify_addition_tensor(wrong, answer).tolist() == [False, True]


def test_dataset_item_shapes() -> None:
    batch = next(iter(_loader(n_operands=4, answer_width=6)))
    assert batch["input_ids"].shape == (4, 4 * 4 + 6)
    assert batch["target_ids"].shape == (4, 6)
    assert batch["answer"].shape == (4,)
    assert batch["input_ids"].dtype == torch.long


def test_forward_shapes_and_loss() -> None:
    model = _model()
    batch = next(iter(_loader(n_operands=4, answer_width=6)))
    seq_len = 4 * 4 + 6
    with torch.inference_mode():
        out = model(batch["input_ids"], target_ids=batch["target_ids"], return_round_logits=True)
    assert out["logits"].shape == (4, seq_len, 10)
    assert out["loss"] is not None
    assert len(out["round_logits"]) == 3
    assert out["round_logits"][0].shape == (4, seq_len, 10)


def test_rope_operand_extrapolation() -> None:
    model = _model()
    # Operand-count-agnostic: more operands than any "train" count must run.
    long_batch = next(iter(_loader(n_operands=12, answer_width=6)))
    with torch.inference_mode():
        out = model(long_batch["input_ids"])
    assert out["logits"].shape == (4, 4 * 12 + 6, 10)


def test_apply_rys_reversible_and_active() -> None:
    model = _model()
    batch = next(iter(_loader(n_operands=4, answer_width=6)))
    ids = batch["input_ids"]
    with torch.inference_mode():
        baseline = model(ids)["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=1):
            no_op = model(ids)["logits"]
        with apply_rys(model, window=(0, 2), n_repeats=2):
            repeated = model(ids)["logits"]
        after = model(ids)["logits"]
    torch.testing.assert_close(baseline, no_op)
    torch.testing.assert_close(baseline, after)
    assert not torch.allclose(baseline, repeated)


def test_no_weight_tying() -> None:
    model = _model()
    w_e = model.model.embed.weight  # (d_model, vocab_in)
    w_u = model.unembed.weight  # (n_digit_classes, d_model)
    assert w_e.shape == (32, 13)
    assert w_u.shape == (10, 32)
    # Distinct parameter objects (no storage sharing / tying).
    assert w_e.data_ptr() != w_u.data_ptr()
    ids = {id(p) for p in model.model.parameters()}
    assert id(model.unembed.weight) not in ids
