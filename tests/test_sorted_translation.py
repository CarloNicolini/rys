"""Tests for the Sorted Translation data and RoPE Transformer."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from rys.sorted_translation_data import (
    SortedTranslationDataset,
    make_sorted_examples,
    make_sorted_splits,
    sortedness,
    verify_sorted_tensor,
)
from rys.sorted_translation_transformer import (
    SortedTranslationConfig,
    SortedTranslationTransformer,
)
from rys.surgery import apply_rys


def _loader(seq_len: int = 8, vocab: int = 16, n: int = 8, batch: int = 4) -> DataLoader:
    examples = make_sorted_examples(n, seq_len, seed=0, vocab=vocab)
    dataset = SortedTranslationDataset(examples)
    return DataLoader(dataset, batch_size=batch, shuffle=False)


def _model(vocab: int = 16) -> SortedTranslationTransformer:
    torch.manual_seed(0)
    config = SortedTranslationConfig(
        vocab_in=vocab,
        d_model=32,
        n_layers=3,
        n_heads=4,
        d_mlp=64,
    )
    return SortedTranslationTransformer(config).eval()


def test_targets_are_true_sort_and_deterministic() -> None:
    a = make_sorted_examples(4, 6, seed=123, vocab=32)
    b = make_sorted_examples(4, 6, seed=123, vocab=32)
    assert a == b
    for ex in a:
        assert list(ex.target) == sorted(ex.inputs)
        assert len(ex.inputs) == len(ex.target) == 6


def test_splits_have_expected_lengths() -> None:
    splits = make_sorted_splits(
        train_len=8,
        ood_lens=(12, 16),
        n_train=10,
        n_val=5,
        n_test=5,
        vocab=32,
        seed=0,
    )
    assert {ex.seq_len for ex in splits["train"]} == {8}
    assert {ex.seq_len for ex in splits["ood_12"]} == {12}
    assert {ex.seq_len for ex in splits["ood_16"]} == {16}


def test_verifier_and_sortedness() -> None:
    inputs = torch.tensor([[3, 1, 2], [5, 5, 1]])
    correct = torch.tensor([[1, 2, 3], [1, 5, 5]])
    wrong = torch.tensor([[1, 3, 2], [5, 1, 5]])
    assert verify_sorted_tensor(correct, inputs).tolist() == [True, True]
    assert verify_sorted_tensor(wrong, inputs).tolist() == [False, False]
    torch.testing.assert_close(sortedness(correct), torch.tensor([1.0, 1.0]))
    # wrong row 0: pairs (1<=3 True, 3<=2 False) -> 0.5
    assert abs(float(sortedness(wrong)[0]) - 0.5) < 1e-6


def test_dataset_item_shapes() -> None:
    batch = next(iter(_loader(seq_len=8, vocab=16)))
    assert batch["input_ids"].shape == (4, 8)
    assert batch["target_ids"].shape == (4, 8)
    assert batch["input_ids"].dtype == torch.long


def test_forward_shapes_and_loss() -> None:
    model = _model(vocab=16)
    batch = next(iter(_loader(seq_len=8, vocab=16)))
    with torch.inference_mode():
        out = model(batch["input_ids"], target_ids=batch["target_ids"], return_round_logits=True)
    assert out["logits"].shape == (4, 8, 16)
    assert out["loss"] is not None
    assert len(out["round_logits"]) == 3
    assert out["round_logits"][0].shape == (4, 8, 16)


def test_rope_length_extrapolation() -> None:
    model = _model(vocab=16)
    # Trained-length-agnostic: a longer sequence than any "train" length must run.
    long_batch = next(iter(_loader(seq_len=20, vocab=16)))
    with torch.inference_mode():
        out = model(long_batch["input_ids"])
    assert out["logits"].shape == (4, 20, 16)


def test_apply_rys_reversible_and_active() -> None:
    model = _model(vocab=16)
    batch = next(iter(_loader(seq_len=8, vocab=16)))
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
    model = _model(vocab=16)
    w_e = model.model.embed.weight  # (d_model, vocab)
    w_u = model.unembed.weight  # (vocab, d_model)
    assert w_e.shape == (32, 16)
    assert w_u.shape == (16, 32)
    # Distinct parameter objects (no storage sharing / tying).
    assert w_e.data_ptr() != w_u.data_ptr()
    ids = {id(p) for p in model.model.parameters()}
    assert id(model.unembed.weight) not in ids
