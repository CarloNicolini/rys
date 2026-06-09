"""Synthetic "Sorted Translation" data for controlled RYS experiments.

The task is an encoder-only, parallel per-position sort. The input is an
unsorted sequence of tokens in ``0..vocab-1``; the target output is the sorted
sequence mapped into a disjoint output vocabulary, i.e. ``sorted(input) + vocab``.

Example with ``vocab=128``::

    input  [45, 12, 89, 3]  ->  output tokens [131, 140, 173, 217]

Internally the model predicts the *class index* ``sorted(input)`` in
``0..vocab-1`` (a ``vocab``-way classification at each position); the output
token is that class plus ``vocab``. Keeping input and output vocabularies
disjoint is what forces the unembedding head ``W_U`` to stay distinct from the
input embedding ``W_E`` (no weight tying), isolating the decoding phase.

Sorting is a deterministic proxy for an iterative compare/swap algorithm: each
Transformer layer behaves like one refinement round, so repeating layers with
RYS buys extra rounds for longer out-of-distribution sequences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_VOCAB = 128


@dataclass(frozen=True)
class SortedExample:
    """One unsorted sequence and its sorted class-index target.

    ``inputs`` and ``target`` hold class indices in ``0..vocab-1``. The output
    *token* for position ``i`` is ``target[i] + vocab``.
    """

    inputs: tuple[int, ...]
    target: tuple[int, ...]
    seq_len: int
    vocab: int
    prompt_id: str


def make_sorted_examples(
    n_examples: int,
    seq_len: int,
    *,
    seed: int = 0,
    vocab: int = DEFAULT_VOCAB,
    prefix: str = "sorted",
) -> list[SortedExample]:
    """Create deterministic unsorted/sorted pairs.

    Tokens are sampled in ``[0, vocab)`` with replacement (duplicates allowed),
    so the model must learn a genuine sort rather than a permutation lookup.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be positive.")
    if seq_len < 1:
        raise ValueError("seq_len must be positive.")
    if vocab < 1:
        raise ValueError("vocab must be positive.")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, vocab, size=(n_examples, seq_len), dtype=np.int64)
    examples: list[SortedExample] = []
    for idx in range(n_examples):
        inputs = draws[idx]
        target = np.sort(inputs)
        examples.append(
            SortedExample(
                inputs=tuple(int(x) for x in inputs),
                target=tuple(int(x) for x in target),
                seq_len=seq_len,
                vocab=vocab,
                prompt_id=f"{prefix}_{idx:06d}",
            )
        )
    return examples


def make_sorted_splits(
    *,
    train_len: int = 16,
    ood_lens: Sequence[int] = (24, 32, 48),
    n_train: int = 8192,
    n_val: int = 2048,
    n_test: int = 2048,
    vocab: int = DEFAULT_VOCAB,
    seed: int = 0,
) -> dict[str, list[SortedExample]]:
    """Return deterministic splits.

    ``train``/``val``/``test`` share the in-distribution ``train_len``; each
    out-of-distribution length gets its own split named ``ood_{length}``. Every
    split is a single fixed length, so batches need no padding or attention
    mask.
    """
    splits: dict[str, list[SortedExample]] = {
        "train": make_sorted_examples(n_train, train_len, seed=seed, vocab=vocab, prefix="train"),
        "val": make_sorted_examples(n_val, train_len, seed=seed + 1, vocab=vocab, prefix="val"),
        "test": make_sorted_examples(n_test, train_len, seed=seed + 2, vocab=vocab, prefix="test"),
    }
    for offset, length in enumerate(ood_lens):
        splits[f"ood_{length}"] = make_sorted_examples(
            n_test,
            length,
            seed=seed + 100 + offset,
            vocab=vocab,
            prefix=f"ood{length}",
        )
    return splits


class SortedTranslationDataset(Dataset):
    """PyTorch dataset wrapping pre-generated sorted-translation examples."""

    def __init__(self, examples: Sequence[SortedExample]) -> None:
        if not examples:
            raise ValueError("SortedTranslationDataset requires at least one example.")
        seq_lens = {ex.seq_len for ex in examples}
        if len(seq_lens) != 1:
            raise ValueError("All examples in a dataset must share the same seq_len.")
        vocabs = {ex.vocab for ex in examples}
        if len(vocabs) != 1:
            raise ValueError("All examples in a dataset must share the same vocab.")
        self.examples = list(examples)
        self.seq_len = seq_lens.pop()
        self.vocab = vocabs.pop()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex.inputs, dtype=torch.long),
            "target_ids": torch.tensor(ex.target, dtype=torch.long),
            "prompt_id": ex.prompt_id,
        }


def verify_sorted_tensor(pred_ids: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-row check that ``pred_ids`` equals ``sort(input_ids)``.

    ``pred_ids`` and ``input_ids`` are class indices of shape ``(batch, seq)``.
    Returns a boolean tensor of shape ``(batch,)``. This accepts *any* correct
    full sort, including when the input contains duplicates.
    """
    if pred_ids.shape != input_ids.shape:
        raise ValueError("pred_ids and input_ids must share shape (batch, seq).")
    expected, _ = torch.sort(input_ids, dim=-1)
    return (pred_ids == expected).all(dim=-1)


def sortedness(pred_ids: torch.Tensor) -> torch.Tensor:
    """Per-row fraction of adjacent pairs in non-decreasing order.

    A fully sorted prediction scores ``1.0``; this quantifies *partial* sorting
    so RYS improvements on longer sequences are visible even before exact match.
    Sequences of length 1 score ``1.0`` by convention.
    """
    if pred_ids.dim() != 2:
        raise ValueError("pred_ids must have shape (batch, seq).")
    if pred_ids.shape[1] < 2:
        return torch.ones(pred_ids.shape[0], device=pred_ids.device)
    non_decreasing = (pred_ids[:, 1:] >= pred_ids[:, :-1]).float()
    return non_decreasing.mean(dim=-1)
