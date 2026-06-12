"""Synthetic "N-ary addition" data for controlled RYS experiments.

The task is to add ``n`` operands of ``digits`` digits each and emit their
sum. The model reads a serialised problem such as ``315 + 120 + 045 + 824 =``
and writes the sum into ``answer_width`` fixed answer slots appended after the
``=`` token. Following the standard addition-Transformer trick, the answer
digits are stored *reversed* (least-significant first), so carries propagate
left-to-right and the solver extrapolates to more operands out of distribution.

Example with ``digits=3``, ``answer_width=6`` and ``n=4``::

    operands [315, 120, 45, 824]  ->  answer 1304
    input tokens  3 1 5 + 1 2 0 + 0 4 5 + 8 2 4 = ANS ANS ANS ANS ANS ANS
    target digits 4 0 3 1 0 0     (reversed sum 1304, zero-padded to width 6)

Internally the model predicts the *digit class* ``0..9`` at each of the trailing
``answer_width`` positions (a 10-way classification). Keeping the input
vocabulary (digits plus ``+``/``=``/``ANS``) separate from the 10-class digit
head forces a distinct unembedding head, isolating the decoding phase exactly as
in the sorted-translation probe.

Addition is a deterministic proxy for an iterative carry-propagation algorithm:
each Transformer layer behaves like one refinement round, so repeating layers
with RYS buys extra rounds for harder out-of-distribution operand counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

# Token vocabulary: digits 0..9 plus the three structural tokens.
PLUS = 10
EQ = 11
ANS = 12
VOCAB_IN = 13
N_DIGIT_CLASSES = 10

DEFAULT_DIGITS = 3
DEFAULT_ANSWER_WIDTH = 6


@dataclass(frozen=True)
class AdditionExample:
    """One n-ary addition problem and its reversed digit-class target.

    ``input_tokens`` holds token ids in ``0..VOCAB_IN-1`` of length
    ``(digits + 1) * n_operands + answer_width`` (each operand is ``digits``
    tokens followed by ``+`` or ``=``, then ``answer_width`` ``ANS`` slots).
    ``target_digits`` holds the sum's digit classes in ``0..9``, reversed
    (least-significant first) and zero-padded to ``answer_width``.
    """

    operands: tuple[int, ...]
    answer: int
    input_tokens: tuple[int, ...]
    target_digits: tuple[int, ...]
    n_operands: int
    digits: int
    answer_width: int
    prompt_id: str


def _int_to_digits(value: int, width: int) -> list[int]:
    """Return ``width`` digits of ``value`` most-significant first, zero-padded."""
    return [int(d) for d in f"{value:0{width}d}"]


def _reversed_answer_digits(answer: int, answer_width: int) -> list[int]:
    """Return the ``answer`` digits reversed (LSB-first), zero-padded to width."""
    return _int_to_digits(answer, answer_width)[::-1]


def _build_input_tokens(operands: Sequence[int], digits: int, answer_width: int) -> list[int]:
    tokens: list[int] = []
    for idx, value in enumerate(operands):
        tokens.extend(_int_to_digits(int(value), digits))
        tokens.append(EQ if idx == len(operands) - 1 else PLUS)
    tokens.extend([ANS] * answer_width)
    return tokens


def make_addition_examples(
    n_examples: int,
    n_operands: int,
    *,
    seed: int = 0,
    digits: int = DEFAULT_DIGITS,
    answer_width: int = DEFAULT_ANSWER_WIDTH,
    prefix: str = "add",
) -> list[AdditionExample]:
    """Create deterministic n-ary addition problems.

    Each operand is sampled uniformly in ``[0, 10**digits)`` with replacement.
    Raises if any sum needs more than ``answer_width`` digits.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be positive.")
    if n_operands < 1:
        raise ValueError("n_operands must be positive.")
    if digits < 1:
        raise ValueError("digits must be positive.")
    if answer_width < 1:
        raise ValueError("answer_width must be positive.")

    max_sum = n_operands * (10**digits - 1)
    if max_sum >= 10**answer_width:
        raise ValueError(
            f"answer_width={answer_width} too small for n_operands={n_operands}, "
            f"digits={digits} (max sum {max_sum} needs {len(str(max_sum))} digits)."
        )

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 10**digits, size=(n_examples, n_operands), dtype=np.int64)
    examples: list[AdditionExample] = []
    for idx in range(n_examples):
        operands = tuple(int(x) for x in draws[idx])
        answer = int(sum(operands))
        examples.append(
            AdditionExample(
                operands=operands,
                answer=answer,
                input_tokens=tuple(_build_input_tokens(operands, digits, answer_width)),
                target_digits=tuple(_reversed_answer_digits(answer, answer_width)),
                n_operands=n_operands,
                digits=digits,
                answer_width=answer_width,
                prompt_id=f"{prefix}_{idx:06d}",
            )
        )
    return examples


def make_addition_splits(
    *,
    train_ns: Sequence[int] = (2, 4, 8, 16, 32),
    ood_ns: Sequence[int] = (48, 64),
    val_n: int = 32,
    n_train: int = 8192,
    n_val: int = 2048,
    n_test: int = 2048,
    digits: int = DEFAULT_DIGITS,
    answer_width: int = DEFAULT_ANSWER_WIDTH,
    seed: int = 0,
) -> dict[str, list[AdditionExample]]:
    """Return deterministic splits.

    ``train`` mixes the operand counts in ``train_ns`` (``n_train`` split evenly
    across them); ``val``/``test`` use the single ``val_n`` operand count; each
    out-of-distribution count gets its own split named ``ood_{n}``. Every split
    is internally single-``n`` so batches need no padding.
    """
    if not train_ns:
        raise ValueError("train_ns must contain at least one operand count.")

    per_n = max(1, n_train // len(train_ns))
    train: list[AdditionExample] = []
    for offset, n_operands in enumerate(train_ns):
        train.extend(
            make_addition_examples(
                per_n,
                n_operands,
                seed=seed + offset,
                digits=digits,
                answer_width=answer_width,
                prefix=f"train{n_operands}",
            )
        )

    splits: dict[str, list[AdditionExample]] = {
        "train": train,
        "val": make_addition_examples(
            n_val, val_n, seed=seed + 1000, digits=digits, answer_width=answer_width, prefix="val"
        ),
        "test": make_addition_examples(
            n_test, val_n, seed=seed + 2000, digits=digits, answer_width=answer_width, prefix="test"
        ),
    }
    for offset, n_operands in enumerate(ood_ns):
        splits[f"ood_{n_operands}"] = make_addition_examples(
            n_test,
            n_operands,
            seed=seed + 3000 + offset,
            digits=digits,
            answer_width=answer_width,
            prefix=f"ood{n_operands}",
        )
    return splits


class AdditionDataset(Dataset):
    """PyTorch dataset wrapping pre-generated n-ary addition examples."""

    def __init__(self, examples: Sequence[AdditionExample]) -> None:
        if not examples:
            raise ValueError("AdditionDataset requires at least one example.")
        n_operands = {ex.n_operands for ex in examples}
        if len(n_operands) != 1:
            raise ValueError("All examples in a dataset must share the same n_operands.")
        widths = {ex.answer_width for ex in examples}
        if len(widths) != 1:
            raise ValueError("All examples in a dataset must share the same answer_width.")
        self.examples = list(examples)
        self.n_operands = n_operands.pop()
        self.answer_width = widths.pop()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex.input_tokens, dtype=torch.long),
            "target_ids": torch.tensor(ex.target_digits, dtype=torch.long),
            "answer": torch.tensor(ex.answer, dtype=torch.long),
            "prompt_id": ex.prompt_id,
        }


class AdditionStream(IterableDataset):
    """On-the-fly stream of random n-ary addition problems (no memorisation).

    Each epoch yields ``samples_per_epoch`` freshly sampled problems for a fixed
    ``n_operands``. A single persistent RNG is advanced across epochs, so the
    model never sees the same training set twice and cannot overfit by rote;
    this keeps train and validation drawn from the same distribution. Intended
    for ``num_workers=0`` (no worker sharding).
    """

    def __init__(
        self,
        n_operands: int,
        samples_per_epoch: int,
        *,
        digits: int = DEFAULT_DIGITS,
        answer_width: int = DEFAULT_ANSWER_WIDTH,
        seed: int = 0,
        prefix: str = "stream",
    ) -> None:
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive.")
        if n_operands < 1:
            raise ValueError("n_operands must be positive.")
        if digits < 1:
            raise ValueError("digits must be positive.")
        if answer_width < 1:
            raise ValueError("answer_width must be positive.")
        max_sum = n_operands * (10**digits - 1)
        if max_sum >= 10**answer_width:
            raise ValueError(
                f"answer_width={answer_width} too small for n_operands={n_operands}, "
                f"digits={digits} (max sum {max_sum})."
            )
        self.n_operands = n_operands
        self.samples_per_epoch = samples_per_epoch
        self.digits = digits
        self.answer_width = answer_width
        self.prefix = prefix
        self._rng = np.random.default_rng(seed)
        self._counter = 0

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            draws = self._rng.integers(0, 10**self.digits, size=self.n_operands, dtype=np.int64)
            operands = tuple(int(x) for x in draws)
            answer = int(sum(operands))
            input_tokens = _build_input_tokens(operands, self.digits, self.answer_width)
            target_digits = _reversed_answer_digits(answer, self.answer_width)
            prompt_id = f"{self.prefix}_{self._counter:09d}"
            self._counter += 1
            yield {
                "input_ids": torch.tensor(input_tokens, dtype=torch.long),
                "target_ids": torch.tensor(target_digits, dtype=torch.long),
                "answer": torch.tensor(answer, dtype=torch.long),
                "prompt_id": prompt_id,
            }


def digits_to_int(digit_ids: torch.Tensor, *, reverse: bool = True) -> torch.Tensor:
    """Decode digit classes of shape ``(batch, width)`` into integers ``(batch,)``.

    When ``reverse`` is true the digits are least-significant first (the storage
    convention used by :func:`make_addition_examples`).
    """
    if digit_ids.dim() != 2:
        raise ValueError("digit_ids must have shape (batch, width).")
    width = digit_ids.shape[1]
    exponents = torch.arange(width, device=digit_ids.device)
    if not reverse:
        exponents = exponents.flip(0)
    place = (10**exponents).to(digit_ids.dtype)
    return (digit_ids * place).sum(dim=-1)


def verify_addition_tensor(
    pred_digit_ids: torch.Tensor,
    answer: torch.Tensor,
    *,
    reverse: bool = True,
) -> torch.Tensor:
    """Per-row check that decoded ``pred_digit_ids`` equals the integer ``answer``.

    ``pred_digit_ids`` are digit classes of shape ``(batch, width)`` and
    ``answer`` is an integer tensor of shape ``(batch,)``. Returns a boolean
    tensor of shape ``(batch,)``.
    """
    if answer.dim() != 1 or answer.shape[0] != pred_digit_ids.shape[0]:
        raise ValueError("answer must have shape (batch,) matching pred_digit_ids.")
    return digits_to_int(pred_digit_ids, reverse=reverse) == answer
