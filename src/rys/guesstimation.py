"""David Ng's guesstimation probe: generative, single-pass, partial-credit.

The RYS effect in the original blog post (Ng, 2026) was discovered with a
deliberately different downstream proxy from a multiple-choice loglikelihood
test. The model is asked a hard arithmetic question, must emit the answer
\\emph{directly} (no chain-of-thought), and is scored by how \\emph{close} the
generated number is to the truth via a partial-credit function. This rewards a
near-miss, so it gives a graded, higher-power signal than a binary correct/wrong
metric and measures what the model produces in one forward pass, which is exactly
what block repetition is meant to improve.

This module reproduces that method:

- :func:`calculate_score` is Ng's exact partial-credit function;
- :func:`make_guesstimation_questions` builds a difficulty-calibrated set so even
  small models have headroom (easy arithmetic up to David-style cube roots);
- :func:`probe_score` greedily generates the answer for every question and
  returns the mean partial-credit score.

Generation uses ``use_cache=False`` so the :func:`rys.surgery.apply_rys` replay
hook stays exact (re-running layers would otherwise corrupt the KV cache).
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


def calculate_score(actual, estimate) -> float:
    """Partial-credit closeness score in [0, 1] (verbatim from Ng, 2026)."""
    try:
        actual_str = str(int(actual))
        estimate_str = str(int(estimate))
    except (ValueError, OverflowError, TypeError):
        return 0.0
    max_length = max(len(actual_str), len(estimate_str))
    actual_padded = actual_str.ljust(max_length, "0")
    estimate_padded = estimate_str.ljust(max_length, "0")
    padding_size = max_length - min(len(actual_str), len(estimate_str))
    actual_int = int(actual_padded)
    estimate_int = int(estimate_padded)
    if max(actual_int, estimate_int) == 0:
        return 0.0
    relative_diff = abs(actual_int - estimate_int) / max(actual_int, estimate_int)
    correction_factor = 1 - (padding_size / max_length)
    score = (1 - relative_diff) * correction_factor
    return max(0.0, min(score, 1.0))


FEWSHOT = (
    "Answer each question with only the number, your single best guess.\n\n"
    "Q: What is 12 multiplied by 12?\nA: 144\n\n"
    "Q: What is the cube root of 27000?\nA: 30\n\n"
    "Q: What is 250 plus 175?\nA: 425\n\n"
)


def make_guesstimation_questions(seed: int = 0) -> list[dict]:
    """Difficulty-calibrated guesstimation set (easy arithmetic to hard roots).

    The mix spans three tiers so weak base models still earn graded partial
    credit on the easy tier while the hard tier matches Ng's cube-root probes.
    """
    rng = random.Random(seed)
    items: list[tuple[str, int]] = []

    # Tier 1: easy magnitudes.
    for _ in range(10):
        a, b = rng.randint(11, 99), rng.randint(2, 9)
        items.append((f"What is {a} multiplied by {b}?", a * b))
    for _ in range(6):
        a, b = rng.randint(20, 99), rng.randint(20, 99)
        items.append((f"What is {a} plus {b}?", a + b))

    # Tier 2: medium.
    for _ in range(8):
        a, b = rng.randint(12, 99), rng.randint(11, 99)
        items.append((f"What is {a} multiplied by {b}?", a * b))
    for _ in range(4):
        n = rng.randint(11, 40)
        items.append((f"What is {n} squared?", n * n))
    for _ in range(4):
        base, n = rng.randint(100, 9999), rng.randint(2, 20)
        items.append((f"What is {base} multiplied by {n}?", base * n))

    # Tier 3: hard, David-style intuitive guesstimation.
    for _ in range(8):
        n = rng.randint(10**9, 10**12)
        items.append((f"What is the cube root of {n}?", round(n ** (1 / 3))))
    for _ in range(4):
        a, b = rng.randint(1000, 99999), rng.randint(1000, 99999)
        items.append((f"What is {a} multiplied by {b}?", a * b))

    return [{"question": q, "answer": int(ans)} for q, ans in items]


def _clone_layer_shared(layer: torch.nn.Module, new_idx: int) -> torch.nn.Module:
    """A layer copy that shares weight tensors but has its own cache index.

    ``copy.copy`` shallow-copies the instance ``__dict__`` but leaves the
    ``_modules``/``_parameters``/``_buffers`` *dicts* shared with the original, so
    mutating the copy would corrupt the source. We give the copy fresh container
    dicts (same tensor values) and a distinct ``layer_idx`` so each occurrence in
    the unrolled path gets its own KV-cache slot, with no extra GPU memory.
    """
    new_layer = copy.copy(layer)
    new_layer._parameters = dict(layer._parameters)
    new_layer._buffers = dict(layer._buffers)
    new_layer._modules = dict(layer._modules)
    new_attn = copy.copy(layer.attention)
    new_attn._parameters = dict(layer.attention._parameters)
    new_attn._buffers = dict(layer.attention._buffers)
    new_attn._modules = dict(layer.attention._modules)
    new_attn.layer_idx = new_idx
    new_layer._modules["attention"] = new_attn
    if "layer_idx" in new_layer.__dict__:
        new_layer.layer_idx = new_idx
    return new_layer


@contextmanager
def rys_unrolled(model: torch.nn.Module, window: tuple[int, int] | None) -> Iterator[None]:
    """Temporarily rebuild ``model.model.layers`` as the unrolled RYS path.

    The half-open RYS window ``(i, j)`` is realised by the layer sequence
    ``[0..j-1, i..j-1, j..L-1]``. Unlike the forward-pre-hook in
    :func:`rys.surgery.apply_rys`, each replayed layer here is a distinct module
    (a shallow copy that *shares the weight tensors*, so no extra memory) with a
    corrected cache index, which lets generation run with the normal KV cache —
    roughly an order of magnitude faster than ``use_cache=False`` replay. The
    computation is identical to ``apply_rys`` with ``n_repeats=2``.

    ``window=None`` is a no-op (baseline model).
    """
    if window is None:
        yield
        return
    start, end = window
    base = model.model.layers
    L = len(base)
    if not (0 <= start < end <= L):
        raise ValueError(f"Bad window {window} for L={L}.")

    path = list(range(0, end)) + list(range(start, end)) + list(range(end, L))
    unrolled = [_clone_layer_shared(base[k], new_idx) for new_idx, k in enumerate(path)]

    orig_layers = model.model.layers
    orig_n = model.config.num_hidden_layers
    model.model.layers = torch.nn.ModuleList(unrolled)
    model.config.num_hidden_layers = len(unrolled)
    try:
        yield
    finally:
        model.model.layers = orig_layers
        model.config.num_hidden_layers = orig_n


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> float | None:
    """Return the first number in ``text`` (commas stripped), or ``None``."""
    match = _NUM_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


@torch.inference_mode()
def generate_numbers(
    model: torch.nn.Module,
    tokenizer,
    questions: list[dict],
    device: str | torch.device | None = None,
    batch_size: int = 32,
    max_new_tokens: int = 12,
    window: tuple[int, int] | None = None,
) -> list[float | None]:
    """Greedily generate the answer for each question and parse the number.

    When ``window`` is given the model is temporarily unrolled into the RYS path
    (:func:`rys_unrolled`), so generation can use the KV cache and remain exact.
    """
    device = device or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    prompts = [FEWSHOT + f"Q: {q['question']}\nA:" for q in questions]

    estimates: list[float | None] = []
    model.eval()
    with rys_unrolled(model, window):
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for row in range(len(batch)):
                new_tokens = out[row, enc["input_ids"].shape[1] :]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                estimates.append(parse_number(text))
    return estimates


def score_estimates(questions: list[dict], estimates: list[float | None]) -> np.ndarray:
    """Per-question partial-credit scores."""
    return np.array(
        [calculate_score(q["answer"], e) for q, e in zip(questions, estimates, strict=True)],
        dtype=float,
    )


def probe_score(
    model: torch.nn.Module,
    tokenizer,
    questions: list[dict],
    device: str | torch.device | None = None,
    batch_size: int = 32,
    max_new_tokens: int = 12,
    window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run the generative probe and return per-question partial-credit scores.

    Pass ``window=(i, j)`` to score under the RYS intervention.
    """
    estimates = generate_numbers(
        model, tokenizer, questions, device=device, batch_size=batch_size,
        max_new_tokens=max_new_tokens, window=window,
    )
    return score_estimates(questions, estimates)
