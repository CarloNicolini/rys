"""Synthetic N-Queens data for controlled RYS experiments.

Parallel to :mod:`rys.sat_data`.  The task is *conditional board completion*:
each instance reveals a few non-attacking queens (``givens``) and the model must
place one queen per remaining row so the full board is valid.  Conditioning on
the givens makes train/val/test/OOD genuinely distinct instances (a bare
``N`` would be a single trivial input), exactly as the CNF formula conditioned
the SAT task.

Representation (v1)
-------------------
One discrete variable per row, taking a column in ``{0, .., N-1}`` (multiclass),
*not* a flat board string.  A board is encoded by per-row fields:

- ``given_col``    column of a given queen, or ``max_n`` (the "unknown" index);
- ``is_given``     1 for revealed rows, 0 otherwise;
- ``row_mask``     1 for real rows (``row < n``);
- ``col_mask``     1 for real columns (``col < n``);
- ``labels``       the canonical solution column per row (diagnostic only).

The **primary metric** is ``valid_board_rate`` from :func:`verify_boards_tensor`
(one queen per row, distinct columns, no diagonal conflict, givens respected),
never exact match to one canonical board — many valid completions exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

UNKNOWN_OFFSET = 0  # placeholder column stored for non-given rows before padding

Board = tuple[int, ...]


@dataclass(frozen=True)
class QueensExample:
    """One N-Queens completion instance."""

    n: int
    solution: Board  # full valid board, column per row
    givens: Board  # length n; column for given rows, -1 otherwise
    prompt_id: str


def board_is_valid(cols: Sequence[int], n: int) -> bool:
    """Exact validity check: one queen per row, distinct columns, no diagonals."""
    cols = list(cols)
    if len(cols) != n:
        return False
    if any(not (0 <= c < n) for c in cols):
        return False
    if len(set(cols)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) == abs(cols[i] - cols[j]):
                return False
    return True


def _complete(n: int, cols: list[int], row: int, rng: np.random.Generator, fixed: dict[int, int]) -> list[int] | None:
    """Randomised backtracking completion from ``row`` onward."""
    if row == n:
        return list(cols)
    candidates = [fixed[row]] if row in fixed else list(rng.permutation(n))
    for c in candidates:
        ok = True
        for r in range(row):
            if cols[r] == c or abs(row - r) == abs(c - cols[r]):
                ok = False
                break
        if not ok:
            continue
        cols[row] = c
        out = _complete(n, cols, row + 1, rng, fixed)
        if out is not None:
            return out
        cols[row] = -1
    return None


def find_solution(n: int, rng: np.random.Generator) -> Board:
    """Return a uniformly-randomised valid board via randomised backtracking."""
    if n < 4 and n not in (1,):
        raise ValueError("N-Queens has no solution for n in {2, 3}.")
    out = _complete(n, [-1] * n, 0, rng, fixed={})
    if out is None:
        raise RuntimeError(f"No N-Queens solution found for n={n}.")
    return tuple(out)


def complete_with_givens(n: int, givens: dict[int, int], rng: np.random.Generator) -> Board | None:
    """Random valid completion consistent with ``givens`` (row -> column)."""
    out = _complete(n, [-1] * n, 0, rng, fixed=givens)
    return tuple(out) if out is not None else None


def make_queens_examples(
    n_examples: int,
    *,
    n: int = 8,
    n_givens: int = 2,
    seed: int = 0,
    prefix: str = "queens",
) -> list[QueensExample]:
    """Create deterministic board-completion instances.

    Each instance: sample a full valid board, reveal ``n_givens`` random rows as
    givens; the canonical label is a (re-)completion consistent with the givens.
    """
    if n_examples < 1:
        raise ValueError("n_examples must be positive.")
    if not 0 <= n_givens < n:
        raise ValueError("n_givens must satisfy 0 <= n_givens < n.")
    rng = np.random.default_rng(seed)
    examples: list[QueensExample] = []
    for idx in range(n_examples):
        base = find_solution(n, rng)
        given_rows = sorted(rng.choice(n, size=n_givens, replace=False).tolist()) if n_givens else []
        fixed = {r: int(base[r]) for r in given_rows}
        solution = _complete(n, [-1] * n, 0, rng, fixed=fixed)
        if solution is None:  # pragma: no cover - givens come from a real board
            solution = list(base)
        givens = tuple(fixed.get(r, -1) for r in range(n))
        examples.append(
            QueensExample(
                n=n,
                solution=tuple(solution),
                givens=givens,
                prompt_id=f"{prefix}_{idx:05d}",
            )
        )
    return examples


def make_queens_splits(
    *,
    n_train: int = 4096,
    n_val: int = 1024,
    n_test: int = 1024,
    n: int = 8,
    n_givens: int = 2,
    seed: int = 0,
) -> dict[str, list[QueensExample]]:
    """Deterministic train/val/test splits (same board size and given count)."""
    return {
        "train": make_queens_examples(n_train, n=n, n_givens=n_givens, seed=seed, prefix="train"),
        "val": make_queens_examples(n_val, n=n, n_givens=n_givens, seed=seed + 1, prefix="val"),
        "test": make_queens_examples(n_test, n=n, n_givens=n_givens, seed=seed + 2, prefix="test"),
    }


class QueensDataset(Dataset):
    """PyTorch dataset over N-Queens completion instances, padded to ``max_n``."""

    def __init__(
        self,
        examples: Sequence[QueensExample],
        *,
        max_n: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("QueensDataset requires at least one example.")
        self.examples = list(examples)
        self.max_n = max_n or max(ex.n for ex in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        m = self.max_n
        unknown = m  # index into an (max_n + 1) embedding table
        given_col = torch.full((m,), unknown, dtype=torch.long)
        is_given = torch.zeros(m, dtype=torch.bool)
        row_mask = torch.zeros(m, dtype=torch.bool)
        col_mask = torch.zeros(m, dtype=torch.bool)
        labels = torch.full((m,), -100, dtype=torch.long)
        for r in range(ex.n):
            row_mask[r] = True
            labels[r] = ex.solution[r]
            if ex.givens[r] >= 0:
                given_col[r] = ex.givens[r]
                is_given[r] = True
        col_mask[: ex.n] = True
        return {
            "given_col": given_col,
            "is_given": is_given,
            "row_mask": row_mask,
            "col_mask": col_mask,
            "labels": labels,
            "n": torch.tensor(ex.n, dtype=torch.long),
            "prompt_id": ex.prompt_id,
        }


def verify_boards_tensor(
    cols: torch.Tensor,
    row_mask: torch.Tensor,
    given_col: torch.Tensor,
    is_given: torch.Tensor,
) -> torch.Tensor:
    """Boolean vector marking boards that satisfy every N-Queens constraint.

    ``cols`` is ``(batch, max_n)`` predicted column per row (ints).  Inactive
    rows (``~row_mask``) are ignored.  Givens must be respected exactly.
    """
    b, r = cols.shape
    ri = torch.arange(r, device=cols.device)
    ci = cols.unsqueeze(2)
    cj = cols.unsqueeze(1)
    rii = ri.view(1, r, 1)
    rjj = ri.view(1, 1, r)
    pair = row_mask.unsqueeze(2) & row_mask.unsqueeze(1) & (rii < rjj)
    same_col = (ci == cj) & pair
    diag = ((rii - rjj).abs() == (ci - cj).abs()) & pair
    conflict = (same_col | diag).reshape(b, -1).any(dim=1)
    givens_ok = ((cols == given_col) | ~is_given | ~row_mask).all(dim=1)
    return (~conflict) & givens_ok


def soft_nqueens_loss(
    logits: torch.Tensor,
    row_mask: torch.Tensor,
    col_mask: torch.Tensor,
    *,
    given_col: torch.Tensor | None = None,
    is_given: torch.Tensor | None = None,
    given_ce_weight: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable relaxation of the N-Queens verifier.

    ``logits`` is ``(batch, max_n, max_n)`` (per row, a distribution over
    columns).  For every active row pair the loss is the negative log-probability
    that the pair is conflict-free, where column- and diagonal-conflict
    probabilities are computed from independent per-row column distributions.
    An optional cross-entropy term anchors the given rows to their column.
    """
    b, r, c = logits.shape
    masked = logits.masked_fill(~col_mask[:, None, :], -1e9)
    p = masked.softmax(dim=-1)  # (B, R, C)
    total = logits.new_zeros(b)
    count = logits.new_zeros(b)
    for i in range(r):
        for j in range(i + 1, r):
            d = j - i
            pi = p[:, i, :]
            pj = p[:, j, :]
            pcol = (pi * pj).sum(-1)
            term1 = torch.zeros_like(pj)
            term1[:, d:] = pj[:, : c - d]  # pj[col - d]
            term2 = torch.zeros_like(pj)
            term2[:, : c - d] = pj[:, d:]  # pj[col + d]
            pdiag = (pi * (term1 + term2)).sum(-1)
            conflict = (pcol + pdiag).clamp(max=1.0 - eps)
            active = (row_mask[:, i] & row_mask[:, j]).float()
            total = total - torch.log((1.0 - conflict).clamp(min=eps)) * active
            count = count + active
    loss = (total / count.clamp(min=1.0)).mean()
    if given_ce_weight > 0 and given_col is not None and is_given is not None:
        target = torch.where(is_given & row_mask, given_col, torch.full_like(given_col, -100))
        ce = F.cross_entropy(masked.reshape(b * r, c), target.reshape(b * r), ignore_index=-100)
        if torch.isfinite(ce):
            loss = loss + given_ce_weight * ce
    return loss
