"""Synthetic 3-SAT data for small controlled RYS experiments.

The generator is intentionally conservative: labels are computed by brute force
for small ``n_vars`` so the experiment has a trustworthy ground truth.  Formulas
are encoded as short token sequences suitable for a tiny Transformer classifier.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

PAD = 0
CLS = 1
CLAUSE = 2
POS = 3
NEG = 4
VAR_BASE = 5

FACTOR_PAD = 0
FACTOR_CLS = 1
FACTOR_LITERAL = 2
FACTOR_QUERY = 3

Clause = tuple[int, int, int]
Formula = tuple[Clause, ...]


@dataclass(frozen=True)
class SatExample:
    """One labelled 3-SAT instance."""

    formula: Formula
    label: int
    n_vars: int
    n_clauses: int
    prompt_id: str


@dataclass(frozen=True)
class SatAssignmentExample:
    """One satisfiable 3-SAT instance with a canonical satisfying assignment."""

    formula: Formula
    assignment: tuple[int, ...]
    n_vars: int
    n_clauses: int
    prompt_id: str


def vocab_size(max_vars: int) -> int:
    """Return the vocabulary size needed to encode variables up to ``max_vars``."""
    if max_vars < 1:
        raise ValueError("max_vars must be positive.")
    return VAR_BASE + max_vars


def sequence_length(n_clauses: int) -> int:
    """Length of the padded token sequence for formulas with ``n_clauses``."""
    if n_clauses < 1:
        raise ValueError("n_clauses must be positive.")
    return 1 + n_clauses * 7


def factorized_sequence_length(n_clauses: int) -> int:
    """Length of the factorized CNF sequence: one CLS plus three literals per clause."""
    if n_clauses < 1:
        raise ValueError("n_clauses must be positive.")
    return 1 + n_clauses * 3


def factorized_assignment_sequence_length(max_vars: int, max_clauses: int) -> int:
    """Length of the factorized formula plus fixed variable-query slots."""
    if max_vars < 1:
        raise ValueError("max_vars must be positive.")
    return factorized_sequence_length(max_clauses) + max_vars


def literal_is_satisfied(literal: int, assignment: np.ndarray) -> bool:
    """Evaluate a signed literal under a boolean assignment indexed from zero."""
    var_idx = abs(literal) - 1
    value = bool(assignment[var_idx])
    return value if literal > 0 else not value


def formula_is_satisfied(formula: Formula, assignment: np.ndarray) -> bool:
    """Return True if every clause has at least one satisfied literal."""
    return all(any(literal_is_satisfied(lit, assignment) for lit in clause) for clause in formula)


def assignment_satisfies_formula(formula: Formula, assignment: Sequence[int | bool]) -> bool:
    """Verify that any proposed assignment satisfies the whole formula."""
    return formula_is_satisfied(formula, np.asarray(assignment, dtype=bool))


def is_satisfiable(formula: Formula, n_vars: int) -> bool:
    """Brute-force satisfiability for small synthetic formulas."""
    if n_vars > 20:
        raise ValueError("Brute-force labels are intended for n_vars <= 20.")
    for mask in range(1 << n_vars):
        assignment = np.array([(mask >> bit) & 1 for bit in range(n_vars)], dtype=bool)
        if formula_is_satisfied(formula, assignment):
            return True
    return False


def first_satisfying_assignment(formula: Formula, n_vars: int) -> tuple[int, ...] | None:
    """Return the lexicographically first satisfying assignment, if one exists."""
    if n_vars > 20:
        raise ValueError("Brute-force assignments are intended for n_vars <= 20.")
    for mask in range(1 << n_vars):
        assignment = np.array([(mask >> bit) & 1 for bit in range(n_vars)], dtype=bool)
        if formula_is_satisfied(formula, assignment):
            return tuple(int(x) for x in assignment)
    return None


def _sample_clause(n_vars: int, rng: np.random.Generator) -> Clause:
    variables = rng.choice(np.arange(1, n_vars + 1), size=3, replace=False)
    signs = rng.choice(np.array([-1, 1]), size=3)
    return tuple(int(sign * var) for sign, var in zip(signs, variables, strict=True))


def _sample_clause_satisfied_by(
    n_vars: int,
    assignment: np.ndarray,
    rng: np.random.Generator,
) -> Clause:
    for _ in range(1_000):
        clause = _sample_clause(n_vars, rng)
        if any(literal_is_satisfied(lit, assignment) for lit in clause):
            return clause
    raise RuntimeError("Failed to sample a planted satisfied clause.")


def random_formula(
    n_vars: int,
    n_clauses: int,
    rng: np.random.Generator,
    *,
    planted_assignment: np.ndarray | None = None,
) -> Formula:
    """Sample a random 3-SAT formula.

    If ``planted_assignment`` is provided, every clause is sampled to be
    satisfied by that assignment, making the whole formula satisfiable.
    """
    if n_vars < 3:
        raise ValueError("3-SAT formulas require at least three variables.")
    if n_clauses < 1:
        raise ValueError("n_clauses must be positive.")

    clauses: list[Clause] = []
    for _ in range(n_clauses):
        if planted_assignment is None:
            clauses.append(_sample_clause(n_vars, rng))
        else:
            clauses.append(_sample_clause_satisfied_by(n_vars, planted_assignment, rng))
    return tuple(clauses)


def labelled_formula(
    n_vars: int,
    n_clauses: int,
    rng: np.random.Generator,
    *,
    target_label: int | None = None,
    max_attempts: int = 10_000,
) -> tuple[Formula, int]:
    """Sample a formula and return its brute-force label.

    ``target_label`` may be ``1`` for satisfiable or ``0`` for unsatisfiable.
    Rejection sampling is acceptable here because the first experiment uses
    tiny formulas and fixed seeds.
    """
    if target_label not in (None, 0, 1):
        raise ValueError("target_label must be None, 0, or 1.")

    for _ in range(max_attempts):
        if target_label == 1 and rng.random() < 0.5:
            planted = rng.integers(0, 2, size=n_vars).astype(bool)
            formula = random_formula(n_vars, n_clauses, rng, planted_assignment=planted)
        else:
            formula = random_formula(n_vars, n_clauses, rng)
        label = int(is_satisfiable(formula, n_vars))
        if target_label is None or label == target_label:
            return formula, label
    raise RuntimeError(
        f"Could not sample target_label={target_label} after {max_attempts} attempts. "
        "Try changing n_vars, n_clauses, or the target balance."
    )


def encode_formula(
    formula: Formula,
    *,
    n_vars: int,
    max_clauses: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a formula as ``(tokens, attention_mask)``.

    Literal ``x_k`` is encoded as ``POS, VAR_BASE + k - 1``; literal ``not x_k``
    as ``NEG, VAR_BASE + k - 1``.  Every clause starts with ``CLAUSE`` and the
    sequence starts with ``CLS``.
    """
    if max_clauses is None:
        max_clauses = len(formula)
    if len(formula) > max_clauses:
        raise ValueError("Formula has more clauses than max_clauses.")

    tokens = [CLS]
    for clause in formula:
        tokens.append(CLAUSE)
        for literal in clause:
            var = abs(literal)
            if not 1 <= var <= n_vars:
                raise ValueError(f"Literal {literal} is outside n_vars={n_vars}.")
            tokens.extend([POS if literal > 0 else NEG, VAR_BASE + var - 1])

    max_len = sequence_length(max_clauses)
    pad_len = max_len - len(tokens)
    if pad_len < 0:
        raise ValueError("Encoded formula exceeded max_len.")
    mask = [1] * len(tokens) + [0] * pad_len
    tokens = tokens + [PAD] * pad_len
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def encode_formula_factorized(
    formula: Formula,
    *,
    n_vars: int,
    max_clauses: int | None = None,
) -> dict[str, torch.Tensor]:
    """Encode a CNF formula as structural fields instead of flat text tokens.

    Each non-padding token carries separate coordinates for variable identity,
    sign, clause identity, literal slot, and token type.  This keeps the model
    close to a standard Transformer while exposing the invariances a SAT model
    should learn.
    """
    if max_clauses is None:
        max_clauses = len(formula)
    if len(formula) > max_clauses:
        raise ValueError("Formula has more clauses than max_clauses.")

    variable_ids = [0]
    sign_ids = [0]
    clause_ids = [0]
    slot_ids = [0]
    token_type_ids = [FACTOR_CLS]

    for clause_idx, clause in enumerate(formula, start=1):
        for slot_idx, literal in enumerate(clause, start=1):
            var = abs(literal)
            if not 1 <= var <= n_vars:
                raise ValueError(f"Literal {literal} is outside n_vars={n_vars}.")
            variable_ids.append(var)
            sign_ids.append(1 if literal > 0 else 2)
            clause_ids.append(clause_idx)
            slot_ids.append(slot_idx)
            token_type_ids.append(FACTOR_LITERAL)

    max_len = factorized_sequence_length(max_clauses)
    pad_len = max_len - len(variable_ids)
    if pad_len < 0:
        raise ValueError("Encoded formula exceeded max_len.")

    def _pad(values: list[int]) -> torch.Tensor:
        return torch.tensor(values + [0] * pad_len, dtype=torch.long)

    attention_mask = torch.tensor([1] * len(variable_ids) + [0] * pad_len, dtype=torch.bool)
    return {
        "variable_ids": _pad(variable_ids),
        "sign_ids": _pad(sign_ids),
        "clause_ids": _pad(clause_ids),
        "slot_ids": _pad(slot_ids),
        "token_type_ids": _pad(token_type_ids),
        "factor_attention_mask": attention_mask,
    }


def encode_formula_assignment_factorized(
    formula: Formula,
    *,
    n_vars: int,
    max_vars: int,
    max_clauses: int,
) -> dict[str, torch.Tensor]:
    """Encode a formula followed by fixed query slots, one per variable.

    Formula literals occupy the same prefix as :func:`encode_formula_factorized`.
    Query slots live after the padded formula region so output position ``k`` is
    always the query for variable ``k + 1``.
    """
    if n_vars > max_vars:
        raise ValueError("n_vars cannot exceed max_vars.")
    if len(formula) > max_clauses:
        raise ValueError("Formula has more clauses than max_clauses.")

    formula_len = factorized_sequence_length(max_clauses)
    max_len = factorized_assignment_sequence_length(max_vars, max_clauses)
    variable_ids = [0] * max_len
    sign_ids = [0] * max_len
    clause_ids = [0] * max_len
    slot_ids = [0] * max_len
    token_type_ids = [FACTOR_PAD] * max_len
    attention_mask = [False] * max_len

    variable_ids[0] = 0
    token_type_ids[0] = FACTOR_CLS
    attention_mask[0] = True

    pos = 1
    for clause_idx, clause in enumerate(formula, start=1):
        for slot_idx, literal in enumerate(clause, start=1):
            var = abs(literal)
            if not 1 <= var <= n_vars:
                raise ValueError(f"Literal {literal} is outside n_vars={n_vars}.")
            variable_ids[pos] = var
            sign_ids[pos] = 1 if literal > 0 else 2
            clause_ids[pos] = clause_idx
            slot_ids[pos] = slot_idx
            token_type_ids[pos] = FACTOR_LITERAL
            attention_mask[pos] = True
            pos += 1

    for var in range(1, n_vars + 1):
        query_pos = formula_len + var - 1
        variable_ids[query_pos] = var
        token_type_ids[query_pos] = FACTOR_QUERY
        attention_mask[query_pos] = True

    return {
        "variable_ids": torch.tensor(variable_ids, dtype=torch.long),
        "sign_ids": torch.tensor(sign_ids, dtype=torch.long),
        "clause_ids": torch.tensor(clause_ids, dtype=torch.long),
        "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        "factor_attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }


def encode_clause_tensor(
    formula: Formula,
    *,
    n_vars: int,
    max_clauses: int,
) -> dict[str, torch.Tensor]:
    """Encode clauses for exact and differentiable assignment verification."""
    if len(formula) > max_clauses:
        raise ValueError("Formula has more clauses than max_clauses.")
    clause_variable_ids = torch.zeros((max_clauses, 3), dtype=torch.long)
    clause_sign_ids = torch.zeros((max_clauses, 3), dtype=torch.long)
    clause_mask = torch.zeros(max_clauses, dtype=torch.bool)
    for clause_idx, clause in enumerate(formula):
        clause_mask[clause_idx] = True
        for slot_idx, literal in enumerate(clause):
            var = abs(literal)
            if not 1 <= var <= n_vars:
                raise ValueError(f"Literal {literal} is outside n_vars={n_vars}.")
            clause_variable_ids[clause_idx, slot_idx] = var
            clause_sign_ids[clause_idx, slot_idx] = 1 if literal > 0 else 2
    return {
        "clause_variable_ids": clause_variable_ids,
        "clause_sign_ids": clause_sign_ids,
        "clause_mask": clause_mask,
    }


def verify_assignment_tensor(
    assignments: torch.Tensor,
    clause_variable_ids: torch.Tensor,
    clause_sign_ids: torch.Tensor,
    clause_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a boolean vector marking assignments that satisfy every clause.

    ``assignments`` has shape ``(batch, max_vars)`` with 0/1 values. Clause
    variables are one-indexed, matching CNF literal notation.
    """
    values = assignments.bool()
    gather_idx = (clause_variable_ids - 1).clamp_min(0)
    expanded = values[:, None, :].expand(-1, gather_idx.shape[1], -1)
    literal_values = torch.gather(expanded, dim=2, index=gather_idx)
    literal_is_positive = clause_sign_ids == 1
    literal_satisfied = torch.where(literal_is_positive, literal_values, ~literal_values)
    clause_satisfied = literal_satisfied.any(dim=-1)
    return (clause_satisfied | ~clause_mask).all(dim=-1)


def multisample_validity(
    sample_assignments: torch.Tensor,
    clause_variable_ids: torch.Tensor,
    clause_sign_ids: torch.Tensor,
    clause_mask: torch.Tensor,
    *,
    assignment_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-formula multi-sample SAT metrics from ``N`` candidate assignments.

    ``sample_assignments`` has shape ``(N, B, max_vars)`` (0/1 values, one draw
    per sample index).  Returns per-formula tensors of shape ``(B,)``:

    - ``valid_any``: 1 if at least one of the ``N`` samples satisfies the formula
      (this is ``valid@N``).
    - ``n_valid``: how many of the ``N`` samples are valid.
    - ``coverage``: number of *distinct* valid assignments found, counting only
      the active variables of each formula.
    """
    n_samples, batch, _ = sample_assignments.shape
    valid = torch.zeros(n_samples, batch, dtype=torch.bool, device=sample_assignments.device)
    for s in range(n_samples):
        valid[s] = verify_assignment_tensor(
            sample_assignments[s], clause_variable_ids, clause_sign_ids, clause_mask
        )
    n_valid = valid.sum(dim=0)
    valid_any = n_valid > 0

    coverage = torch.zeros(batch, dtype=torch.long, device=sample_assignments.device)
    masked = sample_assignments * assignment_mask.unsqueeze(0)  # zero padded vars
    for b in range(batch):
        seen: set[tuple[int, ...]] = set()
        for s in range(n_samples):
            if bool(valid[s, b]):
                seen.add(tuple(int(x) for x in masked[s, b].tolist()))
        coverage[b] = len(seen)
    return {"valid_any": valid_any, "n_valid": n_valid, "coverage": coverage}


def soft_sat_loss(
    logits: torch.Tensor,
    clause_variable_ids: torch.Tensor,
    clause_sign_ids: torch.Tensor,
    clause_mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable relaxation of the hard SAT verifier.

    ``logits`` has shape ``(batch, max_vars, 2)``. The loss is the mean negative
    log probability that each active clause is satisfied, treating per-variable
    truth probabilities as independent.  Minimising it rewards *any* assignment
    that satisfies the formula, not a specific canonical one.

    ``reduction`` is ``"mean"`` (scalar over the whole batch) or ``"none"``
    (per-formula vector of shape ``(batch,)``, the clause-averaged loss for each
    formula — needed for best-of-K / winner-take-all training).
    """
    p_true = logits.softmax(dim=-1)[..., 1]
    gather_idx = (clause_variable_ids - 1).clamp_min(0)
    expanded = p_true[:, None, :].expand(-1, gather_idx.shape[1], -1)
    literal_true_prob = torch.gather(expanded, dim=2, index=gather_idx)
    literal_sat_prob = torch.where(clause_sign_ids == 1, literal_true_prob, 1.0 - literal_true_prob)
    clause_unsat_prob = torch.prod(1.0 - literal_sat_prob, dim=-1)
    clause_sat_prob = (1.0 - clause_unsat_prob).clamp_min(1e-6)
    losses = -torch.log(clause_sat_prob) * clause_mask
    if reduction == "none":
        return losses.sum(dim=1) / clause_mask.sum(dim=1).clamp_min(1)
    if reduction == "mean":
        return losses.sum() / clause_mask.sum().clamp_min(1)
    raise ValueError("reduction must be 'mean' or 'none'.")


def make_sat_examples(
    n_examples: int,
    *,
    n_vars: int,
    n_clauses: int,
    seed: int = 0,
    balanced: bool = True,
    prefix: str = "sat",
) -> list[SatExample]:
    """Create deterministic labelled examples."""
    if n_examples < 1:
        raise ValueError("n_examples must be positive.")
    rng = np.random.default_rng(seed)
    examples: list[SatExample] = []
    for idx in range(n_examples):
        target = idx % 2 if balanced else None
        formula, label = labelled_formula(n_vars, n_clauses, rng, target_label=target)
        examples.append(
            SatExample(
                formula=formula,
                label=label,
                n_vars=n_vars,
                n_clauses=n_clauses,
                prompt_id=f"{prefix}_{idx:05d}",
            )
        )
    return examples


def make_sat_assignment_examples(
    n_examples: int,
    *,
    n_vars: int,
    n_clauses: int,
    seed: int = 0,
    prefix: str = "assign",
    max_attempts: int = 100_000,
) -> list[SatAssignmentExample]:
    """Create deterministic satisfiable formulas with canonical assignment labels."""
    if n_examples < 1:
        raise ValueError("n_examples must be positive.")
    rng = np.random.default_rng(seed)
    examples: list[SatAssignmentExample] = []
    attempts = 0
    while len(examples) < n_examples and attempts < max_attempts:
        attempts += 1
        formula, label = labelled_formula(n_vars, n_clauses, rng, target_label=1)
        if label != 1:
            continue
        assignment = first_satisfying_assignment(formula, n_vars)
        if assignment is None:
            continue
        idx = len(examples)
        examples.append(
            SatAssignmentExample(
                formula=formula,
                assignment=assignment,
                n_vars=n_vars,
                n_clauses=n_clauses,
                prompt_id=f"{prefix}_{idx:05d}",
            )
        )
    if len(examples) != n_examples:
        raise RuntimeError(
            f"Only sampled {len(examples)} examples after {attempts} attempts. "
            "Try reducing n_clauses or increasing max_attempts."
        )
    return examples


def make_sat_splits(
    *,
    n_train: int = 1_024,
    n_val: int = 256,
    n_test: int = 256,
    n_vars: int = 6,
    n_clauses: int = 18,
    seed: int = 0,
) -> dict[str, list[SatExample]]:
    """Return deterministic train/validation/test splits."""
    return {
        "train": make_sat_examples(
            n_train,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed,
            prefix="train",
        ),
        "val": make_sat_examples(
            n_val,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed + 1,
            prefix="val",
        ),
        "test": make_sat_examples(
            n_test,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed + 2,
            prefix="test",
        ),
    }


def make_sat_assignment_splits(
    *,
    n_train: int = 1_024,
    n_val: int = 256,
    n_test: int = 256,
    n_vars: int = 6,
    n_clauses: int = 24,
    seed: int = 0,
) -> dict[str, list[SatAssignmentExample]]:
    """Return deterministic train/validation/test splits for assignment generation."""
    return {
        "train": make_sat_assignment_examples(
            n_train,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed,
            prefix="train",
        ),
        "val": make_sat_assignment_examples(
            n_val,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed + 1,
            prefix="val",
        ),
        "test": make_sat_assignment_examples(
            n_test,
            n_vars=n_vars,
            n_clauses=n_clauses,
            seed=seed + 2,
            prefix="test",
        ),
    }


class SatDataset(Dataset):
    """PyTorch dataset wrapping pre-generated 3-SAT examples."""

    def __init__(
        self,
        examples: Sequence[SatExample],
        *,
        max_vars: int | None = None,
        max_clauses: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("SatDataset requires at least one example.")
        self.examples = list(examples)
        self.max_vars = max_vars or max(ex.n_vars for ex in self.examples)
        self.max_clauses = max_clauses or max(ex.n_clauses for ex in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        tokens, mask = encode_formula(ex.formula, n_vars=self.max_vars, max_clauses=self.max_clauses)
        item = {
            "input_ids": tokens,
            "attention_mask": mask,
            "labels": torch.tensor(ex.label, dtype=torch.long),
            "prompt_id": ex.prompt_id,
        }
        item.update(
            encode_formula_factorized(
                ex.formula,
                n_vars=self.max_vars,
                max_clauses=self.max_clauses,
            )
        )
        return item


class SatAssignmentDataset(Dataset):
    """PyTorch dataset for canonical satisfying-assignment generation."""

    def __init__(
        self,
        examples: Sequence[SatAssignmentExample],
        *,
        max_vars: int | None = None,
        max_clauses: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("SatAssignmentDataset requires at least one example.")
        self.examples = list(examples)
        self.max_vars = max_vars or max(ex.n_vars for ex in self.examples)
        self.max_clauses = max_clauses or max(ex.n_clauses for ex in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ex = self.examples[idx]
        labels = torch.full((self.max_vars,), -100, dtype=torch.long)
        mask = torch.zeros(self.max_vars, dtype=torch.bool)
        labels[: ex.n_vars] = torch.tensor(ex.assignment, dtype=torch.long)
        mask[: ex.n_vars] = True
        item: dict[str, torch.Tensor | str] = {
            "assignment_labels": labels,
            "assignment_mask": mask,
            "labels": labels,
            "prompt_id": ex.prompt_id,
        }
        item.update(
            encode_formula_assignment_factorized(
                ex.formula,
                n_vars=ex.n_vars,
                max_vars=self.max_vars,
                max_clauses=self.max_clauses,
            )
        )
        item.update(
            encode_clause_tensor(
                ex.formula,
                n_vars=ex.n_vars,
                max_clauses=self.max_clauses,
            )
        )
        return item


class ClauseAssignmentDataset(Dataset):
    """Lean assignment dataset for the clause-variable message-passing solver.

    Two differences from :class:`SatAssignmentDataset` make it cheap for large
    RandSATBench runs:

    - It encodes every example *once* in ``__init__`` instead of on every
      ``__getitem__``, and it produces only the tensors the message-passing model
      consumes (the clause adjacency plus the assignment labels/mask), dropping
      the flat- and factorized-token fields that solver never reads.
    - Clause tensors are stored *unpadded* (length ``n_clauses``).  Pair the
      dataset with :func:`collate_clause_assignment` to pad each batch to its own
      clause maximum rather than the global one.  Variable-side tensors keep the
      global ``max_vars`` width because the model's variable stream and the
      assignment head are that wide.
    """

    def __init__(self, examples: Sequence[SatAssignmentExample], *, max_vars: int | None = None) -> None:
        if not examples:
            raise ValueError("ClauseAssignmentDataset requires at least one example.")
        self.examples = list(examples)
        self.max_vars = max_vars or max(ex.n_vars for ex in self.examples)
        self.n_vars = [ex.n_vars for ex in self.examples]
        self._items = [self._encode(ex) for ex in self.examples]

    def _encode(self, ex: SatAssignmentExample) -> dict[str, torch.Tensor | str]:
        if ex.n_vars > self.max_vars:
            raise ValueError(f"Example has n_vars={ex.n_vars} > max_vars={self.max_vars}.")
        labels = torch.full((self.max_vars,), -100, dtype=torch.long)
        mask = torch.zeros(self.max_vars, dtype=torch.bool)
        labels[: ex.n_vars] = torch.tensor(ex.assignment, dtype=torch.long)
        mask[: ex.n_vars] = True
        clause = encode_clause_tensor(ex.formula, n_vars=ex.n_vars, max_clauses=len(ex.formula))
        return {
            "assignment_labels": labels,
            "assignment_mask": mask,
            "labels": labels,
            "prompt_id": ex.prompt_id,
            **clause,
        }

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return self._items[idx]


def collate_clause_assignment(samples: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    """Collate :class:`ClauseAssignmentDataset` items, padding clauses per batch.

    Each clause tensor is padded to the batch's clause maximum (padded clauses
    are fully masked, so the result is numerically identical to global padding).
    Variable-side tensors already share ``max_vars`` and are simply stacked.
    """
    if not samples:
        raise ValueError("collate_clause_assignment received an empty batch.")
    batch = len(samples)
    max_clauses = max(int(sample["clause_mask"].shape[0]) for sample in samples)
    clause_variable_ids = torch.zeros(batch, max_clauses, 3, dtype=torch.long)
    clause_sign_ids = torch.zeros(batch, max_clauses, 3, dtype=torch.long)
    clause_mask = torch.zeros(batch, max_clauses, dtype=torch.bool)
    for row, sample in enumerate(samples):
        n_clauses = int(sample["clause_mask"].shape[0])
        clause_variable_ids[row, :n_clauses] = sample["clause_variable_ids"]
        clause_sign_ids[row, :n_clauses] = sample["clause_sign_ids"]
        clause_mask[row, :n_clauses] = sample["clause_mask"]
    labels = torch.stack([sample["assignment_labels"] for sample in samples])
    assignment_mask = torch.stack([sample["assignment_mask"] for sample in samples])
    return {
        "clause_variable_ids": clause_variable_ids,
        "clause_sign_ids": clause_sign_ids,
        "clause_mask": clause_mask,
        "assignment_labels": labels,
        "assignment_mask": assignment_mask,
        "labels": labels,
        "prompt_id": [str(sample["prompt_id"]) for sample in samples],
    }


class NBucketBatchSampler(Sampler[list[int]]):
    """Batch indices that share a size key (e.g. variable count) together.

    Grouping same-sized instances keeps the per-batch clause padding tight when
    paired with :func:`collate_clause_assignment`.  With ``shuffle`` the order of
    examples within each bucket and the order of the emitted batches are permuted
    every epoch.  Each bucket emits its own trailing short batch unless
    ``drop_last`` is set.
    """

    def __init__(
        self,
        group_ids: Iterable[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.group_ids = list(group_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _buckets(self) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {}
        for idx, group in enumerate(self.group_ids):
            buckets.setdefault(int(group), []).append(idx)
        return buckets

    def _make_batches(self) -> list[list[int]]:
        buckets = self._buckets()
        rng = np.random.default_rng(self.seed + self._epoch) if self.shuffle else None
        batches: list[list[int]] = []
        for group in sorted(buckets):
            members = buckets[group]
            if rng is not None:
                members = [members[i] for i in rng.permutation(len(members))]
            for start in range(0, len(members), self.batch_size):
                batch = members[start : start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if rng is not None:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        batches = self._make_batches()
        self._epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for members in self._buckets().values():
            count = len(members)
            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (count + self.batch_size - 1) // self.batch_size
        return total
