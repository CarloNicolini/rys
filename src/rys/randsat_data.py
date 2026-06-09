"""Load RandSATBench 3-SAT instances for assignment-generation experiments."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from rys.sat_data import Formula, SatAssignmentExample

_FILENAME_RE = re.compile(r"N(\d+)_M(\d+)")


def parse_n_vars_clauses(name: str) -> tuple[int, int]:
    """Parse ``N`` and ``M`` from a RandSATBench CNF filename or path."""
    match = _FILENAME_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse N/M from {name!r}.")
    return int(match.group(1)), int(match.group(2))


def parse_dimacs_cnf(path: Path) -> Formula:
    """Parse a DIMACS CNF file into a 3-SAT formula."""
    literals: list[int] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c") or line.startswith("p"):
            continue
        literals.extend(int(token) for token in line.split())

    clauses: list[tuple[int, int, int]] = []
    current: list[int] = []
    for literal in literals:
        if literal == 0:
            if current:
                if len(current) != 3:
                    raise ValueError(f"Expected 3 literals per clause in {path}, got {len(current)}.")
                clauses.append((current[0], current[1], current[2]))
                current = []
            continue
        current.append(literal)
    if current:
        raise ValueError(f"Unterminated clause in {path}.")
    return tuple(clauses)


def parse_assignment(text: str, n_vars: int) -> tuple[int, ...]:
    """Convert a signed-literal assignment string to 0/1 values indexed from zero."""
    tokens = [int(token) for token in text.split() if token]
    if tokens and tokens[-1] == 0:
        tokens = tokens[:-1]
    if len(tokens) != n_vars:
        raise ValueError(f"Expected {n_vars} assignment literals, got {len(tokens)}.")

    assignment = [0] * n_vars
    for literal in tokens:
        var_idx = abs(literal) - 1
        if not 0 <= var_idx < n_vars:
            raise ValueError(f"Literal {literal} is outside n_vars={n_vars}.")
        assignment[var_idx] = 1 if literal > 0 else 0
    return tuple(assignment)


def _resolve_cnf_path(data_root: Path, cnf_file: str) -> Path:
    path = data_root / cnf_file
    if path.exists():
        return path
    basename = Path(cnf_file).name
    for folder in ("train-final", "test-final"):
        candidate = data_root / folder / basename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find CNF file for {cnf_file!r} under {data_root}.")


def load_randsat_assignment_examples(
    data_root: Path,
    labels_csv: Path,
    *,
    var_values: set[int] | frozenset[int],
    max_examples: int | None = None,
    seed: int = 0,
) -> list[SatAssignmentExample]:
    """Load satisfiable RandSATBench examples with canonical assignments."""
    if not var_values:
        raise ValueError("var_values must be non-empty.")

    frame = pd.read_csv(labels_csv)
    frame = frame[frame["sat"] == 1].copy()
    frame = frame[frame["assignment"].astype(str).str.strip().ne("")]
    if frame.empty:
        raise ValueError(f"No satisfiable rows with assignments found in {labels_csv}.")

    frame[["n_vars", "n_clauses"]] = frame["cnf_file"].apply(
        lambda name: pd.Series(parse_n_vars_clauses(name))
    )
    frame = frame[frame["n_vars"].isin(var_values)]
    if frame.empty:
        raise ValueError(f"No rows matched var_values={sorted(var_values)} in {labels_csv}.")

    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive when provided.")
        frame = frame.iloc[:max_examples]

    examples: list[SatAssignmentExample] = []
    for row in frame.itertuples(index=False):
        cnf_file = str(row.cnf_file)
        n_vars = int(row.n_vars)
        n_clauses = int(row.n_clauses)
        cnf_path = _resolve_cnf_path(data_root, cnf_file)
        formula = parse_dimacs_cnf(cnf_path)
        if len(formula) != n_clauses:
            raise ValueError(
                f"Clause count mismatch for {cnf_file}: filename has M={n_clauses}, "
                f"CNF has {len(formula)} clauses."
            )
        assignment = parse_assignment(str(row.assignment), n_vars)
        examples.append(
            SatAssignmentExample(
                formula=formula,
                assignment=assignment,
                n_vars=n_vars,
                n_clauses=n_clauses,
                prompt_id=cnf_file,
            )
        )
    return examples


def make_randsat_assignment_splits(
    data_root: Path,
    *,
    labels_csv: Path | None = None,
    indist_vars: tuple[int, ...] = (16, 32),
    ood_vars: tuple[int, ...] = (64,),
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    max_indist: int | None = None,
    max_ood: int | None = None,
    seed: int = 0,
) -> dict[str, list[SatAssignmentExample]]:
    """Build train/val/test/ood splits from RandSATBench train labels."""
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be non-negative and sum to less than 1.")

    labels_path = labels_csv or (data_root / "train_labels.csv")
    indist = load_randsat_assignment_examples(
        data_root,
        labels_path,
        var_values=set(indist_vars),
        max_examples=max_indist,
        seed=seed,
    )
    if len(indist) < 3:
        raise ValueError("Need at least three in-distribution examples to split train/val/test.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(indist))
    n_val = max(1, int(round(len(indist) * val_frac)))
    n_test = max(1, int(round(len(indist) * test_frac)))
    n_train = len(indist) - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Split leaves no training examples: n_indist={len(indist)}, "
            f"val_frac={val_frac}, test_frac={test_frac}."
        )

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val : n_train + n_val + n_test]

    ood = load_randsat_assignment_examples(
        data_root,
        labels_path,
        var_values=set(ood_vars),
        max_examples=max_ood,
        seed=seed + 1,
    )
    if not ood:
        raise ValueError(f"No OOD examples found for ood_vars={ood_vars}.")

    return {
        "train": [indist[i] for i in train_idx],
        "val": [indist[i] for i in val_idx],
        "test": [indist[i] for i in test_idx],
        "ood": ood,
    }
