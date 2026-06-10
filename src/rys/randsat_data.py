"""Load RandSATBench 3-SAT instances for assignment-generation experiments."""

from __future__ import annotations

import contextlib
import hashlib
import pickle
import re
from collections.abc import Callable
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


def _read_labels_frame(labels_csv: Path) -> pd.DataFrame:
    """Read the labels CSV once, keep satisfiable rows, and parse N/M vectorised."""
    frame = pd.read_csv(labels_csv)
    frame = frame[frame["sat"] == 1].copy()
    frame = frame[frame["assignment"].astype(str).str.strip().ne("")]
    if frame.empty:
        raise ValueError(f"No satisfiable rows with assignments found in {labels_csv}.")

    extracted = frame["cnf_file"].astype(str).str.extract(_FILENAME_RE)
    if extracted.isna().to_numpy().any():
        bad = frame["cnf_file"][extracted.isna().any(axis=1)].iloc[0]
        raise ValueError(f"Could not parse N/M from {bad!r}.")
    frame["n_vars"] = extracted[0].astype(int)
    frame["n_clauses"] = extracted[1].astype(int)
    return frame


def _cache_path(
    cache_dir: Path,
    data_root: Path,
    labels_csv: Path,
    var_values: set[int] | frozenset[int],
    max_examples: int | None,
    seed: int,
) -> Path:
    """Return the cache file for one parsed example set, keyed by its inputs."""
    labels_csv = Path(labels_csv)
    try:
        stat = labels_csv.stat()
        signature = f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        signature = "nostat"
    key = "|".join(
        [
            str(Path(data_root).resolve()),
            str(labels_csv.resolve()),
            signature,
            ",".join(str(value) for value in sorted(var_values)),
            str(max_examples),
            str(seed),
        ]
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"randsat_examples_{digest}.pkl"


def _load_cached_examples(path: Path) -> list[SatAssignmentExample] | None:
    """Return cached examples, or ``None`` if the cache is missing or unreadable."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return list(payload["examples"])
    except (OSError, pickle.UnpicklingError, EOFError, KeyError, AttributeError):
        return None


def _save_cached_examples(path: Path, examples: list[SatAssignmentExample]) -> None:
    """Best-effort write of parsed examples; never raise into the caller."""
    with contextlib.suppress(OSError, pickle.PicklingError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as handle:
            pickle.dump({"examples": examples}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)


def load_randsat_assignment_examples(
    data_root: Path,
    labels_csv: Path,
    *,
    var_values: set[int] | frozenset[int],
    max_examples: int | None = None,
    seed: int = 0,
    frame: pd.DataFrame | None = None,
    frame_loader: Callable[[], pd.DataFrame] | None = None,
    cache_dir: Path | None = None,
) -> list[SatAssignmentExample]:
    """Load satisfiable RandSATBench examples with canonical assignments.

    Parsing every CNF file is the dominant startup cost on large splits, so when
    ``cache_dir`` is given the parsed example list is cached on disk keyed by the
    labels-file signature, ``var_values``, ``max_examples``, and ``seed``; a cache
    hit skips both the CSV read and the per-file DIMACS parse.  A preloaded
    ``frame`` (or a lazy ``frame_loader``) lets callers share a single CSV read
    across several queries.
    """
    if not var_values:
        raise ValueError("var_values must be non-empty.")

    cache_file = None
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, data_root, labels_csv, var_values, max_examples, seed)
        cached = _load_cached_examples(cache_file)
        if cached is not None:
            return cached

    if frame is None:
        frame = frame_loader() if frame_loader is not None else _read_labels_frame(labels_csv)

    subset = frame[frame["n_vars"].isin(set(var_values))]
    if subset.empty:
        raise ValueError(f"No rows matched var_values={sorted(var_values)} in {labels_csv}.")

    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_examples is not None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive when provided.")
        subset = subset.iloc[:max_examples]

    examples: list[SatAssignmentExample] = []
    for row in subset.itertuples(index=False):
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
    if cache_file is not None:
        _save_cached_examples(cache_file, examples)
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
    cache_dir: Path | None = None,
) -> dict[str, list[SatAssignmentExample]]:
    """Build train/val/test/ood splits from RandSATBench train labels.

    The in-distribution and OOD example queries share a single lazy CSV read, and
    each is cached on disk when ``cache_dir`` is set, so a warm cache builds the
    splits without touching the labels file or the CNF directory.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be non-negative and sum to less than 1.")

    labels_path = labels_csv or (data_root / "train_labels.csv")

    frame_box: list[pd.DataFrame] = []

    def shared_frame() -> pd.DataFrame:
        if not frame_box:
            frame_box.append(_read_labels_frame(labels_path))
        return frame_box[0]

    indist = load_randsat_assignment_examples(
        data_root,
        labels_path,
        var_values=set(indist_vars),
        max_examples=max_indist,
        seed=seed,
        frame_loader=shared_frame,
        cache_dir=cache_dir,
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
        frame_loader=shared_frame,
        cache_dir=cache_dir,
    )
    if not ood:
        raise ValueError(f"No OOD examples found for ood_vars={ood_vars}.")

    return {
        "train": [indist[i] for i in train_idx],
        "val": [indist[i] for i in val_idx],
        "test": [indist[i] for i in test_idx],
        "ood": ood,
    }
