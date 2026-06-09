"""Tests for RandSATBench 3-SAT loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rys.randsat_data import (
    load_randsat_assignment_examples,
    make_randsat_assignment_splits,
    parse_assignment,
    parse_dimacs_cnf,
    parse_n_vars_clauses,
)
from rys.sat_data import assignment_satisfies_formula


def test_parse_n_vars_clauses() -> None:
    assert parse_n_vars_clauses("train-final/N128_M384_id1.cnf") == (128, 384)


def test_parse_dimacs_cnf(tmp_path: Path) -> None:
    cnf = tmp_path / "tiny.cnf"
    cnf.write_text(
        "p cnf 3 2\n"
        "1 -2 3 0\n"
        "-1 2 -3 0\n"
    )
    assert parse_dimacs_cnf(cnf) == ((1, -2, 3), (-1, 2, -3))


def test_parse_dimacs_cnf_rejects_non_3_literal_clause(tmp_path: Path) -> None:
    cnf = tmp_path / "bad.cnf"
    cnf.write_text("p cnf 3 1\n1 2 0\n")
    with pytest.raises(ValueError, match="Expected 3 literals"):
        parse_dimacs_cnf(cnf)


def test_parse_assignment() -> None:
    assert parse_assignment("1 -2 3 0", 3) == (1, 0, 1)


def test_load_filters_unsat_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    cnf_dir = data_root / "train-final"
    cnf_dir.mkdir(parents=True)
    cnf_path = cnf_dir / "N16_M1_id1.cnf"
    cnf_path.write_text("p cnf 16 1\n1 2 3 0\n")

    labels = data_root / "train_labels.csv"
    pd.DataFrame(
        [
            {"cnf_file": "train-final/N16_M1_id1.cnf", "sat": 1, "assignment": "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0"},
            {"cnf_file": "train-final/N16_M1_id2.cnf", "sat": 0, "assignment": ""},
        ]
    ).to_csv(labels, index=False)

    examples = load_randsat_assignment_examples(
        data_root,
        labels,
        var_values={16},
        max_examples=10,
        seed=0,
    )
    assert len(examples) == 1
    assert examples[0].prompt_id == "train-final/N16_M1_id1.cnf"


def test_loaded_examples_satisfy_formula(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    cnf_dir = data_root / "train-final"
    cnf_dir.mkdir(parents=True)
    cnf_path = cnf_dir / "N3_M2_id1.cnf"
    cnf_path.write_text("p cnf 3 2\n1 -2 3 0\n-1 2 -3 0\n")

    labels = data_root / "train_labels.csv"
    pd.DataFrame(
        [
            {
                "cnf_file": "train-final/N3_M2_id1.cnf",
                "sat": 1,
                "assignment": "1 2 -3 0",
            }
        ]
    ).to_csv(labels, index=False)

    examples = load_randsat_assignment_examples(
        data_root,
        labels,
        var_values={3},
        seed=0,
    )
    assert len(examples) == 1
    assert assignment_satisfies_formula(examples[0].formula, examples[0].assignment)


def test_make_randsat_assignment_splits(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    cnf_dir = data_root / "train-final"
    cnf_dir.mkdir(parents=True)

    rows = []
    indist_specs = [
        (16, "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 0"),
        (16, "-1 2 -3 4 5 6 7 8 9 10 11 12 13 14 15 16 0"),
        (32, " ".join(str(i) for i in range(1, 33)) + " 0"),
        (64, " ".join(str(i) for i in range(1, 65)) + " 0"),
    ]
    for idx, (n_vars, assignment) in enumerate(indist_specs, start=1):
        cnf_name = f"N{n_vars}_M1_id{idx}.cnf"
        cnf_path = cnf_dir / cnf_name
        if n_vars >= 3:
            cnf_path.write_text(f"p cnf {n_vars} 1\n1 2 3 0\n")
        else:
            cnf_path.write_text(f"p cnf {n_vars} 1\n1 -2 0\n")
        rows.append(
            {
                "cnf_file": f"train-final/{cnf_name}",
                "sat": 1,
                "assignment": assignment,
            }
        )

    labels = data_root / "train_labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)

    splits = make_randsat_assignment_splits(
        data_root,
        indist_vars=(16, 32),
        ood_vars=(64,),
        val_frac=0.34,
        test_frac=0.33,
        seed=0,
    )
    assert set(splits) == {"train", "val", "test", "ood"}
    assert len(splits["ood"]) == 1
    assert splits["ood"][0].n_vars == 64
    assert sum(len(split) for name, split in splits.items() if name != "ood") == 3
