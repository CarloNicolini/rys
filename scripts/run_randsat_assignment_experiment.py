"""Message-passing SAT assignment experiment on RandSATBench.

This reuses the clause-variable message-passing engine from
``run_sat_messagepassing_experiment`` (permutation-equivariant, iterative,
trained with the soft SAT loss to produce *any* satisfying assignment) but
sources real 3-SAT instances from RandSATBench instead of synthetic formulas.

In-distribution and OOD splits are selected by *variable count* via
``--indist-vars`` / ``--ood-vars`` (e.g. train on N in {16, 32}, evaluate OOD on
N = 64).  All matching examples are loaded unless capped with
``--max-indist`` / ``--max-ood`` (those cap the NUMBER OF EXAMPLES, for quick
smoke runs, not the variable count).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rys.randsat_data import make_randsat_assignment_splits
from rys.sat_data import SatAssignmentDataset

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_sat_messagepassing_experiment as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--depths", type=str, default="8,16,32", help="Message-passing rounds to sweep.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("~/workspace/RandSATBench/datasets/3SAT").expanduser(),
    )
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument(
        "--indist-vars",
        type=str,
        default="16,32",
        help="Variable counts N selected for the in-distribution train/val/test pool.",
    )
    parser.add_argument(
        "--ood-vars",
        type=str,
        default="64",
        help="Variable counts N selected for the out-of-distribution split.",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument(
        "--max-indist",
        type=int,
        default=None,
        help="Optional cap on the NUMBER OF in-distribution EXAMPLES (not variables). Default: load all.",
    )
    parser.add_argument(
        "--max-ood",
        type=int,
        default=None,
        help="Optional cap on the NUMBER OF OOD EXAMPLES (not variables). Default: load all.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-mlp", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-repeat", type=int, default=2)
    parser.add_argument("--capture-batches", type=int, default=4)
    parser.add_argument("--deep-supervision", action="store_true", default=True)
    parser.add_argument("--no-deep-supervision", dest="deep_supervision", action="store_false")
    parser.add_argument("--skip-rys", action="store_true", help="Train and export curves/CKA without RYS matrices.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Load these weights and skip training (single --depths value).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/randsat_messagepassing"),
    )
    return parser.parse_args()


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"Expected at least one integer in {value!r}.")
    return values


def split_quality(splits: dict[str, list]) -> pd.DataFrame:
    rows = []
    for split, examples in splits.items():
        if not examples:
            raise ValueError(f"Split {split!r} is empty.")
        n_vars_values = sorted({ex.n_vars for ex in examples})
        n_clauses_values = [ex.n_clauses for ex in examples]
        assignments = [value for ex in examples for value in ex.assignment]
        rows.append(
            {
                "split": split,
                "n_total": len(examples),
                "n_vars_values": ",".join(str(v) for v in n_vars_values),
                "n_clauses_min": min(n_clauses_values),
                "n_clauses_max": max(n_clauses_values),
                "assignment_one_rate": float(sum(assignments) / len(assignments)),
            }
        )
    return pd.DataFrame(rows)


def make_loaders(args: argparse.Namespace) -> tuple[dict[str, DataLoader], pd.DataFrame, int, int]:
    splits = make_randsat_assignment_splits(
        args.data_root,
        labels_csv=args.labels_csv,
        indist_vars=parse_int_list(args.indist_vars),
        ood_vars=parse_int_list(args.ood_vars),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        max_indist=args.max_indist,
        max_ood=args.max_ood,
        seed=args.seed,
    )
    quality = split_quality(splits)
    max_vars = max(ex.n_vars for split in splits.values() for ex in split)
    max_clauses = max(ex.n_clauses for split in splits.values() for ex in split)
    datasets = {
        name: SatAssignmentDataset(examples, max_vars=max_vars, max_clauses=max_clauses)
        for name, examples in splits.items()
    }
    # pin_memory only helps (and is only supported) for CUDA host->device copies.
    pin_memory = args.pin_memory and torch.cuda.is_available()
    loader_kwargs: dict = {"num_workers": args.num_workers, "pin_memory": pin_memory}
    if args.num_workers > 0:
        # The RYS sweep iterates the eval loaders hundreds of times; keep workers
        # alive so they are not respawned on every pass.
        loader_kwargs["persistent_workers"] = True
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, **loader_kwargs),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, **loader_kwargs),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, **loader_kwargs),
        "ood": DataLoader(datasets["ood"], batch_size=args.batch_size, shuffle=False, **loader_kwargs),
    }
    return loaders, quality, max_vars, max_clauses


def write_report(run_dir: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    indist_vars = parse_int_list(args.indist_vars)
    ood_vars = parse_int_list(args.ood_vars)
    lines = [
        "# RandSATBench Message-Passing SAT Depth Sweep",
        "",
        "Task: produce any assignment that satisfies a satisfiable 3-SAT formula.",
        "Solver: permutation-equivariant clause-variable message passing, trained with the soft SAT loss.",
        "Data source: RandSATBench `train-final/` with CaDiCaL labels from `train_labels.csv`.",
        "Primary metric: valid-assignment rate (verified, not canonical exact match).",
        "",
        f"- Depths (rounds): `{args.depths}`",
        f"- Data root: `{args.data_root}`",
        f"- In-distribution vars: `{indist_vars}`",
        f"- OOD vars: `{ood_vars}`",
        f"- Val/test fractions: `{args.val_frac}` / `{args.test_frac}`",
        f"- Max in-distribution / OOD examples: `{args.max_indist}` / `{args.max_ood}`",
        f"- Deep supervision: `{args.deep_supervision}`",
        "",
        "| depth | val valid | test valid | OOD valid | best val window | best test window | best OOD window |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for summary in summaries:
        baselines = summary["baselines"]
        best = {row["split"]: row for row in summary["best_rows"]}

        def _window(split: str) -> str:
            if summary.get("rys_skipped"):
                return "skipped"
            row = best[split]
            return f"({int(row['start'])}, {int(row['end'])}) Δ={row['delta_valid_assignment_rate']:+.3f}"

        lines.append(
            f"| {summary['depth']} | "
            f"{baselines['val']['valid_assignment_rate']:.3f} | "
            f"{baselines['test']['valid_assignment_rate']:.3f} | "
            f"{baselines['ood']['valid_assignment_rate']:.3f} | "
            f"{_window('val')} | {_window('test')} | {_window('ood')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base.seed_everything(args.seed)
    device = base.resolve_device()
    depths = base.parse_depths(args.depths)
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    loaders, quality, max_vars, max_clauses = make_loaders(args)
    quality.to_csv(run_dir / "dataset_quality.csv", index=False)
    summaries = []
    for depth in depths:
        summaries.append(
            base.train_one_depth(
                args,
                n_rounds=depth,
                loaders=loaders,
                max_vars=max_vars,
                max_clauses=max_clauses,
                device=device,
                run_dir=run_dir,
            )
        )
    (run_dir / "summary.json").write_text(
        json.dumps({"args": vars(args), "summaries": summaries}, indent=2, default=str)
    )
    write_report(run_dir, summaries, args)
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
