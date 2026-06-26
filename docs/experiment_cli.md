# Experiment scripts — command-line reference

All training scripts in `scripts/` expose a [Typer](https://typer.tiangolo.com/) CLI.
The authoritative, always-up-to-date parameter list for any script is:

```bash
python scripts/<script>.py --help
```

This page summarises every script and its options. Options use the `--kebab-case`
form of the Python parameter name (e.g. the parameter `n_vars` is `--n-vars`).
Boolean flags are paired (e.g. `--deep-supervision/--no-deep-supervision`).

## Shared behaviour (all scripts)

- **Training runs on PyTorch Lightning.** Device is auto-selected (MPS → CUDA → CPU);
  there is no device flag.
- Each run writes to `‹output-dir›/‹YYYYMMDD_HHMMSS›/` (depth/mode/regime sweeps add a
  `L‹NN›/` / `‹mode›/` / `‹regime›/` subfolder).
- Per run/sweep-cell you get: the **best-validation** checkpoint `best-*.ckpt` and
  `last.ckpt`; TensorBoard logs under `tb/`; CSV logs under `csv/`; the per-epoch
  `train_history.csv` (now with both `train_*` and `val_*` columns); plus the existing
  `summary.json`, `report.md`, CSVs and PNGs.
- View training curves with: `tensorboard --logdir results/<experiment>`.

### Options shared by most scripts

| Option | Meaning |
| --- | --- |
| `--seed` | Random seed (seeds Python/NumPy/Torch via Lightning). |
| `--n-train` / `--n-val` / `--n-test` | Examples per split (test count is reused for the OOD split). |
| `--batch-size` | Mini-batch size. |
| `--epochs` | Training epochs. |
| `--lr` / `--weight-decay` | AdamW learning rate / weight decay. |
| `--d-model` / `--d-mlp` | Model width / MLP hidden width. |
| `--capture-batches` | Validation batches used to build the CKA connectome. |
| `--output-dir` | Run output directory. |

Unless noted per script, the common defaults are
`n_train=4096, n_val=1024, n_test=1024, batch_size=128, epochs=20, lr=5e-4,
weight_decay=1e-2, d_model=64, d_mlp=128, capture_batches=4`.

---

## SAT classifiers (balanced SAT/UNSAT)

### `run_tiny_sat_experiment.py`
Train a tiny 3-SAT classifier, score candidate RYS windows, export CKA + residual-force.

```bash
python scripts/constrained_satisfaction/run_tiny_sat_experiment.py --architecture factorized --epochs 8
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--architecture` | `factorized` | `flat` or `factorized` (validated). |
| `--n-vars` / `--n-clauses` | `6` / `18` | In-distribution size. Keep the clause/var ratio high enough that UNSAT formulas exist. |
| `--ood-vars` / `--ood-clauses` | `7` / `24` | OOD size. |
| `--n-layers` / `--n-heads` | `6` / `4` | Transformer depth / heads. |
| `--dropout` | `0.0` | Dropout probability. |
| `--min-window` / `--max-window` | `2` / `4` | RYS window length range to score. |
| `--max-repeat` | `4` | Max total traversals of a window. |
| `--top-k-windows` | `8` | Top windows to evaluate. |

Default-deviations: `n_train=1024, n_val=256, n_test=256, batch_size=64, epochs=8, lr=3e-4`.

### `run_rys_accuracy_matrix.py`
Train once (or load `--checkpoint`) and sweep strict upper-triangular RYS windows into a ΔAccuracy matrix.

```bash
python scripts/constrained_satisfaction/run_rys_accuracy_matrix.py --n-layers 32
python scripts/constrained_satisfaction/run_rys_accuracy_matrix.py --checkpoint results/.../best-XX.ckpt  # skip training
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--architecture` | `factorized` | `flat` or `factorized` (validated). |
| `--n-vars` / `--n-clauses` | `4` / `12` | In-distribution size. |
| `--ood-vars` / `--ood-clauses` | `5` / `16` | OOD size. |
| `--n-layers` / `--n-heads` | `32` / `4` | Transformer depth / heads. |
| `--dropout` | `0.0` | Dropout probability. |
| `--max-repeat` | `2` | Total traversals of the inclusive window. |
| `--checkpoint` | `None` | Load a `.ckpt` (or legacy `.pt`) and skip training. |

Default-deviations: `seed=30, n_train=8192, n_val=2048, n_test=2048, epochs=30, lr=3e-4`.

---

## SAT assignment generation

### `run_sat_assignment_experiment.py`
Factorized-CNF transformer that emits a satisfying assignment; depth sweep + RYS Δ matrices.

```bash
python scripts/constrained_satisfaction/run_sat_assignment_experiment.py --depths 8,16,32
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--depths` | `8,16,32` | Comma-separated transformer depths to sweep. |
| `--n-vars` / `--n-clauses` | `6` / `24` | In-distribution size. |
| `--ood-vars` / `--ood-clauses` | `8` / `34` | OOD size. |
| `--n-heads` / `--dropout` | `4` / `0.0` | Heads / dropout. |
| `--max-repeat` | `2` | Total traversals of each RYS window. |
| `--sat-loss-weight` / `--ce-loss-weight` | `1.0` / `0.0` | Soft-SAT vs canonical-CE loss weights. |
| `--skip-rys` | off | Train + export CKA/baselines without RYS matrices. |

Default-deviation: `seed=41, lr=3e-4`.

### `run_sat_messagepassing_experiment.py`
Clause–variable message-passing solver; depth sweep, validity-vs-rounds, CKA, RYS Δ matrices.
By default uses **synthetic** 3-SAT; pass `--labels-csv` for **RandSATBench** instances.

```bash
python scripts/constrained_satisfaction/run_sat_messagepassing_experiment.py --depths 8,16,32
python scripts/constrained_satisfaction/run_sat_messagepassing_experiment.py --depths 16 --checkpoint <best.ckpt>
python scripts/constrained_satisfaction/run_sat_messagepassing_experiment.py --labels-csv ~/workspace/RandSATBench/datasets/3SAT/train_labels.csv --indist-vars 16,32 --randsat-ood-vars 64
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--depths` | `8,16,32` | Message-passing rounds to sweep. |
| `--n-vars` / `--n-clauses` | `6` / `24` | In-distribution size (synthetic mode). |
| `--ood-vars` / `--ood-clauses` | `8` / `34` | OOD size (synthetic mode). |
| `--labels-csv` | `None` | RandSATBench labels CSV; enables real-instance mode. |
| `--data-root` | labels parent | CNF resolution root (RandSATBench mode). |
| `--indist-vars` / `--randsat-ood-vars` | `16,32` / `64` | Variable-count pools in-dist / OOD (RandSATBench). |
| `--val-frac` / `--test-frac` | `0.1` / `0.1` | Split fractions of the in-dist pool (RandSATBench). |
| `--max-indist` / `--max-ood` | `None` / `None` | Cap on number of examples (RandSATBench). |
| `--num-workers` | `4` | DataLoader workers (RandSATBench). |
| `--pin-memory/--no-pin-memory` | on | Pin host memory (RandSATBench, CUDA only). |
| `--dropout` | `0.0` | Dropout probability. |
| `--max-repeat` | `2` | Total traversals of each RYS window. |
| `--deep-supervision/--no-deep-supervision` | on | Average the soft-SAT loss over every round. |
| `--skip-rys` | off | Train + export curves/CKA without RYS matrices. |
| `--checkpoint` | `None` | Load weights, skip training (single `--depths`). |

Default-deviation: `seed=43`.

---

## Other constraint solvers

### `run_coloring_messagepassing_experiment.py`
Graph k-colouring completion; depth sweep + RYS. **RYS windows need `min_span=2`, so use a depth ≥ 4 to get any windows.**

```bash
python scripts/constrained_satisfaction/run_coloring_messagepassing_experiment.py --depths 8,16,32
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--depths` | `8,16,32` | Message-passing rounds to sweep. |
| `--n-vertices` / `--n-edges` | `12` / `24` | In-distribution graph size. Keep `n` large enough to avoid degenerate single-colour planting. |
| `--n-colors` / `--n-givens` | `3` / `3` | Colours k / revealed vertices. |
| `--ood-vertices` / `--ood-edges` / `--ood-givens` | `18` / `40` / `3` | OOD graph. |
| `--n-heads` / `--max-degree` | `4` / `24` | Heads / max degree embedding index. |
| `--given-ce-weight` | `0.25` | Cross-entropy weight anchoring given vertices. |
| `--skip-rys` | off | Skip RYS matrices. |

Default-deviation: `seed=47`.

### `run_nqueens_messagepassing_experiment.py`
N-Queens completion; depth sweep + RYS (same depth ≥ 4 note as colouring).

```bash
python scripts/constrained_satisfaction/run_nqueens_messagepassing_experiment.py --depths 8,16,32 --n 8
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--depths` | `8,16,32` | Message-passing rounds to sweep. |
| `--n` / `--ood-n` | `8` / `8` | Board size in-dist / OOD (`n ≥ 4`). |
| `--n-givens` / `--ood-givens` | `4` / `2` | Revealed queens in-dist / OOD. |
| `--n-heads` | `4` | Attention heads per round. |
| `--given-ce-weight` | `0.25` | Cross-entropy weight anchoring given rows. |
| `--skip-rys` | off | Skip RYS matrices. |

Default-deviation: `seed=45`.

---

## Stochastic / width-axis SAT

### `run_sat_stochastic_experiment.py`
Deterministic vs stochastic message-passing; single-sample validity, valid@N, coverage, and an RYS probe.

```bash
python scripts/constrained_satisfaction/run_sat_stochastic_experiment.py --depth 16 --modes deterministic,stochastic
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--depth` | `16` | Message-passing rounds (single depth). |
| `--modes` | `deterministic,stochastic` | Modes to compare. |
| `--n-vars` / `--n-clauses` | `6` / `24` | In-distribution size. |
| `--ood-vars` / `--ood-clauses` | `8` / `34` | OOD size. |
| `--n-samples` | `20` | N for valid@N / coverage. |
| `--sigma-floor` / `--sigma-reg` | `0.1` / `1e-2` | Variance-floor target / penalty weight. |
| `--rys-window` | `13,15` | Half-open late window for the RYS probe (`end < depth`). |

Default-deviation: `seed=45`.

### `run_sat_coverage_experiment.py`
Variance-floor vs best-of-K coverage training for the stochastic solver.

```bash
python scripts/constrained_satisfaction/run_sat_coverage_experiment.py --depth 16 --regimes floor,best_of_k
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--depth` | `16` | Message-passing rounds (single depth). |
| `--regimes` | `floor,best_of_k` | Training regimes to compare. |
| `--n-vars` / `--n-clauses` | `6` / `24` | In-distribution size. |
| `--ood-vars` / `--ood-clauses` | `8` / `34` | OOD size. |
| `--n-samples` | `20` | N for valid@N / coverage. |
| `--train-k` | `4` | K trajectories per step for best-of-K / diversity. |
| `--floor-sigma` / `--floor-reg` | `0.7` / `0.2` | Target std / variance-floor weight (`floor` regime). |
| `--diversity-weight` | `0.1` | Per-bit spread reward weight. |

Default-deviation: `seed=46, epochs=18`.

---

## Grokking

### `run_sat_grokking_probe.py`
Memorisation/grokking probe on a tiny fixed train set.

```bash
python scripts/constrained_satisfaction/run_sat_grokking_probe.py --n-train 64 --epochs 4000 --weight-decay 1e-2
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--n-train` / `--n-val` | `64` / `512` | Tiny fixed train set / held-out split. |
| `--n-vars` / `--n-clauses` | `6` / `24` | Instance size. |
| `--n-rounds` | `16` | Message-passing depth. |
| `--epochs` | `4000` | Long, to probe delayed generalisation. |
| `--weight-decay` | `0.0` | 0.0 = pure capacity probe; grokking usually needs e.g. `1e-2`. |
| `--batch-size` | `0` | `0` = full-batch over the tiny set. |
| `--deep-supervision/--no-deep-supervision` | on | Deep supervision. |
| `--variable-id-embeddings/--no-variable-id-embeddings` | off | Break permutation-equivariance (Weisfeiler–Leman ceiling test). |
| `--ce-weight` | `0.0` | Canonical-assignment CE term (rounding-gap test). |

Default-deviation: `seed=50`.

---

## Behavioural notes / gotchas

- **Best-checkpoint vs last:** scripts checkpoint and reload the **best validation**
  epoch. For `run_sat_grokking_probe.py` (which studies *delayed* generalisation) the
  best-val checkpoint may precede late memorisation; `last.ckpt` is also saved if you
  want the final-epoch weights.
- **Validation runs every epoch** under Lightning. `--eval-every` (grokking) and
  `--log-every` (sorted translation) are retained for CLI compatibility but no longer
  change validation cadence.
- **Colouring / N-Queens RYS windows** require depth ≥ 4 (`min_span=2`); at depth 2 the
  RYS sweep has no windows.
- **Classifier data generation** (`run_tiny_sat_experiment.py`,
  `run_rys_accuracy_matrix.py`) needs a high enough clause/variable ratio, otherwise the
  balanced sampler cannot find UNSAT formulas and raises
  `Could not sample target_label=0`.
- **`--checkpoint`** accepts the new Lightning `.ckpt` and legacy `.pt` checkpoints.
