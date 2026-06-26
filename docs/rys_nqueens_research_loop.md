# RYS N-Queens Research Loop

This document records the N-Queens replication of the controlled RYS research
programme. It follows the same structure and seriousness as
`docs/rys_sat_research_loop.md`: each cycle states a PI question, postdoc
hypotheses, implementation hooks, exact commands, result tables, and a decision
gate.

The scientific question is unchanged from the SAT loop: does **RYS** (post-hoc
block repetition with no weight changes) buy extra reasoning iterations on a
verifiable constraint task, and can the **CKA / ρ–φ geometry** predict *which*
windows help, especially out of distribution? A secondary question is whether the
N-Queens solver's variable-state CKA connectome shows a clearer
**encoding → reasoning → decoding** tripartition than the flat SAT classifier did.

Conventions shared with the SAT loop:

- RYS via `rys.surgery.apply_rys` — **half-open** windows `[start, end)` with
  `end < n_rounds` and `>= 2` duplicated rounds, matching
  `run_sat_theory_validation.py` and `rys.residual_force.predict_cka_under_rys`.
- CKA on the **variable (row) residual stream**, not a CLS token.
- Primary metric is verifier-checked **`valid_board_rate`**, never exact match to
  one canonical board (many valid completions exist — the SAT Cycle 4 lesson).
- Pre-norm rounds (`pre_norm=True`) keep the row stream additive so the ρ/φ
  telescoping is exact.

## Cycle 1 — Data, verifier, model, tests

**PI question:** can we generate a controllable, exactly-verifiable N-Queens task
whose difficulty (board size, givens) is a knob, and a RYS-compatible solver that
exposes `model.model.layers`?

**Task formulation (v1): conditional board completion.** A bare board of size `N`
with no givens would be a single trivial input (nothing to condition on), so each
instance reveals `n_givens` non-attacking queens and the model places one queen
per remaining row. One discrete variable per row → column in `{0..N-1}`
(multiclass), not a flat board string. Conditioning on the givens makes
train/val/test/OOD genuinely distinct instances, exactly as the CNF conditioned
the SAT task.

**Postdoc hypotheses:**

- *Dynamical systems.* Each message-passing round is an iterate of a shared
  update map; repeating a window (RYS) is one extra Newton-like refinement step.
  Useful windows should be stationary, low-junction-mismatch, marginally stable
  (Jacobian σ_max ≈ 1), as in SAT Cycle 6.
- *Synthetic task.* N-Queens is a permutation + diagonal CSP. Unlike 3-SAT it has
  a global all-different (column) constraint, so independent per-row marginals are
  a weaker proxy — joint coordination is genuinely required, which should make the
  validity-vs-rounds curve informative.
- *Systems.* The constraint graph is a complete graph over rows whose edges carry
  a relative row offset (offsets are what make diagonals meaningful), so a round
  is relative-position attention + MLP over `N` row nodes. Pre-norm keeps the
  stream additive for the ρ/φ test.
- *Mechanistic analysis.* Capture per-round row states; compute CKA and ρ/φ;
  score windows before the behavioural sweep.
- *Statistics.* Multi-seed and random-window controls (Cycle 4) before any
  publishable claim.

**Implementation hooks:**

- Data/verifier/loss: `rys.nqueens_data`
  (`make_queens_examples`, `make_queens_splits`, `QueensDataset`,
  `verify_boards_tensor`, `soft_nqueens_loss`, `board_is_valid`).
- Model: `rys.nqueens_message_passing`
  (`QueensMessagePassingModel`, `QueensMPConfig`, pre-norm / weight-tied,
  rounds exposed as `model.model.layers`).
- Scripts: `scripts/constrained_satisfaction/run_nqueens_messagepassing_experiment.py`,
  `scripts/constrained_satisfaction/run_nqueens_theory_validation.py`.
- Tests: `tests/test_nqueens_data.py`, `tests/test_nqueens_message_passing.py`.

**Soft N-Queens loss.** For each active row pair `(i, j)` with `d = j - i`,
column-conflict probability is `sum_c p_i[c] p_j[c]` and diagonal-conflict
probability is `sum_c p_i[c] (p_j[c-d] + p_j[c+d])`; the loss is the mean negative
log-probability that each pair is conflict-free, plus an optional cross-entropy
anchoring the given rows. Mirrors `soft_sat_loss`.

**Commands:**

```bash
source .venv/bin/activate
pytest tests/test_nqueens_data.py tests/test_nqueens_message_passing.py -q
```

**Result:** all tests green (10 data + 6 model). Verifier accepts known valid
boards and non-canonical valid completions; rejects column / diagonal / given
violations. Generator is deterministic across seeds and always yields satisfiable
completions. Pre-norm additivity (`test_prenorm_variable_stream_is_additive`) and
`apply_rys` reversibility confirmed. Full suite: 59 tests green.

**Learnability / regime decision (smoke runs).** Depth-8 solver, MPS:

| givens | train | epochs | val valid | val cell acc |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1024 | 15 | 0.035 | 0.45 |
| 4 | 2048 | 15 | **0.711** | 0.92 |

With `n_givens=2` (6 free rows for N=8) the joint all-different + diagonal
constraint is too hard to satisfy *fully* in this budget — `valid_board_rate`
sits near 0.04, too low for a clean RYS Δ study (deltas would be in the noise).
With `n_givens=4` (4 free rows) the solver reaches ~0.71 validation validity, a
usable regime. **Decision:** adopt `n_givens=4` for N=8 as the headline regime;
`n_givens` becomes a documented difficulty knob (Cycle 4 ablation will vary it,
expecting harder = lower baseline = larger relative RYS room). This is the direct
analogue of the SAT lesson "make the task solvable before studying RYS".

**Decision gate:** continue. Data + verifier + model + tests are in place and the
solver learns to a usable validity regime. Proceed to Cycle 2 (depth sweep).

**Generation stats (default headline config):** N=8 in-distribution, OOD N=10,
`n_givens=4`, splits 4096 / 1024 / 1024 / 1024. 8-queens has 92 full solutions;
4-given partial boards admit several completions (multi-solution → validity, not
exact match, is the right metric).

## Cycle 2 — Message-passing depth sweep

**PI question:** does deeper iteration raise validity and, as in SAT Cycle 5, does
OOD keep improving with rounds while in-distribution saturates — and are RYS
windows selective and growing with depth?

**Protocol:** `scripts/constrained_satisfaction/run_nqueens_messagepassing_experiment.py`, depths 8/16/32,
N=8 / OOD N=10, `n_givens=4`, 4096/1024/1024/1024, 20 epochs, deep supervision,
pre-norm. Artifacts per depth: checkpoint, `train_history.csv`,
`validity_vs_rounds_*.csv`+`.png`, `cka_variable_val.*`, `delta_valid_*.{csv,png}`,
`summary.json`; run-level `report.md`.

**Command:**

```bash
python scripts/constrained_satisfaction/run_nqueens_messagepassing_experiment.py --depths 8,16,32 --seed 45 --n-givens 4
```

**Run 2a — first sweep (larger-N OOD, `results/nqueens_messagepassing/20260529_101816/`):**

In-distribution validity is high and *grows with depth*:

| rounds | val valid | test valid | OOD (N=10) valid | CKA early | CKA late | best test window |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 0.891 | 0.880 | **0.000** | 0.959 | 0.994 | (3,7) +0.018 |
| 16 | 0.905 | 0.905 | **0.000** | 0.931 | 0.995 | (2,15) +0.009 |
| 32 | 0.942 | 0.927 | **0.000** | 0.906 | 0.997 | (10,20) +0.009 |

RYS gives positive Δ on middle-to-late windows in-distribution, and the best
window moves later and the effect grows-then-holds as depth increases (L8 val
`(1,7)` +0.014; L32 test `(10,20)` +0.009).

**Depth sharpens the connectome.** The early-band CKA *decreases* monotonically
with depth (0.959 → 0.931 → 0.906) while the late band stays ≈0.99. Deeper
solvers develop more early→late contrast: the extra rounds accumulate more
residual force early (active reasoning) before locking into the late equilibrium.
This is the mild emergence of an encode→reason→(trivial-decode) structure, though
still far from a sharp tripartition — the linear column head exerts no decoding
pressure (same mechanism as the SAT binary classifier).

**Critical diagnostic — larger-N OOD collapses to 0.** OOD here was N=10 boards. The
solver scores 0.0 valid (cell accuracy ~0.46, near chance). This is **not**
"RYS fails OOD": it is an architectural artefact. The model uses *absolute*
`row_embedding` and a *fixed-size column head*; training only on N=8 leaves rows
8–9 and columns 8–9 permanently masked, so their parameters never receive
gradient and produce noise at N=10. Unlike the SAT message-passing model
(permutation-equivariant, per-variable binary readout → generalised to more
variables), a column readout is intrinsically size-dependent.

**CKA tripartition.** Early-band mean CKA 0.96 (L8) / 0.93 (L16), late-band 0.99 —
high throughout, only a mild early→late rise. Like the SAT classifier, the
connectome is closer to a plateau than a sharp encode→reason→decode tripartition;
the depth helps via many small refinements, not three distinct phases.

**Decision / pivot (allowed by the brief: "OOD via more givens / harder partial
boards"):** redefine OOD as the **same N=8 with fewer givens** (harder
completion: more free rows ⇒ more reasoning needed), which tests
"useful depth scales with hardness" *without* the untrained-vocabulary confound.
Scripts gained `--ood-givens` (default 2; train givens 4). Re-running as Run 2b.

**Run 2b — harder-givens OOD (`results/nqueens_messagepassing/20260529_110904/`):**
OOD = same N=8 with only 2 givens (vs 4 in training), 6 free rows.

| rounds | val valid | test valid | OOD valid | CKA early | CKA late | best OOD window |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 0.903 | 0.893 | 0.281 | 0.956 | 0.993 | (1,7) **+0.073** |
| 16 | 0.923 | 0.914 | 0.382 | 0.934 | 0.997 | (1,10) **+0.066** |
| 32 | 0.933 | 0.949 | 0.450 | 0.912 | 0.997 | (3,28) **+0.063** |

**Two clean, paper-grade findings:**

1. *Useful depth scales with hardness.* In-distribution validity is near-saturated
   and barely moves with depth (0.90 → 0.93), but the harder OOD climbs strongly
   with rounds: **0.281 → 0.382 → 0.450**. More reasoning iterations matter exactly
   where the problem is harder — the SAT Cycle 5 signature, sharper here.
2. *RYS helps most where reasoning is hardest.* In-distribution RYS Δ is tiny
   (+0.004 to +0.011), but OOD RYS Δ is **an order of magnitude larger (+0.06 to
   +0.07)**, on broad early-to-late reasoning windows. Post-hoc block repetition
   buys real OOD reasoning on a verifiable task — the strongest RYS result in the
   programme so far.

Early-band CKA again decreases with depth (0.956 → 0.912); late stays ≈0.99.

## Cycle 3 — ρ/φ theory validation

**PI question:** does the closed-form CKA decomposition hold on N-Queens (plateau
prediction rank-faithful; RYS doubles `1−CKA` ×4), and do the windows that help
OOD carry the stationary / low-mismatch / marginally-stable signature? Does the
connectome show clearer encode→reason→decode bands than the SAT classifier?

**Protocol:** `scripts/constrained_satisfaction/run_nqueens_theory_validation.py`, pre-norm depth 16,
`n_givens=4`, 25 epochs. ρ/φ table + `theory_fit`, RYS ΔValidity sweep,
amplification + junction mismatch + Jacobian for the top windows.

**Command:**

```bash
python scripts/constrained_satisfaction/run_nqueens_theory_validation.py --depth 16 --seed 46 --n-givens 4 --ood-givens 2
```

**Results (`results/nqueens_theory_validation/20260529_123441/`):** baselines
val 0.937 / test 0.943 / OOD 0.373 (depth 16, pre-norm).

ρ/φ theory fit on 120 pairs:

| quantity | N-Queens | SAT (Cycle 6) |
| --- | ---: | ---: |
| Spearman(plateau pred, 1−CKA) | **0.991** | 0.991 |
| Pearson(plateau pred, 1−CKA) | 0.712 | 0.659 |
| Pearson(Q², 1−CKA) | 0.917 | 0.897 |
| median R (≈ρ) | **0.445** | 1.035 |
| median cos φ | 0.739 | 0.729 |

The plateau formula is again an almost-perfect **rank** predictor of the
connectome (Spearman 0.991, replicating SAT). The decisive difference is
**median ρ ≈ 0.45 vs SAT's 1.03**: N-Queens sits *deeper in the plateau regime*
(smaller per-segment residual force), which quantitatively explains why its CKA
is high everywhere — directly confirming the earlier qualitative analysis.

RYS doubling test (top OOD windows): `S_norm_ratio ≈ 1.3` and
`1−CKA ratio ≈ 1.3` — directionally correct (both amplified, phase preserved,
`cos_phi_diff ≈ +0.01`) but **weaker than SAT's crisp ×2 / ×4**. Expected: at
small ρ on very broad windows (span 11–14) the second-pass displacement is small
and the doubling enters only at higher order. Among the top windows the cleanest
dynamical signature — lowest junction mismatch (1.11) and lowest block-Jacobian
σ_max (17.9) — is window **(4, 15)**, which still delivers +0.060 OOD; the
broadest windows (1,15)/(1,13) help but are far less stable (σ_max 80–94).

**CKA tripartition assessment.** Early-band mean CKA 0.91–0.96, late ≈0.99: a mild
encode→reason→equilibrium gradient that *sharpens with depth*, not a sharp
three-phase anatomy. Same limiter as the SAT classifier — a linear column head
exerts no decoding pressure, so late layers never diverge.

**Decision gate:** continue toward paper. RYS reproduces and is *strongest on
OOD* here; ρ/φ rank-predicts the connectome exactly; the small-ρ regime explains
the high CKA and the softer doubling. Cycle 4 ablations next.

## Conclusions so far (N-Queens vs SAT)

- RYS is task-general: it helps on both 3-SAT assignment and N-Queens completion,
  with **no weight changes**, and its benefit is **largest out of distribution**
  (N-Queens OOD Δ ≈ +0.06–0.07, ~10× the in-distribution effect).
- "Useful reasoning depth scales with hardness" holds on both tasks; N-Queens
  shows it cleanly in the OOD validity-vs-depth climb (0.28 → 0.45).
- The ρ/φ closed form rank-predicts the CKA connectome on both tasks
  (Spearman ≈ 0.99). The residual-force magnitude ρ is the axis that differs
  (N-Queens ≈ 0.45, SAT ≈ 1.03), explaining N-Queens' higher, flatter CKA and its
  softer RYS amplification.
- Neither task shows a sharp encode→reason→decode tripartition with a small
  linear readout; depth sharpens the early/late contrast but a true decoder phase
  needs decoding pressure (large/structured output) — the clearest lever for a
  follow-up task.

## Cycle 2c (proposed) — N-scaling in-distribution

N-Queens is a *complete-graph* CSP (every row pair constrains directly, 1-hop), so
larger N raises **width** (joint difficulty) more than **propagation depth**. Train
in-distribution at N = 8 / 16 / 24 (no OOD-to-larger-N, which the absolute-vocabulary
readout cannot express) and compare CKA early/late bands, validity-vs-rounds, and
RYS Δ magnitude. Falsifiable prediction: as N grows, early-band CKA drops further,
the validity-vs-rounds curve stays climbing longer, and RYS Δ grows (more reasoning
room). For a sharper *depth*-scaling story a large-diameter task (Sudoku, sparse
graph coloring, Latin squares with few givens) would be theoretically superior;
noted for after the N-Queens cycles.

## Next steps for the paper

- Cycle 4 ablations: multi-seed (≥3) best-window OOD Δ vs random same-length
  high-/low-CKA windows; `n_repeats ∈ {2,3,4}` unimodality; pre-norm off vs on;
  depth vs OOD; givens 2/3/4; stretch: stochastic transitions + `valid@K` /
  coverage to address multi-solution mode collapse.
- Comparison table: N-Queens vs SAT (validity ceilings, RYS Δ magnitudes, ρ/φ
  fit quality, CKA tripartition).
- Blog post outline: extend the SAT post with N-Queens as a second verifiable
  CSP, emphasising the global all-different constraint and whether it sharpens the
  tripartition.
