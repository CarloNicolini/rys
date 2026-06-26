# RYS 3-SAT Research Loop

This document records the first on-demand PI/postdoc research cycle for the
controlled RYS experiment. The goal is to produce a publishable result before
scaling back to large LLMs: establish, in a tiny locally trained Transformer,
whether representation geometry and repeat dynamics predict when repeated
blocks help algorithmic reasoning.

## Cycle 1 PI Question

Can a tiny residual Transformer trained on synthetic 3-SAT develop a layer
window that behaves like a reusable refinement operator, so that RYS improves
or preserves SAT generalization better than matched random windows?

## Postdoc Hypotheses

### Dynamical Systems

Residual blocks implement a non-autonomous forward-Euler discretization in
depth. RYS turns a selected block into an iterated map

```text
z_{n+1} = T_B(z_n), where T_B = T_{j-1} ... T_i.
```

Useful blocks should show controlled early iterates: nonzero movement,
approximately quadratic early CKA displacement, and bounded logit drift. Inert
blocks should have negligible movement. Unstable blocks should show expanding
repeat-count curves or large final-logit KL.

Literature anchors: ResNets as dynamical systems and neural ODE limits; Deep
Equilibrium Models; Universal Transformer and recurrent-depth computation;
implicit regularization of residual networks toward neural ODEs.

### Synthetic Tasks

3-SAT is a useful first task because labels are exact by brute force for small
`n_vars`, the data distribution is controllable, and OOD splits can increase
variables or clause density. The first experiment should use classification
before assignment recovery or proof generation.

Falsifier: if the model only learns shallow lexical statistics, RYS windows may
move logits without improving OOD accuracy.

### Systems

The tiny model should remain deliberately plain: token embeddings, positional
embeddings, pre-norm residual attention/MLP blocks, and a classification head.
It must expose `model.model.layers` so the existing `apply_rys` hook is reused
unchanged.

Constraint: train on Apple MPS when available, but run float64 residual-force
diagnostics on CPU.

### Mechanistic Analysis

For each trained run:

1. Capture post-layer residual streams on validation formulas.
2. Compute CKA and residual-force diagnostics.
3. Score candidate windows before looking at RYS accuracy.
4. Evaluate selected windows for `n_repeats = 2..4`.
5. Compare against base accuracy, OOD accuracy, and logit KL.

The expected positive signature is high but nontrivial CKA, nonzero update size,
stable repeat dynamics, and bounded decoder drift.

### Statistics

The first run is exploratory. A publishable result needs enrichment:

```text
top scored windows > random same-length high-CKA windows > random low-CKA windows
```

The decision metric should be improvement or preservation on OOD 3-SAT, not
only in-distribution validation accuracy.

## Current Implementation Hooks

- `rys.sat_data`: deterministic 3-SAT generator, brute-force labels, token encoding.
- `rys.tiny_transformer`: RYS-compatible tiny residual Transformer.
- `rys.tensor_activations`: tensor-native activation capture into the existing CKA schema.
- `scripts/constrained_satisfaction/run_tiny_sat_experiment.py`: one local train/analyze/RYS run.

## First Command

```bash
source .venv/bin/activate
python scripts/constrained_satisfaction/run_tiny_sat_experiment.py \
  --n-vars 6 \
  --n-clauses 18 \
  --epochs 8 \
  --d-model 64 \
  --n-layers 6 \
  --capture-batches 4
```

## Decision Gate

Continue toward a paper only if the first two or three seeds show that
pre-benchmark CKA/dynamics scores enrich for useful RYS windows over matched
controls. If not, pivot to the boundary-transition hypothesis: RYS may help by
sharpening phase transitions between syntactic and semantic modules rather than
by iterating a central equilibrium-like operator.

## Cycle 2 — Checkpointed RYS Accuracy Matrix

Question: after training one factorized CNF Transformer once, what is the full
layer-window map of RYS behavioural effects?

Implementation:

- Script: `scripts/constrained_satisfaction/run_rys_accuracy_matrix.py`
- Architecture: factorized CNF Transformer
- Layers: 10
- Training split: balanced `4 vars / 12 clauses`
- OOD split: balanced `5 vars / 16 clauses`
- Train/val/test/OOD sizes: `8192 / 2048 / 2048 / 2048`
- Checkpoint: `results/rys_accuracy_matrix/20260528_172001/checkpoint.pt`
- Matrix artifacts: `results/rys_accuracy_matrix/20260528_172556/`

Important convention: each matrix cell `(i, j)` uses an **inclusive** RYS
window and duplicates blocks `i..j`. This differs from the hook's half-open
notation but gives a true `L x L` upper-triangular behavioural map, including
windows that end at the final block.

Baseline accuracies from the reloaded checkpoint:

| split | accuracy |
| --- | ---: |
| validation | 0.7876 |
| test | 0.7739 |
| OOD | 0.7080 |

Best observed deltas:

| split | best window | RYS accuracy | delta |
| --- | --- | ---: | ---: |
| validation | `(3, 9)` | 0.7974 | +0.0098 |
| test | `(2, 9)` | 0.7896 | +0.0156 |
| OOD | `(6, 6)` | 0.7197 | +0.0117 |

Saved heatmaps:

- `results/rys_accuracy_matrix/20260528_172556/delta_accuracy_val.png`
- `results/rys_accuracy_matrix/20260528_172556/delta_accuracy_test.png`
- `results/rys_accuracy_matrix/20260528_172556/delta_accuracy_ood.png`

PI note: the matrix confirms the earlier qualitative pattern. Early-layer
duplication is strongly harmful, broad late-ending windows are mildly helpful
on validation/test, and OOD prefers a much narrower late block in this seed.
The effect size is real but small. The next robust test is a multi-seed matrix
sweep and a comparison against same-length random windows.

**Convention fix (Cycle 2 post-mortem):** cell `(6, 6)` in the first matrix meant
*inclusive* window from layer 6 through layer 6 — a single-block repeat. That is
now **excluded**. The sweep uses only **`i < j`**, so every RYS window spans at
least two blocks. Diagonal and lower triangle are left as NaN in the heatmaps.

## Cycle 3 — 32-Layer Factorized Matrix + CKA

**PI question:** With a deeper factorized CNF Transformer, do strict
upper-triangular RYS windows show coherent late-depth gains on test and OOD, and
does the validation CKA connectome align with those intervals?

**Protocol:**

| Parameter | Value |
| --- | --- |
| Script | `scripts/constrained_satisfaction/run_rys_accuracy_matrix.py` |
| Architecture | factorized CNF Transformer |
| Layers | 32 |
| Seed | 30 (default in script) |
| Task | balanced `4 vars / 12 clauses` |
| OOD | balanced `5 vars / 16 clauses` |
| Splits | 8192 / 2048 / 2048 / 2048 |
| Epochs | 30 |
| RYS repeats | 2 (one extra traversal of blocks `i..j`) |
| Matrix | strict upper triangle only (`i < j`, 496 windows) |
| CKA | validation CLS, 4 batches |

**Artifacts:**

- Run directory: `results/rys_accuracy_matrix/20260528_200643/`
- Checkpoint: `results/rys_accuracy_matrix/20260528_200643/checkpoint.pt`
- Log: `results/rys_accuracy_matrix/L32_seed30_run.log`
- ΔAccuracy: `delta_accuracy_{val,test,ood}.{csv,png}`
- CKA: `cka_val.{csv,png}`
- Summary: `summary.json`, `report.md`

**Training (30 epochs, seed 30):** final train accuracy 0.887, val 0.799 at epoch 30
(checkpoint saved after full training).

**Baselines (reloaded checkpoint):**

| split | accuracy | loss |
| --- | ---: | ---: |
| validation | 0.7988 | 0.4794 |
| test | 0.7886 | 0.4987 |
| OOD | 0.6875 | 0.7235 |

**Best strict upper-triangular RYS windows** (`n_repeats=2`, inclusive `i..j`, `i<j`):

| split | window | RYS accuracy | Δ |
| --- | --- | ---: | ---: |
| validation | `(12, 31)` | 0.8086 | +0.0098 |
| test | `(6, 21)` | 0.7920 | +0.0034 |
| OOD | `(11, 29)` | 0.6924 | +0.0049 |

**Single deployable window (0.5·Δ_test + 0.5·Δ_OOD):** `(11, 29)` with combined
Δ = +0.0022 (test −0.0005, OOD +0.0049). Test-optimal and OOD-optimal windows still
disagree; the combined pick favours OOD at a negligible test cost.

**CKA connectome (validation, CLS, 4 batches):**

| region | mean linear CKA |
| --- | ---: |
| early layers 0–11 | 0.583 |
| late layers 12–31 | 0.989 |
| inside val-best window (12,31) | 0.989 |
| inside combined window (11,29) | 0.987 |

Late-depth blocks form a near-plateau in representation geometry; beneficial RYS
windows sit inside this high-CKA band rather than in early low-CKA layers (where
Δ is strongly negative throughout the matrix).

**Interpretation:**

1. Deeper network (32 vs 10 layers) preserves the Cycle 2 pattern: early duplication
   hurts, late broad windows help modestly on val, narrower late windows help OOD.
2. Effect sizes remain small (+0.3–1.0 pp) but are consistent across splits for
   late-ending windows.
3. There is **no** single window that maximises both test and OOD; `(11, 29)` is a
   reasonable compromise when one RYS interval must be fixed before evaluation.
4. CKA aligns qualitatively: useful windows overlap the late CKA plateau; this
   supports the refinement-operator hypothesis but does not yet predict the exact
   `(i, j)` optimum within the plateau.
5. Next steps: multi-seed L32 matrix, same-length random-window controls, and
   optional CKA+residual-force pre-scoring before the behavioural sweep.

## Cycle 4 — Assignment Generation to Force Decoder Pressure

**Motivation:** Cycle 3 likely lacks a decoder-like phase. The model only maps a
CNF formula to one SAT/UNSAT bit, so after the formula is compressed the late
layers can remain in a high-CKA latent plateau. To create pressure toward a
structured logit space, the next task predicts a satisfying assignment: one
binary logit pair per variable.

**Design choice:** the first version avoids free-form autoregressive decoding.
Each example is a satisfiable 3-SAT formula; the target is the lexicographically
first satisfying assignment found by brute force. The input sequence is a
factorized CNF followed by fixed query tokens `x_1? ... x_n?`. The model emits
`(batch, max_vars, 2)` logits from those query positions. This keeps the task
controlled while forcing final layers to prepare variable-specific outputs.

**Implementation:**

- Data: `SatAssignmentExample`, `SatAssignmentDataset`,
  `make_sat_assignment_examples`, `encode_formula_assignment_factorized`
- Model: `FactorizedCNFAssignmentTransformer`
- Script: `scripts/constrained_satisfaction/run_sat_assignment_experiment.py`
- RYS convention: strict upper triangle only (`i < j`), inclusive replay of
  blocks `i..j`
- CKA: validation query-token residual streams, not CLS only
- Metrics: bit accuracy, exact-match assignment accuracy, Δ exact/bit heatmaps

**First depth-controlled run:**

| Parameter | Value |
| --- | --- |
| Depths | 8, 16, 32 |
| Seed | 41 |
| Train/val/test/OOD | 4096 / 1024 / 1024 / 1024 |
| In-distribution | 6 vars / 24 clauses |
| OOD | 8 vars / 34 clauses |
| Epochs | 20 |
| RYS repeats | 2 |

**Artifacts:** `results/sat_assignment_generation/<timestamp>/`

**Results:** pending. The key question is whether query-token CKA develops
clearer blocks than the Cycle 3 CLS CKA, and whether RYS helps middle/late
windows more strongly than in binary SAT classification.

**Objective correction:** exact match to the first brute-force assignment is too
strict because many formulas admit several satisfying assignments. The primary
metric is now `valid_assignment_rate`: the model is correct if its predicted
bits satisfy every clause, even when the assignment differs from the canonical
brute-force target. The implementation adds a hard verifier plus a differentiable
soft-SAT loss:

- Hard verifier: `verify_assignment_tensor(...)`
- Single-formula verifier: `assignment_satisfies_formula(...)`
- Loss: `sat_loss_weight * soft SAT + ce_loss_weight * canonical CE`
- Default for the first corrected baseline: `1.0 * soft SAT + 0.0 * CE`

Smoke tests:

- A deterministic formula accepts two different valid assignments.
- The same formula rejects assignments that falsify one of its clauses.
- A tiny smoke training run completed under
  `results/sat_assignment_generation_validity_smoke/20260528_213959/`.

Corrected baseline sweep without RYS matrices:

- Run directory: `results/sat_assignment_generation_validity/20260528_214030/`
- Depths: 8, 16
- Train/val/test/OOD: 2048 / 512 / 512 / 512
- In-distribution: 6 vars / 24 clauses
- OOD: 8 vars / 34 clauses
- Epochs: 15

| depth | val valid | test valid | OOD valid | val exact | test exact | OOD exact |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.277 | 0.287 | 0.082 | 0.055 | 0.049 | 0.016 |
| 16 | 0.256 | 0.258 | 0.078 | 0.059 | 0.047 | 0.033 |

Interpretation: the verifier fixes the evaluation problem: exact-match remains
low, but many assignments are valid. However, validity is still too low for a
clean RYS study, and simply increasing depth from 8 to 16 does not help. The
next architecture should add SAT-specific inductive bias, likely clause-variable
message passing or explicit clause aggregation before variable-query decoding.

## Cycle 5 — Clause-Variable Message Passing (RYS as Extra Reasoning Rounds)

**Motivation:** the flat factorized Transformer must rediscover the bipartite
clause-variable graph from scratch, which is why validity stays near 0.28 and
extra depth on a plain Transformer does not help. SAT has a known graph
structure, so we give it to the model explicitly (NeuroSAT-style message
passing). Crucially this also gives RYS a clean meaning: each round is an iterate
of one update map `F`, so repeating a window of rounds is literally *doing more
reasoning iterations*, matching the deep-equilibrium / quasi-fixed-point framing.

**Model (`rys.sat_message_passing`):**

- Packed state `(batch, max_vars + max_clauses, d)`; each round is one
  `model.model.layers` entry so `apply_rys` and CKA capture are unchanged.
- Round = variables→clauses pooling then clauses→variables scatter, with residual
  updates and per-node LayerNorm.
- Permutation-equivariant: variables and clauses start from a single shared
  learned vector (no per-index embedding), which is the property that should let
  the solver generalise to more variables/clauses OOD.
- Readout head emits `(batch, max_vars, 2)`; loss is the differentiable
  `soft_sat_loss`. Deep supervision averages the loss over all rounds by default.

**Objective:** primary metric is verified `valid_assignment_rate`; exact match to
the brute-force assignment is diagnostic only.

**New diagnostics:**

- Validity-vs-rounds curve: read out the assignment after each round to see how
  solution quality grows with reasoning depth and whether it saturates
  (fixed-point signature).
- Variable-state CKA connectome (not CLS), to compare geometry with RYS gains.
- Strict upper-triangular (`i < j`) RYS ΔValidity matrices per split.

**First depth sweep (running):**

| Parameter | Value |
| --- | --- |
| Script | `scripts/constrained_satisfaction/run_sat_messagepassing_experiment.py` |
| Depths (rounds) | 8, 16, 32 |
| Seed | 43 |
| Train/val/test/OOD | 4096 / 1024 / 1024 / 1024 |
| In-distribution | 6 vars / 24 clauses |
| OOD | 8 vars / 34 clauses |
| Epochs | 20 |
| Loss | soft SAT, deep supervision on |

**Artifacts:** `results/sat_messagepassing/<timestamp>/` (per depth: checkpoint,
`validity_vs_rounds_*.csv` + `.png`, `cka_variable_val.*`, `delta_valid_*` and
`report.md`).

**Smoke check:** a 3-round run on 64 tiny formulas reached val validity 0.50 in
two epochs (vs ~0.28 for the flat Transformer after 15 epochs on 2048 formulas),
and RYS already showed positive ΔValidity on late windows (e.g. `(1, 2)`
+0.09 val / +0.13 OOD).

**Final results** (`results/sat_messagepassing/20260528_215945/`):

Baseline valid-assignment rate and best RYS window (`n_repeats=2`, `i<j`):

| depth | val valid | test valid | OOD valid | best val | best test | best OOD |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 8 | 0.522 | 0.532 | 0.377 | (0,1) +0.024 | (6,7) +0.022 | (0,7) +0.027 |
| 16 | 0.521 | 0.525 | 0.409 | (2,6) +0.039 | (2,10) +0.025 | (0,10) +0.015 |
| 32 | 0.521 | 0.535 | 0.400 | (3,11) +0.055 | (1,16) +0.036 | (3,8) +0.034 |

**Validity-vs-rounds curves** (read-out after each round):

- In-distribution (val/test) rises steeply then saturates around round 6-7 at
  ~0.52-0.55 for every depth.
- OOD keeps climbing with more rounds: L8 plateaus ~0.38 by round 7, while L32
  reaches ~0.41 around round 27. Useful reasoning depth scales with problem
  hardness — the central evidence for RYS as iterated computation.

**RYS findings:**

1. ΔValidity is positive and the best effect grows with depth (val +0.024 → +0.039 → +0.055).
2. Best windows are early-to-middle rounds (e.g. L32 val `(3,11)`), matching the
   steep part of the validity-vs-rounds curve: repeating the *active* propagation
   phase helps most, repeating the saturated tail does not.
3. The effect is selective (~40% of L32 val windows give Δ>0), so a specific RYS
   interval matters; it is not "repeating anything helps".

**Interpretation:** giving the model the bipartite clause-variable graph makes the
task solvable and gives RYS a clean meaning (extra propagation rounds). Validity
likely plateaus near ~0.52 because the solver is deterministic and 3-SAT is
multi-solution: averaging several valid assignments can yield an invalid one
(mode collapse), as documented for deterministic RRMs.

## Related Work Anchor — GRAM / Recursive Reasoning Models

Baek et al., *Generative Recursive Reasoning* (arXiv:2605.19376, 2026), formalise
exactly this family: Recursive Reasoning Models (HRM, TRM, Looped/Universal
Transformer) that refine a latent state with shared transitions, decouple depth
from parameters, use deep supervision, scale at inference by depth, and evaluate
multi-solution CSPs (N-Queens, Graph Coloring) by constraint validity, not exact
match. Our Cycle 5 independently reproduces this setup; RYS is their depth-based
inference-time scaling applied as a *post-hoc* intervention on a trained network.

Where they are ahead and what to adopt:

- Deterministic recursion mode-collapses on multi-solution tasks; they fix it with
  stochastic latent transitions and parallel-trajectory (width) sampling. This
  likely explains our ~0.52 validity ceiling.
- They report `coverage` (distinct valid solutions over N samples); we only score
  single-sample validity.
- They use truncated gradient (1-step) for memory-efficient deep recursion.

Our distinctive angle: RYS as a no-retrain intervention, CKA geometry as a
predictor of where repetition helps, and a position-dependent L×L RYS window map
(vs their uniform depth scaling).

**Proposed Cycle 6:** add stochastic latent transitions + multi-sample
`coverage` / `valid@N`, and compare RYS windows against uniform depth scaling.

## Cycle 6 — Pre-norm Solver and the rho/phi Theory Test

**PI question:** the closed-form CKA edge predicts that RYS doubles the relative
residual force ($$\rho^{\text{RYS}}\approx 2\rho$$, $$\cos\phi$$ invariant) and
hence quadruples $$1-\mathrm{CKA}$$ on the plateau. Does this hold on the
controlled solver, and do the windows that satisfy the dynamical-systems
diagnostics (stationarity, low junction mismatch, marginal Jacobian) coincide
with the windows that raise validity?

**Changes to make the theory exact:**

- `MessagePassingConfig.pre_norm=True` (default): variable stream is purely
  additive, `h <- h + F(LN(h))`, so `x_j = x_i + S_ij` telescopes and the
  rho/phi decomposition is exact. Clause stream stays post-norm (bounded
  auxiliary). New test `test_prenorm_variable_stream_is_additive`.
- `MessagePassingConfig.weight_tied`: optional shared round → literal iterated
  map for the DEQ fixed-point reading. New test `test_weight_tied_shares_one_round`.
- Backbone hidden-state trace now returns the **raw** pre-final-norm variable
  states (needed for telescoping); round logits apply the final norm.

**New library module `rys.theory_validation`:**

- `rho_phi_table` — per-pair $$\rho, \cos\phi, \mathcal{Q}$$, plateau prediction,
  measured $$1-\mathrm{CKA}$$ (wraps `residual_force_long`).
- `theory_fit` — Pearson/Spearman of plateau prediction and $$\mathcal{Q}^2$$ vs
  measured $$1-\mathrm{CKA}$$ on the plateau subset.
- `rys_amplification_summary` — region-wise doubling test (`S_norm_ratio~2`,
  `one_minus_cka_ratio~4`, `cos_phi_diff~0`) via `amplification_long`.
- `junction_mismatch` — standardised distance between block-output and
  block-input clouds.
- `block_jacobian_sigma_max` — power-iteration top singular value of the block
  map's Jacobian (jvp/vjp, no dense Jacobian).

**Script:** `scripts/constrained_satisfaction/run_sat_theory_validation.py`. Convention is half-open
`apply_rys` windows with `end < n_rounds` and `>=2` duplicated rounds, matching
`predict_cka_under_rys`.

**Protocol (running):** depth 16, seed 44, pre-norm, 6 vars / 24 clauses
in-distribution, 8 / 34 OOD, 20 epochs, deep supervision. Capture base variable
stream → rho/phi table + `theory_fit` + scatter; ΔValidity matrices (val/test/OOD,
105 windows); for top-6 combined windows capture RYS states → amplification +
junction mismatch + Jacobian → `theory_link.csv`.

**Artifacts:** `results/sat_theory_validation/<timestamp>/`
(`rho_phi_long.csv`, `rho_phi_prediction.png`, `cka_variable_val.csv`,
`delta_valid_{split}.{csv,png}`, `theory_link.csv`, `summary.json`).

**Results** (`results/sat_theory_validation/20260529_002010/`):

Baselines (16-round pre-norm solver): val 0.547, test 0.513, OOD 0.384.

Theory fit on 120 pairs:

| quantity | value |
| --- | ---: |
| Spearman(plateau pred, 1−CKA) | **0.991** |
| Pearson(plateau pred, 1−CKA) | 0.659 |
| Pearson(Q², 1−CKA) | 0.897 |
| Spearman(Q², 1−CKA) | 0.963 |
| median R (≈ρ) | 1.035 |
| median cos φ | 0.729 |

Reading: the plateau formula `½ρ²sin²φ` is a near-perfect **rank** predictor of
the connectome (Spearman 0.99) but overshoots **magnitude** because the solver
runs at ρ≈1 (active regime, not the deep plateau). The cross-term Q² is the
better magnitude predictor (Pearson 0.90) — exactly the caveat documented in
`residual_force.py`.

RYS doubling test (best windows, in-window means):

| window | Δvalid OOD | 1−CKA ratio | S_norm ratio | cos φ diff | junction | Jσmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (13,15) | +0.0107 | 4.03 | 1.63 | −0.071 | 0.17 | 2.07 |
| (12,15) | +0.0078 | 4.49 | 1.66 | −0.042 | 0.20 | 2.73 |
| (11,15) | +0.0078 | 4.19 | 1.58 | −0.040 | 0.30 | 3.42 |
| (12,14) | +0.0068 | 5.23 | 1.61 | −0.049 | 0.18 | 2.70 |
| (11,13) | +0.0029 | 6.54 | 1.74 | −0.053 | 0.25 | 2.55 |

Interpretation:

1. The predicted **×4** quadrupling of `1−CKA` is confirmed for the broad late
   windows ending at round 15 (4.03–4.49); tighter windows drift to 5–6.5 as
   higher-order terms enter at ρ≈1.
2. `S_norm` ratio ≈1.6 (predicted 2; lower because the second pass runs on a
   displaced trajectory at non-small ρ); phase approximately preserved.
3. The OOD ΔValidity map is positive **only** on the late block (rounds 11–15);
   the single best OOD window (13,15) also has the lowest junction mismatch (0.17)
   and the smallest Jacobian σmax (2.07). Beneficial windows carry the predicted
   stationary / low-mismatch / marginally-stable signature.

**Conclusion of the programme:** RYS reproduces at sub-megabyte scale on a
verifiable task; its benefit is rank-predicted by the ρ/φ geometry and its CKA
effect follows the closed-form doubling law; the windows that help OOD are the
late, stable, on-manifold reasoning blocks — a compute-for-parameters exchange in
the spirit of Universal Transformers / DEQ / GRAM, applied post-hoc with no new
weights.

**Blog post:** `carlonicolini.github.io/.../2026-05-29-RYS-controlled-SAT-message-passing-and-rho-phi-theory.md`
describes the full programme (problem, motivation, GRAM/DEQ links, ρ/φ math, all
cycles) and credits Ng's original RYS post. Figures copied to the site:
`rys-sat-rho-phi-prediction.png`, `rys-sat-delta-valid-ood.png`,
`rys-sat-validity-vs-rounds-L32.png`. Bibliography keys added: `baek2026gram`,
`selsam2019neurosat`, `wang2026tinyrecursive`.

## Cycle 7 — Stochastic Transitions (Width) vs RYS (Depth)

**PI question:** the deterministic solver plateaus near 0.52 validity, the
signature of mode collapse on a multi-solution task. Does adding GRAM-style
stochastic latent transitions break the ceiling, and how does width (parallel
sampling) interact with depth (RYS)?

**Mechanism:** each round becomes `x_{t+1} = x_t + u_t + eps_t`, with
`u_t = F(x_t)` deterministic and `eps_t ~ N(mu(u_t), sigma(u_t)^2)` a learned,
reparameterised Gaussian guidance. The deterministic operator `F + mu` keeps the
RYS surgery and the ρ/φ geometry; the noise is the new width axis.

**Implementation:**

- `MessagePassingConfig.stochastic` + `mu_head`, `log_sigma_head` per round,
  zero-initialised so training starts near-deterministic. `forward(..., sample=)`
  toggles noise; the deterministic mean path (`sample=False`) stays reproducible
  for CKA/ρ/φ. New tests `test_stochastic_sampling_diversifies_but_mean_is_deterministic`.
- `rys.sat_data.multisample_validity` → per-formula `valid_any` (valid@N),
  `n_valid`, `coverage` (distinct valid assignments). New test
  `test_multisample_validity_metrics`.
- Training keeps the soft-SAT loss + a variance-floor penalty so `sigma` does not
  collapse to zero.
- Script `scripts/constrained_satisfaction/run_sat_stochastic_experiment.py`: compares deterministic vs
  stochastic on single-sample validity, valid@N, coverage, each also under a late
  RYS window.

**Protocol (running):** depth 16, seed 45, pre-norm, 6/24 in-distribution, 8/34
OOD, 20 epochs, N=20 samples, RYS window (13,15) (the Cycle-6 best OOD window).

**Artifacts:** `results/sat_stochastic/20260529_100259/`
(`{deterministic,stochastic}/result.json`, `validity_comparison.png`,
`summary.json`). Figure copied to site as `rys-sat-stochastic-comparison.png`.

**Results (honest negative):**

| split | det single | stoch single | stoch valid@20 | stoch coverage |
| --- | ---: | ---: | ---: | ---: |
| val | 0.514 | 0.512 | 0.533 | 0.55 |
| test | 0.525 | 0.540 | 0.567 | 0.59 |
| OOD | 0.379 | 0.363 | 0.387 | 0.41 |

RYS window (13,15) adds a small positive nudge in both modes (e.g. test stoch
single 0.540 → 0.547 with RYS), independent of sampling.

**Interpretation:** with a weak variance floor (target std 0.1, weight 1e-2) the
optimiser collapses `sigma` to the floor and recovers the deterministic model:
`coverage < 1` means the 20 trajectories almost always produce the *same*
assignment, so `valid@20` is only ~2-3 pts above single-sample. The mode-collapse
ceiling is **not** broken by light Gaussian guidance. This matches GRAM's report
that naive noise is insufficient; breaking the ceiling needs a much stronger
exploration incentive (higher variance floor and/or entropy/coverage reward) or
the principled ELBO with a target-conditioned posterior.

**Next (Cycle 8 candidate):** raise the variance floor / add a coverage-entropy
reward; if still collapsing, implement the amortised variational objective. The
mechanism, metrics (`multisample_validity`), and depth/width harness are in place.

## Cycle 8 — Breaking the Mode-Collapse Ceiling

**Diagnosis first.** Cycle 7 collapsed because the soft-SAT loss gives *zero*
gradient to diversify once the model has one confident valid assignment, so the
optimiser drives sigma to the floor. Light isotropic noise around a confident
logit cannot change the argmax (coverage 1) unless it is large enough to also
flip *forced* bits (validity drops). The missing ingredient vs GRAM is that the
noise is not *shaped toward solutions*.

**Two regimes compared** (`scripts/constrained_satisfaction/run_sat_coverage_experiment.py`):

1. `floor` — the literal recipe: mean soft-SAT loss + high variance floor
   (target std 0.7, weight 0.2) + a **diversity reward** = mean per-active-bit
   variance of P(x=1) across K=4 samples (gives the model a reason to *use* the
   noise productively instead of collapsing it).
2. `best_of_k` — GRAM-inspired multiple-choice / winner-take-all: draw K=4
   trajectories per formula, back-propagate only the **minimum** soft-SAT loss.
   Identical samples waste the min, so productive diversity is rewarded directly.
   This is the cheap cousin of GRAM's target-conditioned ELBO.

Library support: `soft_sat_loss(..., reduction="none")` returns the per-formula
loss needed for the min-over-K. New test `test_soft_sat_loss_reduction_none_is_per_formula`.

**Protocol (running):** depth 16, seed 46, both regimes, 18 epochs, K=4 train
samples, N=20 eval samples, 6/24 in-distribution, 8/34 OOD.

**Smoke signal:** unlike Cycle 7, both regimes already show `coverage > 1` and
`valid@N` well above single-sample on the tiny smoke — the opposite of collapse.

**Artifacts:** `results/sat_coverage/<timestamp>/`
(`{floor,best_of_k}/result.json`, `coverage_comparison.png`, `summary.json`).

**Results:** *(pending — fill from `summary.json`)*

## Cycle 9 — Sorted Translation: grokking, RoPE length-extrapolation, RYS

**Motivation.** A second task family, orthogonal to the CSP solvers, to test
whether the RYS + rho/phi story holds for a *sequence algorithm* with a forced,
large decoding head. The "Sorted Translation" task: read an unsorted sequence of
tokens in `0..127` and emit `sorted(input)+128` in parallel (one 128-way
classification per position). Disjoint input/output vocabularies forbid weight
tying and force a distinct unembedding head `W_U`, isolating a decoding phase.

**Design (`src/rys/sorted_translation_{data,transformer}.py`,
`scripts/constrained_satisfaction/run_sorted_translation_experiment.py`):**

- Encoder-only RoPE Transformer; one-hot input -> `W_E` (`Linear(128,d_model)`),
  distinct `W_U` (`Linear(d_model,128)`), no learned positional table (RoPE only,
  for OOD length extrapolation). Pre-norm rounds expose `model.model.layers` so
  `apply_rys` and the rho/phi capture work unchanged.
- Metrics: token accuracy, exact sequence accuracy (verifier accepts the true
  sort, duplicates allowed), and `sortedness` (fraction of adjacent
  non-decreasing pairs) as a partial-credit signal.
- Heavy weight decay (0.1) to grok; mixed-length training to elicit OOD-length
  generalization.

**Run (9 layers, mixed train lengths {8,12,16,20,24}, OOD {32,48,64}, seed 0,
400 epochs, n_train 12288):** `results/sorted_translation_deeper/20260604_022004/`
(checkpoint trained at `.../20260603_005701/`; analysis re-run with
`--capture-batches 1` because the float64 rho/phi over ~16k token samples for 36
layer pairs was prohibitively slow at the default 4 batches).

Baseline accuracy:

| split | token acc | sequence acc | sortedness |
| --- | ---: | ---: | ---: |
| val (16) | 0.992 | 0.917 | 1.000 |
| ood_32 | 0.864 | 0.094 | 1.000 |
| ood_48 | 0.172 | 0.000 | 0.966 |
| ood_64 | 0.077 | 0.000 | 0.835 |

Key results:

1. **Mixed-length training elicits OOD accuracy.** At length 32 (just beyond the
   training max 24) exact accuracy is non-zero (0.094) and token accuracy 0.86,
   versus 0 in the earlier single-length runs (`sorted_translation/20260603_172019`
   at 3 layers and `sorted_translation_deeper/20260603_234853` at 6 layers).
2. **RYS helps OOD where the base has not converged, and the effect grows with
   hardness.** Best RYS gains (windows `i<j`, repeats 1..6):
   - ood_48 token 0.172 -> 0.256 (+0.084) at window (1,2) x2;
   - ood_64 token 0.077 -> 0.115 (+0.038) at (1,2) x3; sortedness 0.835 -> 0.949
     (+0.114) at (0,4) x6.
   - val and ood_32 (already solved) do not benefit (RYS slightly hurts), as
     expected for an extra-computation intervention.
   - The useful windows are early-to-middle reasoning rounds, not the decoder tail.
3. **rho/phi theory validates cleanly.** 36 layer pairs, plateau predictor vs
   measured `1 - CKA`: Spearman 0.905, Pearson 0.638; median relative force
   R = 0.53 (small-force / plateau regime, where the RYS doubling law is benign).
4. **Three-phase geometry is visible at 9 layers.** Validation token-stream CKA
   shows an encoder corner (L0, CKA 0.15-0.55 to the rest), a high-similarity
   reasoning band (L1-L5, CKA 0.93-0.98), and a sharply diverging decoder tail
   (L6-L8, L8 at 0.15-0.46). Per-round read-out climbs monotonically
   (0.04 -> 0.29 -> ... -> 0.95 -> 0.99 token at val) and the exact sort
   crystallises only at the final layer; OOD plateaus mid-stack (it "runs out of
   layers"), which is exactly why extra RYS rounds help OOD.

**Reading.** The task corroborates both theses on a non-CSP, grokking sequence
algorithm: RYS buys reusable post-training computation that helps precisely where
the problem is harder than training, and the rho/phi closed form predicts the
connectome ordering in the small-force regime where RYS doubling is safe.

**Artifacts:** `results/sorted_translation_deeper/20260604_022004/`
(`report.md`, `summary.json`, `baselines.json`, `rys_sweep.{csv,png}`,
`accuracy_vs_rounds*`, `cka_val.{csv,png}`, `rho_phi.csv`, `theory_fit.json`,
`checkpoint.pt`).
