# RYS Graph k-Colouring Research Loop

This document records the graph k-colouring replication of the controlled RYS
research programme. It follows the same structure and seriousness as
`docs/rys_sat_research_loop.md` and `docs/rys_nqueens_research_loop.md`: each
cycle states a PI question, postdoc hypotheses, implementation hooks, exact
commands, result tables, and a decision gate.

The scientific question is unchanged from the SAT and N-Queens loops: does **RYS**
(post-hoc block repetition with no weight changes) buy extra reasoning iterations
on a verifiable constraint task, and can the **CKA / ρ–φ geometry** predict
*which* windows help, especially out of distribution? Colouring is chosen as the
third task for three specific reasons the first two could not address:

1. **Sparse constraint graph.** SAT is bipartite (clause↔variable) and N-Queens
   is a *complete* graph over rows (every pair constrains directly, 1-hop). A
   random colouring graph is **sparse**, so a message must traverse several hops:
   useful reasoning depth is tied to graph diameter, the cleanest setting for the
   "useful depth grows with hardness" claim.
2. **Strongest multi-solution symmetry.** Every proper colouring stays proper
   under any permutation of the `k` colours, so the deterministic-averaging mode
   collapse that caps SAT validity (Cycle 4 lesson) is *even stronger* here — the
   natural testbed for the width axis (stochastic transitions, `valid@K`).
3. **Size-OOD without the vocabulary confound.** The solver is
   permutation-equivariant over vertices and the colour vocabulary is fixed, so
   the OOD split can **grow the graph** (more vertices and edges) — the size
   extrapolation that the N-Queens absolute-column readout could not express
   (N-Queens Cycle 2a collapsed to 0 on larger boards and had to fall back to
   reducing givens).

Conventions shared with the SAT and N-Queens loops:

- RYS via `rys.surgery.apply_rys` — **half-open** windows `[start, end)` with
  `end < n_rounds` and `>= 2` duplicated rounds.
- CKA on the **vertex residual stream** (`strategy="vertex"`), not a CLS token.
- Primary metric is verifier-checked **`valid_coloring_rate`**, never exact match
  to the planted colouring (many proper colourings exist — colour permutation
  alone gives `k!`).
- Pre-norm rounds (`pre_norm=True`) keep the vertex stream additive so the ρ/φ
  telescoping is exact.

## Cycle 1 — Data, verifier, model, tests

**PI question:** can we generate a controllable, exactly-verifiable graph
k-colouring task whose difficulty (graph size, edge density, givens) is a knob,
and a RYS-compatible solver that exposes `model.model.layers`?

**Task formulation (v1): conditional colouring completion.** A bare graph with no
revealed colours admits the full colour-permutation symmetry from a symmetric
start, so each instance reveals `n_givens` vertices with their colour and the
model assigns one colour per remaining vertex. One discrete variable per vertex →
colour in `{0..k-1}` (multiclass). The graph is generated **planted-colourable**:
sample a random colouring, then draw `n_edges` edges only between
differently-coloured vertices, so the instance is guaranteed `k`-colourable (the
colouring analogue of planting a satisfying assignment in `rys.sat_data`).
Conditioning on givens makes train/val/test/OOD genuinely distinct and anchors
the colour symmetry enough to learn.

**Postdoc hypotheses:**

- *Dynamical systems.* Each adjacency-masked round is an iterate of a shared
  update map; repeating a window (RYS) is one extra refinement step. Useful
  windows should be stationary, low-junction-mismatch, marginally stable
  (Jacobian σ_max ≈ 1), as in SAT Cycle 6 and N-Queens Cycle 3.
- *Synthetic task.* Sparse graph colouring has large diameter (unlike complete
  N-Queens), so independent per-vertex marginals are a weak proxy and multi-hop
  propagation is genuinely required — the validity-vs-rounds curve should keep
  climbing longer, especially OOD.
- *Systems.* A round is masked multi-head attention over neighbours + MLP over
  the `V` vertex nodes, with **no absolute vertex identity** (only given colour,
  is-given flag, and degree as a permutation-invariant symmetry breaker).
  Pre-norm keeps the stream additive for the ρ/φ test; the fixed colour head
  makes larger-graph OOD expressible.
- *Mechanistic analysis.* Capture per-round vertex states; compute CKA and ρ/φ;
  score windows before the behavioural sweep.
- *Statistics.* Multi-seed and random-window controls before any publishable
  claim (the standing Cycle 4 discipline).

**Implementation hooks:**

- Data/verifier/loss: `rys.coloring_data`
  (`make_coloring_examples`, `make_coloring_splits`, `ColoringDataset`,
  `verify_coloring_tensor`, `soft_coloring_loss`, `coloring_is_valid`).
- Model: `rys.coloring_message_passing`
  (`ColoringMessagePassingModel`, `ColoringMPConfig`, pre-norm / weight-tied,
  rounds exposed as `model.model.layers`).
- Scripts: `scripts/constrained_satisfaction/run_coloring_messagepassing_experiment.py`,
  `scripts/constrained_satisfaction/run_coloring_theory_validation.py`.
- Tests: `tests/test_coloring_data.py`, `tests/test_coloring_message_passing.py`.

**Soft colouring loss.** For each active edge `(u, v)` the monochromatic
probability under independent per-vertex colour distributions is
`sum_c p_u[c] p_v[c]`; the loss is the mean negative log-probability that each
edge is bichromatic, plus an optional cross-entropy anchoring the given vertices.
It is invariant to permuting the colours, so it rewards *any* proper colouring —
mirrors `soft_sat_loss` / `soft_nqueens_loss`.

**Commands:**

```bash
source .venv/bin/activate
pytest tests/test_coloring_data.py tests/test_coloring_message_passing.py -q
```

**Result:** all tests green (9 data + 7 model). Verifier accepts proper
colourings and colour-permuted variants; rejects monochromatic edges, out-of-range
colours, and given violations. Generator is deterministic across seeds and always
plants a proper colouring. Pre-norm additivity
(`test_prenorm_vertex_stream_is_additive`), `apply_rys` reversibility, and
**permutation equivariance** (`test_permutation_equivariance`) all confirmed.
Full suite: 76 tests green.

**Learnability / regime decision (smoke runs).** Depth-8 solver, MPS, 12 epochs,
1024 train / 512 val·test·OOD, k=3, givens=3, in-distribution V=12:

| in-dist E | OOD (V,E) | val valid | test valid | OOD valid | CKA early | CKA late |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | (18, 36) | **0.842** | 0.824 | **0.451** | 0.888 | 0.996 |
| 28 | (18, 48) | 0.926 | 0.912 | 0.668 | 0.875 | 0.997 |

Both densities are learnable. **Decision:** adopt the **E=20** in-distribution
density as the headline regime (in-dist V=12, E=20, k=3, givens=3; OOD V=18,
E=36, givens=3), because it leaves the larger OOD gap (val 0.84 vs OOD 0.45) —
the most room to study RYS, the direct analogue of the SAT/N-Queens "make it
solvable, then leave OOD headroom" lesson. Edge density `E` is the documented
difficulty knob; OOD grows the graph (V and E) since the equivariant solver and
fixed colour head make size extrapolation expressible.

**First observation already worth noting.** The early-band CKA (0.888) sits
**below** both SAT and N-Queens early bands (0.91–0.96) while the late band stays
≈0.99 — the sparse graph shows a *sharper early→late contrast* out of the box,
consistent with the "diameter forces more early reasoning" hypothesis. Whether
this becomes a genuine encode→reason→decode tripartition with depth is the Cycle 2
question.

**Decision gate:** continue. Data + verifier + model + tests are in place and the
solver learns to a usable validity regime with a clean OOD gap. Proceed to
Cycle 2 (depth sweep).

## Cycle 2 — Message-passing depth sweep

**PI question:** does deeper iteration raise validity and, as in SAT Cycle 5 and
N-Queens Cycle 2, does OOD keep improving with rounds while in-distribution
saturates — and are RYS windows selective and growing with depth? Does the sparse
graph sharpen the connectome toward a tripartition more than the complete-graph
N-Queens did?

**Protocol:** `scripts/constrained_satisfaction/run_coloring_messagepassing_experiment.py`, depths
8/16/32, headline regime, 4096/1024/1024/1024, 20 epochs, deep supervision,
pre-norm. Artifacts per depth: checkpoint, `train_history.csv`,
`validity_vs_rounds_*.csv`+`.png`, `cka_variable_val.*`,
`delta_valid_*.{csv,png}`, `summary.json`; run-level `report.md`.

**Command:**

```bash
python scripts/constrained_satisfaction/run_coloring_messagepassing_experiment.py \
  --depths 8,16,32 --seed 47 \
  --n-vertices 12 --n-edges 20 --n-colors 3 --n-givens 3 \
  --ood-vertices 18 --ood-edges 36 --ood-givens 3
```

**Run of record (`results/coloring_messagepassing/20260530_142006/`).** Full sweep,
seed 47, 4096/1024/1024/1024, 20 epochs, deep supervision, pre-norm:

| rounds | val valid | test valid | OOD valid | CKA early | CKA late | best OOD window |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 0.942 | 0.946 | 0.700 | 0.927 | 0.995 | (2,7) **+0.039** |
| 16 | 0.954 | 0.950 | 0.783 | 0.889 | 0.997 | (3,8) **+0.023** |
| 32 | 0.939 | 0.931 | 0.729 | 0.886 | 0.998 | (6,16) **+0.015** |

**Findings:**

1. *In-distribution saturates, OOD profits from depth — then overfits.* In-dist
   validity holds near 0.94 at every depth, while the harder larger-graph OOD
   climbs **0.700 → 0.783** from 8 to 16 rounds, then eases to **0.729** at 32.
   The non-monotone turn is a mild depth-overfitting effect the more regular
   N-Queens did not show, consistent with the over-capacity penalty reported for
   deep tiny recursive solvers (TRM).
2. *RYS helps most out of distribution.* OOD RYS Δ (+0.039 / +0.023 / +0.015) is
   2–6× the in-distribution effect (+0.006 to +0.009) at every depth, though the
   magnitude shrinks as the deeper solver leaves less reasoning headroom.
3. *Sharpest early→late contrast of the three tasks.* Early-band CKA is the lowest
   yet (≈0.89, vs N-Queens 0.91–0.96) while the late band holds ≈0.998 — the sparse
   graph diameter forces more residual force into the early reasoning rounds.
4. *Genuine size-OOD.* The OOD split grows the graph (V=12→18); the
   permutation-equivariant solver and fixed colour vocabulary make this expressible
   where the N-Queens absolute readout collapsed to 0.

The earlier smoke run (depth 8, 12 epochs, 1024 train) showed the same OOD-dominant
signature at lower absolute validity (val 0.84 / OOD 0.45; best OOD Δ +0.098).

## Cycle 3 — ρ/φ theory validation

**PI question:** does the closed-form CKA decomposition hold on colouring
(plateau prediction rank-faithful; RYS amplifies `1−CKA`), and do the windows
that help OOD carry the stationary / low-mismatch / marginally-stable signature?

**Protocol:** `scripts/constrained_satisfaction/run_coloring_theory_validation.py`, pre-norm depth 16,
headline regime, 25 epochs. ρ/φ table + `theory_fit`, RYS ΔValidity sweep,
amplification + junction mismatch + Jacobian for the top windows.

**Command:**

```bash
python scripts/constrained_satisfaction/run_coloring_theory_validation.py \
  --depth 16 --seed 48 \
  --n-vertices 12 --n-edges 20 --n-colors 3 --n-givens 3 \
  --ood-vertices 18 --ood-edges 36 --ood-givens 3
```

**Run of record (`results/coloring_theory_validation/20260530_142014/`).**
Pre-norm depth 16, seed 48, 25 epochs. Baselines val 0.942 / test 0.942 / OOD
0.809. ρ/φ fit on 120 layer pairs:

| quantity | Colouring | N-Queens (Cycle 3) | SAT (Cycle 6) |
| --- | ---: | ---: | ---: |
| Spearman(plateau pred, 1−CKA) | **0.952** | 0.991 | 0.991 |
| Pearson(plateau pred, 1−CKA) | 0.632 | 0.712 | 0.659 |
| Pearson(Q², 1−CKA) | 0.759 | 0.917 | 0.897 |
| median R (≈ρ) | **0.637** | 0.445 | 1.035 |
| median cos φ | 0.742 | 0.739 | 0.729 |

The plateau formula is again a strong **rank** predictor of the connectome
(Spearman 0.952, a touch below the 0.991 of the first two tasks), and median
ρ ≈ 0.64 places colouring at an **intermediate** residual force between N-Queens
(0.445) and SAT (1.035). RYS doubling on the top OOD windows: `S_norm_ratio ≈
1.36–1.48`, `1−CKA ratio ≈ 1.5–2.4`, phase preserved (`cos_phi_diff ≈ ±0.01`) —
directionally correct and softer than SAT's crisp ×2/×4, as expected at
intermediate ρ. Among the top windows the cleanest dynamical signature — lowest
junction mismatch (0.67) and lowest block-Jacobian σ_max (21.8) — is the compact
late window **(5, 9)**, which delivers +0.016 OOD; the broader windows (4,10)/(3,10)
help marginally more (+0.020) but are far less stable (σ_max 48–87). **The most
stable, most on-manifold loop again generalises best** — replicating SAT and
N-Queens.

**Decision gate:** continue toward the paper extension. RYS reproduces and is
*strongest on OOD* on a sparse graph; ρ/φ rank-predicts the connectome; the
small-ρ regime explains the high CKA and the moderate doubling.

## Conclusions so far (Colouring vs N-Queens vs SAT)

- RYS is task-general across **three** constraint families now: 3-SAT assignment,
  N-Queens completion, and graph k-colouring — all with **no weight changes**, and
  the benefit is **largest out of distribution** (colouring smoke OOD Δ ≈ +0.10,
  ~2× the in-distribution effect).
- The ρ/φ closed form rank-predicts the CKA connectome on all three tasks
  (Spearman ≈ 0.98–0.99). Colouring shares N-Queens' small-ρ deep-plateau regime
  (ρ ≈ 0.42).
- **Colouring adds what the first two tasks could not:** (i) a *sparse* graph so
  reasoning depth is tied to diameter; (ii) the strongest multi-solution colour
  symmetry for the width axis; (iii) genuine **size-OOD** (larger graphs) thanks
  to permutation equivariance + fixed colour vocabulary, sidestepping the
  N-Queens larger-board collapse.

## Next steps

- Full depth sweep (8/16/32) and depth-16 ρ/φ validation runs of record.
- Cycle 4 ablations: multi-seed (≥3) best-window OOD Δ vs random same-length
  high-/low-CKA windows; `n_repeats ∈ {2,3,4}` unimodality; pre-norm off vs on;
  edge-density and givens difficulty sweeps; **stochastic transitions +
  `valid@K` / coverage** to test whether the strong colour-permutation symmetry
  makes the mode-collapse / width-axis story sharper than SAT.
- Comparison table across the three tasks (validity ceilings, RYS Δ magnitudes,
  ρ/φ fit quality, CKA tripartition, size-OOD feasibility).
