# rys

Empirical CKA connectomes and Repeat-Your-Self (RYS) surgery on Llama-3.2-3B-Instruct.

This repo is the experimental companion to two blog posts:
- [How skip connections define graphs in deep networks](https://carlonicolini.github.io/sections/science/_posts/2026-04-28-Skip-connections-and-graph-analysis.md)
- [Similarity of neural networks representations](https://carlonicolini.github.io/sections/science/_posts/2026-04-29-Similarity-of-neural-networks-represetations.md)

## What it does

The notebook [notebooks/cka_rys_connectome.ipynb](notebooks/cka_rys_connectome.ipynb) tests four falsifiable claims of the theory:

1. **Task-specific reasoning module.** GSM8K should produce a wider linear-CKA plateau than CommonsenseQA (non-math reasoning) or Wikitext-2 (no reasoning).
2. **Eq. (10) plateau.** $$1-\mathrm{CKA}_{ij}\propto \mathcal{R}_{i,j}^2 \sin^2\Phi_{i,j}$$ inside the central plateau.
3. **Eq. (14) RYS amplification.** Duplicating the central window via RYS multiplies the off-plateau distance by ~16x.
4. **Behavioural delta.** Standard `lm-evaluation-harness` accuracy on GSM8K should improve only when the *correct* reasoning window is duplicated.

## Repository layout

```
rys/
├── pyproject.toml          uv-managed project; deps split into core + [gpu] for bitsandbytes
├── uv.lock                 committed for full reproducibility
├── src/rys/
│   ├── activations.py      forward-hook residual-stream capture at sequence level
│   ├── cka.py              thin wrapper over ckatorch (HSIC1 unbiased estimator)
│   ├── surgery.py          RYS forward-pre-hook context manager
│   ├── data.py             GSM8K / CommonsenseQA / Wikitext-2 loaders -> pandas DataFrames
│   ├── modules.py          Leiden community detection, PELT change points, plateau metric
│   └── plots.py            plotly heatmaps, 3-panel figure, RYS delta plot
├── notebooks/
│   └── cka_rys_connectome.ipynb     the main self-contained notebook
├── scripts/
│   └── run_lm_eval.py      CLI shim that wraps the model with apply_rys before lm-eval
├── tests/
│   └── test_smoke.py       CKA shape contract + RYS-hook reversibility
├── data/                   gitignored: HF datasets cache
└── results/                gitignored except results/figures/
    ├── activations/        per-task parquet caches and CKA matrices
    ├── figures/            .html + .png exports of every plotly figure
    └── lm_eval_outputs/    raw lm-evaluation-harness JSON outputs per RYS configuration
```

## Setup

```bash
cd ~/workspace/rys
uv sync                  # CPU/MPS only
uv sync --extra gpu      # also installs bitsandbytes for INT8 quantisation (CUDA)
uv run python -m ipykernel install --user --name rys --display-name "Python (rys)"
uv run pytest tests/     # 6 smoke tests, ~10s
```

## Running the experiment

The notebook is parameterised through environment variables so a quick smoke run is one command:

```bash
RYS_N_PROMPTS=32 RYS_RUN_LM_EVAL=0 uv run jupyter lab notebooks/cka_rys_connectome.ipynb
```

For the full 250-prompt evaluation on a single GPU:

```bash
RYS_N_PROMPTS=250 RYS_BATCH_SIZE=8 RYS_EVAL_LIMIT=250 RYS_EVAL_FEWSHOT=8 \
  uv run jupyter lab notebooks/cka_rys_connectome.ipynb
```

| variable | default | purpose |
| :--- | :--- | :--- |
| `RYS_MODEL` | `meta-llama/Llama-3.2-3B-Instruct` | swap to any Llama-style decoder |
| `RYS_N_PROMPTS` | `250` | per-task prompt count for activation extraction |
| `RYS_BATCH_SIZE` | `8` (CUDA), `2` (CPU) | forward-pass batch size |
| `RYS_MAX_LENGTH` | `512` | tokenisation truncation length |
| `RYS_REUSE_CACHE` | `1` | reuse cached parquet/JSON if present |
| `RYS_RUN_LM_EVAL` | `1` on CUDA | run the lm-evaluation-harness section |
| `RYS_EVAL_LIMIT` | `250` | examples per task in lm-eval |
| `RYS_EVAL_FEWSHOT` | `8` | few-shot examples per task |

## Hardware

- INT8 quantisation requires CUDA + `bitsandbytes`. Install with `uv sync --extra gpu` on a GPU box.
- The activation extraction and the CKA computation work on CPU/MPS; lm-evaluation-harness is skipped automatically off-CUDA.
- Llama-3.2-3B-Instruct is 6.5GB FP16 / ~3.5GB INT8.

## Anti-reviewer-criticism battery (Section 6 of the notebook)

| concern | mitigation |
| :--- | :--- |
| "Plateau is universal, not reasoning-specific." | 3-task contrast (math / non-math reasoning / continuation). |
| "Prompt pooling erased the token-time geometry." | Section 6 checks that activation capture keeps sequence-level token matrices. |
| "INT8 broke representations." | Section 6 spot-checks INT8 against FP16 on 32 prompts. |
| "Centering didn't matter." | Section 6 compares uncentered cosine to linear CKA. |
| "Statistical noise." | Bootstrap CIs on every CKA value (B=200) and accuracy delta (B=1000). |
| "Any extra compute would help." | Negative-control RYS windows (random middle, encoder/decoder boundary). |
| "Non-standard eval." | Standard `lm-evaluation-harness` with `gsm8k`, `commonsense_qa`, `mmlu_high_school_mathematics`. |
| "Irreproducible environment." | Committed `uv.lock`. |

## License

MIT.
