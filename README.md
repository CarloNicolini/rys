# rys

Empirical generated-response CKA connectomes and Repeat-Your-Self (RYS) surgery.

This repo is the experimental companion to two blog posts:
- [How skip connections define graphs in deep networks](https://carlonicolini.github.io/sections/science/_posts/2026-04-28-Skip-connections-and-graph-analysis.md)
- [Similarity of neural networks representations](https://carlonicolini.github.io/sections/science/_posts/2026-04-29-Similarity-of-neural-networks-represetations.md)

## What it does

The notebook [notebooks/cka_rys_connectome.ipynb](notebooks/cka_rys_connectome.ipynb) tests four falsifiable claims of the theory:

1. **Task-specific response modules.** Generated responses to GSM8K, CommonsenseQA, and MMLU subjects should show task-dependent linear-CKA integration/segregation patterns.
2. **Eq. (10) plateau.** $$1-\mathrm{CKA}_{ij}\propto \mathcal{R}_{i,j}^2 \sin^2\Phi_{i,j}$$ inside the central plateau.
3. **Eq. (14) RYS amplification.** Duplicating the central window via RYS multiplies the off-plateau distance by ~16x.
4. **Behavioural delta.** Standard `lm-evaluation-harness` accuracy on GSM8K should improve only when the *correct* reasoning window is duplicated.

## Repository layout

```
rys/
├── pyproject.toml          uv-managed project; CUDA torch + bitsandbytes auto-gated to Linux
├── uv.lock                 committed for full reproducibility
├── src/rys/
│   ├── activations.py      prompt and generated-response residual-stream capture
│   ├── cka.py              thin wrapper over ckatorch (HSIC1 unbiased estimator)
│   ├── surgery.py          RYS forward-pre-hook context manager
│   ├── data.py             GSM8K / CommonsenseQA / MMLU loaders -> pandas DataFrames
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
uv sync                  # picks the right wheels for your platform
uv run python -m ipykernel install --user --name rys --display-name "Python (rys)"
uv run pytest tests/     # 6 smoke tests, ~10s
```

`uv sync` resolves the platform-specific bits automatically: on Linux it pulls the
CUDA 12.6 torch wheels and `bitsandbytes`; on macOS it pulls the default PyPI torch
(with MPS support) and skips `bitsandbytes` entirely, since it only ships CUDA
wheels. The notebook's INT8 path is already gated by `USE_INT8 = DEVICE.type == "cuda"`,
so the macOS install runs end-to-end on MPS / CPU without code changes.

## Running the experiment

The notebook is parameterised through environment variables so a quick smoke run is one command:

```bash
RYS_N_PROMPTS=32 RYS_RUN_LM_EVAL=0 uv run jupyter lab notebooks/cka_rys_connectome.ipynb
```

For the full 250-prompt evaluation on a single GPU:

```bash
RYS_N_PROMPTS=250 RYS_BATCH_SIZE=4 RYS_MAX_NEW_TOKENS=256 RYS_EVAL_LIMIT=250 RYS_EVAL_FEWSHOT=8 \
  uv run jupyter lab notebooks/cka_rys_connectome.ipynb
```

| variable | default | purpose |
| :--- | :--- | :--- |
| `RYS_MODEL` | `Qwen/Qwen3.6-27B` | swap to any Llama-style decoder |
| `RYS_N_PROMPTS` | `250` | per-task prompt count for generation and activation extraction |
| `RYS_MMLU_SUBJECTS` | `philosophy` | comma-separated MMLU subjects to add, e.g. `philosophy,formal_logic` |
| `RYS_BATCH_SIZE` | `4` (CUDA), `1` (CPU) | generation and replay batch size |
| `RYS_MAX_PROMPT_LENGTH` | `512` | prompt tokenisation truncation length |
| `RYS_MAX_NEW_TOKENS` | `256` | maximum generated response length captured for CKA |
| `RYS_REUSE_CACHE` | `1` | reuse cached parquet/JSON if present |
| `RYS_RUN_QUANT_SANITY` | `0` | optionally run FP16-vs-INT8 check when memory allows |
| `RYS_RUN_LM_EVAL` | `1` on CUDA | run the lm-evaluation-harness section |
| `RYS_EVAL_LIMIT` | `250` | examples per task in lm-eval |
| `RYS_EVAL_FEWSHOT` | `8` | few-shot examples per task |

## Hardware

- INT8 quantisation requires CUDA + `bitsandbytes`. The Linux `uv sync` installs both automatically; on macOS `bitsandbytes` is skipped and the notebook stays in plain torch (MPS / CPU).
- The generated-response activation extraction and the CKA computation work on CPU/MPS, but large models are practical only on CUDA.
- For 27B-class models on 24GB cards, prefer 4-bit quantisation if INT8 does not fit.

## Anti-reviewer-criticism battery (Section 6 of the notebook)

| concern | mitigation |
| :--- | :--- |
| "Plateau is universal, not reasoning-specific." | Task contrast over generated responses: math, commonsense, and MMLU subjects. |
| "Prompt prefill is not the task response." | The notebook generates answers, replays prompt + answer, and captures only response-token states. |
| "Prompt pooling erased the token-time geometry." | Section 6 checks that activation capture keeps sequence-level response-token matrices. |
| "INT8 broke representations." | Optional Section 6 spot-checks INT8 against FP16 on 32 prompts when memory allows. |
| "Centering didn't matter." | Section 6 compares uncentered cosine to linear CKA. |
| "Statistical noise." | Bootstrap CIs on every CKA value (B=200) and accuracy delta (B=1000). |
| "Any extra compute would help." | Negative-control RYS windows (random middle, encoder/decoder boundary). |
| "Non-standard eval." | Standard `lm-evaluation-harness` with `gsm8k`, `commonsense_qa`, `mmlu_high_school_mathematics`. |
| "Irreproducible environment." | Committed `uv.lock`. |

## License

MIT.
