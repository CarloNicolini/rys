"""Unified RYS guesstimation probe for any HuggingFace decoder-only model.

This is the faithful replication of the downstream proxy from the original RYS
blog post: the model must emit a numeric answer in a single pass (no
chain-of-thought) and is scored by partial-credit closeness, rather than by a
multiple-choice loglikelihood. The graded generative signal is what discovered
the RYS effect at the 72B scale, so we test whether it reveals a beneficial
window on smaller scale ladders where the binary MC metric did not.

It unifies ``run_pythia_guesstimation_rys.py`` and
``run_qwen3_guesstimation_rys.py``: any Llama/Qwen-style decoder that exposes
``model.model.layers`` works, and Pythia (``GPTNeoXForCausalLM``) is handled by
the aliasing in :func:`rys.gsm8k_mc.load_causal_lm`. The band-Jacobian
:math:`\\sigma_{\\max}(D\\Phi)` matrix --- the contraction/expansion modulus of
each duplicated band --- is computed for every model, not just Pythia.

Per model it records the baseline probe score, the RYS $\\Delta$-score matrix
over all $(i,j)$ windows, the band-Jacobian $\\sigma_{\\max}(D\\Phi)$ matrix, the
CKA connectome and $\\rho/\\Psi$ table, and the geometry->effect rank
correlations with a bootstrap over questions. Outputs land in
``results/LLM/<family>/guesstimation_rys/<tag>/<timestamp>/``.

Examples
--------
::

    uv run python scripts/run_guesstimation_rys.py --model EleutherAI/pythia-70m
    uv run python scripts/run_guesstimation_rys.py --model EleutherAI/pythia-12b --dtype int4 --batch-size 16
    uv run python scripts/run_guesstimation_rys.py --model Qwen/Qwen3-0.6B --dtype bfloat16
    uv run python scripts/run_guesstimation_rys.py --model meta-llama/Llama-3.2-1B --dtype float32

Smoke test (tiny subset, any model, CPU/MPS)::

    uv run python scripts/run_guesstimation_rys.py --model EleutherAI/pythia-70m \\
        --n-questions 4 --stride 2 --capture-n 4 --boot 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from scipy.stats import spearmanr

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix
from rys.eval_core import (
    cka_device_for,
    compute_block_jacobian_sigma_max,
    infer_family,
    precompute_base_states,
    resolve_device,
    resolve_dtype,
    save_heatmap,
    swept_windows,
)
from rys.gsm8k_mc import load_causal_lm
from rys.guesstimation import (
    FEWSHOT,
    generate_numbers,
    make_guesstimation_questions,
    score_estimates,
)
from rys.theory_validation import rho_phi_table, theory_fit


def main(
    model: str = typer.Option("EleutherAI/pythia-70m", help="HuggingFace checkpoint (Pythia, Qwen, Llama, ...)."),
    seed: int = typer.Option(0, help="Question-set seed."),
    batch_size: int = typer.Option(32, help="Questions generated per batch."),
    max_new_tokens: int = typer.Option(12, help="Generated tokens per answer."),
    capture_n: int = typer.Option(32, help="Prompts used for the CKA connectome."),
    capture_batch_size: int = typer.Option(8, help="Batch size for residual capture."),
    stride: int = typer.Option(
        1, help="Sweep windows on a layer stride (use 2+ for very deep models to keep generation tractable)."
    ),
    dtype: str = typer.Option(
        "float32",
        help="Weights precision: 'float32' (safe for small models; bf16 breaks their generation), "
        "'bfloat16' (large models that do not fit in fp32), 'int8', or 'int4' (bitsandbytes NF4, for 12B+).",
    ),
    boot: int = typer.Option(4000, help="Bootstrap resamples over questions."),
    n_questions: int = typer.Option(
        0, help="Cap the question set (0 = use all). Useful for fast smoke tests on CPU/MPS."
    ),
    family: str = typer.Option(
        "",
        help="Output family bucket under results/LLM/<family>/guesstimation_rys/. "
        "Empty string auto-infers from the model name (pythia/qwen/llama/...).",
    ),
    output_dir: Path = typer.Option(Path("results/LLM"), help="Run output root (family is appended)."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


def _run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device()
    rng = np.random.default_rng(args.seed)
    torch_dtype = resolve_dtype(args.dtype)

    tag = args.model.split("/")[-1]
    family = args.family or infer_family(args.model)
    run_dir = args.output_dir / family / "guesstimation_rys" / tag / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device {device}, dtype {args.dtype}, family {family}, run_dir {run_dir}", flush=True)

    model, tok = load_causal_lm(
        args.model,
        device=device,
        dtype=torch_dtype,
        load_in_8bit=args.dtype == "int8",
        load_in_4bit=args.dtype == "int4",
    )
    L = len(model.model.layers)
    questions = make_guesstimation_questions(seed=args.seed)
    if args.n_questions and args.n_questions < len(questions):
        questions = questions[: args.n_questions]
    n_q = len(questions)
    print(f"{tag}: L={L}, {n_q} guesstimation questions", flush=True)

    def probe(window=None) -> np.ndarray:
        ests = generate_numbers(
            model, tok, questions, device=device, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, window=window,
        )
        return score_estimates(questions, ests)

    base = probe()
    print(f"baseline score = {base.mean():.4f}", flush=True)

    windows = swept_windows(L, stride=args.stride)
    per_q = np.zeros((len(windows), n_q), dtype=float)
    jac_sigma = np.full((len(windows),), np.nan)
    # Pick a representative question to calculate the band Jacobian over.
    test_prompt = FEWSHOT + f"Q: {questions[0]['question']}\nA:"
    base_h, causal_mask, pos_emb = precompute_base_states(model, tok, test_prompt)

    for w, (i, j) in enumerate(windows):
        per_q[w] = probe(window=(i, j))
        # sigma_max(DPhi) of the full-sequence band map: the one-step amplification
        # the duplicated pass is subject to (smaller = closer to stable refinement).
        jac_sigma[w] = compute_block_jacobian_sigma_max(
            model, base_h, causal_mask, pos_emb, i, j
        )
        if (w + 1) % 25 == 0:
            print(f"  {w + 1}/{len(windows)} windows", flush=True)

    delta = per_q.mean(1) - base.mean()
    matrix = np.full((L, L), np.nan)
    jac_matrix = np.full((L, L), np.nan)
    rows = []
    for w, (i, j) in enumerate(windows):
        matrix[i, j] = delta[w]
        jac_matrix[i, j] = jac_sigma[w]
        rows.append(
            {
                "start": i,
                "end": j,
                "rys_score": float(per_q[w].mean()),
                "delta_score": float(delta[w]),
                "jacobian_sigma_max": float(jac_sigma[w]),
            }
        )
    long = pd.DataFrame(rows)
    pd.DataFrame(matrix, index=range(L), columns=range(L)).to_csv(run_dir / "delta_score.csv")
    long.to_csv(run_dir / "delta_score_long.csv", index=False)
    pd.DataFrame(jac_matrix, index=range(L), columns=range(L)).to_csv(run_dir / "jacobian_sigma_max.csv")
    save_heatmap(
        pd.DataFrame(matrix),
        run_dir / "delta_score.png",
        title=f"RYS Delta guesstimation score ({tag})",
        cmap="RdBu_r",
        cbar_label="Delta partial-credit score",
        diverging=True,
    )
    save_heatmap(
        pd.DataFrame(jac_matrix),
        run_dir / "jacobian_sigma_max.png",
        title=f"Band Jacobian sigma_max ({tag})",
        cmap="viridis",
        cbar_label="sigma_max(DPhi), full-sequence band map",
        diverging=False,
    )

    # CKA connectome + geometry
    cap_n = min(args.capture_n, n_q)
    cap = pd.DataFrame(
        {"prompt_id": [f"q{i}" for i in range(cap_n)],
         "prompt": [questions[i]["question"] for i in range(cap_n)]}
    )
    acts = capture_residual_stream(model, tok, cap, batch_size=args.capture_batch_size, device=device)
    cka = cka_matrix(acts, unbiased=False, device=cka_device_for(device))
    cka.to_csv(run_dir / "cka.csv")
    save_heatmap(cka, run_dir / "cka.png", title=f"CKA connectome ({tag})", cmap="viridis",
                cbar_label="CKA", diverging=False)
    rp = rho_phi_table(acts)
    rp.to_csv(run_dir / "rho_phi_table.csv", index=False)
    fit = theory_fit(rp)

    # geometry -> effect, with bootstrap over questions
    mg = long.merge(rp, left_on=["start", "end"], right_on=["layer_i", "layer_j"])
    rho = mg["R"].to_numpy()
    ckaw = mg["cka_full"].to_numpy()
    jacw = mg["jacobian_sigma_max"].to_numpy()
    order = [windows.index((int(r.start), int(r.end))) for r in mg.itertuples(index=False)]
    per_q_ord = per_q[order]
    sp_rho = float(spearmanr(mg["delta_score"], rho).statistic)
    sp_cka = float(spearmanr(mg["delta_score"], ckaw).statistic)
    # Contraction hypothesis: less band amplification -> more stable refinement,
    # so we expect a NEGATIVE rank correlation between sigma_max and the effect.
    sp_jac = float(spearmanr(mg["delta_score"], jacw).statistic)
    boot_rho = np.empty(args.boot)
    boot_cka = np.empty(args.boot)
    boot_jac = np.empty(args.boot)
    boot_best = np.empty(args.boot)
    for t in range(args.boot):
        idx = rng.integers(0, n_q, n_q)
        db = per_q_ord[:, idx].mean(1) - base[idx].mean()
        boot_rho[t] = spearmanr(db, rho).statistic
        boot_cka[t] = spearmanr(db, ckaw).statistic
        boot_jac[t] = spearmanr(db, jacw).statistic
        boot_best[t] = db.max()

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

    best_idx = int(mg["delta_score"].idxmax())
    summary = {
        "model": args.model, "family": family, "L": L, "n_questions": n_q, "n_windows": len(windows),
        "dtype": args.dtype, "stride": args.stride,
        "baseline_score": round(float(base.mean()), 4),
        "mean_delta": round(float(delta.mean()), 4),
        "best_delta": round(float(delta.max()), 4),
        "best_window": [int(mg.loc[best_idx, "start"]), int(mg.loc[best_idx, "end"])],
        "best_delta_ci95": ci(boot_best),
        "frac_windows_positive": round(float((delta > 0).mean()), 3),
        "spearman_delta_rho": round(sp_rho, 3), "spearman_delta_rho_ci95": ci(boot_rho),
        "spearman_delta_cka": round(sp_cka, 3), "spearman_delta_cka_ci95": ci(boot_cka),
        "spearman_delta_jac_sigma": round(sp_jac, 3), "spearman_delta_jac_sigma_ci95": ci(boot_jac),
        "theory_fit": fit,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    np.savez_compressed(run_dir / "per_question.npz", base=base, per_window=per_q,
                        start=long["start"].to_numpy(), end=long["end"].to_numpy())
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"Wrote {run_dir}", flush=True)


main.__doc__ = __doc__

if __name__ == "__main__":
    typer.run(main)
