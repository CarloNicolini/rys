"""Apply RYS to Pythia (GPT-NeoX) models on a small GSM8K multiple-choice set.

This is the large-language-model counterpart of
:mod:`scripts.constrained_satisfaction.run_rys_accuracy_matrix`. It extends the RYS theory from the
controlled toy solvers to a real pretrained LLM family (Pythia) on GSM8K,
producing the two headline objects of the paper for a frozen model:

- the **CKA connectome** of the residual-stream layers, and
- the **RYS Delta-Accuracy matrix**: for every half-open band ``(i, j)`` the
  change in GSM8K accuracy when that band is replayed via
  :func:`rys.surgery.apply_rys`.

Correctness is measured without any generation. Each GSM8K problem becomes a
multiple-choice loglikelihood task (gold number vs. numeric distractors), so a
whole problem is scored in one ``use_cache=False`` forward pass that stays
correct under the RYS replay hook. See :mod:`rys.gsm8k_mc`.

Examples
--------
::

    # local smoke test on the smallest model
    uv run python scripts/run_pythia_gsm8k_rys.py --model EleutherAI/pythia-70m \\
        --n 16 --capture-n 16 --batch-size 16

    # full run on a GPU box
    uv run python scripts/run_pythia_gsm8k_rys.py --model EleutherAI/pythia-1b \\
        --n 100 --capture-n 48 --batch-size 32
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

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix
from rys.eval_core import cka_device_for, resolve_device, save_heatmap, swept_windows
from rys.gsm8k_mc import load_pythia, make_gsm8k_mc, mc_accuracy, prepare_mc_batches, score_prepared
from rys.surgery import apply_rys
from rys.theory_validation import junction_mismatch, rho_phi_table, theory_fit


def main(
    model: str = typer.Option("EleutherAI/pythia-70m", help="HuggingFace Pythia checkpoint."),
    n: int = typer.Option(100, help="GSM8K test problems to score."),
    k_distractors: int = typer.Option(4, help="Numeric distractors per problem (chance = 1/(k+1))."),
    fewshot: int = typer.Option(4, help="Few-shot exemplars prepended to every problem."),
    n_repeats: int = typer.Option(2, help="Total traversals of each RYS band (2 = canonical)."),
    batch_size: int = typer.Option(16, help="Candidate sequences scored per forward pass."),
    capture_n: int = typer.Option(48, help="Prompts used for the CKA connectome."),
    capture_batch_size: int = typer.Option(8, help="Batch size for residual-stream capture."),
    max_length: int = typer.Option(1024, help="Prompt truncation length for CKA capture."),
    top_k_windows: int = typer.Option(8, help="Best positive-delta windows kept for geometry diagnostics."),
    seed: int = typer.Option(0, help="Dataset sampling seed."),
    output_dir: Path = typer.Option(Path("results/pythia_gsm8k_rys"), help="Run output root."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


def pick_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def delta_accuracy_matrix(
    model,
    batches: list[dict],
    device: torch.device,
    *,
    n_layers: int,
    n_repeats: int,
    baseline_acc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep every RYS band and tabulate the change in MC accuracy.

    The pre-tokenised ``batches`` are reused for every window (RYS replay does
    not change the inputs), so each window costs only forward passes.
    """
    windows = swept_windows(n_layers)
    matrix = np.full((n_layers, n_layers), np.nan, dtype=float)
    rows = []
    for w_idx, (start, end) in enumerate(windows):
        with apply_rys(model, (start, end), n_repeats=n_repeats):
            scores = score_prepared(model, batches, device=device)
        acc = mc_accuracy(scores)
        delta = acc["acc_norm"] - baseline_acc
        matrix[start, end] = delta
        rows.append(
            {
                "start": start,
                "end": end,
                "n_repeats": n_repeats,
                "baseline_acc_norm": baseline_acc,
                "rys_acc_norm": acc["acc_norm"],
                "rys_acc": acc["acc"],
                "delta_acc_norm": delta,
            }
        )
        print(f"[{w_idx + 1}/{len(windows)}] " + json.dumps(rows[-1]))
    index = list(range(n_layers))
    return pd.DataFrame(matrix, index=index, columns=index), pd.DataFrame(rows)


def compute_cka(
    model,
    tokenizer,
    prompts: pd.DataFrame,
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activations = capture_residual_stream(
        model,
        tokenizer,
        prompts[["prompt_id", "prompt"]],
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    # CKA is O(L^2 * N^2); run it on the model device (GPU) so the many-layer
    # connectome of the larger models stays cheap.
    cka = cka_matrix(activations, unbiased=False, device=cka_device_for(device))
    return cka, activations


def geometry_diagnostics(
    activations: pd.DataFrame,
    long_delta: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """rho/phi theory fit on the connectome plus junction mismatch per top band."""
    rho_phi = rho_phi_table(activations)
    fit = theory_fit(rho_phi)
    best = long_delta.sort_values("delta_acc_norm", ascending=False).head(top_k)
    band_rows = []
    for row in best.itertuples(index=False):
        window = (int(row.start), int(row.end))
        band_rows.append(
            {
                "start": window[0],
                "end": window[1],
                "delta_acc_norm": float(row.delta_acc_norm),
                "junction_mismatch": junction_mismatch(activations, window),
            }
        )
    return rho_phi, fit, band_rows


def sample_generations(model, tokenizer, mc_df, device, *, n: int = 5, max_new_tokens: int = 32) -> list[dict]:
    """Greedy completions on a few prompts, purely for qualitative inspection."""
    out = []
    with torch.inference_mode():
        for row in mc_df.head(n).itertuples(index=False):
            ids = tokenizer(row.prompt, return_tensors="pt").to(device)
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
            out.append({"prompt_id": row.prompt_id, "gold": int(row.gold), "generated": text})
    return out


def write_report(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    n_layers: int,
    baseline: dict,
    long_delta: pd.DataFrame,
    fit: dict,
    band_rows: list[dict],
) -> None:
    chance = 1.0 / (args.k_distractors + 1)
    best = long_delta.loc[long_delta["delta_acc_norm"].idxmax()]
    lines = [
        "# Pythia GSM8K RYS Report",
        "",
        f"- Model: `{args.model}` ({n_layers} layers)",
        f"- GSM8K MC problems: `{args.n}`, distractors: `{args.k_distractors}` (chance = {chance:.3f})",
        f"- Few-shot: `{args.fewshot}`, repeats: `{args.n_repeats}`",
        (
            "- RYS convention: half-open band `(i, j)` replays layers `i..j-1` at the input of layer `j` "
            "(`0 <= i < j < L`); the last layer is never replayed."
        ),
        f"- Windows swept: `{len(swept_windows(n_layers))}`",
        "",
        "## Baseline (no RYS)",
        "",
        f"- acc_norm (per-token mean, headline): **{baseline['acc_norm']:.4f}**",
        f"- acc (summed loglik): {baseline['acc']:.4f}",
        "",
        "## Best RYS band (by Delta acc_norm)",
        "",
        f"- window **({int(best['start'])}, {int(best['end'])})**: "
        f"acc_norm {best['rys_acc_norm']:.4f} (Delta {best['delta_acc_norm']:+.4f})",
        "",
        "## CKA geometry (rho/phi plateau theory fit)",
        "",
        f"- Spearman(plateau predictor, 1-CKA): {fit.get('spearman_plateau_pred', float('nan')):.3f}",
        f"- Pearson(Q^2, 1-CKA): {fit.get('pearson_Q2', float('nan')):.3f}",
        f"- median rho (R): {fit.get('median_R', float('nan')):.3f}, "
        f"median cos(phi): {fit.get('median_cos_phi', float('nan')):.3f}",
        "",
        "## Top bands: Delta vs junction mismatch",
        "",
        "| window | Delta acc_norm | junction mismatch |",
        "| --- | ---: | ---: |",
    ]
    for row in band_rows:
        lines.append(
            f"| ({row['start']}, {row['end']}) | {row['delta_acc_norm']:+.4f} | {row['junction_mismatch']:.3f} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device()
    dtype = pick_dtype(device)

    model_tag = args.model.split("/")[-1]
    run_dir = args.output_dir / model_tag / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}, dtype: {dtype}, run_dir: {run_dir}")

    model, tokenizer = load_pythia(args.model, device=device, dtype=dtype)
    n_layers = len(model.model.layers)
    print(f"Loaded {args.model}: {n_layers} layers")

    mc_df = make_gsm8k_mc(n=args.n, k_distractors=args.k_distractors, fewshot=args.fewshot, seed=args.seed)
    mc_df.drop(columns=["candidates"]).to_json(run_dir / "mc_dataset.json", orient="records", indent=2)
    print(f"Built GSM8K MC set: {len(mc_df)} problems")

    batches = prepare_mc_batches(tokenizer, mc_df, batch_size=args.batch_size)
    baseline = mc_accuracy(score_prepared(model, batches, device=device))
    print("Baseline:", json.dumps(baseline))

    matrix, long_delta = delta_accuracy_matrix(
        model,
        batches,
        device,
        n_layers=n_layers,
        n_repeats=args.n_repeats,
        baseline_acc=baseline["acc_norm"],
    )
    matrix.to_csv(run_dir / "delta_accuracy.csv")
    long_delta.to_csv(run_dir / "delta_accuracy_long.csv", index=False)
    save_heatmap(
        matrix,
        run_dir / "delta_accuracy.png",
        title=f"RYS Delta accuracy_norm ({model_tag}, GSM8K MC)",
        cmap="RdBu_r",
        cbar_label="Delta acc_norm vs baseline",
        diverging=True,
    )

    cka, activations = compute_cka(
        model,
        tokenizer,
        mc_df.head(args.capture_n),
        device,
        batch_size=args.capture_batch_size,
        max_length=args.max_length,
    )
    cka.to_csv(run_dir / "cka.csv")
    save_heatmap(
        cka,
        run_dir / "cka.png",
        title=f"Linear CKA connectome ({model_tag}, GSM8K prompts)",
        cmap="viridis",
        cbar_label="CKA",
        diverging=False,
    )

    rho_phi, fit, band_rows = geometry_diagnostics(activations, long_delta, top_k=args.top_k_windows)
    rho_phi.to_csv(run_dir / "rho_phi_table.csv", index=False)
    pd.DataFrame(band_rows).to_csv(run_dir / "top_bands.csv", index=False)

    generations = sample_generations(model, tokenizer, mc_df, device)
    (run_dir / "generations_sample.json").write_text(json.dumps(generations, indent=2))

    summary = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": str(device),
        "dtype": str(dtype),
        "model": args.model,
        "n_layers": n_layers,
        "chance": 1.0 / (args.k_distractors + 1),
        "baseline": baseline,
        "best_band": long_delta.loc[long_delta["delta_acc_norm"].idxmax()].to_dict(),
        "theory_fit": fit,
        "top_bands": band_rows,
        "n_windows": len(swept_windows(n_layers)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(
        run_dir,
        args=args,
        n_layers=n_layers,
        baseline=baseline,
        long_delta=long_delta,
        fit=fit,
        band_rows=band_rows,
    )
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
