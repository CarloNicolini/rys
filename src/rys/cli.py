"""Unified CLI entrypoint for RYS evaluation workflows."""

from __future__ import annotations

import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer

from rys.activations import capture_generated_residual_stream, capture_residual_stream
from rys.cka import cka_matrix
from rys.combined_probe import build_probe_items, generate_probe_outputs
from rys.eval_core import (
    cka_device_for,
    infer_family,
    repo_root,
    resolve_device,
    resolve_dtype,
    save_heatmap,
    swept_windows,
)
from rys.gsm8k_mc import load_causal_lm
from rys.model_profile import detect_model_profile
from rys.probe_data import load_ng_datasets
from rys.residual_force import residual_force_long

app = typer.Typer(help="RYS eval commands.")


def _write_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _run_probe(
    *,
    model: str,
    dtype: str,
    output_dir: Path,
    family: str | None,
    math_set: str,
    eq_set: str,
    seed: int,
    stride: int,
    batch_size: int,
    max_new_tokens: int,
    capture_batch_size: int,
    generated_capture_max_new_tokens: int,
    n_math: int,
    n_eq: int,
    boot: int,
    prompt_policy: str | None,
    use_no_think_prefix: bool,
) -> Path:
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device()
    torch_dtype = resolve_dtype(dtype)
    model_obj, tokenizer = load_causal_lm(
        model,
        device=device,
        dtype=torch_dtype,
        load_in_8bit=dtype == "int8",
        load_in_4bit=dtype == "int4",
    )
    profile = detect_model_profile(model, tokenizer, force_policy=prompt_policy)
    model_family = family or profile.family
    model_tag = profile.model_tag
    run_dir = output_dir / model_family / "guesstimation_rys" / model_tag / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    root = repo_root()
    math_ds, eq_ds, dataset_records = load_ng_datasets(repo_root=root, math_set=math_set, eq_set=eq_set)
    if n_math > 0:
        math_ds = {k: v for i, (k, v) in enumerate(math_ds.items()) if i < n_math}
    if n_eq > 0:
        eq_ds = {k: v for i, (k, v) in enumerate(eq_ds.items()) if i < n_eq}

    items = build_probe_items(
        math_ds,
        eq_ds,
        tokenizer=tokenizer,
        prompt_policy=profile.policy,
        use_no_think=use_no_think_prefix,
    )
    n_layers = len(model_obj.model.layers)

    baseline_summary, baseline_responses = generate_probe_outputs(
        model_obj,
        tokenizer,
        items,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        window=None,
    )

    windows = swept_windows(n_layers, stride=stride)
    per_window_combined = np.zeros(len(windows), dtype=float)
    rows: list[dict] = []
    combined_results: dict[tuple[int, int], dict] = {}
    math_results: dict[tuple[int, int], dict] = {}
    eq_results: dict[tuple[int, int], dict] = {}

    for idx, window in enumerate(windows):
        summary, responses = generate_probe_outputs(
            model_obj,
            tokenizer,
            items,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
            window=window,
        )
        per_window_combined[idx] = summary["combined_score"]
        delta = summary["combined_score"] - baseline_summary["combined_score"]
        rows.append(
            {
                "start": window[0],
                "end": window[1],
                "rys_score": summary["combined_score"],
                "delta_score": delta,
                "math_score": summary["math_score"],
                "eq_score": summary["eq_score"],
                "jacobian_sigma_max": float("nan"),
            }
        )
        key = tuple(window)
        combined_results[key] = summary | {"responses": responses}
        math_payload = [r for r in responses if r["task"] == "math"]
        eq_payload = [r for r in responses if r["task"] == "eq"]
        math_results[key] = {
            "score": float(np.mean([r["score"] for r in math_payload])) if math_payload else 0.0,
            "responses": math_payload,
        }
        eq_results[key] = {
            "score": float(np.mean([r["score"] for r in eq_payload])) if eq_payload else 0.0,
            "responses": eq_payload,
        }
        if (idx + 1) % 25 == 0:
            typer.echo(f"  {idx + 1}/{len(windows)} windows", err=True)

    long_df = pd.DataFrame(rows)
    delta_matrix = np.full((n_layers, n_layers), np.nan)
    for r in rows:
        delta_matrix[int(r["start"]), int(r["end"])] = r["delta_score"]
    delta_df = pd.DataFrame(delta_matrix, index=range(n_layers), columns=range(n_layers))
    delta_df.to_csv(run_dir / "delta_score.csv")
    long_df.to_csv(run_dir / "delta_score_long.csv", index=False)
    save_heatmap(
        delta_df,
        run_dir / "delta_score.png",
        title=f"RYS Delta combined score ({model_tag})",
        cmap="RdBu_r",
        diverging=True,
    )

    prompts_df = pd.DataFrame(
        {
            "prompt_id": [f"{item.task}_{item.qid}" for item in items],
            "prompt": [item.prompt for item in items],
            "task": [item.task for item in items],
            "gold": [
                item.answer if item.task == "math" else json.dumps(item.reference or {}, sort_keys=True)
                for item in items
            ],
        }
    )
    prefill_acts = capture_residual_stream(
        model_obj,
        tokenizer,
        prompts_df[["prompt_id", "prompt"]],
        batch_size=capture_batch_size,
        device=device,
    )
    cka_prefill = cka_matrix(prefill_acts, unbiased=False, device=cka_device_for(device))
    cka_prefill.to_csv(run_dir / "cka.csv")
    rho_prefill = residual_force_long(prefill_acts, upper_only=True, dtype=torch.float64)
    rho_prefill.to_csv(run_dir / "rho_phi_table.csv", index=False)
    save_heatmap(
        cka_prefill,
        run_dir / "cka.png",
        title=f"CKA connectome ({model_tag})",
        cmap="viridis",
        diverging=False,
    )

    gen_acts, generations = capture_generated_residual_stream(
        model_obj,
        tokenizer,
        prompts_df,
        batch_size=max(1, min(capture_batch_size, len(prompts_df))),
        max_new_tokens=generated_capture_max_new_tokens,
        device=device,
        generation_kwargs={"do_sample": False},
    )
    generations.to_csv(run_dir / "generations.csv", index=False)
    prompts_df.to_csv(run_dir / "generated_prompts.csv", index=False)
    cka_generated = cka_matrix(gen_acts, unbiased=False, device=cka_device_for(device))
    cka_generated.to_csv(run_dir / "cka_generated.csv")
    rho_generated = residual_force_long(gen_acts, upper_only=True, dtype=torch.float64)
    rho_generated.to_csv(run_dir / "rho_phi_generated.csv", index=False)
    save_heatmap(
        cka_generated,
        run_dir / "cka_generated.png",
        title=f"Generated-token CKA ({model_tag})",
        cmap="viridis",
        diverging=False,
    )

    _write_pickle(run_dir / "combined_results.pkl", combined_results)
    _write_pickle(run_dir / "math_results.pkl", math_results)
    _write_pickle(run_dir / "eq_results.pkl", eq_results)
    np.savez_compressed(
        run_dir / "per_question.npz",
        start=long_df["start"].to_numpy(dtype=int),
        end=long_df["end"].to_numpy(dtype=int),
        combined=per_window_combined,
    )

    if len(long_df) >= 3 and long_df["delta_score"].std() > 0 and long_df["math_score"].std() > 0:
        from scipy.stats import spearmanr

        sp_math = float(spearmanr(long_df["delta_score"], long_df["math_score"]).statistic)
        sp_eq = float(spearmanr(long_df["delta_score"], long_df["eq_score"]).statistic)
    else:
        sp_math = float("nan")
        sp_eq = float("nan")

    summary = {
        "mode": "combined_eq_math",
        "model": model,
        "family": model_family,
        "model_tag": model_tag,
        "dtype": dtype,
        "device": str(device),
        "n_layers": n_layers,
        "n_windows": len(windows),
        "math_set": math_set,
        "eq_set": eq_set,
        "n_math": baseline_summary["n_math"],
        "n_eq": baseline_summary["n_eq"],
        "baseline_math_score": baseline_summary["math_score"],
        "baseline_eq_score": baseline_summary["eq_score"],
        "baseline_combined_score": baseline_summary["combined_score"],
        "best_delta": float(long_df["delta_score"].max()) if len(long_df) else 0.0,
        "best_window": (
            [
                int(long_df.loc[long_df["delta_score"].idxmax(), "start"]),
                int(long_df.loc[long_df["delta_score"].idxmax(), "end"]),
            ]
            if len(long_df)
            else None
        ),
        "frac_windows_positive": float((long_df["delta_score"] > 0).mean()) if len(long_df) else 0.0,
        "spearman_delta_math": sp_math,
        "spearman_delta_eq": sp_eq,
        "boot": int(boot),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    artifact_paths = sorted(str(p.relative_to(run_dir)) for p in run_dir.glob("*") if p.is_file())
    manifest = {
        "model_profile": profile.to_dict(),
        "cli_args": {
            "model": model,
            "dtype": dtype,
            "math_set": math_set,
            "eq_set": eq_set,
            "seed": seed,
            "stride": stride,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "capture_batch_size": capture_batch_size,
            "generated_capture_max_new_tokens": generated_capture_max_new_tokens,
            "n_math": n_math,
            "n_eq": n_eq,
            "boot": boot,
        },
        "datasets": [
            {
                "name": rec.name,
                "path": str(rec.path),
                "source": rec.source,
                "sha256": rec.sha256,
                "n_items": rec.n_items,
            }
            for rec in dataset_records
        ],
        "artifacts": artifact_paths,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    typer.echo(f"Wrote {run_dir}")
    return run_dir


@app.command("probe")
def probe_cmd(
    model: str = typer.Option("EleutherAI/pythia-70m", help="HuggingFace model name."),
    dtype: str = typer.Option("float32", help="float32|bfloat16|int8|int4"),
    math_set: str = typer.Option("16", help="Canonical math dataset size: 16 or 120."),
    eq_set: str = typer.Option("16", help="Canonical eq dataset size: 16 or 140."),
    seed: int = typer.Option(0),
    stride: int = typer.Option(1, help="Layer stride for RYS window sweep."),
    batch_size: int = typer.Option(16, help="Generation batch size."),
    max_new_tokens: int = typer.Option(64, help="Generation tokens per probe sample."),
    capture_batch_size: int = typer.Option(8, help="Batch size for activation capture."),
    generated_capture_max_new_tokens: int = typer.Option(64, help="Generated-token capture new tokens."),
    n_math: int = typer.Option(0, help="Cap number of math samples (0 = all)."),
    n_eq: int = typer.Option(0, help="Cap number of eq samples (0 = all)."),
    boot: int = typer.Option(4000, help="Stored for compatibility with existing summaries."),
    family: str = typer.Option("", help="Override result family folder."),
    output_dir: Path = typer.Option(Path("results/LLM"), help="Result root folder."),
    prompt_policy: str = typer.Option(None, help="Force prompt mode: plain or chat."),
    use_no_think_prefix: bool = typer.Option(True, help="Prefix user prompts with /no_think when chat mode is used."),
) -> None:
    _run_probe(
        model=model,
        dtype=dtype,
        output_dir=output_dir,
        family=family or None,
        math_set=math_set,
        eq_set=eq_set,
        seed=seed,
        stride=stride,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        capture_batch_size=capture_batch_size,
        generated_capture_max_new_tokens=generated_capture_max_new_tokens,
        n_math=n_math,
        n_eq=n_eq,
        boot=boot,
        prompt_policy=prompt_policy,
        use_no_think_prefix=use_no_think_prefix,
    )


def _delegate_script(script_path: Path, argv: list[str]) -> None:
    cmd = ["python", str(script_path), *argv]
    subprocess.run(cmd, check=True)


@app.command("guesstimation")
def guesstimation_cmd(
    model: str = typer.Option("EleutherAI/pythia-70m"),
    seed: int = typer.Option(0),
    batch_size: int = typer.Option(32),
    max_new_tokens: int = typer.Option(12),
    capture_n: int = typer.Option(32),
    capture_batch_size: int = typer.Option(8),
    stride: int = typer.Option(1),
    dtype: str = typer.Option("float32"),
    boot: int = typer.Option(4000),
    n_questions: int = typer.Option(0),
    family: str = typer.Option(""),
    output_dir: Path = typer.Option(Path("results/LLM")),
) -> None:
    """Run legacy guesstimation script through unified CLI."""
    root = repo_root()
    script = root / "scripts" / "run_guesstimation_rys.py"
    resolved_family = family or infer_family(model)
    args = [
        "--model",
        model,
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--capture-n",
        str(capture_n),
        "--capture-batch-size",
        str(capture_batch_size),
        "--stride",
        str(stride),
        "--dtype",
        dtype,
        "--boot",
        str(boot),
        "--n-questions",
        str(n_questions),
        "--output-dir",
        str(output_dir),
        "--family",
        resolved_family,
    ]
    _delegate_script(script, args)


@app.command("geometry")
def geometry_cmd(
    model: str = typer.Option("EleutherAI/pythia-70m"),
    dtype: str = typer.Option("float32"),
    capture_n: int = typer.Option(44),
    capture_batch_size: int = typer.Option(8),
    top_k_dims: int = typer.Option(5),
    seed: int = typer.Option(0),
    delta_long: Path = typer.Option(None),
    family: str = typer.Option(""),
    output_dir: Path = typer.Option(Path("results/LLM")),
) -> None:
    """Run robust-geometry controls script through unified CLI."""
    root = repo_root()
    script = root / "scripts" / "run_robust_geometry_controls.py"
    resolved_family = family or infer_family(model)
    args = [
        "--model",
        model,
        "--dtype",
        dtype,
        "--capture-n",
        str(capture_n),
        "--capture-batch-size",
        str(capture_batch_size),
        "--top-k-dims",
        str(top_k_dims),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--family",
        resolved_family,
    ]
    if delta_long is not None:
        args.extend(["--delta-long", str(delta_long)])
    _delegate_script(script, args)


@app.command("all")
def all_cmd(
    model: str = typer.Option("EleutherAI/pythia-70m"),
    dtype: str = typer.Option("float32"),
    math_set: str = typer.Option("16"),
    eq_set: str = typer.Option("16"),
    seed: int = typer.Option(0),
    stride: int = typer.Option(1),
    batch_size: int = typer.Option(16),
    max_new_tokens: int = typer.Option(64),
    capture_batch_size: int = typer.Option(8),
    generated_capture_max_new_tokens: int = typer.Option(64),
    n_math: int = typer.Option(0),
    n_eq: int = typer.Option(0),
    family: str = typer.Option(""),
    output_dir: Path = typer.Option(Path("results/LLM")),
    prompt_policy: str = typer.Option(None),
    use_no_think_prefix: bool = typer.Option(True),
) -> None:
    run_dir = _run_probe(
        model=model,
        dtype=dtype,
        output_dir=output_dir,
        family=family or None,
        math_set=math_set,
        eq_set=eq_set,
        seed=seed,
        stride=stride,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        capture_batch_size=capture_batch_size,
        generated_capture_max_new_tokens=generated_capture_max_new_tokens,
        n_math=n_math,
        n_eq=n_eq,
        boot=4000,
        prompt_policy=prompt_policy,
        use_no_think_prefix=use_no_think_prefix,
    )
    geometry_cmd(
        model=model,
        dtype=dtype,
        capture_n=max(n_math + n_eq, 4) if (n_math > 0 or n_eq > 0) else 44,
        capture_batch_size=capture_batch_size,
        top_k_dims=5,
        seed=seed,
        delta_long=run_dir / "delta_score_long.csv",
        family=family,
        output_dir=output_dir,
    )


def main() -> None:
    app()

