"""RYS on frozen Pythia with David Ng's generative guesstimation probe.

This is the faithful replication of the downstream proxy from the original RYS
blog post: the model must emit a numeric answer in a single pass (no
chain-of-thought) and is scored by partial-credit closeness, rather than by a
multiple-choice loglikelihood. The graded generative signal is what discovered
the RYS effect at the 72B scale, so we test whether it reveals a beneficial
window on the Pythia scale ladder where the binary MC metric did not.

Per model it records the baseline probe score, the RYS $\\Delta$-score matrix over
all $(i,j)$ windows, the CKA connectome and $\\rho/\\Psi$ table, and the
geometry->effect rank correlations with a bootstrap over questions. Outputs land
in ``results/pythia_guesstimation_rys/<model>/<timestamp>/``.

Examples
--------
::

    uv run python scripts/run_pythia_guesstimation_rys.py --model EleutherAI/pythia-70m
    uv run python scripts/run_pythia_guesstimation_rys.py --model EleutherAI/pythia-12b --load-in-4bit --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from scipy.stats import spearmanr

from rys.activations import capture_residual_stream
from rys.cka import cka_matrix
from rys.gsm8k_mc import load_pythia
from rys.guesstimation import generate_numbers, make_guesstimation_questions, score_estimates
from rys.theory_validation import rho_phi_table, theory_fit


def main(
    model: str = typer.Option("EleutherAI/pythia-70m", help="HuggingFace Pythia checkpoint."),
    seed: int = typer.Option(0, help="Question-set seed."),
    batch_size: int = typer.Option(32, help="Questions generated per batch."),
    max_new_tokens: int = typer.Option(12, help="Generated tokens per answer."),
    capture_n: int = typer.Option(32, help="Prompts used for the CKA connectome."),
    capture_batch_size: int = typer.Option(8, help="Batch size for residual capture."),
    stride: int = typer.Option(
        1, help="Sweep windows on a layer stride (use 2 for very deep models to keep generation tractable)."
    ),
    dtype: str = typer.Option(
        "float32",
        help="Weights precision: 'float32' (safe for small models; bf16 breaks their generation), "
        "'bfloat16' (large models that do not fit in fp32), or 'int4' (bitsandbytes NF4, for pythia-12b).",
    ),
    boot: int = typer.Option(4000, help="Bootstrap resamples over questions."),
    output_dir: Path = typer.Option(Path("results/pythia_guesstimation_rys"), help="Run output root."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def swept_windows(L: int, stride: int = 1) -> list[tuple[int, int]]:
    """Half-open windows ``(i, j)``, ``0<=i<j<L``, sampled on a layer stride.

    The last layer ``L-1`` is always kept as a candidate ``j`` so deep models
    still probe coda-crossing windows even under a coarse stride.
    """
    ends = sorted(set(range(0, L, stride)) | {L - 1})
    starts = sorted(set(range(0, L, stride)))
    return [(i, j) for i in starts for j in ends if i < j]


def save_heatmap(matrix, path, *, title, cmap, cbar_label, diverging):
    values = matrix.to_numpy(dtype=float) if isinstance(matrix, pd.DataFrame) else np.asarray(matrix, float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    if diverging:
        bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
        vmin, vmax = -bound, bound
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("end layer j (half-open band [i, j))")
    ax.set_ylabel("start layer i")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device()
    rng = np.random.default_rng(args.seed)
    load_in_4bit = args.dtype == "int4"
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int4": None}[args.dtype]

    tag = args.model.split("/")[-1]
    run_dir = args.output_dir / tag / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device {device}, dtype {args.dtype}, run_dir {run_dir}", flush=True)

    model, tok = load_pythia(args.model, device=device, dtype=torch_dtype, load_in_4bit=load_in_4bit)
    L = len(model.model.layers)
    questions = make_guesstimation_questions(seed=args.seed)
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
    for w, (i, j) in enumerate(windows):
        per_q[w] = probe(window=(i, j))
        if (w + 1) % 25 == 0:
            print(f"  {w + 1}/{len(windows)} windows", flush=True)

    delta = per_q.mean(1) - base.mean()
    matrix = np.full((L, L), np.nan)
    rows = []
    for w, (i, j) in enumerate(windows):
        matrix[i, j] = delta[w]
        rows.append({"start": i, "end": j, "rys_score": float(per_q[w].mean()), "delta_score": float(delta[w])})
    long = pd.DataFrame(rows)
    pd.DataFrame(matrix, index=range(L), columns=range(L)).to_csv(run_dir / "delta_score.csv")
    long.to_csv(run_dir / "delta_score_long.csv", index=False)
    save_heatmap(
        pd.DataFrame(matrix),
        run_dir / "delta_score.png",
        title=f"RYS Delta guesstimation score ({tag})",
        cmap="RdBu_r",
        cbar_label="Delta partial-credit score",
        diverging=True,
    )

    # CKA connectome + geometry
    cap = pd.DataFrame(
        {"prompt_id": [f"q{i}" for i in range(min(args.capture_n, n_q))],
         "prompt": [questions[i]["question"] for i in range(min(args.capture_n, n_q))]}
    )
    acts = capture_residual_stream(model, tok, cap, batch_size=args.capture_batch_size, device=device)
    cka_device = device if device.type == "cuda" else "cpu"
    cka = cka_matrix(acts, unbiased=False, device=cka_device)
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
    order = [windows.index((int(r.start), int(r.end))) for r in mg.itertuples(index=False)]
    per_q_ord = per_q[order]
    sp_rho = float(spearmanr(mg["delta_score"], rho).statistic)
    sp_cka = float(spearmanr(mg["delta_score"], ckaw).statistic)
    boot_rho = np.empty(args.boot)
    boot_cka = np.empty(args.boot)
    boot_best = np.empty(args.boot)
    for t in range(args.boot):
        idx = rng.integers(0, n_q, n_q)
        db = per_q_ord[:, idx].mean(1) - base[idx].mean()
        boot_rho[t] = spearmanr(db, rho).statistic
        boot_cka[t] = spearmanr(db, ckaw).statistic
        boot_best[t] = db.max()

    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]

    summary = {
        "model": args.model, "L": L, "n_questions": n_q, "n_windows": len(windows),
        "dtype": args.dtype, "stride": args.stride,
        "baseline_score": round(float(base.mean()), 4),
        "mean_delta": round(float(delta.mean()), 4),
        "best_delta": round(float(delta.max()), 4),
        "best_window": [int(mg.loc[mg["delta_score"].idxmax(), "start"]), int(mg.loc[mg["delta_score"].idxmax(), "end"])],
        "best_delta_ci95": ci(boot_best),
        "frac_windows_positive": round(float((delta > 0).mean()), 3),
        "spearman_delta_rho": round(sp_rho, 3), "spearman_delta_rho_ci95": ci(boot_rho),
        "spearman_delta_cka": round(sp_cka, 3), "spearman_delta_cka_ci95": ci(boot_cka),
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
