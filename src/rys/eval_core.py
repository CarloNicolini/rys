"""Shared helpers for LLM RYS evaluation scripts and the ``rys-eval`` CLI.

This module is the single source of truth for the small utilities that were
previously copy-pasted across ``scripts/run_*_guesstimation_rys.py``,
``scripts/run_robust_geometry_controls.py``, and ``src/rys/cli.py``:

- device / dtype resolution
- model family + tag inference
- RYS half-open window enumeration
- heatmap saving
- delta-long discovery (joining geometry to behavioural deltas)
- functional Spearman correlation between deltas and predictors
- band-Jacobian :math:`\\sigma_{\\max}(D\\Phi)` estimation setup

Keeping them here lets the scripts and the CLI import one implementation instead
of drifting copies, following the DRY principle.
"""

from __future__ import annotations

import glob
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

DTYPE_CHOICES = ("float32", "bfloat16", "int8", "int4")
_DTYPE_MAP: dict[str, torch.dtype | None] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": None,
    "int4": None,
}


def resolve_device() -> torch.device:
    """Pick the best available acceleration device: CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str) -> torch.dtype | None:
    """Map a dtype string to a ``torch.dtype`` (``None`` for quantised loads)."""
    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {DTYPE_CHOICES}.")
    return _DTYPE_MAP[dtype]


def infer_family(model_name: str) -> str:
    """Default family bucket: the org prefix when a slash is present.

    ``meta-llama/Llama-3.2-1B`` -> ``meta-llama``; ``pythia-70m`` -> ``pythia-70m``.
    """
    if "/" in model_name:
        return model_name.split("/", 1)[0].lower()
    return model_name.lower()


def infer_model_tag(model_name: str) -> str:
    """The checkpoint tag: the part after the org slash, or the whole name."""
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def swept_windows(n_layers: int, stride: int = 1) -> list[tuple[int, int]]:
    """Half-open RYS windows ``(i, j)`` with ``0 <= i < j < L`` on a layer stride.

    The last layer ``L-1`` is always kept as a candidate ``j`` so deep models
    still probe coda-crossing windows even under a coarse stride.
    """
    ends = sorted(set(range(0, n_layers, stride)) | {n_layers - 1})
    starts = sorted(set(range(0, n_layers, stride)))
    return [(i, j) for i in starts for j in ends if i < j]


def save_heatmap(
    matrix,
    path: Path,
    *,
    title: str,
    cmap: str = "viridis",
    cbar_label: str = "",
    diverging: bool = False,
) -> None:
    """Render a square matrix/DataFrame to a PNG, skipping empty (all-NaN) inputs."""
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


def square_from_long(
    long_df: pd.DataFrame,
    n_layers: int,
    value_col: str,
    layer_col_i: str = "layer_i",
    layer_col_j: str = "layer_j",
) -> pd.DataFrame:
    """Pivot a long-format pair table into an ``L x L`` DataFrame."""
    mat = np.full((n_layers, n_layers), np.nan)
    for _, r in long_df.iterrows():
        mat[int(r[layer_col_i]), int(r[layer_col_j])] = r[value_col]
    return pd.DataFrame(mat, index=range(n_layers), columns=range(n_layers))


def latest_delta_long(family: str, tag: str, repo_root: Path | None = None) -> Path | None:
    """Most recent ``delta_score_long.csv`` for ``<family>/<tag>`` under ``results/LLM``.

    ``repo_root`` defaults to the directory containing this package's parent
    (the repository root with ``results/`` and ``pyproject.toml``).
    """
    root = repo_root or _default_repo_root()
    patterns = [
        str(root / "results" / "LLM" / family / "guesstimation_rys" / tag / "*" / "delta_score_long.csv"),
        str(root / "results" / "LLM" / family / "guesstimation_rys" / tag.lower() / "*" / "delta_score_long.csv"),
    ]
    runs = sorted(path for pattern in patterns for path in glob.glob(pattern))
    return Path(runs[-1]) if runs else None


def functional_corr(
    delta_long: pd.DataFrame,
    predictors: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Spearman correlations between every predictor column and the RYS delta.

    Joins ``delta_long`` (keyed by ``start``/``end``) to ``predictors`` (keyed
    by ``layer_i``/``layer_j``) and returns the merged frame plus a summary dict
    with one ``spearman_delta_<col>`` entry per predictor column, plus the best
    window coordinates.
    """
    merged = delta_long.merge(
        predictors, left_on=["start", "end"], right_on=["layer_i", "layer_j"], how="inner"
    )

    def corr(col: str) -> float:
        if len(merged) < 3 or merged["delta_score"].std() == 0 or merged[col].std() == 0:
            return float("nan")
        return float(spearmanr(merged["delta_score"], merged[col]).statistic)

    summary: dict[str, float] = {"n_windows": int(len(merged))}
    predictor_cols = [c for c in predictors.columns if c not in {"layer_i", "layer_j"} and c in merged.columns]
    for col in predictor_cols:
        summary[f"spearman_delta_{col}"] = corr(col)
    if len(merged):
        summary["best_delta"] = float(merged["delta_score"].max())
        best = merged.loc[merged["delta_score"].idxmax()]
        summary["best_window_start"] = int(best["start"])
        summary["best_window_end"] = int(best["end"])
    return merged, summary


def cka_device_for(device: torch.device) -> torch.device:
    """CKA is O(L^2 N^2); run on GPU when available, else CPU."""
    return device if device.type == "cuda" else torch.device("cpu")


def precompute_base_states(model, tokenizer, prompt):
    """Run one forward pass to cache hidden states, the causal mask, and rotary embeddings.

    Works for any decoder-only HF model whose base (``model.model``) exposes
    both ``layers`` and ``rotary_emb`` (Pythia after the load-time alias,
    Qwen3, Llama, Mistral, Gemma, ...).

    Returns
    -------
    hidden_states : tuple of tensors (one per layer boundary, length L+1)
    causal_mask   : additive causal mask compatible with the layer forward signature
    position_embeddings : (cos, sin) from the model's rotary embedding module
    """
    from transformers.masking_utils import create_causal_mask

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    seq_len = inputs.input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_embeds = outputs.hidden_states[0]
    causal_mask = create_causal_mask(
        config=model.config,
        inputs_embeds=dummy_embeds,
        attention_mask=inputs.attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    rotary_emb = model.model.rotary_emb
    position_embeddings = rotary_emb(dummy_embeds, position_ids=position_ids)
    return outputs.hidden_states, causal_mask, position_embeddings


def compute_block_jacobian_sigma_max(
    model,
    base_hidden_states,
    causal_mask,
    position_embeddings,
    start_layer: int,
    end_layer: int,
    n_iter: int = 20,
) -> float:
    r"""Largest singular value of the band map :math:`\Phi_{[i, j)}` Jacobian.

    The RYS surgery duplicates the band by feeding the layer-``end`` state back
    through layers ``[start, end)``, so the object that governs whether that
    extra pass refines or blows up the representation is :math:`D\Phi`,
    evaluated at the band input :math:`x_i`. We return
    :math:`\sigma_{\max}(D\Phi)`, estimated matrix-free by power iteration via
    :func:`rys.theory_validation.block_jacobian_sigma_max`.
    """
    from rys.theory_validation import block_jacobian_sigma_max

    layers = model.model.layers
    h_in = base_hidden_states[start_layer]
    position_ids = torch.arange(h_in.shape[1], device=h_in.device).unsqueeze(0)

    def block_map(h):
        for k in range(start_layer, end_layer):
            out = layers[k](
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            h = out[0] if isinstance(out, tuple) else out
        return h

    return block_jacobian_sigma_max(block_map, h_in, n_iter=n_iter)


def write_summary(run_dir: Path, summary: dict) -> None:
    """Write a JSON summary with stable, ``default=str`` serialisation."""
    (run_dir / "summary.json").write_text(_json_dumps(summary))


def _json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


def _default_repo_root() -> Path:
    """Walk up from this file to the directory containing ``pyproject.toml``."""
    p = Path(__file__).resolve()
    for cand in [p.parent, *p.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "results").exists():
            return cand
    raise FileNotFoundError("Could not resolve repository root (need pyproject.toml and results/).")


def repo_root(start: Path | None = None) -> Path:
    """Resolve the repository root from ``start`` (default: current directory)."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "src").exists():
            return cand
    raise FileNotFoundError("Could not resolve repository root from current directory.")


def timestamped_run_dir(output_dir: Path, family: str, experiment: str, tag: str) -> Path:
    """Build and create a timestamped run directory under ``results/LLM`` layout."""
    import time

    run_dir = output_dir / family / experiment / tag / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_capture_prompts(questions: Iterable[dict], n: int) -> pd.DataFrame:
    """Build the ``[prompt_id, prompt]`` frame used for CKA capture."""
    qs = list(questions)[:n]
    return pd.DataFrame(
        {"prompt_id": [f"q{i}" for i in range(len(qs))], "prompt": [q["question"] for q in qs]}
    )
