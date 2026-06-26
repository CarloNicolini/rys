"""RYS on base Qwen3 models with the guesstimation partial-credit probe.

Thin Qwen3-defaulting wrapper around the unified
:mod:`scripts.run_guesstimation_rys` probe. Outputs are identical in shape to
the Pythia variant because the underlying implementation is shared.

Examples
--------
::

    uv run python scripts/run_qwen3_guesstimation_rys.py --model Qwen/Qwen3-0.6B --dtype bfloat16
    uv run python scripts/run_qwen3_guesstimation_rys.py --model Qwen/Qwen3-14B --dtype int8 --batch-size 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import typer


def _load_unified_run():
    """Import :mod:`scripts.run_guesstimation_rys` regardless of the launch cwd."""
    import importlib.util

    target = Path(__file__).resolve().parent / "run_guesstimation_rys.py"
    spec = importlib.util.spec_from_file_location("rys._unified_guesstimation", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._run


_run = _load_unified_run()


def main(
    model: str = typer.Option("Qwen/Qwen3-0.6B", help="HuggingFace Qwen3 base checkpoint."),
    seed: int = typer.Option(0, help="Question-set seed."),
    batch_size: int = typer.Option(16, help="Questions generated per batch."),
    max_new_tokens: int = typer.Option(12, help="Generated tokens per answer."),
    capture_n: int = typer.Option(32, help="Prompts used for the CKA connectome."),
    capture_batch_size: int = typer.Option(4, help="Batch size for residual capture."),
    stride: int = typer.Option(
        1, help="Sweep windows on a layer stride (use 2 for 8B/14B if the full sweep is too slow)."
    ),
    dtype: str = typer.Option(
        "bfloat16",
        help="Weights precision: 'float32', 'bfloat16', 'int8', or 'int4'. Use int8 first for Qwen3-14B.",
    ),
    boot: int = typer.Option(4000, help="Bootstrap resamples over questions."),
    n_questions: int = typer.Option(0, help="Cap the question set (0 = use all). Useful for fast smoke tests."),
    family: str = typer.Option("", help="Output family bucket; auto-inferred if empty."),
    output_dir: Path = typer.Option(Path("results/LLM"), help="Run output root (family is appended)."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


main.__doc__ = __doc__

if __name__ == "__main__":
    typer.run(main)
