"""RYS on frozen Pythia with David Ng's generative guesstimation probe.

Thin Pythia-defaulting wrapper around the unified
:mod:`scripts.run_guesstimation_rys` probe (which handles any decoder-only
HF model). Kept as a separate entrypoint so the historical
``--model EleutherAI/pythia-70m`` invocation and the ``results/pythia...``
defaults still work; all logic lives in one place.

Examples
--------
::

    uv run python scripts/run_pythia_guesstimation_rys.py --model EleutherAI/pythia-70m
    uv run python scripts/run_pythia_guesstimation_rys.py --model EleutherAI/pythia-12b --dtype int4 --batch-size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import typer


def _load_unified_run():
    """Import :mod:`scripts.run_guesstimation_rys` regardless of the launch cwd.

    When launched as ``python scripts/run_pythia_guesstimation_rys.py`` the
    repo root is not on ``sys.path`` and ``import scripts...`` fails; when
    launched via ``uv run`` it is. We resolve by file path to stay robust.
    """
    import importlib.util

    here = Path(__file__).resolve().parent
    target = here / "run_guesstimation_rys.py"
    spec = importlib.util.spec_from_file_location("rys._unified_guesstimation", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._run


_run = _load_unified_run()


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
    n_questions: int = typer.Option(0, help="Cap the question set (0 = use all). Useful for fast smoke tests."),
    family: str = typer.Option("", help="Output family bucket; auto-inferred if empty."),
    output_dir: Path = typer.Option(Path("results/LLM"), help="Run output root (family is appended)."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


main.__doc__ = __doc__

if __name__ == "__main__":
    typer.run(main)
