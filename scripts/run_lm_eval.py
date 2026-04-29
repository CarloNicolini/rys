"""CLI shim: load Llama-3.2-3B (optionally INT8), apply RYS, run lm-evaluation-harness.

Usage
-----
::

    uv run python scripts/run_lm_eval.py \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --tasks gsm8k commonsense_qa mmlu_high_school_mathematics \\
        --window 12 20 --n-repeats 2 \\
        --limit 250 --output results/lm_eval_outputs/rys_12_20.json

The script is intentionally thin: all mutable state lives in the
:func:`rys.surgery.apply_rys` context manager so the same model object can be
re-used across multiple RYS configurations from a higher-level driver script
(or from the notebook) without re-loading weights.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rys.surgery import apply_rys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--num-fewshot", type=int, default=8)
    parser.add_argument("--limit", type=int, default=250, help="Examples per task; full split if 0.")
    parser.add_argument("--batch-size", type=str, default="auto")
    parser.add_argument(
        "--window",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Half-open layer interval to duplicate; omit to evaluate the base model.",
    )
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--int8", action="store_true", help="Load with bitsandbytes 8-bit.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_hflm(model_id: str, *, int8: bool):
    """Instantiate ``lm_eval.models.huggingface.HFLM`` once and return it."""
    from lm_eval.models.huggingface import HFLM

    kwargs: dict = {"pretrained": model_id, "trust_remote_code": False}
    if int8:
        kwargs["load_in_8bit"] = True
    else:
        kwargs["dtype"] = "bfloat16"
    return HFLM(**kwargs)


def main() -> None:
    args = parse_args()
    from lm_eval import simple_evaluate

    print(f"Loading model {args.model} (int8={args.int8})...")
    hflm = build_hflm(args.model, int8=args.int8)
    underlying = hflm.model  # the bare HF causal-LM

    config = {
        "model": args.model,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "limit": None if args.limit <= 0 else args.limit,
        "window": tuple(args.window) if args.window else None,
        "n_repeats": args.n_repeats,
    }
    print("Eval config:", json.dumps(config, indent=2))

    t0 = time.time()
    if args.window is None or args.n_repeats == 1:
        results = simple_evaluate(
            model=hflm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=config["limit"],
            batch_size=args.batch_size,
        )
    else:
        with apply_rys(underlying, tuple(args.window), n_repeats=args.n_repeats):
            results = simple_evaluate(
                model=hflm,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=config["limit"],
                batch_size=args.batch_size,
            )
    elapsed = time.time() - t0

    payload = {"config": config, "results": results.get("results", {}), "elapsed_s": elapsed}
    print(json.dumps(payload, indent=2, default=str))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
