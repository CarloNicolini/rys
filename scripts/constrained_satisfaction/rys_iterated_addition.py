"""Train a weight-tied *looped* adder whose depth is the RYS iteration axis.

Motivation
----------
The fixed-depth baseline (``run_addition_experiment.py``) learns a parallel,
operand-count-bound template: depth-demand grows with ``n`` but the trained
layers are heterogeneous expanders (block-Jacobian spectral radius far above 1),
so replaying a band (RYS) diverges instead of continuing the computation, and
out-of-distribution operand counts collapse.

This script trains a small **band** of layers that is applied a *variable*
number of times, with a fixed-point / anytime objective, so that "more RYS
rounds" genuinely means "more iterations of one stable update". The band is the
half-open window ``(0, block_size)`` that :func:`rys.surgery.apply_rys` would
replay, so running the band ``R`` times here is exactly RYS with ``n_repeats=R``.

Honesty contract
----------------
- Train only on ``--train-ns`` (default 2..8). ``--dev-ns`` (default 9) is the
  only extrapolation signal used for any decision. ``--ood-ns`` (default
  10,12,16) is reported but never used for training or selection.
- Optional count/length invariance augmentation only inserts null (zero)
  operands *within* the trained operand-count range, so no longer sequence than
  ``max(train_ns)`` is ever shown. Longer sequences at eval are genuine
  extrapolation; the extra rounds to solve them must come from RYS.

The "RYS test" is then: with the band and round count chosen from in-distribution
behaviour, does held-out accuracy rise monotonically as the band is iterated
more times?
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rys.addition_data import ANS, EQ, PLUS, verify_addition_tensor
from rys.addition_transformer import AdditionConfig, AdditionTransformer


def parse_ns(value: str) -> list[int]:
    return [int(p) for p in value.split(",") if p.strip()]


def build_batch(
    rng: np.random.Generator,
    n_operands: int,
    batch_size: int,
    *,
    digits: int,
    answer_width: int,
    invariance: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised batch of fixed operand-count addition problems.

    With ``invariance`` a random per-row fraction of operand slots is zeroed
    (null operands), teaching count-invariance without ever growing the
    sequence beyond ``n_operands`` operands.
    """
    draws = rng.integers(0, 10**digits, size=(batch_size, n_operands), dtype=np.int64)
    if invariance and n_operands >= 2:
        null_frac = rng.uniform(0.0, 0.6, size=(batch_size, 1))
        null_mask = rng.random((batch_size, n_operands)) < null_frac
        draws = np.where(null_mask, 0, draws)
    answers = draws.sum(axis=1)

    seq_len = (digits + 1) * n_operands + answer_width
    tokens = np.empty((batch_size, seq_len), dtype=np.int64)
    for i in range(n_operands):
        base = i * (digits + 1)
        for j in range(digits):
            place = 10 ** (digits - 1 - j)
            tokens[:, base + j] = (draws[:, i] // place) % 10
        tokens[:, base + digits] = EQ if i == n_operands - 1 else PLUS
    tokens[:, (digits + 1) * n_operands :] = ANS

    targets = np.empty((batch_size, answer_width), dtype=np.int64)
    for p in range(answer_width):
        targets[:, p] = (answers // (10**p)) % 10

    return (
        torch.from_numpy(tokens),
        torch.from_numpy(targets),
        torch.from_numpy(answers),
    )


def run_band(
    model: AdditionTransformer,
    input_ids: torch.Tensor,
    n_rounds: int,
    *,
    answer_width: int,
    inject: bool,
    every_layer: bool = False,
) -> list[torch.Tensor]:
    """Apply the layer band ``n_rounds`` times; return answer-slot logits.

    Running the band ``n_rounds`` times is identical to ``apply_rys`` over window
    ``(0, len(layers))`` with ``n_repeats=n_rounds``. Input injection re-adds the
    token embedding after every band application (looped-transformer style).

    With ``every_layer`` a readout is collected after *every* layer application
    (deep supervision, matching the baseline's per-layer iterative-solver
    objective); otherwise one readout per completed band (the unit RYS replays,
    used for evaluation so that round index == RYS repeats).
    """
    cfg = model.config

    def readout(h: torch.Tensor) -> torch.Tensor:
        return model.unembed(model.model.norm(h[:, -answer_width:, :]))

    one_hot = F.one_hot(input_ids, num_classes=cfg.vocab_in).to(model.model.embed.weight.dtype)
    emb = model.model.dropout(model.model.embed(one_hot))
    hidden = emb
    readouts: list[torch.Tensor] = []
    for _ in range(n_rounds):
        for layer in model.model.layers:
            hidden = layer(hidden, attention_mask=None)[0]
            if every_layer:
                readouts.append(readout(hidden))
        if inject:
            hidden = hidden + emb
        if not every_layer:
            readouts.append(readout(hidden))
    return readouts


@torch.inference_mode()
def evaluate(
    model: AdditionTransformer,
    rng: np.random.Generator,
    n_operands: int,
    *,
    eval_rounds: list[int],
    digits: int,
    answer_width: int,
    inject: bool,
    n_examples: int,
    device: torch.device,
    batch_size: int = 512,
) -> dict[int, float]:
    """Exact-accuracy of each round count in ``eval_rounds`` for one operand count."""
    max_rounds = max(eval_rounds)
    correct = {r: 0 for r in eval_rounds}
    total = 0
    done = 0
    while done < n_examples:
        bs = min(batch_size, n_examples - done)
        ids, _, ans = build_batch(
            rng, n_operands, bs, digits=digits, answer_width=answer_width, invariance=False
        )
        ids = ids.to(device)
        ans = ans.to(device)
        readouts = run_band(model, ids, max_rounds, answer_width=answer_width, inject=inject)
        for r in eval_rounds:
            preds = readouts[r - 1].argmax(dim=-1)
            correct[r] += int(verify_addition_tensor(preds, ans).sum().item())
        total += bs
        done += bs
    return {r: round(correct[r] / total, 4) for r in eval_rounds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--digits", type=int, default=3)
    ap.add_argument("--answer-width", type=int, default=6)
    ap.add_argument("--train-ns", type=str, default="2,3,4,5,6,7,8")
    ap.add_argument("--dev-ns", type=str, default="9")
    ap.add_argument("--ood-ns", type=str, default="10,12,16")
    ap.add_argument(
        "--axis", choices=["operands", "digits"], default="operands",
        help="OOD axis: operand count (vary n, fixed digits) or carry-chain length (fixed n, vary digits).",
    )
    ap.add_argument("--n-operands", type=int, default=3, help="Fixed operand count when --axis digits.")
    ap.add_argument("--train-digits", type=str, default="1,2,3,4,5")
    ap.add_argument("--dev-digits", type=str, default="6")
    ap.add_argument("--ood-digits", type=str, default="7,8,10")
    ap.add_argument("--block-size", type=int, default=2, help="Layers in the repeated band.")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-mlp", type=int, default=1024)
    ap.add_argument("--inject", action="store_true", help="Re-add input embedding each round.")
    ap.add_argument("--invariance", action="store_true", help="Null-operand augmentation.")
    ap.add_argument("--rounds-min", type=int, default=5, help="Min band iterations at train time.")
    ap.add_argument("--rounds-max", type=int, default=10, help="Max band iterations at train time.")
    ap.add_argument("--fixed-rounds", type=int, default=0, help="If >0, use this many band iterations every step (overrides random).")
    ap.add_argument("--curriculum", action="store_true", help="Grow difficulty: start with the easiest spec, add the next every --curriculum-epochs.")
    ap.add_argument("--curriculum-epochs", type=int, default=20, help="Epochs before introducing the next-harder spec.")
    ap.add_argument(
        "--sup-mode", type=str, default="anytime", choices=["anytime", "last", "tail"],
        help="Which rounds to supervise: anytime (all, weighted by round), last (final round only), tail (last few).",
    )
    ap.add_argument("--tail-k", type=int, default=3, help="Rounds supervised when --sup-mode tail.")
    ap.add_argument(
        "--per-layer-readout", action="store_true",
        help="Supervise after every layer (deep supervision, like the baseline) rather than once per band.",
    )
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=16384, help="Examples per epoch.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--eval-every-epochs", type=int, default=10)
    ap.add_argument("--eval-rounds", type=str, default="3,4,5,6,8,10,12,14,16,20")
    ap.add_argument("--eval-examples", type=int, default=2048)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_rounds = parse_ns(args.eval_rounds)

    # A "spec" is (n_operands, digits, answer_width). The training step samples a
    # spec; eval iterates labelled specs. This unifies the operand-count axis
    # (vary n) and the carry-chain axis (vary digits) behind one loop.
    def aw_for_digits(d: int) -> int:
        return d + 2  # enough for sums of a few operands, plus margin

    if args.axis == "operands":
        train_specs = [(n, args.digits, args.answer_width) for n in sorted(parse_ns(args.train_ns))]
        eval_specs = (
            [("train", n, args.digits, args.answer_width) for n in parse_ns(args.train_ns)]
            + [("dev", n, args.digits, args.answer_width) for n in parse_ns(args.dev_ns)]
            + [("ood", n, args.digits, args.answer_width) for n in parse_ns(args.ood_ns)]
        )

        def label(split: str, n: int, d: int, aw: int) -> str:
            return f"{split}_n{n}"
    else:
        no = args.n_operands
        train_specs = [(no, d, aw_for_digits(d)) for d in sorted(parse_ns(args.train_digits))]
        eval_specs = (
            [("train", no, d, aw_for_digits(d)) for d in parse_ns(args.train_digits)]
            + [("dev", no, d, aw_for_digits(d)) for d in parse_ns(args.dev_digits)]
            + [("ood", no, d, aw_for_digits(d)) for d in parse_ns(args.ood_digits)]
        )

        def label(split: str, n: int, d: int, aw: int) -> str:
            return f"{split}_d{d}"

    config = AdditionConfig(
        d_model=args.d_model,
        n_layers=args.block_size,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=0.0,
        pre_norm=True,
        weight_tied=False,  # block_size distinct layers form the repeated band
        use_rope=False,  # NoPE
        causal=True,
    )
    model = AdditionTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    out_dir = Path(args.out_dir) / f"{args.tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    steps_per_epoch = max(1, args.n_train // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    train_rng = np.random.default_rng(args.seed + 1)
    eval_rng_seed = args.seed + 9999

    header = {
        "event": "config",
        "tag": args.tag,
        "n_params": n_params,
        "config": config.__dict__,
        "args": vars(args),
        "device": str(device),
    }
    print(json.dumps(header), flush=True)
    (out_dir / "config.json").write_text(json.dumps(header, indent=2, default=str))

    def do_eval(epoch: int) -> dict:
        model.eval()
        rows = {}
        for split, n, d, aw in eval_specs:
            rng = np.random.default_rng(eval_rng_seed + n * 100 + d)
            rows[label(split, n, d, aw)] = evaluate(
                model, rng, n,
                eval_rounds=eval_rounds, digits=d, answer_width=aw,
                inject=args.inject, n_examples=args.eval_examples, device=device,
            )
        model.train()
        rec = {"event": "eval", "epoch": epoch, "rounds": eval_rounds, "exact": rows}
        with metrics_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        return rec

    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        if args.curriculum:
            k = min(epoch // args.curriculum_epochs + 1, len(train_specs))
            active_specs = train_specs[:k]
        else:
            active_specs = train_specs
        for _ in range(steps_per_epoch):
            n, d, aw = random.choice(active_specs)
            ids, target, _ = build_batch(
                train_rng, n, args.batch_size,
                digits=d, answer_width=aw,
                invariance=(args.invariance and args.axis == "operands"),
            )
            ids = ids.to(device)
            target = target.to(device)
            R = args.fixed_rounds if args.fixed_rounds > 0 else random.randint(args.rounds_min, args.rounds_max)
            readouts = run_band(model, ids, R, answer_width=aw, inject=args.inject, every_layer=args.per_layer_readout)
            tgt = target.reshape(-1)
            if args.sup_mode == "last":
                # Supervise only the final round; random R supplies fixed-point
                # stability across depths without the impossible early-round pressure.
                loss = F.cross_entropy(readouts[-1].reshape(-1, config.n_digit_classes), tgt)
            elif args.sup_mode == "tail":
                k = min(args.tail_k, R)
                loss = torch.stack(
                    [F.cross_entropy(o.reshape(-1, config.n_digit_classes), tgt) for o in readouts[-k:]]
                ).mean()
            else:  # anytime, weighted toward later rounds
                num = 0.0
                den = 0.0
                for r, logits in enumerate(readouts, start=1):
                    ce = F.cross_entropy(logits.reshape(-1, config.n_digit_classes), tgt)
                    num = num + r * ce
                    den += r
                loss = num / den
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
            step += 1
        if epoch % args.eval_every_epochs == 0 or epoch == args.epochs:
            rec = do_eval(epoch)
            print(json.dumps({"event": "epoch", "epoch": epoch, "train_loss": round(running / steps_per_epoch, 4), "lr": scheduler.get_last_lr()[0]}), flush=True)

    torch.save({"state_dict": model.state_dict(), "config": config.__dict__, "args": vars(args)}, out_dir / "model.pt")
    print(json.dumps({"event": "done", "out_dir": str(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
