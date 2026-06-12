"""
N-ary addition experiment: operand-count generalization + RYS.

Trains an encoder-only RoPE Transformer to add ``n`` 3-digit operands on a uniform mixture of operand counts ``n in {2,4,8,16,32}`` directly on the input-output pair (no chain-of-thought), then uses the post-training RYS hook to repeat middle layers and add longer out-of-distribution operand counts with no retraining.

Per run it records:
- baseline accuracy per split (digit / exact);
- an accuracy-vs-rounds curve (read out after each layer) for every split;
- the validation token-stream CKA connectome and a rho/phi theory fit;
- an RYS sweep over windows x n_repeats for every OOD operand count, with the
  headline "exact accuracy vs n_repeats of the best window" plot.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader

from rys.addition_data import (
    AdditionDataset,
    AdditionStream,
    make_addition_examples,
    verify_addition_tensor,
)
from rys.addition_transformer import AdditionConfig, AdditionTransformer
from rys.cka import cka_matrix
from rys.surgery import apply_rys
from rys.theory_validation import rho_phi_table, theory_fit
from rys.training.device_cache import cache_batches_on_device
from rys.training.modules import AdditionLitModule
from rys.training.rys_logging import log_rys_progress
from rys.training.trainer import best_checkpoint_path, build_trainer, load_rys_model


def main(
    seed: int = typer.Option(0, help="Random seed."),
    train_ns: str = typer.Option("2,3,4,5,6,7,8", help="Comma list of training operand counts (dense, on-the-fly)."),
    ood_ns: str = typer.Option("10,12,16", help="Comma list of out-of-distribution operand counts."),
    val_ns: str = typer.Option("2,3,4,5,6,7,8", help="Comma list of in-distribution validation operand counts (stratified)."),
    test_ns: str = typer.Option("2,3,4,5,6,7,8", help="Comma list of in-distribution test operand counts (stratified)."),
    digits: int = typer.Option(3, help="Digits per operand."),
    answer_width: int = typer.Option(6, help="Number of answer-slot tokens (max sum width)."),
    n_train: int = typer.Option(8192, help="Training samples PER EPOCH (split evenly across train-ns, resampled on-the-fly)."),
    n_val: int = typer.Option(512, help="Validation examples (per operand count)."),
    n_test: int = typer.Option(512, help="Test/OOD examples (per operand count)."),
    batch_size: int = typer.Option(128, help="Batch size."),
    num_workers: int = typer.Option(
        4,
        help="DataLoader worker processes for validation/eval loaders (training streams stay at 0).",
    ),
    epochs: int = typer.Option(200, help="Training epochs (also the cosine schedule horizon)."),
    lr: float = typer.Option(1e-4, help="AdamW peak learning rate (cosine-decayed to 0)."),
    warmup_epochs: int = typer.Option(0, help="Linear LR warmup epochs before cosine decay (0 disables)."),
    loss_weighting: str = typer.Option(
        "none", help="Per-n training loss weighting: 'none' or 'linear' (weight proportional to operand count)."
    ),
    weight_decay: float = typer.Option(0.0, help="AdamW weight decay."),
    d_model: int = typer.Option(512, help="Model width (must be divisible by n-heads)."),
    n_layers: int = typer.Option(6, help="Number of transformer layers."),
    n_heads: int = typer.Option(8, help="Attention heads per layer."),
    d_mlp: int = typer.Option(2048, help="MLP hidden width."),
    dropout: float = typer.Option(0.0, help="Dropout probability."),
    weight_tied: bool = typer.Option(
        False, "--weight-tied/--no-weight-tied", help="Share one round across depth (iterated map)."
    ),
    pre_norm: bool = typer.Option(True, "--pre-norm/--post-norm", help="Pre-norm (additive residual) vs post-norm."),
    nope: bool = typer.Option(
        True, "--nope/--rope", help="NoPE (no positional encoding) vs RoPE. The paper uses NoPE."
    ),
    causal: bool = typer.Option(
        True,
        "--causal/--bidirectional",
        help="Causal (decoder-only) attention. Required for NoPE to break permutation symmetry.",
    ),
    deep_supervision: bool = typer.Option(
        True,
        "--deep-supervision/--no-deep-supervision",
        help="Supervise the answer readout after every layer (iterative-solver objective).",
    ),
    curriculum: bool = typer.Option(
        True,
        "--curriculum/--no-curriculum",
        help="Start training on the smallest operand count and add one larger n every --curriculum-epochs (cumulative).",
    ),
    curriculum_epochs: int = typer.Option(
        15, help="Epochs to train before introducing the next-larger operand count."
    ),
    max_repeat: int = typer.Option(6, help="Maximum total traversals of a window in the RYS sweep."),
    skip_rys: bool = typer.Option(
        False, "--skip-rys/--no-skip-rys", help="Skip the RYS sweep (Phase 1: just check decent performance)."
    ),
    capture_batches: int = typer.Option(4, help="Validation batches used for the CKA connectome."),
    checkpoint: Path | None = typer.Option(None, help="Load weights and skip training."),
    output_dir: Path = typer.Option(Path("results/addition"), help="Run output directory."),
) -> None:
    args = argparse.Namespace(**locals())
    _run(args)


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ns(value: str) -> list[int]:
    ns = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ns or any(n < 1 for n in ns):
        raise ValueError("operand counts must be positive integers.")
    return ns


def _eval_loader_kwargs(args: argparse.Namespace) -> dict:
    kwargs: dict = {"num_workers": args.num_workers}
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def make_loaders(
    args: argparse.Namespace,
) -> tuple[list[DataLoader], dict[str, DataLoader], list[str], list[int], list[str]]:
    """Build on-the-fly training loaders and fixed, per-n stratified eval loaders.

    Training uses one :class:`AdditionStream` loader per operand count in
    ``--train-ns``; each epoch resamples fresh problems so the model cannot
    memorise the training set (train and validation stay drawn from the same
    distribution). Validation and test are stratified per operand count in
    ``--val-ns`` / ``--test-ns`` (in-distribution), and each ``--ood-ns`` count
    is a separate extrapolation split. Every eval split is internally single-``n``.

    Returns the train loaders, the eval-loader dict keyed by split name, the
    ordered list of all eval split names, the training operand counts, and the
    subset of split names used for in-distribution validation.
    """
    train_ns = sorted(parse_ns(args.train_ns))
    val_ns = parse_ns(args.val_ns)
    test_ns = parse_ns(args.test_ns)
    ood_ns = parse_ns(args.ood_ns)
    per_n = max(1, args.n_train // len(train_ns))

    train_loaders: list[DataLoader] = []
    for offset, n_operands in enumerate(train_ns):
        stream = AdditionStream(
            n_operands,
            per_n,
            digits=args.digits,
            answer_width=args.answer_width,
            seed=args.seed + offset,
            prefix=f"train{n_operands}",
        )
        train_loaders.append(DataLoader(stream, batch_size=args.batch_size, num_workers=0))

    eval_loaders: dict[str, DataLoader] = {}
    eval_splits: list[str] = []

    def add_eval(name: str, examples: list) -> None:
        eval_loaders[name] = DataLoader(
            AdditionDataset(examples),
            batch_size=args.batch_size,
            shuffle=False,
            **_eval_loader_kwargs(args),
        )
        eval_splits.append(name)

    val_split_names: list[str] = []
    for offset, n_operands in enumerate(val_ns):
        examples = make_addition_examples(
            args.n_val, n_operands, seed=args.seed + 1000 + offset,
            digits=args.digits, answer_width=args.answer_width, prefix=f"val{n_operands}",
        )
        name = f"val_n{n_operands}"
        add_eval(name, examples)
        val_split_names.append(name)
    for offset, n_operands in enumerate(test_ns):
        examples = make_addition_examples(
            args.n_test, n_operands, seed=args.seed + 2000 + offset,
            digits=args.digits, answer_width=args.answer_width, prefix=f"test{n_operands}",
        )
        add_eval(f"test_n{n_operands}", examples)
    for offset, n_operands in enumerate(ood_ns):
        examples = make_addition_examples(
            args.n_test, n_operands, seed=args.seed + 3000 + offset,
            digits=args.digits, answer_width=args.answer_width, prefix=f"ood{n_operands}",
        )
        add_eval(f"ood_n{n_operands}", examples)

    return train_loaders, eval_loaders, eval_splits, train_ns, val_split_names


class AdditionDataModule(L.LightningDataModule):
    """Serve training loaders with an optional cumulative operand-count curriculum.

    ``train_loaders`` are ordered by ascending ``n``. With ``curriculum`` enabled,
    epoch ``e`` trains on the first ``e // curriculum_epochs + 1`` operand counts
    (so the model first masters small ``n`` and then fine-tunes on each larger one
    in turn, keeping the easier counts to avoid forgetting). Requires the trainer
    to reload dataloaders every epoch. Validation always covers every operand
    count so the per-``n`` metrics stay comparable across the curriculum.
    """

    def __init__(
        self,
        train_loaders: list[DataLoader],
        train_ns: list[int],
        val_loader: CombinedLoader,
        *,
        curriculum: bool,
        curriculum_epochs: int,
    ) -> None:
        super().__init__()
        self.train_loaders = train_loaders
        self.train_ns = train_ns
        self.val_loader = val_loader
        self.curriculum = curriculum
        self.curriculum_epochs = max(1, curriculum_epochs)

    def active_count(self, epoch: int) -> int:
        if not self.curriculum:
            return len(self.train_loaders)
        return min(epoch // self.curriculum_epochs + 1, len(self.train_loaders))

    def train_dataloader(self) -> list[DataLoader]:
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        k = self.active_count(epoch)
        if self.curriculum:
            print(json.dumps({"epoch": epoch, "curriculum_active_ns": self.train_ns[:k]}))
        return self.train_loaders[:k]

    def val_dataloader(self) -> CombinedLoader:
        return self.val_loader


def build_model(args: argparse.Namespace) -> tuple[AdditionTransformer, dict]:
    config = AdditionConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
        pre_norm=args.pre_norm,
        weight_tied=args.weight_tied,
        use_rope=not args.nope,
        causal=args.causal,
    )
    return AdditionTransformer(config), config.__dict__


def metrics_from_logits(
    logits: torch.Tensor,
    answer: torch.Tensor,
    target_ids: torch.Tensor,
) -> dict[str, float]:
    answer_width = target_ids.shape[1]
    preds = logits[:, -answer_width:, :].argmax(dim=-1)
    digit_correct = int((preds == target_ids).sum().item())
    digit_total = int(target_ids.numel())
    exact_correct = int(verify_addition_tensor(preds, answer).sum().item())
    exact_total = int(target_ids.shape[0])
    return {
        "digit_correct": digit_correct,
        "digit_total": digit_total,
        "exact_correct": exact_correct,
        "exact_total": exact_total,
    }


def fold(metrics: dict[str, float], loss_sum: float) -> dict[str, float]:
    return {
        "loss": loss_sum / metrics["exact_total"],
        "digit_accuracy": metrics["digit_correct"] / metrics["digit_total"],
        "exact_accuracy": metrics["exact_correct"] / metrics["exact_total"],
    }


@torch.inference_mode()
def evaluate(
    model: AdditionTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    window: tuple[int, int] | None = None,
    n_repeats: int = 1,
) -> dict[str, float]:
    model.eval()
    agg = {"digit_correct": 0, "digit_total": 0, "exact_correct": 0, "exact_total": 0}
    loss_sum = 0.0
    use_rys = window is not None and n_repeats > 1

    def run() -> None:
        nonlocal loss_sum
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            answer = batch["answer"].to(device)
            out = model(input_ids, target_ids=target_ids)
            loss_sum += float(out["loss"].item()) * target_ids.shape[0]
            for key, value in metrics_from_logits(out["logits"], answer, target_ids).items():
                agg[key] += value

    if use_rys:
        with apply_rys(model, window=window, n_repeats=n_repeats):
            run()
    else:
        run()
    return fold(agg, loss_sum)


@torch.inference_mode()
def accuracy_vs_rounds(
    model: AdditionTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    n_layers: int,
) -> pd.DataFrame:
    model.eval()
    per_round = [
        {"digit_correct": 0, "digit_total": 0, "exact_correct": 0, "exact_total": 0}
        for _ in range(n_layers)
    ]
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        answer = batch["answer"].to(device)
        out = model(input_ids, return_round_logits=True)
        for idx, logits in enumerate(out["round_logits"]):
            for key, value in metrics_from_logits(logits, answer, target_ids).items():
                per_round[idx][key] += value
    rows = []
    for idx, agg in enumerate(per_round):
        folded = fold(agg, 0.0)
        rows.append(
            {
                "round": idx + 1,
                "digit_accuracy": folded["digit_accuracy"],
                "exact_accuracy": folded["exact_accuracy"],
            }
        )
    return pd.DataFrame(rows)


@torch.inference_mode()
def capture_token_states(
    model: AdditionTransformer,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        out = model(input_ids, return_hidden_states=True)
        trace = out["hidden_states"]
        prompt_ids = [str(x) for x in batch["prompt_id"]]
        for layer_idx, hidden in enumerate(trace):
            for row_idx, prompt_id in enumerate(prompt_ids):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "layer": layer_idx,
                        "activation": hidden[row_idx].detach().cpu().numpy(),
                        "strategy": "token",
                    }
                )
    return pd.DataFrame(rows)


def rys_windows(n_layers: int) -> list[tuple[int, int]]:
    """Half-open windows ``(start, end)`` valid for apply_rys (``end <= n_layers-1``)."""
    return [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers)]


def rys_sweep(
    model: AdditionTransformer,
    loaders: dict[str, DataLoader],
    device: torch.device,
    *,
    eval_splits: list[str],
    n_layers: int,
    max_repeat: int,
) -> pd.DataFrame:
    rows = []
    windows = rys_windows(n_layers)
    for split in eval_splits:
        cached = cache_batches_on_device(loaders[split], device)
        base = evaluate(model, cached, device)
        for window in log_rys_progress(windows, device=device, split=split, n_layers=n_layers):
            for n_repeats in range(1, max_repeat + 1):
                if n_repeats == 1:
                    metrics = base
                else:
                    metrics = evaluate(model, cached, device, window=window, n_repeats=n_repeats)
                row = {
                    "split": split,
                    "start": window[0],
                    "end": window[1],
                    "n_repeats": n_repeats,
                    "exact_accuracy": metrics["exact_accuracy"],
                    "digit_accuracy": metrics["digit_accuracy"],
                    "baseline_exact_accuracy": base["exact_accuracy"],
                    "delta_exact_accuracy": metrics["exact_accuracy"] - base["exact_accuracy"],
                }
                rows.append(row)
                print(json.dumps(row))
    return pd.DataFrame(rows)


def save_round_curve(curves: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, frame in curves.items():
        ax.plot(frame["round"], frame["exact_accuracy"], marker="o", label=split)
    ax.set_title("Exact accuracy vs layer rounds")
    ax.set_xlabel("round (layer read-out)")
    ax.set_ylabel("exact accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rys_plot(sweep: pd.DataFrame, path: Path) -> dict[str, dict]:
    """Plot exact accuracy vs n_repeats for each split's best window."""
    best_windows: dict[str, dict] = {}
    fig, ax = plt.subplots(figsize=(7, 5))
    for split in sweep["split"].unique():
        sub = sweep[sweep["split"] == split]
        # Best window = the one with the highest exact accuracy at max n_repeats.
        max_rep = sub["n_repeats"].max()
        at_max = sub[sub["n_repeats"] == max_rep]
        best_row = at_max.loc[at_max["exact_accuracy"].idxmax()]
        start, end = int(best_row["start"]), int(best_row["end"])
        curve = sub[(sub["start"] == start) & (sub["end"] == end)].sort_values("n_repeats")
        ax.plot(curve["n_repeats"], curve["exact_accuracy"], marker="o", label=f"{split} ({start},{end})")
        best_windows[split] = {
            "start": start,
            "end": end,
            "best_exact_accuracy": float(best_row["exact_accuracy"]),
            "baseline_exact_accuracy": float(best_row["baseline_exact_accuracy"]),
            "delta_exact_accuracy": float(best_row["delta_exact_accuracy"]),
            "n_repeats": int(max_rep),
        }
    ax.set_title("RYS: exact accuracy vs repeats (best window per split)")
    ax.set_xlabel("n_repeats of window")
    ax.set_ylabel("exact accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return best_windows


def save_cka_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="viridis", vmin=float(values.min()), vmax=float(values.max()))
    ax.set_title("Token-stream CKA (validation)")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label="CKA")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(run_dir: Path, *, args: argparse.Namespace, baselines: dict, best_windows: dict, fit: dict) -> None:
    lines = [
        "# N-ary addition — RYS-controlled experiment",
        "",
        "Encoder-only RoPE Transformer adding `n` 3-digit operands and writing the",
        "sum into fixed answer slots (reversed digit classes). One-hot input embedding,",
        "distinct 10-class digit head (no weight tying), trained directly on the",
        "input-output pair with no chain-of-thought.",
        "",
        f"- Train operand counts: `{args.train_ns}`; val counts: `{args.val_ns}`; OOD counts: `{args.ood_ns}`",
        f"- Digits: `{args.digits}`; answer width: `{args.answer_width}`",
        f"- Layers: `{args.n_layers}`, heads `{args.n_heads}`, d_model `{args.d_model}`, d_mlp `{args.d_mlp}`",
        f"- Weight decay: `{args.weight_decay}`; pre_norm `{args.pre_norm}`; weight_tied `{args.weight_tied}`",
        "",
        "## Baseline accuracy per split",
        "",
        "| split | digit acc | exact acc |",
        "| --- | ---: | ---: |",
    ]
    for split, metrics in baselines.items():
        lines.append(
            f"| {split} | {metrics['digit_accuracy']:.4f} | {metrics['exact_accuracy']:.4f} |"
        )
    lines += [
        "",
        "## Best RYS window per split (exact accuracy)",
        "",
        "| split | window | repeats | baseline exact acc | RYS exact acc | delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split, info in best_windows.items():
        lines.append(
            f"| {split} | ({info['start']}, {info['end']}) | {info['n_repeats']} | "
            f"{info['baseline_exact_accuracy']:.4f} | {info['best_exact_accuracy']:.4f} | "
            f"{info['delta_exact_accuracy']:+.4f} |"
        )
    lines += [
        "",
        "## rho/phi theory fit (validation token stream)",
        "",
        f"- Pairs (total / plateau): {fit['n_pairs_total']} / {fit['n_pairs_plateau']}",
        f"- Pearson(measured, plateau predictor): {fit['pearson_plateau_pred']:.4f}",
        f"- Spearman(measured, plateau predictor): {fit['spearman_plateau_pred']:.4f}",
        f"- Median R (relative force): {fit['median_R']:.4f}; median cos phi: {fit['median_cos_phi']:.4f}",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    device = resolve_device()
    run_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loaders, loaders, eval_splits, train_ns, val_split_names = make_loaders(args)
    # Representative in-distribution split (largest training operand count) used
    # for the per-layer round curve and the CKA connectome.
    representative_split = f"val_n{max(train_ns)}"
    model, config = build_model(args)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"n_params": n_params, "train_ns": train_ns, "config": config, "device": str(device)}, default=str))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        lit = AdditionLitModule(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            deep_supervision=args.deep_supervision,
            warmup_epochs=args.warmup_epochs,
            loss_weighting=args.loss_weighting,
            model_config=config,
        )
        trainer = build_trainer(
            run_dir,
            max_epochs=args.epochs,
            monitor=lit.primary_metric,
            mode=lit.primary_mode,
            reload_dataloaders_every_n_epochs=1 if args.curriculum else 0,
        )
        # Combine the per-n validation loaders into a single dict-batch loader so
        # one validation step covers every operand count and the LitModule can log
        # per-n metrics (val/exact_accuracy_val_nK) each epoch.
        val_loader = CombinedLoader(
            {name: loaders[name] for name in val_split_names}, mode="max_size_cycle"
        )
        datamodule = AdditionDataModule(
            train_loaders,
            train_ns,
            val_loader,
            curriculum=args.curriculum,
            curriculum_epochs=args.curriculum_epochs,
        )
        trainer.fit(lit, datamodule=datamodule)
        checkpoint_path = best_checkpoint_path(trainer)

    reloaded, _ = load_rys_model(
        checkpoint_path,
        lambda _config: build_model(args)[0],
        map_location=device,
    )
    reloaded.to(device)

    baselines = {split: evaluate(reloaded, loaders[split], device) for split in eval_splits}
    (run_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))

    curves = {
        split: accuracy_vs_rounds(reloaded, loaders[split], device, n_layers=args.n_layers)
        for split in eval_splits
    }
    for split, frame in curves.items():
        frame.to_csv(run_dir / f"accuracy_vs_rounds_{split}.csv", index=False)
    save_round_curve(curves, run_dir / "accuracy_vs_rounds.png")

    activations = capture_token_states(
        reloaded, loaders[representative_split], device, max_batches=args.capture_batches
    )
    cka = cka_matrix(activations, unbiased=False, device="cpu")
    cka.to_csv(run_dir / "cka_val.csv")
    save_cka_heatmap(cka, run_dir / "cka_val.png")
    rho_phi = rho_phi_table(activations)
    rho_phi.to_csv(run_dir / "rho_phi.csv", index=False)
    fit = theory_fit(rho_phi)
    (run_dir / "theory_fit.json").write_text(json.dumps(fit, indent=2))

    if args.skip_rys:
        best_windows: dict = {}
    else:
        # Phase 1: restrict the sweep to the OOD splits plus the representative
        # in-distribution split to keep the MPS cost bounded.
        sweep_splits = [s for s in eval_splits if s.startswith("ood_")] + [representative_split]
        sweep = rys_sweep(
            reloaded,
            loaders,
            device,
            eval_splits=sweep_splits,
            n_layers=args.n_layers,
            max_repeat=args.max_repeat,
        )
        sweep.to_csv(run_dir / "rys_sweep.csv", index=False)
        best_windows = save_rys_plot(sweep, run_dir / "rys_sweep.png")

    summary = {
        "args": vars(args),
        "device": str(device),
        "n_params": n_params,
        "config": config,
        "checkpoint": str(checkpoint_path),
        "baselines": baselines,
        "best_windows": best_windows,
        "theory_fit": fit,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(run_dir, args=args, baselines=baselines, best_windows=best_windows, fit=fit)
    print(f"Wrote {run_dir}")


main.__doc__ = __doc__


if __name__ == "__main__":
    typer.run(main)
