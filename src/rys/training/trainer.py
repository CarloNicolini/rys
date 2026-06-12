"""Trainer construction and checkpoint loading utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from rys.training.callbacks import MetricsHistoryCallback


def build_trainer(
    run_dir: Path,
    *,
    max_epochs: int,
    monitor: str,
    mode: str = "max",
    extra_log_fields: dict[str, Any] | None = None,
    enable_progress_bar: bool = True,
    reload_dataloaders_every_n_epochs: int = 0,
) -> L.Trainer:
    """Return the standard RYS Lightning Trainer."""
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_key = f"val/{monitor}"
    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best-{epoch:02d}",
        monitor=monitor_key,
        mode=mode,
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    history = MetricsHistoryCallback(
        run_dir / "train_history.csv",
        extra_fields=extra_log_fields,
    )
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
        logger=[
            TensorBoardLogger(save_dir=run_dir, name="tb", version=""),
            CSVLogger(save_dir=run_dir, name="csv", version=""),
        ],
        callbacks=[checkpoint, history],
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_progress_bar=enable_progress_bar,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
    )


def best_checkpoint_path(trainer: L.Trainer) -> Path:
    """Return the best checkpoint path, falling back to the last checkpoint."""
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            path = callback.best_model_path or callback.last_model_path
            if path:
                return Path(path)
    raise RuntimeError("Trainer has no ModelCheckpoint with a saved path.")


def resolve_training_checkpoint(trainer: L.Trainer, run_dir: Path) -> Path:
    """Return the best available checkpoint after training or an interrupt.

    Prefer the ModelCheckpoint callback paths; if those are empty (e.g. abrupt
    interrupt), fall back to ``best-*.ckpt`` then ``last.ckpt`` under ``run_dir``.
    """
    try:
        return best_checkpoint_path(trainer)
    except RuntimeError:
        pass
    best_matches = sorted(run_dir.glob("best-*.ckpt"))
    if best_matches:
        return best_matches[-1]
    last = run_dir / "last.ckpt"
    if last.exists():
        return last
    raise RuntimeError(f"No checkpoint found in {run_dir}.")


def load_rys_model(
    checkpoint_path: Path,
    build_model: Callable[[dict[str, Any] | None], torch.nn.Module],
    *,
    map_location: torch.device | str | None = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a bare RYS ``nn.Module`` from a Lightning ``.ckpt`` or legacy ``.pt`` file."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if "state_dict" in payload:
        hparams = dict(payload.get("hyper_parameters", {}))
        config = hparams.get("model_config")
        state_dict = {
            key.removeprefix("net."): value
            for key, value in payload["state_dict"].items()
            if key.startswith("net.")
        }
        model = build_model(config)
        model.load_state_dict(state_dict)
        return model, {"checkpoint": str(checkpoint_path), "config": config, "hyper_parameters": hparams}

    if "model_state_dict" in payload:
        config = payload.get("config")
        model = build_model(config if isinstance(config, dict) else None)
        model.load_state_dict(payload["model_state_dict"])
        return model, dict(payload)

    raise ValueError(f"Unrecognised checkpoint format: {checkpoint_path}")
