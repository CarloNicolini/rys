"""Callbacks that preserve the existing RYS stdout/CSV logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch


class MetricsHistoryCallback(L.Callback):
    """Write epoch metrics as JSON lines and ``train_history.csv`` rows."""

    def __init__(self, output_path: Path, *, extra_fields: dict[str, Any] | None = None) -> None:
        self.output_path = output_path
        self.extra_fields = extra_fields or {}
        self.rows: list[dict[str, Any]] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Lightning runs the validation loop *before* this hook, so by now both
        # the train epoch aggregates and the fresh val metrics are in
        # ``callback_metrics`` -- writing the row here keeps them on one line.
        if trainer.sanity_checking:
            return
        row: dict[str, Any] = {
            "epoch": int(trainer.current_epoch + 1),
            **_metrics_row(trainer.callback_metrics, prefix="train/"),
            **_metrics_row(trainer.callback_metrics, prefix="val/"),
        }
        self.rows.append(row)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.output_path, index=False)
        print(json.dumps({**self.extra_fields, **row}))


def _metrics_row(metrics: dict[str, object], *, prefix: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in metrics.items():
        if not key.startswith(prefix) or key.endswith("_step"):
            continue
        scalar = _as_scalar(value)
        if scalar is None:
            continue
        row[_history_key(key)] = scalar
    return row


def _as_scalar(value: object) -> float | int | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        number = float(value.detach().cpu())
    elif isinstance(value, int | float):
        number = float(value)
    else:
        return None
    if number.is_integer():
        return int(number)
    return number


def _history_key(key: str) -> str:
    key = key.replace("/", "_")
    if key.endswith("_epoch"):
        key = key[: -len("_epoch")]
    return key
