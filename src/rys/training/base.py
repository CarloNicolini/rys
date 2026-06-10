"""Shared LightningModule base for RYS experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any

import lightning as L
import torch


def config_to_dict(config: object) -> dict[str, Any]:
    """Return a checkpoint-friendly config dictionary."""
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


class RysLitModule(L.LightningModule, ABC):
    """Base wrapper that keeps the original RYS module available as ``self.net``."""

    primary_metric: str = "loss"
    primary_mode: str = "min"

    def __init__(
        self,
        net: torch.nn.Module,
        *,
        lr: float,
        weight_decay: float,
        model_config: object | None = None,
        extra_hparams: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        hparams = {
            "lr": lr,
            "weight_decay": weight_decay,
            "model_config": config_to_dict(model_config if model_config is not None else getattr(net, "config", {})),
        }
        if extra_hparams:
            hparams.update(extra_hparams)
        self.save_hyperparameters(hparams)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.net(*args, **kwargs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self._log_metrics("train", loss, metrics, batch)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, metrics = self._shared_step(batch, "val")
        self._log_metrics("val", loss, metrics, batch)
        return loss

    def _log_metrics(
        self,
        stage: str,
        loss: torch.Tensor,
        metrics: dict[str, torch.Tensor | float | int],
        batch: dict[str, Any],
    ) -> None:
        values: dict[str, torch.Tensor | float | int] = {f"{stage}/loss": loss.detach()}
        values.update({f"{stage}/{key}": value for key, value in metrics.items()})
        self.log_dict(
            values,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            batch_size=self.batch_size(batch),
        )

    @staticmethod
    def batch_size(batch: Any) -> int:
        # ``batch`` is usually a dict, but the multi-loader path (e.g. sorted
        # translation mixed lengths) hands Lightning a list of per-loader dicts.
        if isinstance(batch, dict):
            items: Any = batch.values()
        elif isinstance(batch, list | tuple):
            items = batch
        else:
            items = [batch]
        for value in items:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, dict | list | tuple):
                size = RysLitModule.batch_size(value)
                if size > 1:
                    return size
        return 1

    @abstractmethod
    def _shared_step(
        self,
        batch: dict[str, Any],
        stage: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int]]:
        """Return loss and scalar metrics for one train/validation batch."""
