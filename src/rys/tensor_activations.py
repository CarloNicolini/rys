"""Activation capture for tensor-native toy models."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch


def _prompt_ids(batch: dict[str, object], batch_idx: int, batch_size: int) -> list[str]:
    ids = batch.get("prompt_id")
    if ids is None:
        return [f"batch{batch_idx:04d}_{i:03d}" for i in range(batch_size)]
    if isinstance(ids, tuple):
        return [str(x) for x in ids]
    if isinstance(ids, list):
        return [str(x) for x in ids]
    return [str(x) for x in ids]  # type: ignore[union-attr]


def _select_activation(
    hidden: torch.Tensor,
    mask: torch.Tensor | None,
    row: int,
    *,
    token_strategy: str,
) -> np.ndarray:
    if token_strategy == "cls":
        return hidden[row, 0].detach().cpu().numpy()
    if token_strategy == "all":
        if mask is None:
            selected = hidden[row]
        else:
            selected = hidden[row, mask[row].bool()]
        return selected.detach().cpu().numpy()
    raise ValueError("token_strategy must be 'cls' or 'all'.")


def capture_tensor_residual_stream(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor | list[str]]],
    *,
    device: str | torch.device = "cpu",
    architecture: str = "flat",
    token_strategy: str = "cls",
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Capture post-layer residual streams from a tensor dataloader.

    The output schema matches :mod:`rys.cka`: each row contains
    ``prompt_id``, ``layer``, and an ``activation`` vector or matrix.  The
    model must accept ``return_hidden_states=True`` and return a mapping with a
    ``hidden_states`` list, as :class:`rys.tiny_transformer.TinySatTransformer`
    does.
    """
    was_training = model.training
    model.eval()
    model.to(device)

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(batches):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if architecture == "flat":
                input_ids = batch["input_ids"].to(device)  # type: ignore[union-attr]
                attention_mask = batch.get("attention_mask")
                kwargs = {"input_ids": input_ids}
            elif architecture == "factorized":
                input_ids = batch["variable_ids"].to(device)  # type: ignore[union-attr]
                attention_mask = batch.get("factor_attention_mask")
                kwargs = {
                    "variable_ids": input_ids,
                    "sign_ids": batch["sign_ids"].to(device),  # type: ignore[union-attr]
                    "clause_ids": batch["clause_ids"].to(device),  # type: ignore[union-attr]
                    "slot_ids": batch["slot_ids"].to(device),  # type: ignore[union-attr]
                    "token_type_ids": batch["token_type_ids"].to(device),  # type: ignore[union-attr]
                }
            else:
                raise ValueError("architecture must be 'flat' or 'factorized'.")

            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(device)
            else:
                attention_mask = None

            outputs = model(
                **kwargs,
                attention_mask=attention_mask,
                return_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"]
            if not isinstance(hidden_states, list):
                raise RuntimeError("Model did not return a hidden_states list.")

            ids = _prompt_ids(batch, batch_idx, input_ids.shape[0])
            for layer_idx, hidden in enumerate(hidden_states):
                for row_idx, prompt_id in enumerate(ids):
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "layer": layer_idx,
                            "activation": _select_activation(
                                hidden,
                                attention_mask,
                                row_idx,
                                token_strategy=token_strategy,
                            ),
                            "strategy": token_strategy,
                        }
                    )

    if was_training:
        model.train()
    return pd.DataFrame(rows, columns=["prompt_id", "layer", "activation", "strategy"])
