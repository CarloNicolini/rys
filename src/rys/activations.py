"""Hook-based residual-stream capture for decoder-only LLMs.

The module exposes a single function `capture_residual_stream` that runs a forward
pass of a HuggingFace causal-LM over a batch of prompts and returns the
post-block residual stream activations of every requested layer, after a chosen
token-level aggregation.

Design choices
--------------
- Uses `register_forward_hook` on `model.model.layers[l]`. The hook output is the
  full residual-stream tensor of shape `(batch, seq, d)` *after* block ``l``.
- All token aggregation is left-padding aware via the `attention_mask` returned
  by the tokenizer. We assume `tokenizer.padding_side == "left"`, which is the
  HuggingFace default for decoder-only generation.
- The output is a long-format DataFrame with one row per (prompt, layer, strategy)
  triple. The activation vector is stored as an `np.ndarray` of shape `(d,)`
  inside an ``object``-dtype column, which is the Pandas-idiomatic way to keep
  per-row arrays without flattening them across the table.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack
from typing import Literal

import numpy as np
import pandas as pd
import torch

TokenStrategy = Literal["last", "mean", "answer_tokens"]


def _resolve_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the residual-block list for a HuggingFace decoder-only model.

    Llama / Mistral / Qwen all expose this as ``model.model.layers``. We keep
    the lookup explicit rather than walking attributes blindly.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            "Expected `model.model.layers` (Llama-style decoder). "
            f"Got model of type {type(model).__name__}."
        )
    return model.model.layers


def _aggregate(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy: TokenStrategy,
) -> torch.Tensor:
    """Reduce a (B, T, d) hidden tensor to (B, d) given an attention mask.

    Notes
    -----
    - ``last``: take the last *non-pad* token of every sequence. With left
      padding this is simply position ``-1``; we still consult the mask so the
      function is also correct for right-padding inputs.
    - ``mean``: mean over non-pad positions, with masked-fill of pad to zero.
    - ``answer_tokens``: same as ``mean`` here. The notebook only requests this
      strategy when prompts have already been augmented with the gold answer in
      the same sequence — the caller is responsible for that framing.
    """
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)  # (B, T, 1)
    if strategy == "last":
        # Index of the last True element per row.
        last_idx = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, gather_idx).squeeze(1)
    if strategy in ("mean", "answer_tokens"):
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    raise ValueError(f"Unknown token strategy {strategy!r}")


def capture_residual_stream(
    model: torch.nn.Module,
    tokenizer,
    prompts: pd.DataFrame,
    layer_indices: Iterable[int] | None = None,
    aggregate: TokenStrategy = "last",
    batch_size: int = 8,
    max_length: int = 1024,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> pd.DataFrame:
    """Run a forward pass and capture per-layer residual-stream activations.

    Parameters
    ----------
    model
        HuggingFace ``LlamaForCausalLM``-style model (any decoder-only variant
        that exposes ``model.model.layers`` works).
    tokenizer
        Matching tokenizer; must have ``padding_side == 'left'``.
    prompts
        DataFrame with at least the columns ``prompt_id`` and ``prompt``. The
        index is reset internally so the returned DataFrame's ``prompt_id``
        column is the only stable identifier.
    layer_indices
        Subset of layers to capture. ``None`` means every layer. Negative
        indices are interpreted Python-style (``-1`` = last).
    aggregate
        Token-level reduction strategy; see :func:`_aggregate`.
    batch_size
        Number of prompts processed per forward pass. Memory scales with
        ``batch_size * seq_len * L * d * 4 bytes`` for FP32 outputs.
    max_length
        Truncate prompts longer than this (in tokens).
    device
        Device for the input tensors. Defaults to the model's first parameter
        device.
    dtype
        Cast captured activations to this dtype before moving to CPU. Defaults
        to ``torch.float32`` for numerical stability of downstream CKA.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns
        ``[prompt_id, layer, strategy, activation]`` where ``activation`` is an
        ``np.ndarray`` of shape ``(d,)`` per row.
    """
    if "prompt_id" not in prompts.columns or "prompt" not in prompts.columns:
        raise ValueError("`prompts` must have columns 'prompt_id' and 'prompt'.")

    layers = _resolve_layers(model)
    L = len(layers)
    if layer_indices is None:
        layer_indices = list(range(L))
    layer_indices = sorted({i if i >= 0 else L + i for i in layer_indices})
    if any(i < 0 or i >= L for i in layer_indices):
        raise IndexError(f"Layer indices out of range for L={L}: {layer_indices}")

    device = device or next(model.parameters()).device
    dtype = dtype or torch.float32

    # Pre-allocate a buffer for one batch at a time.
    captured: dict[int, list[np.ndarray]] = {layer_id: [] for layer_id in layer_indices}

    # Hooks fire in order; we close over a mutable batch holder to receive
    # the input mask and aggregate in-place.
    holder: dict[str, torch.Tensor | None] = {"mask": None}

    def make_hook(layer_id: int):
        def _hook(_module, _inputs, output):
            # Llama returns a tuple (hidden, ...) per block. Some custom variants
            # return a tensor directly. Handle both.
            hidden = output[0] if isinstance(output, tuple) else output
            mask = holder["mask"]
            assert mask is not None, "Attention mask must be set before hook fires."
            agg = _aggregate(hidden, mask, aggregate).detach().to(dtype).cpu().numpy()
            captured[layer_id].append(agg)

        return _hook

    model.eval()
    with ExitStack() as stack:
        for layer_id in layer_indices:
            handle = layers[layer_id].register_forward_hook(make_hook(layer_id))
            stack.callback(handle.remove)

        prompts_reset = prompts.reset_index(drop=True)
        with torch.inference_mode():
            for start in range(0, len(prompts_reset), batch_size):
                batch = prompts_reset.iloc[start : start + batch_size]
                tokenized = tokenizer(
                    batch["prompt"].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                holder["mask"] = tokenized["attention_mask"]
                model(**tokenized)
                holder["mask"] = None

    # Assemble the long-format DataFrame.
    rows = []
    prompt_ids = prompts_reset["prompt_id"].to_numpy()
    for layer_id in layer_indices:
        arrs = np.concatenate([np.asarray(a) for a in captured[layer_id]], axis=0)
        for pid, vec in zip(prompt_ids, arrs, strict=True):
            rows.append((pid, int(layer_id), aggregate, vec))
    return pd.DataFrame(rows, columns=["prompt_id", "layer", "strategy", "activation"])
