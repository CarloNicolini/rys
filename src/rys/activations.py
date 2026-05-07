"""Hook-based residual-stream capture for decoder-only LLMs.

The module exposes a single function `capture_residual_stream` that runs a forward
pass of a HuggingFace causal-LM over a batch of prompts and returns the
post-block residual stream activations of every requested layer at sequence
level.

Design choices
--------------
- Uses `register_forward_hook` on `model.model.layers[l]`. The hook output is the
  full residual-stream tensor of shape `(batch, seq, d)` *after* block ``l``.
- Pad tokens are removed with the `attention_mask` returned by the tokenizer.
- The output is a long-format DataFrame with one row per (prompt, layer)
  pair. The activation matrix is stored as an `np.ndarray` of shape `(n_tokens, d)`
  inside an ``object``-dtype column, which is the Pandas-idiomatic way to keep
  per-row arrays without flattening them across the table. Prompt-level averages
  can be computed downstream when they are scientifically needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack

import numpy as np
import pandas as pd
import torch


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


def _valid_token_matrices(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Return one ``(n_tokens, d)`` tensor per prompt after dropping pads."""
    keep = attention_mask.bool()
    return [row[mask] for row, mask in zip(hidden, keep, strict=True)]


def capture_residual_stream(
    model: torch.nn.Module,
    tokenizer,
    prompts: pd.DataFrame,
    layer_indices: Iterable[int] | None = None,
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
        ``[prompt_id, layer, activation]`` where ``activation`` is an
        ``np.ndarray`` of shape ``(n_tokens, d)`` per row.
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
    # the input mask and strip padding in-place.
    holder: dict[str, torch.Tensor | None] = {"mask": None}

    def make_hook(layer_id: int):
        def _hook(_module, _inputs, output):
            # Llama returns a tuple (hidden, ...) per block. Some custom variants
            # return a tensor directly. Handle both.
            hidden = output[0] if isinstance(output, tuple) else output
            mask = holder["mask"]
            assert mask is not None, "Attention mask must be set before hook fires."
            seqs = _valid_token_matrices(hidden, mask)
            arrays = [seq.detach().to(dtype).cpu().numpy() for seq in seqs]
            captured[layer_id].extend(arrays)

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
                model(**tokenized, use_cache=False)
                holder["mask"] = None

    # Assemble the long-format DataFrame.
    rows = []
    prompt_ids = prompts_reset["prompt_id"].to_numpy()
    for layer_id in layer_indices:
        for pid, matrix in zip(prompt_ids, captured[layer_id], strict=True):
            rows.append((pid, int(layer_id), matrix))
    return pd.DataFrame(rows, columns=["prompt_id", "layer", "activation"])
