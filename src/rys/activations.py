"""Hook-based residual-stream capture for decoder-only LLMs.

The module exposes prompt-prefill and generated-response capture functions for
HuggingFace causal LMs. Both return post-block residual stream activations of
every requested layer at sequence level.

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
- Generated-response capture first samples a completion, then replays the full
  prompt plus completion without KV cache and selects only the completion tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack

import numpy as np
import pandas as pd
import torch

from rys.surgery import is_in_rys_replay


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


def _selected_token_matrices(
    hidden: torch.Tensor,
    selection_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Return one ``(n_selected_tokens, d)`` tensor per prompt."""
    keep = selection_mask.bool()
    if keep.shape[0] != hidden.shape[0]:
        raise ValueError(
            "Selection mask batch size does not match hidden-state batch size: "
            f"{keep.shape[0]} != {hidden.shape[0]}."
        )
    if keep.shape[1] != hidden.shape[1]:
        keep = keep[:, -hidden.shape[1] :]
    return [row[mask] for row, mask in zip(hidden, keep, strict=True)]


def _normalise_layer_indices(
    layers: torch.nn.ModuleList,
    layer_indices: Iterable[int] | None,
) -> list[int]:
    L = len(layers)
    if layer_indices is None:
        return list(range(L))
    resolved = sorted({i if i >= 0 else L + i for i in layer_indices})
    if any(i < 0 or i >= L for i in resolved):
        raise IndexError(f"Layer indices out of range for L={L}: {resolved}")
    return resolved


def _pad_token_sequences(
    sequences: list[torch.Tensor],
    response_masks: list[torch.Tensor],
    *,
    pad_token_id: int,
    padding_side: str,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad token-id sequences and matching response masks into a batch."""
    max_len = max(seq.numel() for seq in sequences)
    input_rows, attention_rows, response_rows = [], [], []
    for seq, response_mask in zip(sequences, response_masks, strict=True):
        pad_len = max_len - seq.numel()
        pads = torch.full((pad_len,), pad_token_id, dtype=seq.dtype, device=seq.device)
        pad_mask = torch.zeros(pad_len, dtype=torch.bool, device=seq.device)
        token_mask = torch.ones(seq.numel(), dtype=torch.bool, device=seq.device)
        if padding_side == "left":
            input_rows.append(torch.cat([pads, seq]))
            attention_rows.append(torch.cat([pad_mask, token_mask]))
            response_rows.append(torch.cat([pad_mask, response_mask]))
        else:
            input_rows.append(torch.cat([seq, pads]))
            attention_rows.append(torch.cat([token_mask, pad_mask]))
            response_rows.append(torch.cat([response_mask, pad_mask]))
    return (
        torch.stack(input_rows).to(device),
        torch.stack(attention_rows).to(device),
        torch.stack(response_rows).to(device),
    )


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
    layer_indices = _normalise_layer_indices(layers, layer_indices)

    device = device or next(model.parameters()).device
    dtype = dtype or torch.float32

    # Pre-allocate a buffer for one batch at a time.
    captured: dict[int, list[np.ndarray]] = {layer_id: [] for layer_id in layer_indices}

    # Hooks fire in order; we close over a mutable batch holder to receive
    # the input mask and strip padding in-place.
    holder: dict[str, torch.Tensor | None] = {"mask": None}

    def make_hook(layer_id: int):
        def _hook(_module, _inputs, output):
            if is_in_rys_replay():
                return
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


def capture_generated_residual_stream(
    model: torch.nn.Module,
    tokenizer,
    prompts: pd.DataFrame,
    layer_indices: Iterable[int] | None = None,
    batch_size: int = 4,
    max_prompt_length: int = 1024,
    max_new_tokens: int = 256,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    generation_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate responses, then capture residual streams on response tokens.

    This is the task-evoked counterpart of :func:`capture_residual_stream`.
    Generation decides what the model actually says; a second no-cache forward
    pass over ``prompt + response`` captures complete hidden states for the
    generated response span only. Generation also defaults to ``use_cache=False``
    because the hook-based RYS surgery replays full residual streams.

    Returns
    -------
    activations, generations
        ``activations`` has columns ``[prompt_id, layer, activation]`` with
        ``activation`` shaped ``(n_generated_tokens, d)``. ``generations`` stores
        ``[prompt_id, task, generated_text, n_generated_tokens]`` plus ``gold``
        when available.
    """
    if "prompt_id" not in prompts.columns or "prompt" not in prompts.columns:
        raise ValueError("`prompts` must have columns 'prompt_id' and 'prompt'.")

    layers = _resolve_layers(model)
    layer_indices = _normalise_layer_indices(layers, layer_indices)
    device = device or next(model.parameters()).device
    dtype = dtype or torch.float32
    generation_kwargs = {"use_cache": False, **(generation_kwargs or {})}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    captured: dict[int, list[np.ndarray]] = {layer_id: [] for layer_id in layer_indices}
    generation_rows = []
    holder: dict[str, torch.Tensor | None] = {"selection": None}

    def make_hook(layer_id: int):
        def _hook(_module, _inputs, output):
            if is_in_rys_replay():
                return
            hidden = output[0] if isinstance(output, tuple) else output
            selection = holder["selection"]
            if selection is None:
                return
            seqs = _selected_token_matrices(hidden, selection)
            arrays = [seq.detach().to(dtype).cpu().numpy() for seq in seqs]
            captured[layer_id].extend(arrays)

        return _hook

    model.eval()
    prompts_reset = prompts.reset_index(drop=True)
    with torch.inference_mode():
        for start in range(0, len(prompts_reset), batch_size):
            batch = prompts_reset.iloc[start : start + batch_size]
            tokenized = tokenizer(
                batch["prompt"].tolist(),
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            ).to(device)
            input_width = tokenized["input_ids"].shape[1]

            # Generate without activation hooks attached. This guarantees the
            # potentially many decoding-step forward passes never touch
            # ``captured``.
            generated = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs,
            )

            replay_sequences, response_masks = [], []
            for row, (_, item) in enumerate(batch.iterrows()):
                prompt_ids = tokenized["input_ids"][row][tokenized["attention_mask"][row].bool()]
                generated_ids = generated[row, input_width:]
                if tokenizer.eos_token_id is not None:
                    eos_positions = (generated_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
                    if len(eos_positions):
                        generated_ids = generated_ids[: int(eos_positions[0])]
                full_ids = torch.cat([prompt_ids, generated_ids])
                response_mask = torch.cat(
                    [
                        torch.zeros(prompt_ids.numel(), dtype=torch.bool, device=device),
                        torch.ones(generated_ids.numel(), dtype=torch.bool, device=device),
                    ]
                )
                replay_sequences.append(full_ids)
                response_masks.append(response_mask)
                generation_rows.append(
                    {
                        "prompt_id": item["prompt_id"],
                        "task": item.get("task", ""),
                        "gold": item.get("gold", ""),
                        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                        "n_generated_tokens": int(generated_ids.numel()),
                    }
                )

            input_ids, attention_mask, selection_mask = _pad_token_sequences(
                replay_sequences,
                response_masks,
                pad_token_id=pad_token_id,
                padding_side=tokenizer.padding_side,
                device=device,
            )

            # Snapshot per-layer counts so we can assert exactly ``len(batch)``
            # new entries land per layer in this replay forward.
            counts_before = {layer_id: len(captured[layer_id]) for layer_id in layer_indices}

            with ExitStack() as stack:
                for layer_id in layer_indices:
                    handle = layers[layer_id].register_forward_hook(make_hook(layer_id))
                    stack.callback(handle.remove)
                holder["selection"] = selection_mask
                try:
                    model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                finally:
                    holder["selection"] = None

            for layer_id in layer_indices:
                added = len(captured[layer_id]) - counts_before[layer_id]
                if added != len(batch):
                    raise RuntimeError(
                        f"Layer {layer_id} captured {added} entries for a batch of "
                        f"{len(batch)} prompts; expected one entry per prompt. This "
                        "usually means a forward hook fired more often than expected."
                    )

    rows = []
    prompt_ids = prompts_reset["prompt_id"].to_numpy()
    for layer_id in layer_indices:
        if len(captured[layer_id]) != len(prompt_ids):
            raise RuntimeError(
                f"Layer {layer_id} accumulated {len(captured[layer_id])} entries "
                f"for {len(prompt_ids)} prompts."
            )
        for pid, matrix in zip(prompt_ids, captured[layer_id], strict=True):
            rows.append((pid, int(layer_id), matrix))
    return (
        pd.DataFrame(rows, columns=["prompt_id", "layer", "activation"]),
        pd.DataFrame(generation_rows),
    )
