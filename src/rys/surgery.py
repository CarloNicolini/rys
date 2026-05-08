"""RYS surgery as a forward-pre-hook context manager.

The blog-post equation $$\\mathbf{S}^{(1)}_{i,j}\\approx \\mathbf{S}^{(0)}_{i,j}$$
says we can implement *Repeat Your Self* by pushing the residual stream back
through the same parameters of layers ``[i, j)`` once it has reached layer
``j``. This module does exactly that and nothing else: it does not allocate
any new weights, it does not alter the model's static graph, and it is fully
reversible by exiting the context.

Implementation
--------------
We attach a single ``forward_pre_hook`` to ``model.model.layers[end]`` (where
``end == window[1]``). When the hook fires, it:

1. takes the residual-stream input ``h`` to that layer;
2. for ``k`` in ``range(n_repeats - 1)``, pushes ``h`` back through layers
   ``[start, ..., end-1]`` *in order*, accumulating residual contributions
   exactly as the standard recursion would;
3. returns the modified input so layer ``end`` sees the duplicated stream.

The hook handles the standard Llama signature ``forward(hidden_states,
attention_mask=..., position_ids=..., ...)`` by forwarding the same keyword
arguments to the inner replay loop.

Notes
-----
- ``n_repeats == 1`` is a no-op, useful as a sanity check.
- The window is a *half-open* interval ``[start, end)``: layers ``start`` through
  ``end - 1`` are duplicated, and layer ``end`` is the layer at whose input the
  duplicated stream is injected.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

class _RYSReplayState:
    """Module-level reentrant counter for nested RYS layer replay.

    Activation hooks read this to skip the duplicated calls performed inside
    :func:`apply_rys`'s pre-hook, without having to introspect the model
    object (which can be wrapped by bitsandbytes / device_map).
    """

    depth = 0


def is_in_rys_replay() -> bool:
    """Return ``True`` when the calling hook is inside an internal RYS replay."""
    return _RYSReplayState.depth > 0


def _resolve_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            "Expected `model.model.layers`. Got "
            f"{type(model).__name__}; provide a Llama-style decoder."
        )
    return model.model.layers


def _causal_allowed_mask(attention_mask: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Build a bool SDPA mask with True entries where attention is allowed."""
    batch, seq_len = hidden.shape[:2]
    key_is_real = attention_mask.bool()
    if key_is_real.shape != (batch, seq_len):
        key_is_real = key_is_real[:, -seq_len:]

    causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden.device).tril()
    allowed = causal.unsqueeze(0) & key_is_real[:, None, :]

    # Fully padded query rows can otherwise have no allowed keys, which some
    # SDPA backends dislike. Those rows are ignored downstream, so self-attend.
    empty_rows = ~allowed.any(dim=-1)
    if empty_rows.any():
        diag = torch.eye(seq_len, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        allowed = torch.where(empty_rows.unsqueeze(-1), diag.expand(batch, -1, -1), allowed)
    return allowed[:, None, :, :]


def _replay_kwargs(kwargs: dict, hidden: torch.Tensor) -> dict:
    """Return kwargs safe for the inner RYS replay loop.

    Transformer generation may pass integer masks into decoder layers. The
    normal model path can tolerate those in some implementations, but replaying
    a layer directly can route them to PyTorch SDPA, which accepts only bool or
    floating masks. We normalize only the replay copy so the outer model call
    remains untouched.
    """
    replay = dict(kwargs)
    mask = replay.get("attention_mask")
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            replay["attention_mask"] = _causal_allowed_mask(mask.to(hidden.device), hidden)
        elif not (mask.dtype == torch.bool or mask.is_floating_point()):
            replay["attention_mask"] = mask.bool()

    # Hook-based RYS is a full-sequence replay. Reusing generation KV caches in
    # the duplicated block would update/cache the wrong trajectory.
    if "past_key_values" in replay:
        replay["past_key_values"] = None
    if "use_cache" in replay:
        replay["use_cache"] = False

    return replay


@contextmanager
def apply_rys(
    model: torch.nn.Module,
    window: tuple[int, int],
    n_repeats: int = 2,
) -> Iterator[None]:
    """Context manager that activates RYS replay over ``window = (start, end)``.

    Parameters
    ----------
    model
        Llama-style decoder; mutated only by hook registration (reversed on exit).
    window
        ``(start, end)`` half-open layer interval. Layers ``start..end-1`` are
        the ones whose forward pass is replayed.
    n_repeats
        Total number of times the window is traversed. ``1`` is the standard
        forward pass; ``2`` is the canonical RYS surgery in {% cite ng2026rys %}.

    Yields
    ------
    None
        Inside the ``with`` block the model has the RYS hook installed.

    Raises
    ------
    ValueError
        If the window is malformed (start >= end, out-of-range indices, or if
        ``n_repeats < 1``).
    """
    start, end = window
    layers = _resolve_layers(model)
    L = len(layers)
    if not (0 <= start < end <= L):
        raise ValueError(f"Bad window {window} for L={L}.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1.")

    if n_repeats == 1:
        # No-op surgery; still yield so the caller's code path is uniform.
        yield
        return

    def pre_hook(_module, args, kwargs):
        # `args` is empty when transformers calls layers with kwargs only.
        # We cover both cases for forward compatibility across versions.
        if args:
            hidden = args[0]
            rest_args = args[1:]
        elif "hidden_states" in kwargs:
            hidden = kwargs["hidden_states"]
            rest_args = ()
        else:
            raise RuntimeError("Could not locate hidden_states in layer call.")

        for _ in range(n_repeats - 1):
            for k in range(start, end):
                _RYSReplayState.depth += 1
                try:
                    out = layers[k](hidden, *rest_args, **_replay_kwargs(kwargs, hidden))
                    hidden = out[0] if isinstance(out, tuple) else out
                finally:
                    _RYSReplayState.depth -= 1

        if args:
            return (hidden, *rest_args), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs["hidden_states"] = hidden
        return args, new_kwargs

    handle = layers[end].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        # Reset the counter in case an exception left it dangling.
        _RYSReplayState.depth = 0
