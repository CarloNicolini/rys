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


def _resolve_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            "Expected `model.model.layers`. Got "
            f"{type(model).__name__}; provide a Llama-style decoder."
        )
    return model.model.layers


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
                out = layers[k](hidden, *rest_args, **kwargs)
                hidden = out[0] if isinstance(out, tuple) else out

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
