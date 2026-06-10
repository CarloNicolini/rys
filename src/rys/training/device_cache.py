"""Cache a DataLoader's batches on a device for repeated evaluation.

The RYS sweep evaluates the same split under hundreds of layer windows. Re-running
the DataLoader every window (host collation plus host->device copies) starves the
GPU: it waits on the input pipeline instead of running forward passes. Moving the
batches onto the device once and reusing them keeps the GPU fed across the whole
sweep. Because ``Tensor.to(device)`` returns the tensor unchanged when it already
lives on ``device``, the cached batches drop straight into the existing per-batch
``.to(device)`` calls without any change to the evaluation math.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def cache_batches_on_device(loader: Iterable[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    """Return the loader's batches with every tensor moved to ``device`` once.

    Non-tensor values (e.g. ``prompt_id`` lists) are kept as-is. The result is a
    plain list, so it can be iterated as many times as there are RYS windows.
    """
    cached: list[dict[str, Any]] = []
    for batch in loader:
        cached.append(
            {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        )
    return cached
