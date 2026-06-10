"""Progress logging for the RYS window-sweep (evaluation) phase.

The RYS sweep evaluates many duplicated-layer windows back-to-back. On small
models the work is GPU-light and otherwise silent, which makes it look like
nothing is happening (and that the GPU is idle). These helpers emit a start
banner naming the device actually in use, a bounded number of progress lines,
and a completion banner so the phase is observable across every experiment.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

T = TypeVar("T")


def log_rys_progress(
    windows: Sequence[T],
    *,
    device: Any,
    max_updates: int = 20,
    **fields: Any,
) -> Iterator[T]:
    """Yield RYS windows while emitting structured progress logs.

    Prints a ``start`` line (device + window count), at most ``max_updates``
    ``progress`` lines (with elapsed/ETA seconds), and a ``done`` line. Extra
    ``fields`` (e.g. ``split``, ``depth``) are echoed on every line so the
    output stays greppable. The progress line for window ``i`` is emitted after
    that window has been consumed, so it reflects completed work.
    """
    total = len(windows)
    every = max(1, total // max_updates)
    start = time.perf_counter()
    print(json.dumps({"phase": "rys", "event": "start", "device": str(device), "n_windows": total, **fields}))
    for index, window in enumerate(windows, start=1):
        yield window
        if index == total or index % every == 0:
            elapsed = time.perf_counter() - start
            rate = index / elapsed if elapsed > 0 else 0.0
            eta = (total - index) / rate if rate > 0 else 0.0
            print(
                json.dumps(
                    {
                        "phase": "rys",
                        "event": "progress",
                        "window": index,
                        "n_windows": total,
                        "elapsed_s": round(elapsed, 2),
                        "eta_s": round(eta, 2),
                        **fields,
                    }
                )
            )
    print(
        json.dumps(
            {
                "phase": "rys",
                "event": "done",
                "n_windows": total,
                "elapsed_s": round(time.perf_counter() - start, 2),
                **fields,
            }
        )
    )
