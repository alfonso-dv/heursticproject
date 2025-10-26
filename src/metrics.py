from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

import statistics as stats


@dataclass
class SearchResult:
    """
    Canonical record for a single A* run (used by experiment.py).
    """
    solved: bool
    depth: int
    expanded: int
    runtime_s: float
    heuristic: str
    start_state: tuple[int, ...]


def time_call(fn, *args, **kwargs) -> Tuple[float, Any]:
    """
    Run `fn(*args, **kwargs)` and return (elapsed_seconds, result).
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0, out)


def mean_std(values: Iterable[float | int]) -> Tuple[float, float]:
    """
    Return (mean, stddev) for a non-empty iterable of numbers.
    Uses population stddev if only one value is given (std = 0.0).
    """
    vals = list(values)
    if not vals:
        raise ValueError("mean_std() requires at least one value")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(stats.mean(vals)), float(stats.stdev(vals))


# ------------------------------- Self-test -----------------------------------

if __name__ == "__main__":
    # quick check
    dt, out = time_call(sum, [1, 2, 3])
    print(f"time={dt:.6f}s, out={out}")
    m, s = mean_std([1, 2, 3, 4])
    print(f"mean={m}, std={s}")