"""Pure reduction and aggregation rules for video filtering."""

from __future__ import annotations

import math

from capicove.processing.video.filtering.video_filter_contracts import (
    Aggregation,
    Reduction,
)


def in_range(value: float, minimum: float | None, maximum: float | None) -> bool:
    """Return whether a finite value is inside optional closed bounds."""
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if minimum is not None and value < minimum:
        return False
    return maximum is None or value <= maximum


def reduce_values(values: tuple[float, ...], mode: Reduction) -> float:
    """Reduce a non-empty tuple of finite values deterministically."""
    if not values:
        raise ValueError("cannot reduce empty values")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("values must be finite")
    if mode is Reduction.AVG:
        return sum(values) / len(values)
    if mode is Reduction.MAX:
        return max(values)
    return min(values)


def aggregate(values: tuple[bool, ...], mode: Aggregation) -> bool:
    """Aggregate a non-empty tuple of boolean decisions."""
    if not values:
        raise ValueError("cannot aggregate empty values")
    if mode is Aggregation.ALL:
        return all(values)
    return any(values)
