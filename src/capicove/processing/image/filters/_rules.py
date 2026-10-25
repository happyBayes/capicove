"""Pure rules shared by image filters."""

from __future__ import annotations

import math

from capicove.processing.image.models import Aggregation


def validate_range(minimum: float, maximum: float, name: str) -> None:
    """Validate one finite inclusive range."""
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError(f"{name} bounds must be finite")
    if minimum > maximum:
        raise ValueError(f"minimum {name} must not exceed maximum {name}")


def in_range(value: float, minimum: float, maximum: float) -> bool:
    """Return whether a finite value is within an inclusive range."""
    return math.isfinite(value) and minimum <= value <= maximum


def aggregate(values: tuple[bool, ...], mode: Aggregation) -> bool:
    """Aggregate a non-empty ordered collection of decisions."""
    mode = Aggregation(mode)
    return any(values) if mode is Aggregation.ANY else all(values)
