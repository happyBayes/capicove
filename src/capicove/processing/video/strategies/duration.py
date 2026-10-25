"""Pure fixed-duration video range planning."""

from __future__ import annotations

import math

from capicove.processing.video.models import DurationSplitConfig, TimeRange
from capicove.processing.video.strategies.errors import PlanningFailure


FLOAT_TOLERANCE_SECONDS = 1e-9


def plan_duration_ranges(
    source_duration: float,
    config: DurationSplitConfig,
) -> tuple[TimeRange, ...]:
    """Plan deterministic fixed-duration ranges without media dependencies."""
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise PlanningFailure("source duration must be finite and positive")
    if source_duration <= config.split_duration + FLOAT_TOLERANCE_SECONDS:
        return ()

    ranges: list[TimeRange] = []
    step = config.split_duration - config.overlap_duration
    start = 0.0
    while start < source_duration - FLOAT_TOLERANCE_SECONDS:
        end = min(start + config.split_duration, source_duration)
        length = end - start
        if length <= FLOAT_TOLERANCE_SECONDS:
            break
        is_tail = end >= source_duration - FLOAT_TOLERANCE_SECONDS
        if is_tail and length + FLOAT_TOLERANCE_SECONDS < config.min_last_split_duration:
            break
        ranges.append(TimeRange(start, end))
        start += step
    return tuple(ranges)
