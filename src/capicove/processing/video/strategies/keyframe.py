"""Pure keyframe video range planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

from capicove.processing.video.models import (
    KeyframeSplitConfig,
    TimeRange,
    VideoSplitWarning,
    VideoSplitWarningCode,
)
from capicove.processing.video.strategies.errors import PlanningFailure


@dataclass(frozen=True, slots=True)
class KeyframePlan:
    """Return planned keyframe ranges and visible fallback state."""

    ranges: tuple[TimeRange, ...]
    warning: VideoSplitWarning | None = None
    degraded: bool = False


def plan_keyframe_ranges(
    source_duration: float,
    timestamps: tuple[float, ...],
    config: KeyframeSplitConfig | None = None,
) -> KeyframePlan:
    """Validate keyframes and build adjacent ranges without sorting metadata."""
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise PlanningFailure("source duration must be finite and positive")
    if not timestamps:
        return KeyframePlan(
            ranges=(),
            warning=VideoSplitWarning(
                code=VideoSplitWarningCode.NO_KEYFRAMES,
                message="No keyframes were reported; the original video was retained.",
            ),
            degraded=True,
        )

    normalized: list[float] = []
    for timestamp in timestamps:
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > source_duration:
            raise PlanningFailure(
                "keyframe timestamps must be finite and within source duration",
                VideoSplitWarningCode.INVALID_TIMESTAMPS,
            )
        if normalized and timestamp < normalized[-1]:
            raise PlanningFailure(
                "keyframe timestamps must be chronological",
                VideoSplitWarningCode.INVALID_TIMESTAMPS,
            )
        if not normalized or timestamp != normalized[-1]:
            normalized.append(timestamp)

    if normalized[0] != 0.0:
        normalized.insert(0, 0.0)
    if normalized[-1] != source_duration:
        normalized.append(source_duration)
    minimum = config.min_segment_duration if config is not None else 0.0
    if minimum == 0.0:
        ranges = tuple(TimeRange(start, end) for start, end in pairwise(normalized) if end > start)
        return KeyframePlan(ranges=ranges)

    ranges_list: list[TimeRange] = []
    start = normalized[0]
    for boundary in normalized[1:-1]:
        if boundary - start >= minimum and source_duration - boundary >= minimum:
            ranges_list.append(TimeRange(start, boundary))
            start = boundary
    ranges_list.append(TimeRange(start, source_duration))
    ranges = tuple(ranges_list)
    return KeyframePlan(ranges=ranges)
