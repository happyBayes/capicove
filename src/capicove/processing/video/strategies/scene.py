"""Scene detector configuration and range normalization."""

from __future__ import annotations

import math

from capicove.processing.video.models import SceneDetectorName, SceneSplitConfig, TimeRange
from capicove.processing.video.strategies.errors import PlanningFailure


FLOAT_TOLERANCE_SECONDS = 1e-9
BACKEND_DRIFT_TOLERANCE_SECONDS = 1e-3
ALLOWED_OPTIONS: dict[SceneDetectorName, frozenset[str]] = {
    SceneDetectorName.CONTENT: frozenset({"threshold", "min_scene_len", "luma_only"}),
    SceneDetectorName.ADAPTIVE: frozenset({"adaptive_threshold", "min_scene_len", "window_width"}),
    SceneDetectorName.THRESHOLD: frozenset({"threshold", "min_scene_len", "fade_bias"}),
}


def validate_scene_config(config: SceneSplitConfig) -> None:
    """Reject unknown detectors and detector-specific options."""
    if not isinstance(config.detector, SceneDetectorName):
        raise PlanningFailure("scene detector is not supported")
    unknown = set(config.options).difference(ALLOWED_OPTIONS[config.detector])
    if unknown:
        raise PlanningFailure(f"unknown scene option: {min(unknown)}")


def normalize_scene_ranges(
    source_duration: float,
    ranges: tuple[TimeRange, ...],
) -> tuple[TimeRange, ...]:
    """Validate full chronological scene coverage and suppress ineffective splits."""
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise PlanningFailure("source duration must be finite and positive")
    if len(ranges) <= 1:
        return ()
    if abs(ranges[0].start_seconds) > FLOAT_TOLERANCE_SECONDS:
        raise PlanningFailure("scene ranges must begin at zero")
    previous_end = 0.0
    for item in ranges:
        if abs(item.start_seconds - previous_end) > FLOAT_TOLERANCE_SECONDS:
            raise PlanningFailure("scene ranges must be chronological and contiguous")
        if item.end_seconds > source_duration + BACKEND_DRIFT_TOLERANCE_SECONDS:
            raise PlanningFailure("scene range exceeds source duration")
        previous_end = item.end_seconds
    if abs(previous_end - source_duration) > BACKEND_DRIFT_TOLERANCE_SECONDS:
        raise PlanningFailure("scene ranges must cover source duration")
    return (*ranges[:-1], TimeRange(ranges[-1].start_seconds, source_duration))
