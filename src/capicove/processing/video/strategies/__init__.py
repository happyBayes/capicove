"""Pure range planners for Capicove video splitting."""

from capicove.processing.video.strategies.duration import plan_duration_ranges
from capicove.processing.video.strategies.errors import PlanningFailure
from capicove.processing.video.strategies.keyframe import KeyframePlan, plan_keyframe_ranges
from capicove.processing.video.strategies.scene import (
    normalize_scene_ranges,
    validate_scene_config,
)


__all__ = [
    "KeyframePlan",
    "PlanningFailure",
    "normalize_scene_ranges",
    "plan_duration_ranges",
    "plan_keyframe_ranges",
    "validate_scene_config",
]
