"""Immutable runtime policy for bounded video search."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoLanePolicy:
    """Configure one known candidate lane."""

    lane_id: str
    weight: float
    degradable: bool
    dense: bool = False

    def __post_init__(self) -> None:
        if self.lane_id not in {
            "keyframe_visual",
            "segment_visual",
            "asr_lexical",
            "asr_dense",
            "ocr_lexical",
            "ocr_dense",
        }:
            raise ValueError("unknown lane")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("weight must be finite and positive")
        if not isinstance(self.degradable, bool) or not isinstance(self.dense, bool):
            raise ValueError("lane flags must be bool")
        if self.lane_id.endswith("visual") and self.degradable:
            raise ValueError("visual lane is required")


@dataclass(frozen=True, slots=True)
class VideoRetrievalProfile:
    """Keep candidate budgets outside public query arguments."""

    profile_id: str
    lanes: tuple[VideoLanePolicy, ...]
    visual_depth_floor: int = 100
    visual_depth_factor: int = 10
    maximum_visual_depth: int = 500
    text_depth_floor: int = 80
    text_depth_factor: int = 8
    maximum_text_depth: int = 300
    maximum_temporal_candidates: int = 500
    rrf_k: int = 60
    compact_response_max_bytes: int = 16_384
    deadline_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("invalid profile id")
        lanes = tuple(self.lanes)
        if any(not isinstance(lane, VideoLanePolicy) for lane in lanes) or len(
            {lane.lane_id for lane in lanes}
        ) != len(lanes):
            raise ValueError("invalid or duplicate lane")
        if not {"keyframe_visual", "segment_visual"} <= {lane.lane_id for lane in lanes}:
            raise ValueError("visual lanes required")
        object.__setattr__(self, "lanes", lanes)
        for name in (
            "visual_depth_floor",
            "visual_depth_factor",
            "maximum_visual_depth",
            "text_depth_floor",
            "text_depth_factor",
            "maximum_text_depth",
            "maximum_temporal_candidates",
            "rrf_k",
            "compact_response_max_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"invalid {name}")
        if (
            self.visual_depth_floor > self.maximum_visual_depth
            or self.text_depth_floor > self.maximum_text_depth
        ):
            raise ValueError("floor exceeds maximum")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("invalid deadline")

    @classmethod
    def video_local_ann(cls) -> VideoRetrievalProfile:
        """Return the versioned initial policy; qualification is separate."""
        return cls(
            "video-local-ann",
            (
                VideoLanePolicy("keyframe_visual", 1.0, False),
                VideoLanePolicy("segment_visual", 1.0, False),
                VideoLanePolicy("asr_lexical", 1.0, True),
                VideoLanePolicy("ocr_lexical", 0.5, True),
            ),
        )

    @classmethod
    def video_local_exact(cls) -> VideoRetrievalProfile:
        """Return the bounded exact reference profile."""
        return cls("video-local-exact", cls.video_local_ann().lanes)

    def visual_depth(self, top_k: int) -> int:
        """Compute a closed visual lane budget."""
        _validate_top_k(top_k)
        return min(
            self.maximum_visual_depth,
            max(self.visual_depth_floor, top_k * self.visual_depth_factor),
        )

    def text_depth(self, top_k: int) -> int:
        """Compute a closed text lane budget."""
        _validate_top_k(top_k)
        return min(
            self.maximum_text_depth, max(self.text_depth_floor, top_k * self.text_depth_factor)
        )


def _validate_top_k(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("invalid top_k")


class StaticVideoRetrievalProfileRegistry:
    """Resolve configured profiles without model or index side effects."""

    def __init__(self, profiles: tuple[VideoRetrievalProfile, ...] | None = None) -> None:
        values = (
            profiles
            if profiles is not None
            else (
                VideoRetrievalProfile.video_local_exact(),
                VideoRetrievalProfile.video_local_ann(),
            )
        )
        self._profiles = {item.profile_id: item for item in values}
        if len(self._profiles) != len(values):
            raise ValueError("duplicate profile")

    def resolve(self, profile_id: str) -> VideoRetrievalProfile:
        """Resolve one explicitly registered profile."""
        return self._profiles[profile_id]
