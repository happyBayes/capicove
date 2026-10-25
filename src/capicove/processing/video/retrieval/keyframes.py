"""Deterministic scene-balanced semantic frame selection."""

import itertools
from dataclasses import asdict, dataclass
from math import isfinite

from capicove.contracts.retrieval import TemporalRange
from capicove.processing.video.retrieval.models import FrameObservation


@dataclass(frozen=True, slots=True)
class KeyframeSelectionConfig:
    """All output-sensitive selection policy settings."""

    max_coverage_gap_ms: int = 8000
    center_fraction: float = 0.5
    policy_version: str = "scene-balanced-v1"

    def __post_init__(self) -> None:
        if (
            self.max_coverage_gap_ms <= 0
            or not isfinite(self.center_fraction)
            or not 0 < self.center_fraction <= 1
            or not self.policy_version
        ):
            raise ValueError("invalid keyframe selection policy")


def keyframe_policy_manifest(config: KeyframeSelectionConfig) -> dict[str, object]:
    """Return normalized identity inputs."""
    return asdict(config)


def validate_scenes(scenes: tuple[TemporalRange, ...]) -> None:
    """Reject overlapping or unordered scene coordinates."""
    if any(a.end_ms > b.start_ms for a, b in itertools.pairwise(scenes)):
        raise ValueError("scenes must be sorted and non-overlapping")


def select_retrieval_keyframes(
    scenes: tuple[TemporalRange, ...],
    frames: tuple[FrameObservation, ...],
    config: KeyframeSelectionConfig,
) -> tuple[FrameObservation, ...]:
    """Select center-quality representatives and fill attainable coverage gaps."""
    validate_scenes(scenes)
    times = [f.timestamp_ms for f in frames]
    if times != sorted(set(times)):
        raise ValueError("frames must have sorted unique timestamps")
    eligible = tuple(
        f for f in frames if not f.black_frame and not f.corrupt and f.quality_score > 0
    )
    if not eligible:
        raise ValueError("no eligible frame")
    if not scenes:
        scenes = (TemporalRange(0, eligible[-1].timestamp_ms + 1),)
    selected: dict[int, FrameObservation] = {}
    for scene in scenes:
        pool = [f for f in eligible if scene.start_ms <= f.timestamp_ms < scene.end_ms]
        if not pool:
            raise ValueError("scene has no eligible frame")
        center = (scene.start_ms + scene.end_ms) / 2
        radius = (scene.end_ms - scene.start_ms) * config.center_fraction / 2
        near = [f for f in pool if abs(f.timestamp_ms - center) <= radius] or pool
        best = min(
            near,
            key=lambda f: (
                -f.quality_score,
                abs(f.timestamp_ms - center),
                f.timestamp_ms,
                f.preview_candidate_ref,
            ),
        )
        chosen = {best.timestamp_ms: best}
        # Include scene edges in coverage checks; greedily split oversized gaps.
        while True:
            positions = [scene.start_ms, *sorted(chosen), scene.end_ms]
            additions = []
            for left, right in itertools.pairwise(positions):
                if right - left <= config.max_coverage_gap_ms:
                    continue
                candidates = [
                    f
                    for f in pool
                    if left < f.timestamp_ms < right and f.timestamp_ms not in chosen
                ]
                if candidates:
                    additions.append(
                        min(
                            candidates,
                            key=lambda f: (
                                abs(f.timestamp_ms - (left + right) / 2),
                                -f.quality_score,
                                f.timestamp_ms,
                            ),
                        )
                    )
            if not additions:
                break
            chosen.update((f.timestamp_ms, f) for f in additions)
        selected.update(chosen)
    return tuple(selected[t] for t in sorted(selected))
