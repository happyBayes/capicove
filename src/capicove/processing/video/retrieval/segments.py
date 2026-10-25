"""Pure temporal candidate construction and visual pooling."""

from dataclasses import asdict, dataclass
from math import isfinite, sqrt

from capicove.contracts.retrieval import ProcessingFingerprint, SearchableUnit, TemporalRange
from capicove.processing.video.retrieval.keyframes import validate_scenes
from capicove.processing.video.retrieval.models import (
    CandidateVideoSegment,
    RetrievalKeyframe,
    sha256,
)


@dataclass(frozen=True, slots=True)
class SegmentBuildConfig:
    """Bounded scene/transcript segment policy."""

    minimum_duration_ms: int = 2000
    target_duration_ms: int = 15000
    maximum_duration_ms: int = 30000
    transcript_snap_tolerance_ms: int = 2000
    policy_version: str = "scene-transcript-segments-v1"
    strong_boundaries_ms: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if (
            not 0
            < self.minimum_duration_ms
            <= self.target_duration_ms
            <= self.maximum_duration_ms
            <= 30000
            or self.transcript_snap_tolerance_ms < 0
        ):
            raise ValueError("invalid segment durations")


def build_candidate_segments(
    *,
    source_hash: str,
    duration_ms: int,
    scenes: tuple[TemporalRange, ...],
    keyframes: tuple[RetrievalKeyframe, ...],
    asr_units: tuple[SearchableUnit, ...],
    ocr_units: tuple[SearchableUnit, ...],
    config: SegmentBuildConfig,
    processing_fingerprint: ProcessingFingerprint,
) -> tuple[CandidateVideoSegment, ...]:
    """Build immutable ranges and require contained visual evidence for each."""
    sha256(source_hash)
    scenes = scenes or (TemporalRange(0, duration_ms),)
    validate_scenes(scenes)
    if any(s.end_ms > duration_ms for s in scenes):
        raise ValueError("scene exceeds source duration")
    ranges: list[tuple[int, int]] = []
    for scene in scenes:
        start = scene.start_ms
        while start < scene.end_ms:
            end = min(start + config.target_duration_ms, scene.end_ms)
            ranges.append((start, end))
            start = end
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if (
            merged
            and start == merged[-1][1]
            and start not in config.strong_boundaries_ms
            and (
                end - start < config.minimum_duration_ms
                or merged[-1][1] - merged[-1][0] < config.minimum_duration_ms
            )
            and end - merged[-1][0] <= config.maximum_duration_ms
        ):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    boundaries = sorted(
        {
            t
            for u in asr_units
            if u.text_locator and u.text_locator.time_range
            for t in (u.text_locator.time_range.start_ms, u.text_locator.time_range.end_ms)
        }
    )
    for index in range(len(merged) - 1):
        start, edge = merged[index]
        next_start, end = merged[index + 1]
        if edge != next_start or edge in config.strong_boundaries_ms:
            continue
        valid = [
            t
            for t in boundaries
            if abs(t - edge) <= config.transcript_snap_tolerance_ms
            and config.minimum_duration_ms <= t - start <= config.maximum_duration_ms
            and config.minimum_duration_ms <= end - t <= config.maximum_duration_ms
            and any(s.start_ms < t < s.end_ms and s.start_ms < edge < s.end_ms for s in scenes)
        ]
        if valid:
            split = min(valid, key=lambda t: (abs(t - edge), t))
            merged[index], merged[index + 1] = (start, split), (split, end)
    result = []
    for start, end in merged:
        if end - start < config.minimum_duration_ms and end != duration_ms:
            raise ValueError("non-terminal segment below minimum duration")
        linked = sorted(
            (k for k in keyframes if start <= k.timestamp_ms < end),
            key=lambda k: (k.timestamp_ms, k.unit_id),
        )
        if not linked:
            raise ValueError("segment requires a contained keyframe")
        preview = min(linked, key=lambda k: (-k.quality_score, k.timestamp_ms, k.unit_id))

        def overlap(
            units: tuple[SearchableUnit, ...], start: int = start, end: int = end
        ) -> tuple[str, ...]:
            return tuple(
                u.unit_id
                for u in units
                if u.text_locator
                and u.text_locator.time_range
                and u.text_locator.time_range.start_ms < end
                and u.text_locator.time_range.end_ms > start
            )

        identity = ProcessingFingerprint.from_manifest(
            {
                "source": source_hash,
                "start": start,
                "end": end,
                "policy": asdict(config),
                "fingerprint": processing_fingerprint.sha256,
            }
        )
        result.append(
            CandidateVideoSegment(
                "segment:" + identity.sha256,
                TemporalRange(start, end),
                preview.preview_resource_ref,
                tuple(k.unit_id for k in linked),
                overlap(asr_units),
                overlap(ocr_units),
                processing_fingerprint,
            )
        )
    return tuple(result)


def pool_segment_vector(
    vectors: tuple[tuple[float, ...], ...], quality_weights: tuple[float, ...]
) -> tuple[float, ...]:
    """Normalize a positive quality-weighted mean in the unchanged base space."""
    if not vectors or not vectors[0]:
        raise ValueError("vectors must be non-empty")
    if len(vectors) != len(quality_weights):
        raise ValueError("vectors and weights must have same length")
    dimension = len(vectors[0])
    if any(len(v) != dimension or any(not isfinite(x) for x in v) for v in vectors) or any(
        not isfinite(w) or w <= 0 for w in quality_weights
    ):
        raise ValueError("invalid dimensions, finite values or positive weights")
    scale = max(quality_weights)
    weights = tuple(w / scale for w in quality_weights)
    pooled = tuple(
        sum(v[i] * w for v, w in zip(vectors, weights, strict=True)) / sum(weights)
        for i in range(dimension)
    )
    norm = sqrt(sum(x * x for x in pooled))
    if not isfinite(norm) or norm == 0:
        raise ValueError("pooled vector must have finite non-zero norm")
    return tuple(x / norm for x in pooled)
