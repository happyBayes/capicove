"""Bounded temporal assembly over published catalog relations, without media I/O."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.contracts import SearchableUnit, TargetType, TemporalRange
from capicove.retrieval.video_fusion import FusedVideoCandidate, VideoCandidate
from capicove.retrieval.video_ports import VideoUnitCatalogReader


@dataclass(frozen=True, slots=True)
class RefinedVideoCandidate:
    """A published segment and the evidence that influenced its rank."""

    unit: SearchableUnit
    score: float
    contributions: tuple[VideoCandidate, ...]
    evidence_unit_ids: tuple[str, ...]


def temporal_iou(left: TemporalRange | None, right: TemporalRange | None) -> float:
    """Compute half-open interval intersection over union."""
    if left is None or right is None:
        return 0.0
    intersection = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    union = max(left.end_ms, right.end_ms) - min(left.start_ms, right.start_ms)
    return intersection / union


def _sort_key(item: RefinedVideoCandidate) -> tuple[float, int, int, int, str]:
    span = item.unit.time_range
    return (
        -item.score,
        -len({c.lane_id for c in item.contributions}),
        0 if span is None else span.end_ms - span.start_ms,
        0 if span is None else span.start_ms,
        item.unit.unit_id,
    )


def deduplicate_segments(
    candidates: tuple[RefinedVideoCandidate, ...], *, threshold: float = 0.8
) -> tuple[RefinedVideoCandidate, ...]:
    """Select stable representatives without crossing source identities."""
    if not 0 < threshold <= 1:
        raise ValueError("invalid IoU threshold")
    kept: list[RefinedVideoCandidate] = []
    for candidate in sorted(candidates, key=_sort_key):
        if any(
            other.unit.artifact_ref == candidate.unit.artifact_ref
            and other.unit.source_hash == candidate.unit.source_hash
            and temporal_iou(other.unit.time_range, candidate.unit.time_range) >= threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return tuple(kept)


class TemporalCandidateAssembler:
    """Map ranked evidence only through an immutable catalog."""

    def __init__(self, catalog: VideoUnitCatalogReader) -> None:
        self._catalog = catalog

    def assemble(
        self, *, fused: tuple[FusedVideoCandidate, ...], maximum_candidates: int = 500
    ) -> tuple[RefinedVideoCandidate, ...]:
        """Bound both input work and the number of segment units retrieved."""
        if isinstance(maximum_candidates, bool) or not 1 <= maximum_candidates <= 500:
            raise ValueError("invalid temporal candidate limit")
        evidence = fused[:maximum_candidates]
        units = {
            unit.unit_id: unit
            for unit in self._catalog.get_many(tuple(c.unit_id for c in evidence))
        }
        if len(units) != len(evidence):
            raise ValueError("index_corrupt")
        keyframes = tuple(
            key for key, unit in units.items() if unit.unit_type is TargetType.KEYFRAME
        )
        text = tuple(key for key, unit in units.items() if unit.unit_type is TargetType.TEXT_CHUNK)
        key_relations = self._catalog.segments_for_keyframes(keyframes)
        text_relations = self._catalog.segments_for_texts(text)
        by_segment: dict[str, list[FusedVideoCandidate]] = {}
        for candidate in evidence:
            unit = units[candidate.unit_id]
            related: tuple[str, ...]
            if unit.unit_type is TargetType.VIDEO_SEGMENT:
                related = (unit.unit_id,)
            elif unit.unit_type is TargetType.KEYFRAME:
                related = key_relations.get(unit.unit_id, ())
            else:
                related = tuple(key for key, _ in text_relations.get(unit.unit_id, ()))
            if len(related) > maximum_candidates:
                raise ValueError("index_corrupt")
            for segment_id in related:
                if segment_id not in by_segment and len(by_segment) >= maximum_candidates:
                    continue
                by_segment.setdefault(segment_id, []).append(candidate)
        segments = self._catalog.get_many(tuple(by_segment))
        if len(segments) != len(by_segment):
            raise ValueError("index_corrupt")
        result: list[RefinedVideoCandidate] = []
        for segment in segments:
            span = segment.time_range
            if segment.unit_type is not TargetType.VIDEO_SEGMENT or span is None:
                raise ValueError("index_corrupt")
            contributors = by_segment[segment.unit_id]
            lane_scores: dict[str, float] = {}
            all_contributions: list[VideoCandidate] = []
            for item in contributors:
                source = units[item.unit_id]
                if (
                    source.artifact_ref != segment.artifact_ref
                    or source.source_hash != segment.source_hash
                ):
                    raise ValueError("index_corrupt")
                if source.unit_type is TargetType.KEYFRAME and (
                    source.timestamp_ms is None
                    or not span.start_ms <= source.timestamp_ms < span.end_ms
                ):
                    raise ValueError("index_corrupt")
                if source.unit_type is TargetType.TEXT_CHUNK:
                    locator = source.text_locator
                    if locator is None or temporal_iou(span, locator.time_range) <= 0:
                        raise ValueError("index_corrupt")
                # Repeated evidence in one lane cannot dominate through volume.
                weighted_scores = dict(item.lane_scores)
                for contribution in item.contributions:
                    lane_scores[contribution.lane_id] = max(
                        lane_scores.get(contribution.lane_id, 0),
                        weighted_scores[contribution.lane_id],
                    )
                    all_contributions.append(contribution)
            result.append(
                RefinedVideoCandidate(
                    segment,
                    sum(lane_scores.values()),
                    tuple(all_contributions),
                    tuple(item.unit_id for item in contributors),
                )
            )
        return deduplicate_segments(tuple(result))
