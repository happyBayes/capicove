"""Temporal mapping must use published units and respect source identity."""

from dataclasses import replace

from capicove.contracts import ProcessingFingerprint, SearchableUnit, TargetType, TemporalRange
from capicove.retrieval.temporal import RefinedVideoCandidate, deduplicate_segments, temporal_iou


def segment(name: str, start: int, end: int, artifact: str = "a") -> SearchableUnit:
    return SearchableUnit(
        name,
        TargetType.VIDEO_SEGMENT,
        "a" * 64,
        artifact,
        "e",
        ("resource:preview:k",),
        "r",
        "v",
        ProcessingFingerprint("b" * 64),
        time_range=TemporalRange(start, end),
    )


def test_iou_and_artifact_isolation() -> None:
    a = RefinedVideoCandidate(segment("a", 0, 10000), 1.0, (), ())
    b = RefinedVideoCandidate(segment("b", 1000, 10000), 1.0, (), ())
    assert temporal_iou(a.unit.time_range, b.unit.time_range) == 0.9
    assert [v.unit.unit_id for v in deduplicate_segments((a, b))] == ["b"]
    other = replace(b, unit=segment("other", 1000, 10000, "other"))
    assert len(deduplicate_segments((a, other))) == 2
