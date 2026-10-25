"""Boundary rejection, policy identity and deterministic segment tests."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange
from capicove.processing.video.retrieval._adapters.local import (
    LocalContentStore,
    LocalDerivedCacheCatalog,
)
from capicove.processing.video.retrieval.keyframes import (
    KeyframeSelectionConfig,
    keyframe_policy_manifest,
    select_retrieval_keyframes,
)
from capicove.processing.video.retrieval.models import (
    FrameObservation,
    RetrievalKeyframe,
    VideoDecodeObservation,
)
from capicove.processing.video.retrieval.segments import (
    SegmentBuildConfig,
    build_candidate_segments,
)


@pytest.mark.parametrize("quality", [-1.0, 2.0, float("nan"), float("inf")])
def test_frame_quality_rejected(quality: float) -> None:
    with pytest.raises(ValueError):
        FrameObservation(0, quality, "preview")


def test_keyframe_selection_rejections_and_fallback() -> None:
    frames = (FrameObservation(10, 0.8, "a"), FrameObservation(20, 0.8, "b"))
    config = KeyframeSelectionConfig()
    assert select_retrieval_keyframes((), frames, config)
    for bad in (tuple(reversed(frames)), (frames[0], frames[0])):
        with pytest.raises(ValueError, match="sorted"):
            select_retrieval_keyframes((), bad, config)
    with pytest.raises(ValueError, match="eligible"):
        select_retrieval_keyframes((), (replace(frames[0], black_frame=True),), config)
    with pytest.raises(ValueError, match="non-overlapping"):
        select_retrieval_keyframes((TemporalRange(0, 30), TemporalRange(10, 20)), frames, config)
    baseline = ProcessingFingerprint.from_manifest(keyframe_policy_manifest(config))
    for changed in (
        replace(config, max_coverage_gap_ms=4000),
        replace(config, center_fraction=0.25),
        replace(config, policy_version="v2"),
    ):
        assert ProcessingFingerprint.from_manifest(keyframe_policy_manifest(changed)) != baseline


def test_short_scene_merge_and_strong_boundary() -> None:
    fp = ProcessingFingerprint("b" * 64)
    scenes = (TemporalRange(0, 1000), TemporalRange(1000, 5000))
    frames = tuple(
        RetrievalKeyframe(str(t), t, scene, "resource:preview:" + ("a" * 64), "a" * 64, 0.8, fp)
        for t, scene in ((500, scenes[0]), (2500, scenes[1]))
    )
    kwargs: dict[str, Any] = dict(
        source_hash="a" * 64,
        duration_ms=5000,
        scenes=scenes,
        keyframes=frames,
        asr_units=(),
        ocr_units=(),
        processing_fingerprint=fp,
    )
    first = build_candidate_segments(**kwargs, config=SegmentBuildConfig())
    assert [s.time_range for s in first] == [TemporalRange(0, 5000)]
    assert first == build_candidate_segments(**kwargs, config=SegmentBuildConfig())
    with pytest.raises(ValueError, match="minimum"):
        build_candidate_segments(**kwargs, config=SegmentBuildConfig(strong_boundaries_ms=(1000,)))


def test_decode_observation_order_and_containment() -> None:
    frame = FrameObservation(10, 0.8, "a")
    for duration, frames in ((10, (frame,)), (20, (frame, frame))):
        with pytest.raises(ValueError):
            VideoDecodeObservation(duration, frames, ())


def test_content_and_cache_are_immutable(tmp_path: Path) -> None:
    store = LocalContentStore(tmp_path / "objects", max_object_bytes=8)
    ref = store.put(b"abc")
    assert store.put(b"abc") == ref
    assert store.read(ref) == b"abc"
    with pytest.raises(ValueError, match="capacity"):
        store.put(b"longer than eight bytes")
    with pytest.raises(ValueError):
        store.read("resource:sha256:../secret")
    catalog = LocalDerivedCacheCatalog(tmp_path / "cache")
    fp = ProcessingFingerprint("b" * 64)
    assert catalog.lookup(fp) is None
    catalog.publish(fp, ref)
    assert catalog.lookup(fp) == ref
    with pytest.raises(ValueError, match="conflict"):
        catalog.publish(fp, "different")
