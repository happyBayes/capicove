"""Behavioral tests for offline retrieval preparation."""

import itertools

import pytest

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange
from capicove.processing.video.retrieval.keyframes import (
    KeyframeSelectionConfig,
    select_retrieval_keyframes,
)
from capicove.processing.video.retrieval.models import FrameObservation, RetrievalKeyframe
from capicove.processing.video.retrieval.segments import (
    SegmentBuildConfig,
    build_candidate_segments,
    pool_segment_vector,
)


def test_center_quality_and_coverage() -> None:
    frames = tuple(
        FrameObservation(t, q, str(t)) for t, q in ((1000, 0.95), (3500, 0.8), (4200, 0.9))
    )
    assert select_retrieval_keyframes(
        (TemporalRange(0, 8000),), frames, KeyframeSelectionConfig()
    ) == (frames[2],)
    long = tuple(FrameObservation(t, 0.8, str(t)) for t in (1000, 8000, 16000, 24000))
    selected = select_retrieval_keyframes(
        (TemporalRange(0, 25000),), long, KeyframeSelectionConfig()
    )
    assert max(b.timestamp_ms - a.timestamp_ms for a, b in itertools.pairwise(selected)) <= 8000


def test_pooling_rejects_bad_values_and_normalizes() -> None:
    assert pool_segment_vector(((1.0, 0.0), (0.0, 1.0)), (3.0, 1.0)) == pytest.approx(
        (0.948683298, 0.316227766)
    )
    for vectors, weights in (
        ((), ()),
        (((0.0, 0.0),), (1.0,)),
        (((float("nan"),),), (1.0,)),
        (((1.0,),), (0.0,)),
    ):
        with pytest.raises(ValueError):
            pool_segment_vector(vectors, weights)


def test_segment_requires_contained_frame() -> None:
    fp = ProcessingFingerprint("b" * 64)
    frame = RetrievalKeyframe("k", 1000, TemporalRange(0, 65000), "resource:p", "a" * 64, 0.8, fp)
    with pytest.raises(ValueError, match="contained keyframe"):
        build_candidate_segments(
            source_hash="a" * 64,
            duration_ms=65000,
            scenes=(TemporalRange(0, 65000),),
            keyframes=(frame,),
            asr_units=(),
            ocr_units=(),
            config=SegmentBuildConfig(),
            processing_fingerprint=fp,
        )
