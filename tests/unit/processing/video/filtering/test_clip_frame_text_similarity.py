from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from capicove.processing.video import AcceptedVideoSource, VideoSource
from capicove.processing.video.filtering._adapters.huggingface_frame_scorers import (
    HuggingFaceClipSimilarityEvaluator,
    ResolvedHuggingFaceModel,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    FrameTextSimilarityCriterion,
    LocalModelReference,
    Reduction,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    InvalidVideoFilterInputError,
    SampledFrame,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _frames() -> tuple[SampledFrame, ...]:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return (
        SampledFrame(0, 0.0, 4, 4, image),
        SampledFrame(1, 1.0, 4, 4, image),
    )


class FakeClipResolver:
    def resolve(
        self,
        reference: LocalModelReference,
        *,
        model_kind: str,
    ) -> ResolvedHuggingFaceModel:
        assert model_kind == "clip"
        return ResolvedHuggingFaceModel(
            reference=reference.reference,
            revision=reference.revision,
            manifest_hash="d" * 64,
            model=object(),
            processor=object(),
            id2label={},
        )


def _criterion(reduction: Reduction = Reduction.AVG) -> FrameTextSimilarityCriterion:
    return FrameTextSimilarityCriterion(
        "similarity",
        LocalModelReference("local/clip"),
        FrameSamplingConfig(),
        min_score=0.5,
        reduction=reduction,
    )


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource("source-1", "a" * 64, "video/mp4", data=b"video")


def _metadata() -> VideoFilterMetadata:
    return VideoFilterMetadata(1.0, 4, 4, 4, 4, backend="fake")


def test_similarity_preserves_raw_logits_and_reduces_normalized_scores() -> None:
    evaluator = HuggingFaceClipSimilarityEvaluator(
        FakeClipResolver(),
        inference=lambda resolved, frames, text: ([0.2, 0.8], [20.0, 80.0]),
    )

    result = evaluator.evaluate(
        VideoFilterItem("item-1", VideoSource(data=b"video"), text="a blue square"),
        _source(),
        _metadata(),
        _frames(),
        _criterion(),
        _deadline(),
    )

    assert result.score == pytest.approx(0.5)
    assert result.passed is True
    assert [metric.value for metric in result.frame_metrics] == [0.2, 0.8]
    assert [metric.raw_value for metric in result.frame_metrics] == [20.0, 80.0]


def test_similarity_uses_each_items_associated_text() -> None:
    texts: list[str] = []

    def inference(
        resolved: ResolvedHuggingFaceModel,
        frames: tuple[SampledFrame, ...],
        text: str,
    ) -> tuple[list[float], list[float]]:
        del resolved, frames
        texts.append(text)
        return [0.6, 0.6], [6.0, 6.0]

    evaluator = HuggingFaceClipSimilarityEvaluator(FakeClipResolver(), inference=inference)
    for item_id, text in (("item-1", "a dog"), ("item-2", "a car")):
        evaluator.evaluate(
            VideoFilterItem(item_id, VideoSource(data=b"video"), text=text),
            _source(),
            _metadata(),
            _frames(),
            _criterion(),
            _deadline(),
        )

    assert texts == ["a dog", "a car"]


def test_similarity_rejects_missing_text_at_evaluator_boundary() -> None:
    evaluator = HuggingFaceClipSimilarityEvaluator(
        FakeClipResolver(),
        inference=lambda resolved, frames, text: ([0.5, 0.5], [5.0, 5.0]),
    )

    with pytest.raises(InvalidVideoFilterInputError, match="text"):
        evaluator.evaluate(
            VideoFilterItem("item-1", VideoSource(data=b"video")),
            _source(),
            _metadata(),
            _frames(),
            _criterion(),
            _deadline(),
        )


def test_similarity_max_reduction_is_not_fixed_logit_scaling() -> None:
    evaluator = HuggingFaceClipSimilarityEvaluator(
        FakeClipResolver(),
        inference=lambda resolved, frames, text: ([0.1, 0.7], [1.0, 2.0]),
    )

    result = evaluator.evaluate(
        VideoFilterItem("item-1", VideoSource(data=b"video"), text="a subject"),
        _source(),
        _metadata(),
        _frames(),
        _criterion(Reduction.MAX),
        _deadline(),
    )

    assert result.score == 0.7
    assert result.score != 0.02
