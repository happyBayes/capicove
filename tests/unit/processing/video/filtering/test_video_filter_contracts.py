import importlib
from datetime import UTC, datetime, timedelta

import pytest

from capicove.processing.video import VideoSource
from capicove.processing.video.filtering.video_filter_contracts import (
    AestheticsCriterion,
    Aggregation,
    FrameSamplingConfig,
    FrameTextSimilarityCriterion,
    LocalModelReference,
    MetadataCriterion,
    MotionCriterion,
    NsfwCriterion,
    OcrAreaCriterion,
    Reduction,
    SamplingMethod,
    TagCriterion,
    VideoFilterItem,
    VideoFilterRequest,
    WatermarkCriterion,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _item(*, text: str | None = None) -> VideoFilterItem:
    return VideoFilterItem("item-1", VideoSource(data=b"video"), text=text)


def _model() -> LocalModelReference:
    return LocalModelReference("local/model", revision="v1")


def test_request_defaults_to_all_and_accepts_every_criterion_type() -> None:
    sampling = FrameSamplingConfig(SamplingMethod.UNIFORM, frame_count=3)
    criteria = (
        MetadataCriterion("metadata", min_duration_seconds=1.0),
        MotionCriterion("motion", sampling=sampling, min_score=0.1),
        OcrAreaCriterion("ocr", sampling=sampling, max_area_ratio=0.5),
        AestheticsCriterion("aesthetics", _model(), sampling, min_score=0.4),
        NsfwCriterion("nsfw", _model(), sampling, target_label="nsfw", max_score=0.5),
        WatermarkCriterion(
            "watermark",
            _model(),
            sampling,
            target_label="watermark",
            max_probability=0.8,
        ),
        TagCriterion(
            "tags",
            _model(),
            sampling,
            tags=("People",),
            vocabulary_path="local/tags.txt",
        ),
        FrameTextSimilarityCriterion("similarity", _model(), sampling, min_score=0.1),
    )

    request = VideoFilterRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        items=(_item(text="a person walking"),),
        criteria=criteria,
        deadline=_deadline(),
    )

    assert request.aggregation is Aggregation.ALL
    assert criteria[1].videos_aggregation is Aggregation.ALL
    assert criteria[2].reduction is Reduction.AVG
    assert criteria[6].tags == ("people",)
    assert criteria[6].vocabulary_path == "local/tags.txt"


@pytest.mark.parametrize(
    ("items", "criteria", "message"),
    [
        ((), (MetadataCriterion("metadata", min_duration_seconds=1.0),), "item"),
        ((_item(),), (), "criterion"),
    ],
)
def test_request_rejects_empty_required_collections(
    items: tuple[VideoFilterItem, ...],
    criteria: tuple[MetadataCriterion, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VideoFilterRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            items=items,
            criteria=criteria,
            deadline=_deadline(),
        )


def test_request_rejects_duplicate_criterion_ids() -> None:
    with pytest.raises(ValueError, match="criterion_id"):
        VideoFilterRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            items=(_item(),),
            criteria=(
                MetadataCriterion("duplicate", min_duration_seconds=1.0),
                MetadataCriterion("duplicate", max_duration_seconds=10.0),
            ),
            deadline=_deadline(),
        )


def test_request_rejects_naive_deadline() -> None:
    with pytest.raises(ValueError, match="deadline"):
        VideoFilterRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            items=(_item(),),
            criteria=(MetadataCriterion("metadata", min_duration_seconds=1.0),),
            deadline=datetime.now(),
        )


def test_similarity_requires_text_for_every_item() -> None:
    with pytest.raises(ValueError, match="text"):
        VideoFilterRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            items=(_item(text=" "),),
            criteria=(
                FrameTextSimilarityCriterion(
                    "similarity",
                    _model(),
                    FrameSamplingConfig(),
                ),
            ),
            deadline=_deadline(),
        )


@pytest.mark.parametrize(
    "sampling",
    [
        pytest.param(FrameSamplingConfig, id="zero-frame-count"),
    ],
)
def test_sampling_validation_is_exercised(sampling: object) -> None:
    del sampling
    with pytest.raises(ValueError, match="frame_count"):
        FrameSamplingConfig(frame_count=0)
    with pytest.raises(ValueError, match="max_frames"):
        FrameSamplingConfig(max_frames=257)


def test_criteria_reject_reversed_or_empty_ranges() -> None:
    with pytest.raises(ValueError, match="range"):
        MetadataCriterion(
            "metadata",
            min_duration_seconds=10.0,
            max_duration_seconds=1.0,
        )
    with pytest.raises(ValueError, match="at least one"):
        MetadataCriterion("metadata")


def test_model_reference_rejects_remote_urls() -> None:
    with pytest.raises(ValueError, match="local"):
        LocalModelReference("https://example.com/model")


def test_filter_pipeline_and_video_tagger_are_deliberate_public_exports() -> None:
    video = importlib.import_module("capicove.processing.video")
    filtering = importlib.import_module("capicove.processing.video.filtering")

    assert video.VideoFilterPipeline.__name__ == "VideoFilterPipeline"
    assert filtering.VideoTagger.__name__ == "VideoTagger"
