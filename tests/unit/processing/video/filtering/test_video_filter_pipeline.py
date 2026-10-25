from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.processing.video import VideoSource
from capicove.processing.video.filtering.video_filter_contracts import (
    AestheticsCriterion,
    Aggregation,
    FrameSamplingConfig,
    LocalModelReference,
    MotionCriterion,
    OcrAreaCriterion,
    VideoFilterFailureCode,
    VideoFilterItem,
    VideoFilterRequest,
    VideoFilterStatus,
)
from capicove.processing.video.filtering.video_filter_pipeline import VideoFilterPipeline
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
)
from tests.unit.processing.video.fakes import FakePathAuthorizer
from tests.unit.processing.video.filtering.video_filter_fakes import (
    FakeEvaluator,
    FakeFrameSampler,
    FakeMetadataProbe,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _request(
    *,
    aggregation: Aggregation = Aggregation.ALL,
    videos_aggregation: Aggregation = Aggregation.ALL,
    two_items: bool = False,
) -> VideoFilterRequest:
    sampling = FrameSamplingConfig(frame_count=2)
    items = [VideoFilterItem("item-1", VideoSource(data=b"same-video"))]
    if two_items:
        items.append(VideoFilterItem("item-2", VideoSource(data=b"same-video")))
    return VideoFilterRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        items=tuple(items),
        criteria=(
            MotionCriterion(
                "motion",
                sampling=sampling,
                videos_aggregation=videos_aggregation,
            ),
            OcrAreaCriterion(
                "ocr",
                sampling=sampling,
                videos_aggregation=videos_aggregation,
            ),
        ),
        deadline=_deadline(),
        aggregation=aggregation,
    )


def _pipeline(evaluator: FakeEvaluator) -> tuple[VideoFilterPipeline, FakeFrameSampler]:
    sampler = FakeFrameSampler()
    return (
        VideoFilterPipeline(
            metadata_probe=FakeMetadataProbe(),
            frame_sampler=sampler,
            evaluators={
                "motion": evaluator,
                "ocr_area": evaluator,
                "aesthetics": evaluator,
            },
            path_authorizer=FakePathAuthorizer(),
        ),
        sampler,
    )


def test_pipeline_reuses_equal_sampling_for_duplicate_content() -> None:
    decisions = {
        (item_id, criterion_id): True
        for item_id in ("item-1", "item-2")
        for criterion_id in ("motion", "ocr")
    }
    pipeline, sampler = _pipeline(FakeEvaluator(decisions))

    result = pipeline.evaluate(_request(two_items=True))

    assert result.status is VideoFilterStatus.SUCCEEDED
    assert result.keep is True
    assert len(sampler.calls) == 1
    assert [item.item_id for item in result.item_results] == ["item-1", "item-2"]


def test_pipeline_defaults_to_all_criteria() -> None:
    pipeline, _ = _pipeline(FakeEvaluator({("item-1", "motion"): True, ("item-1", "ocr"): False}))

    result = pipeline.evaluate(_request())

    assert [criterion.passed for criterion in result.criterion_results] == [True, False]
    assert result.keep is False


def test_pipeline_supports_any_criterion_and_video_aggregation() -> None:
    decisions = {
        ("item-1", "motion"): True,
        ("item-2", "motion"): False,
        ("item-1", "ocr"): False,
        ("item-2", "ocr"): False,
    }
    pipeline, _ = _pipeline(FakeEvaluator(decisions))

    result = pipeline.evaluate(
        _request(
            aggregation=Aggregation.ANY,
            videos_aggregation=Aggregation.ANY,
            two_items=True,
        )
    )

    assert [criterion.passed for criterion in result.criterion_results] == [True, False]
    assert result.keep is True


def test_pipeline_failure_retains_completed_diagnostics_without_decision() -> None:
    error = BackendUnavailableFilterError("OCR backend is unavailable")
    decisions = {
        ("item-1", "motion"): True,
        ("item-2", "motion"): True,
        ("item-1", "ocr"): True,
        ("item-2", "ocr"): True,
    }
    pipeline, _ = _pipeline(FakeEvaluator(decisions, failure=("item-2", "ocr", error)))

    result = pipeline.evaluate(_request(two_items=True))

    assert result.status is VideoFilterStatus.FAILED
    assert result.keep is None
    assert [criterion.criterion_id for criterion in result.criterion_results] == ["motion"]
    assert len(result.item_results) == 2
    assert result.failures[0].code is VideoFilterFailureCode.BACKEND_UNAVAILABLE
    assert result.failures[0].item_id == "item-2"
    assert result.failures[0].criterion_id == "ocr"


def test_pipeline_rejects_expired_deadline_before_probe() -> None:
    evaluator = FakeEvaluator({})
    pipeline, sampler = _pipeline(evaluator)
    request = VideoFilterRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        items=(VideoFilterItem("item-1", VideoSource(data=b"video")),),
        criteria=(
            AestheticsCriterion(
                "aesthetics",
                LocalModelReference("local/model"),
                FrameSamplingConfig(),
            ),
        ),
        deadline=datetime.now(UTC) - timedelta(seconds=1),
    )

    result = pipeline.evaluate(request)

    assert result.status is VideoFilterStatus.FAILED
    assert result.failures[0].code is VideoFilterFailureCode.TIMEOUT
    assert sampler.calls == []
    assert evaluator.calls == []


def test_pipeline_authorizes_model_paths_before_source_or_evaluator_work() -> None:
    evaluator = FakeEvaluator({})
    sampler = FakeFrameSampler()
    probe = FakeMetadataProbe()
    authorizer = FakePathAuthorizer(deny=True)
    pipeline = VideoFilterPipeline(
        metadata_probe=probe,
        frame_sampler=sampler,
        evaluators={"aesthetics": evaluator},
        path_authorizer=authorizer,
    )
    model_path = Path("/approved/models/aesthetics")
    request = VideoFilterRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        items=(VideoFilterItem("item-1", VideoSource(data=b"video")),),
        criteria=(
            AestheticsCriterion(
                "aesthetics",
                LocalModelReference(str(model_path)),
                FrameSamplingConfig(),
            ),
        ),
        deadline=_deadline(),
    )

    result = pipeline.evaluate(request)

    assert result.status is VideoFilterStatus.FAILED
    assert result.failures[0].code is VideoFilterFailureCode.PERMISSION_DENIED
    assert result.failures[0].criterion_id == "aesthetics"
    assert authorizer.reads == [model_path]
    assert probe.calls == []
    assert sampler.calls == []
    assert evaluator.calls == []
