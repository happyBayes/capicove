from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from capicove.processing.video import AcceptedVideoSource, VideoSource
from capicove.processing.video.filtering._adapters.opencv_farneback_motion import (
    OpenCvFarnebackMotionEvaluator,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    MotionCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    DecodeFailedFilterError,
    SampledFrame,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource("source-1", "a" * 64, "video/mp4", data=b"video")


def _metadata() -> VideoFilterMetadata:
    return VideoFilterMetadata(1.0, 32, 32, 32, 32, backend="fake")


def _frame(index: int, image: np.ndarray) -> SampledFrame:
    return SampledFrame(index, index / 2, 32, 32, image)


def test_static_frames_have_near_zero_motion() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    frames = (_frame(0, image.copy()), _frame(1, image.copy()))

    result = OpenCvFarnebackMotionEvaluator().evaluate(
        VideoFilterItem("item-1", source=_video_source()),
        _source(),
        _metadata(),
        frames,
        MotionCriterion("motion", FrameSamplingConfig(), max_score=0.01),
        _deadline(),
    )

    assert result.score == pytest.approx(0.0, abs=1e-6)
    assert result.passed is True
    assert result.frame_metrics[0].timestamp_seconds == 0.5


def test_translation_has_positive_motion_and_relative_normalization() -> None:
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    second = np.zeros_like(first)
    first[8:20, 4:16] = 255
    second[8:20, 8:20] = 255
    frames = (_frame(0, first), _frame(1, second))
    evaluator = OpenCvFarnebackMotionEvaluator()

    absolute = evaluator.evaluate(
        VideoFilterItem("item-1", source=_video_source()),
        _source(),
        _metadata(),
        frames,
        MotionCriterion("motion", FrameSamplingConfig(), min_score=0.01),
        _deadline(),
    )
    relative = evaluator.evaluate(
        VideoFilterItem("item-1", source=_video_source()),
        _source(),
        _metadata(),
        frames,
        MotionCriterion(
            "motion-relative",
            FrameSamplingConfig(),
            min_score=0.0,
            relative=True,
        ),
        _deadline(),
    )

    assert absolute.score is not None and absolute.score > 0.01
    assert relative.score == pytest.approx(absolute.score / np.hypot(32, 32))
    assert absolute.frame_metrics[0].raw_value == absolute.score


def test_motion_requires_two_frames() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    with pytest.raises(DecodeFailedFilterError, match="two"):
        OpenCvFarnebackMotionEvaluator().evaluate(
            VideoFilterItem("item-1", source=_video_source()),
            _source(),
            _metadata(),
            (_frame(0, image),),
            MotionCriterion("motion", FrameSamplingConfig()),
            _deadline(),
        )


def _video_source() -> VideoSource:
    return VideoSource(data=b"video")
