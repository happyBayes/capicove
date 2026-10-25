import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from capicove.processing.video import AcceptedVideoSource, VideoSource
from capicove.processing.video.filtering._adapters.easyocr_text_area import (
    EasyOcrTextAreaEvaluator,
    TextDetections,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    OcrAreaCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    SampledFrame,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource("source-1", "a" * 64, "video/mp4", data=b"video")


def _item() -> VideoFilterItem:
    return VideoFilterItem("item-1", VideoSource(data=b"video"))


def _metadata() -> VideoFilterMetadata:
    return VideoFilterMetadata(1.0, 10, 10, 10, 10, backend="fake")


def _frames(count: int = 1) -> tuple[SampledFrame, ...]:
    return tuple(
        SampledFrame(index, float(index), 10, 10, np.zeros((10, 10, 3), dtype=np.uint8))
        for index in range(count)
    )


def test_overlapping_rectangles_are_counted_once() -> None:
    def detector(image: object) -> TextDetections:
        del image
        return (((0, 5, 0, 5), (3, 8, 0, 5)), ())

    result = EasyOcrTextAreaEvaluator(detector=detector).evaluate(
        _item(),
        _source(),
        _metadata(),
        _frames(),
        OcrAreaCriterion("ocr", FrameSamplingConfig(), max_area_ratio=0.4),
        _deadline(),
    )

    assert result.score == pytest.approx(0.4)
    assert result.passed is True


def test_quadrilateral_is_clipped_and_ratio_is_bounded() -> None:
    def detector(image: object) -> TextDetections:
        del image
        return ((), (((-5, -5), (15, -5), (15, 15), (-5, 15)),))

    result = EasyOcrTextAreaEvaluator(detector=detector).evaluate(
        _item(),
        _source(),
        _metadata(),
        _frames(),
        OcrAreaCriterion("ocr", FrameSamplingConfig(), min_area_ratio=1.0),
        _deadline(),
    )

    assert result.score == 1.0
    assert result.passed is True


def test_empty_detections_reduce_to_zero() -> None:
    result = EasyOcrTextAreaEvaluator(detector=lambda image: ((), ())).evaluate(
        _item(),
        _source(),
        _metadata(),
        _frames(2),
        OcrAreaCriterion("ocr", FrameSamplingConfig(), max_area_ratio=0.0),
        _deadline(),
    )

    assert result.score == 0.0
    assert [metric.value for metric in result.frame_metrics] == [0.0, 0.0]


def test_default_reader_never_enables_model_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = Mock(return_value=SimpleNamespace(detect=lambda image: ([[]], [[]])))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=reader))
    evaluator = EasyOcrTextAreaEvaluator()
    for _ in range(2):
        evaluator.evaluate(
            _item(),
            _source(),
            _metadata(),
            _frames(),
            OcrAreaCriterion("ocr", FrameSamplingConfig()),
            _deadline(),
        )
    assert reader.call_count == 1
    assert reader.call_args.kwargs["download_enabled"] is False


def test_missing_local_detector_has_actionable_sanitized_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = Mock(side_effect=FileNotFoundError("private/model/cache"))
    monkeypatch.setitem(sys.modules, "easyocr", SimpleNamespace(Reader=reader))
    with pytest.raises(BackendUnavailableFilterError, match="provision") as error:
        EasyOcrTextAreaEvaluator().evaluate(
            _item(),
            _source(),
            _metadata(),
            _frames(),
            OcrAreaCriterion("ocr", FrameSamplingConfig()),
            _deadline(),
        )
    assert "private/model/cache" not in str(error.value)
