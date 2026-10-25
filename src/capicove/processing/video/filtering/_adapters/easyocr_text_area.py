"""EasyOCR-compatible text-area evaluator for sampled video frames."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TypeAlias

from capicove.processing.video.filtering.video_filter_contracts import (
    FrameMetric,
    OcrAreaCriterion,
    VideoCriterionResult,
    VideoFilterCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    DecodeFailedFilterError,
    InvalidVideoFilterInputError,
    SampledFrame,
    VideoFilterTimeoutError,
)
from capicove.processing.video.filtering.video_filter_rules import in_range, reduce_values
from capicove.processing.video.models import AcceptedVideoSource


Rectangle: TypeAlias = tuple[float, float, float, float]
Point: TypeAlias = tuple[float, float]
Quadrilateral: TypeAlias = tuple[Point, Point, Point, Point]
TextDetections: TypeAlias = tuple[tuple[Rectangle, ...], tuple[Quadrilateral, ...]]
TextDetector: TypeAlias = Callable[[object], TextDetections]


class EasyOcrTextAreaEvaluator:
    """Calculate bounded text-area union ratios without recognizing text."""

    def __init__(
        self, *, detector: TextDetector | None = None, backend_label: str = "easyocr-detector"
    ) -> None:
        self._detector = detector
        self._backend_label = backend_label
        self._reader_cache: dict[tuple[str, ...], object] = {}

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        """Evaluate OCR area for one accepted source."""
        del source, metadata
        if not isinstance(criterion, OcrAreaCriterion):
            raise InvalidVideoFilterInputError("OCR area requires an OCR-area criterion.")
        if not frames:
            raise DecodeFailedFilterError("OCR area scoring requires at least one frame.")
        detector = self._detector or self._easyocr_detector(criterion.languages)
        metrics: list[FrameMetric] = []
        for frame in frames:
            if datetime.now(UTC) >= deadline:
                raise VideoFilterTimeoutError("The OCR-area deadline expired.")
            rectangles, quadrilaterals = detector(frame.payload)
            ratio = _area_ratio(
                frame.width,
                frame.height,
                rectangles,
                quadrilaterals,
            )
            metrics.append(
                FrameMetric(
                    frame_index=frame.index,
                    timestamp_seconds=frame.timestamp_seconds,
                    value=ratio,
                )
            )
        score = reduce_values(tuple(metric.value for metric in metrics), criterion.reduction)
        return VideoCriterionResult(
            item_id=item.item_id,
            criterion_id=criterion.criterion_id,
            passed=in_range(score, criterion.min_area_ratio, criterion.max_area_ratio),
            score=score,
            frame_metrics=tuple(metrics),
            details={"languages": criterion.languages, "backend": self._backend_label},
        )

    def _easyocr_detector(self, languages: tuple[str, ...]) -> TextDetector:
        try:
            easyocr = importlib.import_module("easyocr")
        except ModuleNotFoundError as error:
            raise BackendUnavailableFilterError(
                "EasyOCR is unavailable; install capicove[video-filter-ocr]."
            ) from error
        reader = self._reader_cache.get(languages)
        if reader is None:
            try:
                reader = easyocr.Reader(
                    lang_list=list(languages),
                    recognizer=False,
                    verbose=False,
                    gpu=False,
                    download_enabled=False,
                )
            except (OSError, RuntimeError) as error:
                raise BackendUnavailableFilterError(
                    "EasyOCR could not load its local detector; provision verified model "
                    "files in the EasyOCR model cache before running this criterion."
                ) from error
            self._reader_cache[languages] = reader

        def detect(image: object) -> TextDetections:
            horizontal, free_form = reader.detect(image)  # type: ignore[attr-defined]
            rectangles = tuple(tuple(float(value) for value in box) for box in horizontal[0])
            quadrilaterals = tuple(
                tuple((float(point[0]), float(point[1])) for point in points)
                for points in free_form[0]
            )
            return rectangles, quadrilaterals  # type: ignore[return-value]

        return detect


def _area_ratio(
    width: int,
    height: int,
    rectangles: tuple[Rectangle, ...],
    quadrilaterals: tuple[Quadrilateral, ...],
) -> float:
    if width <= 0 or height <= 0:
        raise DecodeFailedFilterError("OCR frames require positive dimensions.")
    try:
        cv2 = importlib.import_module("cv2")
        np = importlib.import_module("numpy")
    except ModuleNotFoundError as error:
        raise BackendUnavailableFilterError(
            "OpenCV and NumPy are required; install capicove[video-filter-ocr]."
        ) from error
    mask = np.zeros((height, width), dtype=np.uint8)
    for xmin, xmax, ymin, ymax in rectangles:
        x1 = max(0, min(width, math.floor(xmin)))
        x2 = max(0, min(width, math.ceil(xmax)))
        y1 = max(0, min(height, math.floor(ymin)))
        y2 = max(0, min(height, math.ceil(ymax)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    for points in quadrilaterals:
        clipped = np.array(
            [
                [
                    max(0, min(width - 1, round(point[0]))),
                    max(0, min(height - 1, round(point[1]))),
                ]
                for point in points
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [clipped], 1)
    return min(1.0, max(0.0, float(mask.sum()) / (width * height)))
