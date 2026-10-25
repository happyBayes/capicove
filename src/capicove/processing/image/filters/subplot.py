"""Filter images containing grid-like subplot layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from capicove.processing.image.analyzers.opencv import OpenCvSubplotAnalyzer
from capicove.processing.image.invocation import check_image_deadline
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    Aggregation,
    FilterDecision,
    ImageMetric,
    ImageSource,
)


@dataclass(frozen=True, slots=True)
class ImageSubplotFilter:
    """Reject images whose straight-line evidence forms a subplot grid."""

    min_horizontal_lines: int = 3
    min_vertical_lines: int = 3
    min_confidence: float = 0.5
    aggregation: Aggregation = Aggregation.ANY
    canny_threshold1: int = 70
    canny_threshold2: int = 190
    hough_threshold: int = 110
    min_line_length: int = 110
    max_line_gap: int = 18
    angle_tolerance: float = 4.0

    def __post_init__(self) -> None:
        if self.min_horizontal_lines <= 0 or self.min_vertical_lines <= 0:
            raise ValueError("minimum subplot line counts must be positive")
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("minimum subplot confidence must be within [0, 1]")

    def evaluate(self, sources: tuple[ImageSource, ...]) -> FilterDecision:
        """Analyze ordered image sources and reject configured subplot collections."""
        aggregation = Aggregation(self.aggregation)
        parameters = (
            ("aggregation", aggregation.value),
            ("angle_tolerance", self.angle_tolerance),
            ("canny_threshold1", self.canny_threshold1),
            ("canny_threshold2", self.canny_threshold2),
            ("hough_threshold", self.hough_threshold),
            ("max_line_gap", self.max_line_gap),
            ("min_confidence", self.min_confidence),
            ("min_horizontal_lines", self.min_horizontal_lines),
            ("min_line_length", self.min_line_length),
            ("min_vertical_lines", self.min_vertical_lines),
        )
        if not sources:
            return FilterDecision(
                "image_subplot_filter",
                True,
                False,
                aggregation,
                (),
                (),
                parameters=parameters,
            )
        loaded = tuple(load_image(source) for source in sources)
        analyzer = OpenCvSubplotAnalyzer(
            min_horizontal_lines=self.min_horizontal_lines,
            min_vertical_lines=self.min_vertical_lines,
            canny_threshold1=self.canny_threshold1,
            canny_threshold2=self.canny_threshold2,
            hough_threshold=self.hough_threshold,
            min_line_length=self.min_line_length,
            max_line_gap=self.max_line_gap,
            angle_tolerance=self.angle_tolerance,
        )
        results_list = []
        for item in loaded:
            check_image_deadline()
            results_list.append(analyzer.analyze(item.image))
            check_image_deadline()
        results = tuple(results_list)
        flags = tuple(
            result.confidence >= self.min_confidence
            and result.horizontal_lines >= self.min_horizontal_lines
            and result.vertical_lines >= self.min_vertical_lines
            for result in results
        )
        contains_subplot = any(flags) if aggregation is Aggregation.ANY else all(flags)
        return FilterDecision(
            filter_name="image_subplot_filter",
            keep=not contains_subplot,
            applicable=True,
            aggregation=aggregation,
            source_hashes=tuple(item.sha256 for item in loaded),
            metrics=(
                ImageMetric(
                    "image_subplot_confidence",
                    tuple(result.confidence for result in results),
                    "opencv-hough-lines",
                    cv2_version(),
                ),
                ImageMetric(
                    "horizontal_line_count",
                    tuple(float(result.horizontal_lines) for result in results),
                    "opencv-hough-lines",
                    cv2_version(),
                ),
                ImageMetric(
                    "vertical_line_count",
                    tuple(float(result.vertical_lines) for result in results),
                    "opencv-hough-lines",
                    cv2_version(),
                ),
            ),
            parameters=parameters,
        )


def cv2_version() -> str:
    """Return the directly loaded OpenCV version for metric lineage."""
    import cv2

    return cv2.__version__
