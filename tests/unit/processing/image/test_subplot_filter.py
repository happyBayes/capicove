"""Tests for deterministic OpenCV subplot detection."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from capicove.processing.image.analyzers.opencv import (
    OpenCvSubplotAnalyzer,
    SubplotMetric,
)
from capicove.processing.image.filters import ImageSubplotFilter
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import authored_source


def _grid_source() -> ImageSource:
    image = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    for position in (50, 150, 250, 350):
        draw.line((20, position, 380, position), fill="black", width=4)
        draw.line((position, 20, position, 380), fill="black", width=4)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ImageSource(data=buffer.getvalue(), media_type="image/png")


def test_subplot_filter_rejects_regular_grid_image() -> None:
    decision = ImageSubplotFilter(
        min_horizontal_lines=3,
        min_vertical_lines=3,
        min_confidence=0.5,
        min_line_length=100,
    ).evaluate((_grid_source(),))

    assert decision.keep is False
    assert decision.metrics[0].values[0] >= 0.5
    assert dict(decision.parameters)["min_horizontal_lines"] == 3


def test_subplot_analyzer_returns_zero_evidence_for_blank_image() -> None:
    image = Image.new("RGB", (400, 400), "white")

    metric = OpenCvSubplotAnalyzer().analyze(image)

    assert metric == SubplotMetric(0.0, 0, 0)


def test_subplot_filter_processes_authored_image_deterministically() -> None:
    source = authored_source("large")
    image_filter = ImageSubplotFilter()

    first = image_filter.evaluate((source,))
    second = image_filter.evaluate((source,))

    assert first == second
    assert len(first.source_hashes) == 1
