"""Tests for deterministic image-pair difference-area generation."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from capicove.processing.image.mappers import ImgDiffDifferenceAreaGeneratorMapper
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import authored_source


def _png_source(image: Image.Image) -> ImageSource:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ImageSource(data=buffer.getvalue(), media_type="image/png")


def test_difference_area_mapper_finds_changed_rectangle() -> None:
    left = Image.new("RGB", (200, 120), "white")
    right = left.copy()
    ImageDraw.Draw(right).rectangle((50, 30, 109, 79), fill="black")

    result = ImgDiffDifferenceAreaGeneratorMapper(
        pixel_threshold=10,
        min_area_ratio=0.01,
    ).map((_png_source(left), _png_source(right)))

    assert len(result.boxes) == 1
    box = result.boxes[0]
    assert 45 <= box.x <= 55
    assert 25 <= box.y <= 35
    assert box.width >= 55
    assert box.height >= 45
    assert result.operation_version == 1
    assert dict(result.parameters)["pixel_threshold"] == 10
    assert tuple(stage.stage for stage in result.stage_evidence) == (
        "pair_pre_gate",
        "candidate_proposal",
        "crop_dissimilarity",
        "non_maximum_suppression",
    )


def test_difference_area_mapper_returns_empty_for_identical_content() -> None:
    source = authored_source("landscape")

    result = ImgDiffDifferenceAreaGeneratorMapper().map((source, source))

    assert result.boxes == ()
    assert result.source_hashes[0] == result.source_hashes[1]


def test_difference_area_mapper_requires_exactly_two_images() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        ImgDiffDifferenceAreaGeneratorMapper().map(())


def test_difference_area_mapper_detects_equal_luminance_color_change() -> None:
    left = Image.new("RGB", (100, 100), (255, 0, 0))
    right = Image.new("RGB", (100, 100), (0, 130, 0))

    result = ImgDiffDifferenceAreaGeneratorMapper(
        pixel_threshold=10,
        min_area_ratio=0.01,
    ).map((_png_source(left), _png_source(right)))

    assert len(result.boxes) == 1


def test_difference_area_mapper_rejects_unaligned_dimensions() -> None:
    left = _png_source(Image.new("RGB", (100, 100), "white"))
    right = _png_source(Image.new("RGB", (200, 100), "white"))

    with pytest.raises(ValueError, match="dimensions"):
        ImgDiffDifferenceAreaGeneratorMapper().map((left, right))
