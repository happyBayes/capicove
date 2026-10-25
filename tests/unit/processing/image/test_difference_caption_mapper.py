"""Tests for structured difference-caption generation."""

from __future__ import annotations

from io import BytesIO
from typing import cast

from PIL import Image, ImageDraw

from capicove.processing.image.mappers import (
    ImgDiffDifferenceAreaGeneratorMapper,
    ImgDiffDifferenceCaptionGeneratorMapper,
)
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.fakes import FixedImageTextScorer


class StructuredCaptionProvider:
    """Return structured deterministic captions for cropped pairs."""

    provider_id = "structured-test-captioner"
    provider_version = "1"
    model_id = "authored-test-model"
    model_revision = "1"
    model_manifest_sha256 = "b" * 64
    preprocess_revision = "authored-caption-crops-v1"
    device = "cpu"

    def __init__(self) -> None:
        self.crop_sizes: list[tuple[int, int]] = []

    def caption(self, image: Image.Image) -> str:
        self.crop_sizes.append(image.size)
        maximum = cast(int, image.convert("L").getextrema()[1])
        return "bright region" if maximum > 0 else "dark region"

    def describe_difference(
        self,
        left: Image.Image,
        right: Image.Image,
        left_caption: str,
        right_caption: str,
    ) -> str:
        del left, right
        return f"{left_caption} changed to {right_caption}"


class IdenticalCaptionProvider(StructuredCaptionProvider):
    """Return the same caption for both sides of a candidate region."""

    def caption(self, image: Image.Image) -> str:
        self.crop_sizes.append(image.size)
        return "the same object"


def _png_source(image: Image.Image) -> ImageSource:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ImageSource(data=buffer.getvalue(), media_type="image/png")


def test_difference_caption_mapper_crops_regions_and_returns_structured_text() -> None:
    left = Image.new("RGB", (160, 100), "white")
    right = left.copy()
    ImageDraw.Draw(right).rectangle((40, 20, 99, 69), fill="black")
    sources = (_png_source(left), _png_source(right))
    areas = ImgDiffDifferenceAreaGeneratorMapper(pixel_threshold=10, min_area_ratio=0.01).map(
        sources
    )
    provider = StructuredCaptionProvider()

    result = ImgDiffDifferenceCaptionGeneratorMapper(
        provider,
        validator=FixedImageTextScorer((1.0,)),
    ).map(sources, areas)

    assert len(result.regions) == 1
    assert result.regions[0].difference_caption == "bright region changed to dark region"
    assert result.regions[0].left_match_score == 1.0
    assert result.regions[0].right_match_score == 1.0
    assert result.regions[0].caption_difference_score > 0
    assert provider.crop_sizes[0] == provider.crop_sizes[1]
    assert result.preprocess_revision == "authored-caption-crops-v1"
    assert result.device == "cpu"
    assert dict(result.parameters)["prompt_revision"] == "difference-caption-v2-multi-image"


def test_difference_caption_mapper_skips_provider_for_empty_areas() -> None:
    source = _png_source(Image.new("RGB", (20, 20), "white"))
    sources = (source, source)
    areas = ImgDiffDifferenceAreaGeneratorMapper().map(sources)
    provider = StructuredCaptionProvider()

    result = ImgDiffDifferenceCaptionGeneratorMapper(
        provider,
        validator=FixedImageTextScorer((1.0,)),
    ).map(sources, areas)

    assert result.regions == ()
    assert provider.crop_sizes == []


def test_difference_caption_mapper_rejects_regions_with_identical_captions() -> None:
    left = Image.new("RGB", (160, 100), "white")
    right = left.copy()
    ImageDraw.Draw(right).rectangle((40, 20, 99, 69), fill="black")
    sources = (_png_source(left), _png_source(right))
    areas = ImgDiffDifferenceAreaGeneratorMapper(pixel_threshold=10, min_area_ratio=0.01).map(
        sources
    )

    result = ImgDiffDifferenceCaptionGeneratorMapper(
        IdenticalCaptionProvider(),
        validator=FixedImageTextScorer((1.0,)),
    ).map(
        sources,
        areas,
    )

    assert result.regions == ()


def test_difference_caption_mapper_rejects_caption_that_does_not_match_crop() -> None:
    left = Image.new("RGB", (160, 100), "white")
    right = left.copy()
    ImageDraw.Draw(right).rectangle((40, 20, 99, 69), fill="black")
    sources = (_png_source(left), _png_source(right))
    areas = ImgDiffDifferenceAreaGeneratorMapper(pixel_threshold=10, min_area_ratio=0.01).map(
        sources
    )

    result = ImgDiffDifferenceCaptionGeneratorMapper(
        StructuredCaptionProvider(),
        validator=FixedImageTextScorer((0.1,)),
        min_caption_match=0.5,
    ).map(sources, areas)

    assert result.regions == ()
