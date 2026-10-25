"""Parity tests for the retrieval-owned OpenCV query pHash adapter."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image, ImageDraw

from capicove.processing.image.deduplication.perceptual import OpenCvPhashProvider
from capicove.retrieval._adapters.opencv_phash import (
    OPENCV_PHASH_64_V1_SPACE,
    OPENCV_PHASH_HASH_SPACE,
    OpenCvPerceptualQueryEncoder,
    OpenCvQueryPhashAdapter,
    ResearchPhashValue,
)


def _authored_pattern(kind: str) -> Image.Image:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    if kind == "solid":
        draw.rectangle((0, 0, 63, 63), fill=(20, 80, 160))
    elif kind == "checkerboard":
        for y in range(0, 64, 8):
            for x in range(0, 64, 8):
                if (x // 8 + y // 8) % 2:
                    draw.rectangle((x, y, x + 7, y + 7), fill="black")
    elif kind == "stripes":
        for x in range(0, 64, 8):
            if (x // 8) % 2:
                draw.rectangle((x, 0, x + 7, 63), fill=(220, 40, 40))
    elif kind == "diagonal":
        draw.line((0, 63, 63, 0), fill="black", width=5)
        draw.ellipse((16, 16, 40, 40), fill=(40, 160, 80))
    else:  # pragma: no cover - protected by parametrization
        raise AssertionError(f"unknown authored pattern: {kind}")
    return image


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("solid", 0x8000000000000000),
        ("checkerboard", 0x8055005500550055),
        ("stripes", 0xD500000000000000),
        ("diagonal", 0xB070CFDE34353127),
    ],
)
def test_query_phash_matches_golden_and_processing_provider(kind: str, expected: int) -> None:
    image = _authored_pattern(kind)
    query_provider = OpenCvQueryPhashAdapter()
    processing_provider = OpenCvPhashProvider()

    query_hash = query_provider.phash(image)

    assert query_provider.space == OPENCV_PHASH_64_V1_SPACE
    assert query_provider.provider_id == processing_provider.provider_id
    assert query_provider.provider_version == processing_provider.provider_version
    assert query_provider.hash_version == "phash-64-v1"
    assert isinstance(query_hash, ResearchPhashValue)
    assert query_hash.space == OPENCV_PHASH_64_V1_SPACE
    assert query_hash.value == expected
    assert query_hash.value == processing_provider.phash(image)
    assert 0 <= query_hash.value < 2**64


def test_query_phash_rejects_non_image_input() -> None:
    with pytest.raises(TypeError, match="PIL Image"):
        OpenCvQueryPhashAdapter().phash(object())  # type: ignore[arg-type]


def test_query_phash_rejects_oversized_image_before_full_raster_conversion() -> None:
    oversized = Image.new("1", (40_000_001, 1))

    with pytest.raises(ValueError, match="40,000,000-pixel"):
        OpenCvQueryPhashAdapter().phash(oversized)


def test_hash_space_identity_rejects_non_integer_width() -> None:
    with pytest.raises(ValueError, match="64-bit width"):
        replace(OPENCV_PHASH_64_V1_SPACE, bit_width=64.0)  # type: ignore[arg-type]


def test_production_query_encoder_matches_processing_provider() -> None:
    image = _authored_pattern("diagonal")
    encoder = OpenCvPerceptualQueryEncoder(OPENCV_PHASH_HASH_SPACE)

    query = encoder.encode(
        image,
        source_sha256="a" * 64,
        max_hamming_distance=2,
        deadline=datetime.now(UTC) + timedelta(minutes=1),
    )

    assert query.space == OPENCV_PHASH_HASH_SPACE
    assert query.value == OpenCvPhashProvider().phash(image)
    assert query.max_hamming_distance == 2


def test_production_query_encoder_honors_expired_deadline_before_work() -> None:
    encoder = OpenCvPerceptualQueryEncoder(OPENCV_PHASH_HASH_SPACE)

    with pytest.raises(TimeoutError, match="deadline"):
        encoder.encode(
            _authored_pattern("solid"),
            source_sha256="a" * 64,
            max_hamming_distance=2,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )
