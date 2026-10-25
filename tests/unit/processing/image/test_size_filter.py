"""Tests for encoded image-size filtering."""

from __future__ import annotations

import pytest

from capicove.processing.image.filters import ImageSizeFilter
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import authored_image_bytes


def test_size_filter_measures_encoded_source_bytes() -> None:
    content = authored_image_bytes("landscape")
    size = len(content)

    decision = ImageSizeFilter(min_size=size, max_size=size).evaluate(
        (ImageSource(data=content, media_type="image/png"),)
    )

    assert decision.keep is True
    assert decision.metrics[0].values == (float(size),)
    assert dict(decision.parameters)["min_size_bytes"] == size


def test_size_filter_parses_binary_units() -> None:
    source = ImageSource(data=b"not decoded", media_type="image/jpeg")

    decision = ImageSizeFilter(min_size="1KiB", max_size="2KiB").evaluate_sizes((source,), (1536,))

    assert decision.keep is True


def test_size_filter_rejects_unknown_units() -> None:
    with pytest.raises(ValueError, match="size"):
        ImageSizeFilter(max_size="1XB")
