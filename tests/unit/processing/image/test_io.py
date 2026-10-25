"""Tests for direct Pillow image loading and evidence extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.processing.image.io import ImageDecodeError, load_image
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import (
    authored_image_bytes,
    write_authored_image,
)


def test_load_image_records_dimensions_bytes_and_hash(tmp_path: Path) -> None:
    path = write_authored_image(tmp_path / "landscape.png")
    loaded = load_image(ImageSource(path=str(path)))

    assert (loaded.width, loaded.height) == (640, 427)
    assert loaded.byte_size > 0
    assert len(loaded.sha256) == 64
    assert loaded.image.mode == "RGB"


def test_load_image_accepts_bytes_without_changing_source_bytes() -> None:
    data = authored_image_bytes("portrait")

    loaded = load_image(ImageSource(data=data, media_type="image/png"))

    assert (loaded.width, loaded.height) == (640, 960)
    assert loaded.source_bytes == data


def test_load_image_rejects_invalid_content() -> None:
    with pytest.raises(ImageDecodeError, match="decode"):
        load_image(ImageSource(data=b"not an image"))
