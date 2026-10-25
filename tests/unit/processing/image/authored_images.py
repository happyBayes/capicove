"""Generate small deterministic image fixtures without repository source data."""

from __future__ import annotations

from functools import cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from capicove.processing.image.models import ImageSource


_SIZES = {
    "landscape": (640, 427),
    "portrait": (640, 960),
    "large": (900, 700),
}


@cache
def authored_image_bytes(kind: str = "landscape", image_format: str = "PNG") -> bytes:
    """Return deterministic encoded bytes for a synthetic geometric scene."""
    width, height = _SIZES[kind]
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (width // 10, height // 8, width // 2, height // 2),
        fill=(30, 90, 180),
    )
    draw.ellipse(
        (width // 2, height // 3, width * 9 // 10, height * 5 // 6),
        fill=(220, 80, 40),
    )
    draw.line((0, height - 1, width - 1, 0), fill="black", width=max(2, width // 100))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def authored_source(kind: str = "landscape") -> ImageSource:
    """Return one in-memory authored PNG source."""
    return ImageSource(data=authored_image_bytes(kind), media_type="image/png")


def write_authored_image(path: Path, kind: str = "landscape") -> Path:
    """Write one authored image only inside a pytest-owned temporary directory."""
    path.write_bytes(authored_image_bytes(kind))
    return path
