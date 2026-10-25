"""Direct Pillow image loading with immutable source evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from capicove.processing.image.invocation import account_decoded_image, check_image_deadline
from capicove.processing.image.models import ImageSource


_MAX_DECODED_PIXELS = 100_000_000


class ImageDecodeError(ValueError):
    """Raised when an image source cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class LoadedImage:
    """Hold a normalized decoded image and its immutable source evidence."""

    image: Image.Image
    source_bytes: bytes
    sha256: str
    media_type: str
    width: int
    height: int
    byte_size: int


def load_image(source: ImageSource) -> LoadedImage:
    """Decode one local image and apply its EXIF orientation."""
    check_image_deadline()
    source_bytes = _read_source(source)
    try:
        with Image.open(BytesIO(source_bytes)) as opened:
            image_format = opened.format
            if opened.width * opened.height > _MAX_DECODED_PIXELS:
                raise ImageDecodeError("decoded image exceeds the pixel limit")
            normalized = ImageOps.exif_transpose(opened).convert("RGB").copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ImageDecodeError("unable to decode image content") from error

    account_decoded_image(normalized.width, normalized.height)
    media_type = source.media_type or Image.MIME.get(image_format or "", "application/octet-stream")
    loaded = LoadedImage(
        image=normalized,
        source_bytes=source_bytes,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        media_type=media_type,
        width=normalized.width,
        height=normalized.height,
        byte_size=len(source_bytes),
    )
    check_image_deadline()
    return loaded


def account_image_source_dimensions(source: ImageSource) -> None:
    """Charge one encoded image's declared dimensions without decoding its pixels."""
    check_image_deadline()
    source_bytes = _read_source(source)
    try:
        with Image.open(BytesIO(source_bytes)) as opened:
            width, height = opened.size
            if width * height > _MAX_DECODED_PIXELS:
                raise ImageDecodeError("decoded image exceeds the pixel limit")
    except (OSError, UnidentifiedImageError) as error:
        raise ImageDecodeError("unable to decode image content") from error
    account_decoded_image(width, height)
    check_image_deadline()


def _read_source(source: ImageSource) -> bytes:
    if source.data is not None:
        return source.data
    if source.path is None:  # pragma: no cover - protected by ImageSource validation
        raise ImageDecodeError("image source has no readable representation")
    try:
        return Path(source.path).read_bytes()
    except OSError as error:
        raise ImageDecodeError(f"unable to read image source: {source.path}") from error
