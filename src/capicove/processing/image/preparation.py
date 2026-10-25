"""Prepare image payloads for bounded transport and model processing."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Collection
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_IMAGE_INPUT_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000

_MAX_COMPRESSION_QUALITY = 95
_MIN_COMPRESSION_QUALITY = 40
_MIN_COMPRESSION_SCALE = 0.25
_COMPRESSION_SCALE_STEP = 0.8
_FORMAT_MEDIA_TYPES = {
    "BMP": "image/bmp",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImagePreparationError(ValueError):
    """Base error for invalid or unsafe image preparation requests."""


class ImageInputTooLargeError(ImagePreparationError):
    """Raised when an input exceeds the configured ingress byte limit."""


class ImagePixelLimitError(ImagePreparationError):
    """Raised when a decoded image exceeds the configured pixel limit."""


class ImageFormatError(ImagePreparationError):
    """Raised when an image format is not accepted by a transport."""


class ImageCompressionError(ImagePreparationError):
    """Raised when an image cannot be encoded within the target byte limit."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """Hold source-byte evidence and the bounded inference representation.

    Both SHA-256 fields attest byte strings: ``original_sha256`` hashes the caller
    payload and ``processed_sha256`` hashes ``encoded_bytes``. ``image`` is the
    normalized decode used for inference; after compression it is re-decoded from
    those exact encoded bytes so lossy codec changes are represented faithfully.
    """

    image: Image.Image
    encoded_bytes: bytes
    media_type: str
    original_sha256: str
    processed_sha256: str
    original_byte_size: int
    processed_byte_size: int
    width: int
    height: int
    was_compressed: bool


def prepare_image(
    payload: bytes,
    /,
    *,
    target_bytes: int = DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES,
    max_input_bytes: int = DEFAULT_MAX_IMAGE_INPUT_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    allowed_formats: Collection[str] | None = None,
    single_frame: bool = False,
) -> PreparedImage:
    """Validate, normalize, and optionally compress one image payload.

    The original bytes are never modified. If they exceed ``target_bytes``, the
    returned image is encoded as JPEG for opaque images or lossy WebP for images
    with alpha. The ingress and decoded-pixel limits are enforced before any
    potentially expensive compression work.
    """
    if not isinstance(payload, bytes):
        raise TypeError("image payload must be bytes")
    if not payload:
        raise ImagePreparationError("image must not be empty")
    if target_bytes <= 0 or max_input_bytes <= 0 or max_image_pixels <= 0:
        raise ValueError("image preparation limits must be positive")
    if len(payload) > max_input_bytes:
        raise ImageInputTooLargeError(f"image exceeds the {max_input_bytes}-byte input limit")

    normalized, image_format = _decode_image(
        payload,
        max_image_pixels=max_image_pixels,
        allowed_formats=allowed_formats,
        single_frame=single_frame,
    )
    if len(payload) <= target_bytes:
        processed_bytes = payload
        media_type = _media_type(image_format)
        processed_image = normalized
        was_compressed = False
    else:
        processed_bytes, media_type, processed_image = _compress_image(
            normalized,
            target_bytes=target_bytes,
        )
        was_compressed = True

    return PreparedImage(
        image=processed_image,
        encoded_bytes=processed_bytes,
        media_type=media_type,
        original_sha256=hashlib.sha256(payload).hexdigest(),
        processed_sha256=hashlib.sha256(processed_bytes).hexdigest(),
        original_byte_size=len(payload),
        processed_byte_size=len(processed_bytes),
        width=processed_image.width,
        height=processed_image.height,
        was_compressed=was_compressed,
    )


def _decode_image(
    payload: bytes,
    *,
    max_image_pixels: int,
    allowed_formats: Collection[str] | None,
    single_frame: bool,
) -> tuple[Image.Image, str]:
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            image_format = probe.format
            if image_format is None:
                raise ImageFormatError("image format is unknown")
            if allowed_formats is not None and image_format not in allowed_formats:
                raise ImageFormatError(f"unsupported image format: {image_format}")
            if single_frame and getattr(probe, "n_frames", 1) != 1:
                raise ImageFormatError("expected one image frame")
            width, height = probe.size
            if width * height > max_image_pixels:
                raise ImagePixelLimitError(f"image exceeds the {max_image_pixels}-pixel limit")
            probe.verify()
        with Image.open(io.BytesIO(payload)) as decoded:
            oriented = ImageOps.exif_transpose(decoded)
            has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
            mode = "RGBA" if has_alpha else "RGB"
            return oriented.convert(mode).copy(), image_format
    except ImagePreparationError:
        raise
    except Image.DecompressionBombError as error:
        raise ImagePixelLimitError(f"image exceeds the {max_image_pixels}-pixel limit") from error
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError, EOFError) as error:
        raise ImagePreparationError("unable to decode image content") from error


def _compress_image(
    image: Image.Image,
    *,
    target_bytes: int,
) -> tuple[bytes, str, Image.Image]:
    has_alpha = "A" in image.getbands()
    output_format = "WEBP" if has_alpha else "JPEG"
    media_type = _media_type(output_format)
    scale = 1.0

    while True:
        candidate = _scaled_image(image, scale)
        encoded = _best_encoding(
            candidate,
            output_format=output_format,
            target_bytes=target_bytes,
        )
        if encoded is not None:
            with Image.open(io.BytesIO(encoded)) as decoded:
                processed = decoded.convert(candidate.mode).copy()
            return encoded, media_type, processed
        if scale <= _MIN_COMPRESSION_SCALE:
            break
        next_scale = max(_MIN_COMPRESSION_SCALE, scale * _COMPRESSION_SCALE_STEP)
        if next_scale == scale:
            break
        scale = next_scale

    raise ImageCompressionError(f"unable to compress image below the {target_bytes}-byte target")


def _scaled_image(image: Image.Image, scale: float) -> Image.Image:
    if scale == 1.0:
        return image
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _best_encoding(
    image: Image.Image,
    *,
    output_format: str,
    target_bytes: int,
) -> bytes | None:
    low = _MIN_COMPRESSION_QUALITY
    high = _MAX_COMPRESSION_QUALITY
    best: bytes | None = None
    while low <= high:
        quality = (low + high) // 2
        encoded = _encode(image, output_format=output_format, quality=quality)
        if len(encoded) <= target_bytes:
            best = encoded
            low = quality + 1
        else:
            high = quality - 1
    return best


def _encode(image: Image.Image, *, output_format: str, quality: int) -> bytes:
    output = io.BytesIO()
    if output_format == "JPEG":
        image.convert("RGB").save(
            output,
            format=output_format,
            quality=quality,
            optimize=True,
            progressive=True,
        )
    else:
        image.save(
            output,
            format=output_format,
            quality=quality,
            method=6,
        )
    return output.getvalue()


def _media_type(image_format: str) -> str:
    return _FORMAT_MEDIA_TYPES.get(
        image_format,
        Image.MIME.get(image_format, "application/octet-stream"),
    )
