"""Tests for shared image preparation and compression."""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image, ImageOps

from capicove.processing.image.preparation import (
    DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES,
    ImageCompressionError,
    ImageFormatError,
    ImageInputTooLargeError,
    ImagePixelLimitError,
    prepare_image,
)


def _image_bytes(*, size: tuple[int, int], image_format: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (17, 101, 231)).save(output, format=image_format)
    return output.getvalue()


def test_prepare_image_compresses_synthetic_image_above_four_mib() -> None:
    payload = _image_bytes(size=(1_200, 1_200), image_format="BMP")
    assert len(payload) > DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES

    prepared = prepare_image(payload)

    assert prepared.was_compressed is True
    assert len(prepared.encoded_bytes) <= DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES
    assert prepared.media_type == "image/jpeg"
    assert prepared.image.size == (1_200, 1_200)
    assert prepared.original_byte_size == len(payload)
    assert prepared.processed_byte_size == len(prepared.encoded_bytes)
    assert prepared.original_sha256 == hashlib.sha256(payload).hexdigest()
    assert prepared.processed_sha256 == hashlib.sha256(prepared.encoded_bytes).hexdigest()
    with Image.open(io.BytesIO(prepared.encoded_bytes)) as encoded:
        encoded_rgb = encoded.convert("RGB")
        assert prepared.image.mode == encoded_rgb.mode
        assert prepared.image.tobytes() == encoded_rgb.tobytes()


def test_prepare_image_preserves_small_image_bytes() -> None:
    payload = _image_bytes(size=(16, 12), image_format="JPEG")

    prepared = prepare_image(payload)

    assert prepared.was_compressed is False
    assert prepared.encoded_bytes == payload
    assert prepared.media_type == "image/jpeg"
    assert prepared.image.size == (16, 12)


def test_prepare_image_rejects_input_bytes_before_decoding() -> None:
    with pytest.raises(ImageInputTooLargeError, match="8-byte input limit"):
        prepare_image(b"not-an-image", max_input_bytes=8)


def test_prepare_image_rejects_decoded_pixel_limit() -> None:
    payload = _image_bytes(size=(3, 3), image_format="PNG")

    with pytest.raises(ImagePixelLimitError, match="8-pixel limit"):
        prepare_image(payload, max_image_pixels=8)


def test_prepare_image_rejects_disallowed_content_format() -> None:
    payload = _image_bytes(size=(3, 3), image_format="BMP")

    with pytest.raises(ImageFormatError, match="BMP"):
        prepare_image(payload, allowed_formats={"JPEG", "PNG"})


def test_prepare_image_fails_closed_when_target_is_impossible() -> None:
    payload = _image_bytes(size=(16, 16), image_format="BMP")

    with pytest.raises(ImageCompressionError, match="below the 1-byte target"):
        prepare_image(payload, target_bytes=1)


def test_prepare_image_returns_decoded_compressed_alpha_pixels() -> None:
    output = io.BytesIO()
    Image.new("RGBA", (64, 64), (17, 101, 231, 127)).save(output, format="PNG")

    prepared = prepare_image(output.getvalue(), target_bytes=128)

    assert prepared.was_compressed is True
    assert prepared.media_type == "image/webp"
    assert prepared.image.mode == "RGBA"
    with Image.open(io.BytesIO(prepared.encoded_bytes)) as encoded:
        assert prepared.image.tobytes() == encoded.convert("RGBA").tobytes()


def test_prepare_image_returns_pixels_from_forced_scaled_encoding() -> None:
    size = (256, 256)
    pixels = bytes((index * 37 + (index // 3) * 17) % 256 for index in range(size[0] * size[1] * 3))
    source = Image.frombytes("RGB", size, pixels)
    output = io.BytesIO()
    source.save(output, format="BMP")

    prepared = prepare_image(output.getvalue(), target_bytes=1_000)

    assert prepared.was_compressed is True
    assert prepared.image.width < source.width
    assert prepared.image.height < source.height
    with Image.open(io.BytesIO(prepared.encoded_bytes)) as encoded:
        assert prepared.image.tobytes() == encoded.convert("RGB").tobytes()


def test_prepare_image_applies_exif_orientation_without_changing_small_source_bytes() -> None:
    source = Image.new("RGB", (3, 2), "black")
    source.putpixel((0, 0), (255, 0, 0))
    exif = source.getexif()
    exif[274] = 6
    output = io.BytesIO()
    source.save(output, format="JPEG", exif=exif)
    payload = output.getvalue()

    prepared = prepare_image(payload)

    with Image.open(io.BytesIO(payload)) as decoded:
        expected = ImageOps.exif_transpose(decoded).convert("RGB")
        assert prepared.image.size == (2, 3)
        assert prepared.image.tobytes() == expected.tobytes()
    assert prepared.image.getexif().get(274) is None
    assert prepared.encoded_bytes == payload
    assert prepared.processed_sha256 == hashlib.sha256(payload).hexdigest()
