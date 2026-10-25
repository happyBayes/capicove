"""Tests for local pHash visual near-duplicate grouping."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from capicove.processing.image.deduplication import (
    ImagePerceptualDeduplicator,
    ImageRecord,
)
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import authored_image_bytes, authored_source


class WidthHasher:
    """Use image width as a deterministic synthetic 64-bit hash."""

    provider_id = "width-test-hasher"
    provider_version = "1"
    hash_version = "test"

    def phash(self, image: Image.Image) -> int:
        return image.width


def _source(width: int) -> ImageSource:
    buffer = BytesIO()
    Image.new("RGB", (width, 8), "white").save(buffer, format="PNG")
    return ImageSource(data=buffer.getvalue(), media_type="image/png")


def test_perceptual_deduplicator_groups_same_visual_content() -> None:
    same = authored_source("landscape")
    different = authored_source("portrait")
    records = (
        ImageRecord("b", (same,)),
        ImageRecord("a", (same,)),
        ImageRecord("c", (different,)),
    )

    result = ImagePerceptualDeduplicator().group(records)

    assert len(result.groups) == 1
    assert result.groups[0].kept_record_id == "a"
    assert result.groups[0].duplicate_record_ids == ("b",)
    assert dict(result.record_source_hashes)["a"]


def test_perceptual_deduplicator_uses_connected_components_for_near_duplicates() -> None:
    records = (
        ImageRecord("c", (_source(7),)),
        ImageRecord("a", (_source(1),)),
        ImageRecord("b", (_source(3),)),
    )

    result = ImagePerceptualDeduplicator(
        max_hamming_distance=1,
        hasher=WidthHasher(),
    ).group(records)

    assert result.groups[0].kept_record_id == "a"
    assert result.groups[0].duplicate_record_ids == ("b", "c")


def test_perceptual_deduplicator_keeps_empty_image_records() -> None:
    result = ImagePerceptualDeduplicator().group((ImageRecord("empty", ()),))

    assert result.groups == ()
    assert result.retained_record_ids == ("empty",)


def test_perceptual_deduplicator_groups_resized_reencoded_visual_content() -> None:
    original_bytes = authored_image_bytes("landscape")
    with Image.open(BytesIO(original_bytes)) as opened:
        resized = opened.convert("RGB").resize((320, 214))
    buffer = BytesIO()
    resized.save(buffer, format="JPEG", quality=85)
    records = (
        ImageRecord("original", (ImageSource(data=original_bytes, media_type="image/png"),)),
        ImageRecord(
            "transformed",
            (ImageSource(data=buffer.getvalue(), media_type="image/jpeg"),),
        ),
    )

    result = ImagePerceptualDeduplicator(max_hamming_distance=2).group(records)

    assert result.groups[0].kept_record_id == "original"
    assert result.groups[0].duplicate_record_ids == ("transformed",)
