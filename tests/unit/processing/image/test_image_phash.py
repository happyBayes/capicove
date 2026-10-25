"""Tests for public batch pHash production and optional LSH grouping."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image

from capicove.processing.image.deduplication import ImagePhash, ImageRecord
from capicove.processing.image.models import ImageSource


class WidthHasher:
    """Use image width as an independently predictable synthetic hash."""

    provider_id = "width-test-hasher"
    provider_version = "1"
    hash_version = "test-phash-64-v1"

    def phash(self, image: Image.Image) -> int:
        return image.width


class SparseWidthHasher(WidthHasher):
    """Map one fixture width to a hash outside both radius-one buckets."""

    def phash(self, image: Image.Image) -> int:
        if image.width == 64:
            return 0xFFFFFFFFFFFFFFFF
        return super().phash(image)


def _source(width: int, color: str = "white") -> ImageSource:
    buffer = BytesIO()
    Image.new("RGB", (width, 8), color).save(buffer, format="PNG")
    return ImageSource(data=buffer.getvalue(), media_type="image/png")


def test_image_phash_returns_fixed_width_hashes_and_lineage() -> None:
    source = _source(15)

    result = ImagePhash(hasher=WidthHasher()).process(
        (ImageRecord("record-1", (source,), "caption"),)
    )

    assert result.records[0].record_id == "record-1"
    assert result.records[0].hashes == ("000000000000000f",)
    assert result.records[0].source_hashes == (hashlib.sha256(source.data or b"").hexdigest(),)
    assert result.records[0].text_hash == hashlib.sha256(b"caption").hexdigest()
    assert result.provider_id == "width-test-hasher"
    assert result.provider_version == "1"
    assert result.hash_version == "test-phash-64-v1"
    assert result.bit_width == 64
    assert result.max_hamming_distance == 0
    assert result.candidate_strategy == "exhaustive"
    assert result.candidate_count == 0
    assert result.comparison_count == 0


def test_image_phash_groups_batch_records_and_keeps_empty_records() -> None:
    same = _source(9)
    records = (
        ImageRecord("b", (same,)),
        ImageRecord("empty", ()),
        ImageRecord("a", (same,)),
    )

    result = ImagePhash(hasher=WidthHasher()).process(records)

    assert result.groups[0].kept_record_id == "a"
    assert result.groups[0].duplicate_record_ids == ("b",)
    assert result.retained_record_ids == ("a", "empty")
    assert result.records[1].hashes == ()
    assert result.candidate_count == 1
    assert result.comparison_count == 1


def test_image_phash_lsh_matches_exhaustive_connected_components() -> None:
    records = (
        ImageRecord("c", (_source(7),)),
        ImageRecord("a", (_source(1),)),
        ImageRecord("unrelated", (_source(64),)),
        ImageRecord("b", (_source(3),)),
    )

    exhaustive = ImagePhash(
        max_hamming_distance=1,
        hasher=SparseWidthHasher(),
    ).process(records)
    accelerated = ImagePhash(
        max_hamming_distance=1,
        hasher=SparseWidthHasher(),
        enable_lsh=True,
    ).process(records)

    assert accelerated.groups == exhaustive.groups
    assert accelerated.retained_record_ids == exhaustive.retained_record_ids
    assert accelerated.candidate_strategy == "lsh"
    assert accelerated.comparison_count < exhaustive.comparison_count


def test_image_phash_lsh_preserves_multi_image_and_text_semantics() -> None:
    records = (
        ImageRecord("a", (_source(1), _source(2)), "same"),
        ImageRecord("b", (_source(3), _source(2)), "same"),
        ImageRecord("other-text", (_source(3), _source(2)), "different"),
        ImageRecord("other-count", (_source(3),), "same"),
    )

    exhaustive = ImagePhash(
        max_hamming_distance=1,
        hasher=WidthHasher(),
    ).process(records)
    accelerated = ImagePhash(
        max_hamming_distance=1,
        hasher=WidthHasher(),
        enable_lsh=True,
    ).process(records)

    assert accelerated.groups == exhaustive.groups
    assert accelerated.groups[0].kept_record_id == "a"
    assert accelerated.groups[0].duplicate_record_ids == ("b",)
    assert accelerated.retained_record_ids == ("a", "other-count", "other-text")


def test_image_phash_rejects_boolean_hamming_distance() -> None:
    with pytest.raises(ValueError, match="max_hamming_distance"):
        ImagePhash(max_hamming_distance=True)

    with pytest.raises(ValueError, match="max_hamming_distance"):
        ImagePhash(hasher=WidthHasher()).process(
            (ImageRecord("record", (_source(1),)),),
            max_hamming_distance=False,
        )
