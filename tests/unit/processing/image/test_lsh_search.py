"""Tests for deterministic binary LSH candidate search."""

from __future__ import annotations

import pytest

from capicove.processing.image.deduplication.lsh import (
    HashRecord,
    ImageLshSearch,
    candidate_pairs,
)


def test_lsh_search_returns_exact_matches_in_stable_order() -> None:
    records = (
        HashRecord("far", (0xFFFFFFFFFFFFFFFF,)),
        HashRecord("z-near", (0x0000000000000001,)),
        HashRecord("a-near", (0x0000000000000002,)),
    )

    result = ImageLshSearch().search(
        (0,),
        records,
        hash_version="phash-64-v1",
        max_hamming_distance=1,
        top_k=2,
    )

    assert [(match.record_id, match.distance) for match in result.matches] == [
        ("a-near", 1),
        ("z-near", 1),
    ]
    assert result.hash_version == "phash-64-v1"
    assert result.bit_width == 64
    assert result.candidate_strategy == "lsh"
    assert result.candidate_count == 2
    assert result.comparison_count == 2


def test_lsh_search_exactly_verifies_bucket_collisions() -> None:
    records = (
        HashRecord("inside", (0x0000000000000001,)),
        HashRecord("collision", (0x0000000000000003,)),
        HashRecord("outside-buckets", (0xFFFFFFFFFFFFFFFF,)),
    )

    result = ImageLshSearch().search(
        (0,),
        records,
        hash_version="phash-64-v1",
        max_hamming_distance=1,
    )

    assert [(match.record_id, match.distance) for match in result.matches] == [
        ("inside", 1),
    ]
    assert result.candidate_count == 2
    assert result.comparison_count == 2


def test_lsh_search_recalls_multi_hash_neighbor_across_changed_bands() -> None:
    records = (
        HashRecord("neighbor", (0x0000000000000001, 0x0000000000000001)),
        HashRecord("far", (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF)),
    )

    result = ImageLshSearch().search(
        (0, 0),
        records,
        hash_version="phash-64-v1",
        max_hamming_distance=2,
    )

    assert [(match.record_id, match.distance) for match in result.matches] == [
        ("neighbor", 2),
    ]
    assert result.bit_width == 128


def test_lsh_search_excludes_incompatible_text_hashes_before_probing() -> None:
    records = (
        HashRecord("same-text", (0,), "a" * 64),
        HashRecord("other-text", (0,), "b" * 64),
        HashRecord("no-text", (0,)),
    )

    result = ImageLshSearch().search(
        (0,),
        records,
        hash_version="phash-64-v1",
        query_text_hash="a" * 64,
    )

    assert [match.record_id for match in result.matches] == ["same-text"]
    assert result.candidate_count == 1


def test_lsh_search_uses_exhaustive_fallback_for_full_radius() -> None:
    records = (
        HashRecord("all-ones", (0xFFFFFFFFFFFFFFFF,)),
        HashRecord("zero", (0,)),
    )

    result = ImageLshSearch().search(
        (0,),
        records,
        hash_version="phash-64-v1",
        max_hamming_distance=64,
        top_k=1,
    )

    assert [(match.record_id, match.distance) for match in result.matches] == [
        ("zero", 0),
    ]
    assert result.candidate_strategy == "exhaustive"
    assert result.candidate_count == 2
    assert result.comparison_count == 2


def test_candidate_pairs_are_unique_and_recall_transitive_edges() -> None:
    records = (
        HashRecord("c", (0b111,)),
        HashRecord("a", (0b001,)),
        HashRecord("b", (0b011,)),
        HashRecord("far", (0xFFFFFFFFFFFFFFFF,)),
    )

    pairs = candidate_pairs(records, max_hamming_distance=1)

    assert pairs.strategy == "lsh"
    assert pairs.pairs == (("a", "b"), ("b", "c"))
    assert pairs.candidate_count >= 2


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ((HashRecord("same", (0,)), HashRecord("same", (1,))), "unique"),
        ((HashRecord("bad", (-1,)),), "unsigned 64-bit"),
        ((HashRecord("bad", (2**64,)),), "unsigned 64-bit"),
        ((HashRecord("bad", (True,)),), "unsigned 64-bit"),
        ((HashRecord("bad", (0,), "not-a-sha"),), "text_hash"),
    ],
)
def test_lsh_search_rejects_invalid_candidate_records(
    records: tuple[HashRecord, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageLshSearch().search((0,), records, hash_version="phash-64-v1")


@pytest.mark.parametrize(
    ("query", "hash_version", "max_distance", "top_k", "message"),
    [
        ((), "phash-64-v1", 0, 100, "query_hashes"),
        ((0,), "", 0, 100, "hash_version"),
        ((0,), "phash-64-v1", -1, 100, "max_hamming_distance"),
        ((0,), "phash-64-v1", 65, 100, "max_hamming_distance"),
        ((0,), "phash-64-v1", 0, 0, "top_k"),
        ((0,), "phash-64-v1", 0, 1001, "top_k"),
    ],
)
def test_lsh_search_rejects_invalid_query_options(
    query: tuple[int, ...],
    hash_version: str,
    max_distance: int,
    top_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ImageLshSearch().search(
            query,
            (),
            hash_version=hash_version,
            max_hamming_distance=max_distance,
            top_k=top_k,
        )
