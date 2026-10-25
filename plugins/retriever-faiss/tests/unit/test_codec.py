"""Tests for provider-owned ordinal and manifest codecs."""

from capicove_retriever_faiss.codec import (
    decode_visual_manifest,
    encode_ordinal,
    encode_visual_manifest,
    validate_ordinals,
)
from helpers import entry, manifest


def test_manifest_codec_round_trips_public_contract_identity() -> None:
    entries = (entry("unit-a", (1.0, 0.0)),)
    original = manifest(entries)

    decoded = decode_visual_manifest(encode_visual_manifest(original))

    assert decoded == original
    assert decoded.manifest_sha256 == original.manifest_sha256


def test_ordinals_are_sequential_unique_and_target_bound() -> None:
    entries = (
        entry("unit-a", (1.0, 0.0)),
        entry("unit-b", (0.0, 1.0)),
    )
    generation = manifest(entries)
    rows = tuple(
        encode_ordinal(index, item, generation.corpus_refs) for index, item in enumerate(entries)
    )

    assert validate_ordinals(rows, generation) == ()
    rows[1]["ordinal"] = 7
    assert validate_ordinals(rows, generation) == ("ordinal_mismatch",)
