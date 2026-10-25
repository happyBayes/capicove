"""Canonical visual hashes remain bounded independently of vector corpus size."""

import hashlib
import json
import struct
import tracemalloc
from dataclasses import replace

import pytest

from capicove.retrieval.video_models import (
    VisualIndexEntry,
    _unit_payload,
    searchable_units_manifest_sha256,
    visual_entries_content_sha256,
)
from tests.unit.retrieval.test_video_models import unit


def test_streaming_unit_digest_preserves_original_canonical_array() -> None:
    units = (unit(), replace(unit(), unit_id="frame-中文"))
    old_payload = json.dumps(
        [_unit_payload(item) for item in units],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert searchable_units_manifest_sha256(iter(units)) == hashlib.sha256(old_payload).hexdigest()
    assert searchable_units_manifest_sha256(()) == hashlib.sha256(b"[]").hexdigest()


def test_streaming_digest_preserves_original_canonical_wire_format() -> None:
    entries = (
        VisualIndexEntry(unit(), (1.0, -0.0)),
        VisualIndexEntry(replace(unit(), unit_id="frame-中文"), (0.1, -0.3)),
    )
    payload = [
        {
            "unit_id": entry.unit.unit_id,
            "target_type": entry.unit.unit_type.value,
            "vector_float32_be": [struct.pack("!f", value).hex() for value in entry.vector],
        }
        for entry in entries
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    assert visual_entries_content_sha256(iter(entries)) == hashlib.sha256(encoded).hexdigest()
    assert visual_entries_content_sha256(()) == hashlib.sha256(b"[]").hexdigest()


@pytest.mark.slow
def test_visual_hash_does_not_materialize_all_vector_strings() -> None:
    template = unit()
    template = replace(
        template, embedding_spaces=(replace(template.embedding_spaces[0], dimension=768),)
    )
    vector = (0.125,) * 768
    entries = (VisualIndexEntry(replace(template, unit_id=f"k-{i}"), vector) for i in range(512))
    tracemalloc.start()
    try:
        digest = visual_entries_content_sha256(entries)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(digest) == 64
    assert peak < 16 * 1024**2


@pytest.mark.slow
def test_unit_manifest_hash_streams_large_metadata_corpus() -> None:
    template = unit()
    units = (replace(template, unit_id=f"k-{i}") for i in range(10_000))
    tracemalloc.start()
    try:
        digest = searchable_units_manifest_sha256(units)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(digest) == 64
    assert peak < 8 * 1024**2
