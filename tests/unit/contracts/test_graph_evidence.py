"""Exercise graph evidence validation and canonical serialization."""

import dataclasses
import json
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphScope, InputSegment, SourceSpan, TextUnit
from capicove.contracts.retrieval import CharacterRange, TextChunkLocator, TextOrigin
from capicove.graph_build.codec import (
    canonical_bytes,
    evidence_from_json,
    evidence_to_json,
    fingerprint,
)


def make_unit(**changes: Any) -> TextUnit:
    values: dict[str, Any] = {
        "text_unit_id": "unit:authored",
        "document_ref": "document:authored",
        "scope": GraphScope("workspace:authored", "scope:private"),
        "source_sha256": "a" * 64,
        "representation_ref": "representation:authored",
        "core_locator": TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(1, 3)),
        "segments": [
            InputSegment(
                "core",
                "core",
                "中🙂",
                [SourceSpan("evidence:authored", "representation:authored", 1, 3)],  # type: ignore[arg-type]
            )
        ],
        "section_path": ["Practice"],
        "ordinal": 0,
        "token_count": 2,
        "chunking_fingerprint": "b" * 64,
    }
    values.update(changes)
    return TextUnit(**values)


@pytest.mark.parametrize("start,end", [(4, 4), (-1, 2), (3, 1), (True, 3), (0, 2.0)])
def test_rejects_invalid_source_ranges(start: Any, end: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        SourceSpan("evidence:1", "representation:1", start, end)


@pytest.mark.parametrize(
    "changes",
    [
        {"block_id": "b"},
        {"page_index": 0},
        {"block_id": "b", "page_index": -1},
        {"block_id": "b", "page_index": True},
        {"evidence_ref": " "},
        {"representation_ref": 4},
    ],
)
def test_source_span_rejects_invalid_provenance(changes: dict[str, Any]) -> None:
    values = dict(evidence_ref="e", representation_ref="r", start_char=0, end_char=1)
    values.update(changes)
    with pytest.raises((TypeError, ValueError)):
        SourceSpan(**values)  # type: ignore[arg-type]


def test_scope_identity_and_immutable_unicode_evidence() -> None:
    unit = make_unit()
    assert unit.scope != GraphScope("workspace:authored", "scope:other")
    assert unit.scope != GraphScope("workspace:other", "scope:private")
    assert unit.segments[0].text == "A中🙂B"[1:3]
    assert isinstance(unit.segments, tuple)
    assert isinstance(unit.section_path, tuple)
    assert isinstance(unit.segments[0].source_spans, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        unit.ordinal = 7  # type: ignore[misc]


@pytest.mark.parametrize("value", ["", " ", 1, None])
def test_scope_rejects_invalid_references(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        GraphScope(value, "s")


@pytest.mark.parametrize(
    "changes",
    [
        {"ordinal": True},
        {"token_count": 0},
        {"source_sha256": "bad"},
        {"chunking_fingerprint": "A" * 64},
        {"scope": "scope"},
        {"segments": []},
        {"core_locator": "locator"},
        {"section_path": [""]},
        {"representation_ref": "other"},
    ],
)
def test_text_unit_rejects_invalid_metadata(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_unit(**changes)


def test_input_segments_defensively_copy_and_validate_source_lengths() -> None:
    spans = [SourceSpan("e", "r", 0, 1)]
    segment = InputSegment("s", "context", "A", spans)  # type: ignore[arg-type]
    spans.clear()
    assert len(segment.source_spans) == 1
    for role, text, sources in [
        ("unknown", "A", segment.source_spans),
        ("core", "AB", segment.source_spans),
        ("core", "", ()),
    ]:
        with pytest.raises(ValueError):
            InputSegment("s", role, text, sources)


def test_text_unit_requires_unique_segments_and_contiguous_core() -> None:
    core = make_unit().segments[0]
    with pytest.raises(ValueError):
        make_unit(segments=[core, core])
    with pytest.raises(ValueError):
        make_unit(core_locator=TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 2)))
    with pytest.raises(ValueError):
        make_unit(segments=[dataclasses.replace(core, role="context")])


@pytest.mark.parametrize("origin", list(TextOrigin))
def test_evidence_round_trip_preserves_locator_semantics(origin: TextOrigin) -> None:
    from capicove.contracts.retrieval import TemporalRange

    locator = TextChunkLocator(
        origin,
        CharacterRange(1, 3),
        page_number=1 if origin is TextOrigin.OCR else None,
        time_range=TemporalRange(0, 1000) if origin is TextOrigin.ASR else None,
    )
    unit = make_unit(core_locator=locator)
    for value in (unit.scope, SourceSpan("e", "r", 1, 3, "block", 0), unit.segments[0], unit):
        assert evidence_from_json(evidence_to_json(value)) == value


def test_json_identity_ignores_mapping_order_without_unicode_normalization() -> None:
    assert canonical_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert canonical_bytes({"text": "中🙂"}) == '{"text":"中🙂"}'.encode()
    assert canonical_bytes("é") != canonical_bytes("e\u0301")
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert len(fingerprint(None)) == 64
    assert fingerprint({"scope": "a"}) != fingerprint({"scope": "b"})


@pytest.mark.parametrize("value", [{1: "x"}, float("nan"), float("inf"), b"x", {1}, object()])
def test_canonical_json_rejects_non_json_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_bytes(value)


def test_canonical_json_bounds_depth_and_cycles() -> None:
    value: Any = 0
    for _ in range(16):
        value = [value]
    canonical_bytes(value)
    with pytest.raises(ValueError):
        canonical_bytes([value])
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        canonical_bytes(cycle)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"kind":"scope","value":{"workspace_ref":"w","scope_ref":"s","extra":0}}',
        '{"schema_version":true,"kind":"scope","value":{"workspace_ref":"w","scope_ref":"s"}}',
        '{"schema_version":2,"kind":"scope","value":{}}',
        '{"schema_version":1,"kind":"unknown","value":{}}',
        '{"schema_version":1,"schema_version":1,"kind":"scope","value":{}}',
        "[]",
        "NaN",
    ],
)
def test_decoder_rejects_unversioned_or_open_payloads(payload: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        evidence_from_json(payload)


def test_decoder_rejects_unknown_nested_locator_fields() -> None:
    payload = json.loads(evidence_to_json(make_unit()))
    payload["value"]["core_locator"]["unexpected"] = 1
    with pytest.raises(ValueError):
        evidence_from_json(json.dumps(payload))
