"""Tests for canonical trace record encoding and strict decoding."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from capicove.contracts import PrivacyClass, ToolStatus
from capicove.traces.codec import (
    canonical_json_bytes,
    canonical_record_bytes,
    decode_trace_record,
    encode_trace_record,
    trace_record_sha256,
)
from capicove.traces.content import MessageRole, TextContent, TraceError, TraceModality
from capicove.traces.events import MessageObserved, ToolResultObserved, TraceRecordType
from capicove.traces.models import TraceRecord


SHA256 = "a" * 64


def make_record() -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.MESSAGE_OBSERVED,
        record_version=1,
        record_id="record-1",
        episode_id="episode-1",
        trace_id="trace-1",
        sequence=0,
        occurred_at=datetime(2026, 8, 18, 12, 30, 4, 5, tzinfo=UTC),
        payload=MessageObserved(MessageRole.USER, (TextContent("你好"),)),
        modalities=(TraceModality.IMAGE,),
        privacy_class=PrivacyClass.INTERNAL,
    )


def test_canonical_encoding_is_compact_utf8_stable_and_round_trips() -> None:
    record = make_record()

    encoded = encode_trace_record(record)
    raw = canonical_record_bytes(record)

    assert raw == canonical_json_bytes(encoded)
    assert b"\n" not in raw
    assert "你好".encode() in raw
    assert b"2026-08-18T12:30:04.000005Z" in raw
    assert raw.startswith(b'{"episode_id":"episode-1","error":null')
    assert trace_record_sha256(record) == trace_record_sha256(record)
    assert decode_trace_record(json.loads(raw)) == record


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"record_type": "unknown"}),
        lambda value: value.update({"payload": {}}),
        lambda value: value.update({"occurred_at": "2026-08-18T12:30:04"}),
    ],
)
def test_decoder_rejects_unknown_or_malformed_records(mutate: object) -> None:
    value = encode_trace_record(make_record())
    mutate(value)  # type: ignore[operator]

    with pytest.raises(ValueError):
        decode_trace_record(value)


def test_canonical_json_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_decoder_rejects_result_status_error_mismatch() -> None:
    record = TraceRecord(
        schema_version=1,
        record_type=TraceRecordType.TOOL_RESULT,
        record_version=1,
        record_id="record-2",
        episode_id="episode-1",
        trace_id="trace-1",
        sequence=1,
        occurred_at=datetime(2026, 8, 18, tzinfo=UTC),
        payload=ToolResultObserved("capicove.image.ocr", 1, ToolStatus.FAILED, SHA256),
        request_id="request-1",
        error=TraceError("internal", "failed", False),
        previous_record_sha256=SHA256,
    )
    value = encode_trace_record(record)
    value["error"] = None

    with pytest.raises(ValueError, match="failed or rejected"):
        decode_trace_record(value)
