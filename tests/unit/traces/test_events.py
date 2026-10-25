"""Tests for typed trace events and persisted record envelopes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from capicove.batch import BatchState
from capicove.contracts import PrivacyClass, ToolStatus
from capicove.traces.content import (
    MediaContent,
    MediaReference,
    MessageRole,
    TextContent,
    TraceError,
    TraceModality,
)
from capicove.traces.events import (
    ArtifactProduced,
    BatchItemObserved,
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    EpisodeStatus,
    MessageObserved,
    ToolCallObserved,
    ToolResultObserved,
    TraceRecordType,
    record_type_for_payload,
)
from capicove.traces.models import TraceRecord, TraceRecordDraft


SHA256 = "a" * 64
NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


def _payloads() -> tuple[object, ...]:
    return (
        EpisodeStarted(
            goal="Process visible media.",
            workspace_id="workspace-1",
            actor_id="actor-1",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_sha256=SHA256,
            active_asset_versions={"asset-1": 1},
        ),
        MessageObserved(MessageRole.USER, (TextContent("hello"), MediaContent("media-1"))),
        ToolCallObserved("capicove.image.ocr", 1, {"language": "eng"}, SHA256, 1),
        ToolResultObserved(
            "capicove.image.ocr",
            1,
            ToolStatus.SUCCEEDED,
            SHA256,
            result={"text": "hello"},
            metrics={"duration": 0.5},
        ),
        ArtifactProduced("artifact:one", "media-1"),
        BatchItemObserved(
            plan_id=f"plan:{SHA256}",
            item_id=f"item:{SHA256}",
            state=BatchState.SUCCEEDED,
            attempts=1,
            cache_hit=True,
            modality=TraceModality.IMAGE,
            media_type="image/png",
            size_bytes=12,
            source_sha256=SHA256,
            cache_key=f"cache:{SHA256}",
        ),
        EpisodeFeedback("quality", {"score": 1}, "reviewer"),
        EpisodeCompleted(EpisodeStatus.SUCCEEDED, "done"),
    )


def _draft(payload: object, **changes: object) -> TraceRecordDraft:
    values: dict[str, object] = {
        "episode_id": "episode-1",
        "trace_id": "trace-1",
        "occurred_at": NOW,
        "payload": payload,
    }
    if isinstance(payload, (ToolCallObserved, ToolResultObserved)):
        values["request_id"] = "request-1"
    if isinstance(payload, ToolResultObserved) and payload.status is not ToolStatus.SUCCEEDED:
        values["error"] = TraceError("internal", "failed", False)
    values.update(changes)
    return TraceRecordDraft(**values)  # type: ignore[arg-type]


def _record(payload: object, **changes: object) -> TraceRecord:
    draft = _draft(payload)
    values = {
        "schema_version": 1,
        "record_type": record_type_for_payload(draft.payload),
        "record_version": 1,
        "record_id": "record-1",
        "episode_id": draft.episode_id,
        "trace_id": draft.trace_id,
        "sequence": 0,
        "occurred_at": draft.occurred_at,
        "payload": draft.payload,
        "request_id": draft.request_id,
        "error": draft.error,
        "privacy_class": PrivacyClass.INTERNAL,
    }
    values.update(changes)
    return TraceRecord(**values)  # type: ignore[arg-type]


def test_draft_accepts_every_record_payload_and_infers_record_type() -> None:
    expected = tuple(TraceRecordType)
    actual = tuple(record_type_for_payload(_draft(payload).payload) for payload in _payloads())
    assert actual == expected


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EpisodeStarted(" ", "w", "a", "e", 1, SHA256, {}),
        lambda: EpisodeStarted("g", "", "a", "e", 1, SHA256, {}),
        lambda: ToolCallObserved("invalid", 1, {}, SHA256, 1),
        lambda: ToolCallObserved("capicove.valid", 0, {}, SHA256, 1),
        lambda: ToolCallObserved("capicove.valid", 1, {}, SHA256, 0),
        lambda: BatchItemObserved(
            f"plan:{SHA256}",
            f"item:{SHA256}",
            BatchState.SUCCEEDED,
            -1,
            False,
            TraceModality.IMAGE,
            "image/png",
            1,
            SHA256,
            f"cache:{SHA256}",
        ),
        lambda: BatchItemObserved(
            f"plan:{SHA256}",
            f"item:{SHA256}",
            BatchState.SUCCEEDED,
            1,
            False,
            TraceModality.IMAGE,
            " ",
            1,
            SHA256,
            f"cache:{SHA256}",
        ),
        lambda: BatchItemObserved(
            f"plan:{SHA256}",
            f"item:{SHA256}",
            BatchState.FAILED,
            1,
            True,
            TraceModality.IMAGE,
            "image/png",
            1,
            SHA256,
            f"cache:{SHA256}",
        ),
        lambda: MessageObserved(MessageRole.USER, ()),
    ],
)
def test_payloads_reject_invalid_identity_counts_and_content(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_envelope_enforces_tool_status_and_error_consistency() -> None:
    succeeded = ToolResultObserved(
        "capicove.image.ocr", 1, ToolStatus.SUCCEEDED, SHA256, result={"text": "ok"}
    )
    with pytest.raises(ValueError, match="successful"):
        _draft(succeeded, error=TraceError("internal", "bad", False))

    failed = ToolResultObserved("capicove.image.ocr", 1, ToolStatus.FAILED, SHA256)
    with pytest.raises(ValueError, match="failed or rejected"):
        _draft(failed, error=None)


def test_tool_records_require_request_id() -> None:
    call = ToolCallObserved("capicove.image.ocr", 1, {}, SHA256, 1)
    with pytest.raises(ValueError, match="request_id"):
        _draft(call, request_id=None)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sequence": -1}, "sequence"),
        ({"occurred_at": datetime(2026, 8, 18)}, "occurred_at"),
        ({"previous_record_sha256": "bad"}, "previous_record_sha256"),
        ({"modalities": (TraceModality.IMAGE, TraceModality.IMAGE)}, "modalities"),
    ],
)
def test_persisted_record_rejects_invalid_envelope_values(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _record(EpisodeCompleted(EpisodeStatus.SUCCEEDED), **changes)


def test_persisted_record_rejects_payload_record_type_mismatch() -> None:
    with pytest.raises(ValueError, match="record_type"):
        _record(
            EpisodeCompleted(EpisodeStatus.SUCCEEDED),
            record_type=TraceRecordType.MESSAGE_OBSERVED,
        )


def test_event_mappings_are_defensively_frozen() -> None:
    arguments: dict[str, object] = {"nested": [1]}
    call = ToolCallObserved("capicove.image.ocr", 1, arguments, SHA256, 1)
    nested = arguments["nested"]
    assert isinstance(nested, list)
    nested.append(2)

    assert call.arguments["nested"] == (1,)
    with pytest.raises(TypeError):
        call.arguments["new"] = True  # type: ignore[index]


def test_media_identity_is_immutable_inside_record() -> None:
    media = MediaReference("media-1", TraceModality.IMAGE, "image/png", SHA256, 12)
    record = _record(
        MessageObserved(MessageRole.USER, (MediaContent("media-1"),)),
        media=(media,),
        modalities=(TraceModality.IMAGE,),
    )

    assert record.media == (media,)
