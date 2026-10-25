"""Tests for explicit multi-step trace recording sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from capicove.batch import BatchState
from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)
from capicove.tools import AUDIO_TRANSCRIBE_SPEC
from capicove.traces import (
    BatchItemObserved,
    EpisodeFeedback,
    EpisodeStatus,
    MediaContent,
    MediaReference,
    MessageRole,
    TextContent,
    ToolResultObserved,
    TraceModality,
    TraceRecord,
    TraceRecordDraft,
    TraceSanitizer,
)
from capicove.traces.codec import trace_record_sha256
from capicove.traces.ports import TraceStore
from capicove.traces.recorder import TraceRecorder, TraceRecordingError, TraceSession


SHA256 = "a" * 64
NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


class FakeStore:
    """Persist recorder drafts in memory while assigning real envelopes."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[TraceRecordDraft, str]] = []
        self.records: list[TraceRecord] = []

    def append(self, draft: TraceRecordDraft, /, *, idempotency_key: str) -> TraceRecord:
        if self.fail:
            raise RuntimeError("private database path must not escape")
        self.calls.append((draft, idempotency_key))
        sequence = len(self.records)
        record = TraceRecord(
            schema_version=1,
            record_type=draft.record_type,
            record_version=1,
            record_id=f"record-{sequence}",
            episode_id=draft.episode_id,
            trace_id=draft.trace_id,
            sequence=sequence,
            occurred_at=draft.occurred_at,
            payload=draft.payload,
            modalities=draft.modalities,
            step_id=draft.step_id,
            request_id=draft.request_id,
            parent_record_id=draft.parent_record_id,
            media=draft.media,
            lineage=draft.lineage,
            warnings=draft.warnings,
            error=draft.error,
            privacy_class=draft.privacy_class,
            previous_record_sha256=(
                None if not self.records else trace_record_sha256(self.records[-1])
            ),
        )
        self.records.append(record)
        return record

    def get_episode(self, episode_id: str, /) -> tuple[TraceRecord, ...]:
        return tuple(record for record in self.records if record.episode_id == episode_id)

    def open_snapshot(self, query: object = None, /) -> object:
        del query
        raise NotImplementedError


def _recorder(store: FakeStore) -> TraceRecorder:
    return TraceRecorder(
        cast(TraceStore, store),
        sanitizer=TraceSanitizer(),
        clock=lambda: NOW,
    )


def _session(store: FakeStore) -> TraceSession:
    return _recorder(store).start_session(
        episode_id="episode-1",
        trace_id="trace-1",
        goal="Transcribe visible media.",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        environment_version=1,
        tool_catalog_sha256=SHA256,
        active_asset_versions={},
    )


def _context(*, trace_id: str = "trace-1", request_id: str = "request-1") -> InvocationContext:
    return InvocationContext(
        request_id=request_id,
        trace_id=trace_id,
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        deadline=NOW + timedelta(minutes=1),
    )


def _result(
    *,
    trace_id: str = "trace-1",
    request_id: str = "request-1",
    status: ToolStatus = ToolStatus.SUCCEEDED,
) -> ToolResult:
    return ToolResult(
        request_id=request_id,
        trace_id=trace_id,
        tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
        tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
        status=status,
        result={"text": "hello", "lineage": {"backend": "local"}}
        if status is ToolStatus.SUCCEEDED
        else None,
        error=None
        if status is ToolStatus.SUCCEEDED
        else ToolError(ToolErrorCode.INTERNAL, "failed", False),
        started_at=NOW,
        completed_at=NOW,
    )


def test_start_session_records_one_stable_episode_start() -> None:
    store = FakeStore()
    first = _session(store)

    assert first.episode_id == "episode-1"
    assert len(store.calls) == 1
    assert store.calls[0][1] == "episode.started"

    another = FakeStore()
    _session(another)
    assert another.calls[0][1] == store.calls[0][1]


def test_message_preserves_interleaved_content_and_attaches_matching_media() -> None:
    store = FakeStore()
    session = _session(store)
    media = MediaReference("media-1", TraceModality.AUDIO, "audio/wav", SHA256, 5)

    record = session.record_message(
        message_id="message-1",
        role=MessageRole.USER,
        content=(TextContent("before"), MediaContent("media-1"), TextContent("after")),
        media=(media,),
    )

    assert record.payload.content == (  # type: ignore[union-attr]
        TextContent("before"),
        MediaContent("media-1"),
        TextContent("after"),
    )
    assert record.media == (media,)
    assert store.calls[-1][1] == "message:message-1"


def test_tool_calls_results_and_exception_use_stable_keys_and_public_error() -> None:
    store = FakeStore()
    session = _session(store)
    context = _context()
    session.record_tool_call(
        context=context,
        spec=AUDIO_TRANSCRIBE_SPEC,
        arguments={"source": "[REDACTED]"},
        media=(),
        attempt=2,
    )
    session.record_tool_result(result=_result(), spec=AUDIO_TRANSCRIBE_SPEC, attempt=2)
    assert tuple(key for _, key in store.calls[-2:]) == (
        "tool.call:request-1:2",
        "tool.result:request-1:2",
    )

    second_context = _context(request_id="request-2")
    session.record_tool_call(
        context=second_context,
        spec=AUDIO_TRANSCRIBE_SPEC,
        arguments={},
        media=(),
        attempt=1,
    )
    exception_record = session.record_tool_exception(
        context=second_context, spec=AUDIO_TRANSCRIBE_SPEC, attempt=1
    )
    assert exception_record.error is not None
    assert exception_record.error.code == "internal"
    assert exception_record.error.message == "Tool execution raised an unexpected exception."
    assert "private" not in exception_record.error.message


def test_tool_results_and_feedback_are_sanitized_before_persistence() -> None:
    """Catch raw credentials, paths, and inline media crossing the store boundary."""
    store = FakeStore()
    session = _session(store)
    context = _context()
    session.record_tool_call(
        context=context,
        spec=AUDIO_TRANSCRIBE_SPEC,
        arguments={},
        media=(),
        attempt=1,
    )
    result = ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
        tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
        status=ToolStatus.SUCCEEDED,
        result={
            "path": "/Users/alice/private/transcript.json",
            "api_key": "sk-abcdefghijklmnop",  # pragma: allowlist secret
            "inline": {"base64": "aGVsbG8="},
            "lineage": {"backend": "Bearer abcdefghijklmnop"},
        },
        started_at=NOW,
        completed_at=NOW,
    )

    recorded = session.record_tool_result(
        result=result,
        spec=AUDIO_TRANSCRIBE_SPEC,
        attempt=1,
    )
    session.complete(status=EpisodeStatus.SUCCEEDED)
    feedback = session.record_feedback(
        feedback_id="privacy",
        payload=EpisodeFeedback(
            "review",
            {
                "password": "open sesame",  # pragma: allowlist secret
                "note": "See C:\\Users\\alice\\private.txt",
            },
            "reviewer",
        ),
    )

    assert isinstance(recorded.payload, ToolResultObserved)
    assert (
        recorded.payload.result
        == {
            "path": "[REDACTED_ABSOLUTE_PATH]",
            "api_key": "[REDACTED]",
            "inline": {
                "base64": {
                    "type": "redacted_media",
                    "size_bytes": 5,
                    "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",  # pragma: allowlist secret
                }
            },
            "lineage": {"backend": "[REDACTED_BEARER_TOKEN]"},
        }
    )
    assert isinstance(feedback.payload, EpisodeFeedback)
    assert feedback.payload.value == {
        "password": "[REDACTED]",
        "note": "See [REDACTED_ABSOLUTE_PATH]",
    }


def test_artifact_batch_feedback_completion_and_post_completion_rules() -> None:
    store = FakeStore()
    session = _session(store)
    media = MediaReference("media-1", TraceModality.IMAGE, "image/png", SHA256, 5)
    session.record_artifact(event_id="artifact-1", artifact_ref="artifact:one", media=media)
    batch = BatchItemObserved(
        plan_id=f"plan:{SHA256}",
        item_id=f"item:{SHA256}",
        state=BatchState.SKIPPED,
        attempts=0,
        cache_hit=False,
        modality=TraceModality.IMAGE,
        media_type="image/png",
        size_bytes=5,
        source_sha256=SHA256,
        cache_key=f"cache:{SHA256}",
    )
    session.record_batch_item(event_id="batch-1", payload=batch)
    session.complete(status=EpisodeStatus.SUCCEEDED, outcome="done")
    session.record_feedback(
        feedback_id="feedback-1",
        payload=EpisodeFeedback("quality", {"score": 1}, "reviewer"),
    )
    assert tuple(key for _, key in store.calls[-4:]) == (
        "artifact:artifact-1",
        "batch-1",
        "episode.completed",
        "feedback:feedback-1",
    )
    with pytest.raises(TraceRecordingError, match="completed"):
        session.record_message(
            message_id="late", role=MessageRole.USER, content=(TextContent("late"),)
        )


def test_artifact_requires_payload_and_media_derivation_to_match() -> None:
    session = _session(FakeStore())
    media = MediaReference(
        "media-1",
        TraceModality.IMAGE,
        "image/png",
        SHA256,
        5,
        derived_from=("source-1",),
    )

    with pytest.raises(TraceRecordingError, match="derivation"):
        session.record_artifact(
            event_id="artifact-1",
            artifact_ref="artifact:one",
            media=media,
            derived_from=("different-source",),
        )


def test_session_rejects_identity_mismatch_and_duplicate_message_media() -> None:
    store = FakeStore()
    session = _session(store)
    with pytest.raises(TraceRecordingError, match="trace_id"):
        session.record_tool_call(
            context=_context(trace_id="other"),
            spec=AUDIO_TRANSCRIBE_SPEC,
            arguments={},
            media=(),
            attempt=1,
        )

    media = MediaReference("media-1", TraceModality.IMAGE, "image/png", SHA256, 5)
    with pytest.raises(TraceRecordingError, match="duplicate"):
        session.record_message(
            message_id="message-1",
            role=MessageRole.USER,
            content=(MediaContent("media-1"), MediaContent("media-1")),
            media=(media,),
        )


def test_store_failure_is_wrapped_without_being_swallowed() -> None:
    with pytest.raises(TraceRecordingError) as captured:
        _session(FakeStore(fail=True))
    assert "private database path" not in str(captured.value)
