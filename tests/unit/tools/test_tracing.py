"""Tests for the shared fail-closed Tool tracing boundary."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)
from capicove.tools import AUDIO_TRANSCRIBE_SPEC, VIDEO_SPLIT_SPEC
from capicove.tools.tracing import (
    TracedToolHandler,
    TraceProtocolError,
    invoke_one_step_tool,
    tool_catalog_sha256,
)
from capicove.traces import (
    EpisodeStatus,
    ToolCallObserved,
    ToolResultObserved,
    TraceRecorder,
    TraceRecordingError,
    TraceSanitizer,
)
from capicove.traces._adapters.sqlite import SqliteTraceStore
from capicove.traces.models import TraceRecord, TraceRecordDraft


SHA256 = "a" * 64
NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


def _context() -> InvocationContext:
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        deadline=NOW + timedelta(minutes=1),
    )


def _result(
    *,
    status: ToolStatus = ToolStatus.SUCCEEDED,
    trace_id: str = "trace-1",
) -> ToolResult:
    return ToolResult(
        request_id="request-1",
        trace_id=trace_id,
        tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
        tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
        status=status,
        result={"text": "visible"} if status is ToolStatus.SUCCEEDED else None,
        error=None
        if status is ToolStatus.SUCCEEDED
        else ToolError(ToolErrorCode.INTERNAL, "public failure", False),
        started_at=NOW,
        completed_at=NOW,
    )


def _session(tmp_path: Path, sanitizer: TraceSanitizer | None = None):  # type: ignore[no-untyped-def]
    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    recorder = TraceRecorder(
        store,
        sanitizer=sanitizer or TraceSanitizer(),
        clock=lambda: NOW,
    )
    session = recorder.start_session(
        episode_id="episode-1",
        trace_id="trace-1",
        goal="Run a Tool.",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        environment_version=1,
        tool_catalog_sha256=SHA256,
        active_asset_versions={},
    )
    return store, recorder, session


class _FailingAppendStore:
    def __init__(self, delegate: SqliteTraceStore, failing_keys: frozenset[str]) -> None:
        self.delegate = delegate
        self.failing_keys = failing_keys

    def append(
        self,
        draft: TraceRecordDraft,
        /,
        *,
        idempotency_key: str,
    ) -> TraceRecord:
        if idempotency_key in self.failing_keys or any(
            idempotency_key.startswith(prefix)
            for prefix in self.failing_keys
            if prefix.endswith(":")
        ):
            raise RuntimeError("simulated persistence failure")
        return self.delegate.append(draft, idempotency_key=idempotency_key)

    def get_episode(self, episode_id: str, /) -> tuple[TraceRecord, ...]:
        return self.delegate.get_episode(episode_id)

    def open_snapshot(self, query: object, /) -> object:
        return self.delegate.open_snapshot(query)  # type: ignore[arg-type]


def test_wrapper_persists_sanitized_call_before_raw_execution_and_exact_result(
    tmp_path: Path,
) -> None:
    store, recorder, session = _session(tmp_path)
    raw_arguments = {"source": {"type": "path", "path": "/private/source.wav"}}
    observations: list[object] = []
    expected = _result()

    def handler(context: InvocationContext, arguments: object, /) -> ToolResult:
        observations.append(store.get_episode("episode-1"))
        assert context is _context_value
        assert arguments is raw_arguments
        return expected

    _context_value = _context()
    wrapped = TracedToolHandler(
        AUDIO_TRANSCRIBE_SPEC,
        handler,
        session,
        recorder.sanitizer,
    )
    actual = wrapped(_context_value, raw_arguments)

    assert actual is expected
    call_seen_before_execution = observations[0]
    assert isinstance(call_seen_before_execution, tuple)
    assert isinstance(call_seen_before_execution[-1].payload, ToolCallObserved)
    assert "/private/source.wav" not in repr(call_seen_before_execution[-1].payload.arguments)
    records = store.get_episode("episode-1")
    assert isinstance(records[-1].payload, ToolResultObserved)


@pytest.mark.parametrize("status", [ToolStatus.FAILED, ToolStatus.REJECTED])
def test_wrapper_records_canonical_failed_and_rejected_results(
    tmp_path: Path, status: ToolStatus
) -> None:
    store, recorder, session = _session(tmp_path)
    expected = _result(status=status)
    wrapped = TracedToolHandler(
        AUDIO_TRANSCRIBE_SPEC,
        lambda context, arguments: expected,
        session,
        recorder.sanitizer,
    )

    assert wrapped(_context(), {"source": {"path": "relative.wav"}}) is expected
    record = store.get_episode("episode-1")[-1]
    assert isinstance(record.payload, ToolResultObserved)
    assert record.payload.status is status
    assert record.error is not None


def test_unexpected_exception_records_fixed_internal_result_then_reraises_original(
    tmp_path: Path,
) -> None:
    store, recorder, session = _session(tmp_path)
    original = LookupError("private stack detail")

    def handler(context: object, arguments: object, /) -> ToolResult:
        del context, arguments
        raise original

    wrapped = TracedToolHandler(
        AUDIO_TRANSCRIBE_SPEC,
        handler,
        session,
        recorder.sanitizer,
    )
    with pytest.raises(LookupError) as captured:
        wrapped(_context(), {})

    assert captured.value is original
    record = store.get_episode("episode-1")[-1]
    assert record.error is not None
    assert record.error.message == "Tool execution raised an unexpected exception."
    assert "private stack detail" not in repr(record)


def test_post_call_trace_failure_preserves_primary_error_and_keeps_episode_incomplete(
    tmp_path: Path,
) -> None:
    delegate = SqliteTraceStore(tmp_path / "traces.sqlite3")
    store = _FailingAppendStore(delegate, frozenset({"tool.result:"}))
    recorder = TraceRecorder(
        store,  # type: ignore[arg-type]
        sanitizer=TraceSanitizer(),
        clock=lambda: NOW,
    )
    session = recorder.start_session(
        episode_id="episode-1",
        trace_id="trace-1",
        goal="Run a Tool.",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        environment_version=1,
        tool_catalog_sha256=SHA256,
        active_asset_versions={},
    )
    original = LookupError("primary Tool failure")

    def handler(context: object, arguments: object, /) -> ToolResult:
        del context, arguments
        raise original

    wrapped = TracedToolHandler(
        AUDIO_TRANSCRIBE_SPEC,
        handler,
        session,
        recorder.sanitizer,
    )

    with pytest.raises(LookupError) as captured:
        wrapped(_context(), {})

    assert captured.value is original
    with pytest.raises(TraceRecordingError, match="open Tool call"):
        session.complete(status=EpisodeStatus.FAILED)
    assert len(delegate.get_episode("episode-1")) == 2


def test_one_step_completion_trace_failure_does_not_mask_tool_error(
    tmp_path: Path,
) -> None:
    delegate = SqliteTraceStore(tmp_path / "traces.sqlite3")
    store = _FailingAppendStore(
        delegate,
        frozenset({"tool.result:"}),
    )
    recorder = TraceRecorder(
        store,  # type: ignore[arg-type]
        sanitizer=TraceSanitizer(),
        clock=lambda: NOW,
    )
    original = LookupError("primary Tool failure")

    def handler(context: object, arguments: object, /) -> ToolResult:
        del context, arguments
        raise original

    with pytest.raises(LookupError) as captured:
        invoke_one_step_tool(
            recorder=recorder,
            spec=AUDIO_TRANSCRIBE_SPEC,
            handler=handler,
            context=_context(),
            arguments={},
            episode_id="episode-1",
        )

    assert captured.value is original
    assert len(delegate.get_episode("episode-1")) == 2


def test_identity_mismatch_records_internal_result_and_raises_protocol_error(
    tmp_path: Path,
) -> None:
    store, recorder, session = _session(tmp_path)
    wrapped = TracedToolHandler(
        AUDIO_TRANSCRIBE_SPEC,
        lambda context, arguments: _result(trace_id="other-trace"),
        session,
        recorder.sanitizer,
    )

    with pytest.raises(TraceProtocolError, match="identity"):
        wrapped(_context(), {})
    record = store.get_episode("episode-1")[-1]
    assert record.error is not None
    assert record.error.code == "internal"


def test_sanitization_failure_prevents_underlying_execution(tmp_path: Path) -> None:
    sanitizer = TraceSanitizer(max_text_bytes=4)
    _, recorder, session = _session(tmp_path, sanitizer)
    executed = False

    def handler(context: object, arguments: object, /) -> ToolResult:
        nonlocal executed
        executed = True
        return _result()

    wrapped = TracedToolHandler(
        AUDIO_TRANSCRIBE_SPEC,
        handler,
        session,
        recorder.sanitizer,
    )
    with pytest.raises(ValueError):
        wrapped(_context(), {"ordinary": "12345"})
    assert not executed


def test_tool_catalog_hash_is_order_independent_and_covers_public_contract() -> None:
    first = tool_catalog_sha256((AUDIO_TRANSCRIBE_SPEC, VIDEO_SPLIT_SPEC))
    reversed_hash = tool_catalog_sha256((VIDEO_SPLIT_SPEC, AUDIO_TRANSCRIBE_SPEC))
    changed = tool_catalog_sha256(
        (replace(AUDIO_TRANSCRIBE_SPEC, description="Changed description."), VIDEO_SPLIT_SPEC)
    )

    assert first == reversed_hash
    assert first != changed
    assert len(first) == 64


def test_one_step_invocation_records_explicit_attempt_episode(tmp_path: Path) -> None:
    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    recorder = TraceRecorder(store, sanitizer=TraceSanitizer(), clock=lambda: NOW)
    expected = _result()

    actual = invoke_one_step_tool(
        recorder=recorder,
        spec=AUDIO_TRANSCRIBE_SPEC,
        handler=lambda context, arguments: expected,
        context=_context(),
        arguments={},
        attempt=3,
        episode_id="episode-attempt-3",
    )

    assert actual is expected
    records = store.get_episode("episode-attempt-3")
    assert isinstance(records[1].payload, ToolCallObserved)
    assert records[1].payload.attempt == 3
    assert records[0].trace_id == "trace-1"


def test_one_step_invocation_rejects_invalid_attempt_before_starting_episode(
    tmp_path: Path,
) -> None:
    store = SqliteTraceStore(tmp_path / "traces.sqlite3")
    recorder = TraceRecorder(store, sanitizer=TraceSanitizer(), clock=lambda: NOW)

    with pytest.raises(ValueError, match="positive integer"):
        invoke_one_step_tool(
            recorder=recorder,
            spec=AUDIO_TRANSCRIBE_SPEC,
            handler=lambda context, arguments: _result(),
            context=_context(),
            arguments={},
            attempt=0,
            episode_id="invalid-attempt",
        )

    assert store.get_episode("invalid-attempt") == ()
