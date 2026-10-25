"""Tests for explicit CLI trace composition and one-step completion."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from capicove.cli import main
from capicove.cli.tracing import invoke_cli_tool
from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)
from capicove.tools.audio_transcribe import AUDIO_TRANSCRIBE_SPEC
from capicove.traces import EpisodeCompleted, EpisodeStatus, TraceRecorder, TraceSanitizer
from capicove.traces._adapters.sqlite import SqliteTraceStore


_NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)


def _context(trace_id: str) -> InvocationContext:
    return InvocationContext(
        request_id=f"request-{trace_id}",
        trace_id=trace_id,
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="local",
        deadline=_NOW + timedelta(minutes=1),
        permissions=AUDIO_TRANSCRIBE_SPEC.required_permissions,
        privacy_class=AUDIO_TRANSCRIBE_SPEC.privacy_class,
    )


def _recorder(path: Path) -> TraceRecorder:
    return TraceRecorder(
        SqliteTraceStore(path.resolve()),
        sanitizer=TraceSanitizer(),
        clock=lambda: _NOW,
    )


@pytest.mark.parametrize(
    ("tool_status", "episode_status"),
    [
        (ToolStatus.SUCCEEDED, EpisodeStatus.SUCCEEDED),
        (ToolStatus.FAILED, EpisodeStatus.FAILED),
        (ToolStatus.REJECTED, EpisodeStatus.REJECTED),
    ],
)
def test_invoke_cli_tool_maps_all_terminal_statuses(
    tmp_path: Path,
    tool_status: ToolStatus,
    episode_status: EpisodeStatus,
) -> None:
    path = tmp_path / f"{tool_status.value}.sqlite3"
    context = _context(tool_status.value)

    def handler(
        observed_context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        del arguments
        return ToolResult(
            request_id=observed_context.request_id,
            trace_id=observed_context.trace_id,
            tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
            tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
            status=tool_status,
            result={} if tool_status is ToolStatus.SUCCEEDED else None,
            error=(
                None
                if tool_status is ToolStatus.SUCCEEDED
                else ToolError(ToolErrorCode.INTERNAL, "public failure", False)
            ),
            started_at=_NOW,
            completed_at=_NOW,
        )

    result = invoke_cli_tool(
        recorder=_recorder(path),
        spec=AUDIO_TRANSCRIBE_SPEC,
        handler=handler,
        context=context,
        arguments={"source": {"path": "media.wav"}, "token": "secret-value"},
    )

    assert result.status is tool_status
    records = SqliteTraceStore(path.resolve()).get_episode(context.trace_id)
    assert len(records) == 4
    completion = records[-1].payload
    assert isinstance(completion, EpisodeCompleted)
    assert completion.status is episode_status
    assert "secret-value" not in repr(records)


def test_unexpected_handler_exception_records_internal_result_and_failed_completion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "exception.sqlite3"
    context = _context("exception")

    def handler(
        observed_context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        del observed_context, arguments
        raise RuntimeError("private exception text")

    with pytest.raises(RuntimeError, match="private exception text"):
        invoke_cli_tool(
            recorder=_recorder(path),
            spec=AUDIO_TRANSCRIBE_SPEC,
            handler=handler,
            context=context,
            arguments={"source": {"path": "media.wav"}},
        )

    records = SqliteTraceStore(path.resolve()).get_episode(context.trace_id)
    assert len(records) == 4
    assert records[2].error is not None
    assert records[2].error.code == "internal"
    assert "private exception text" not in repr(records)
    completion = records[-1].payload
    assert isinstance(completion, EpisodeCompleted)
    assert completion.status is EpisodeStatus.FAILED


def test_no_recorder_calls_the_exact_handler_without_storage(tmp_path: Path) -> None:
    context = _context("direct")
    calls = 0

    def handler(
        observed_context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        nonlocal calls
        calls += 1
        assert observed_context is context
        assert arguments == {"source": "direct"}
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=AUDIO_TRANSCRIBE_SPEC.tool_id,
            tool_version=AUDIO_TRANSCRIBE_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED,
            result={},
            started_at=_NOW,
            completed_at=_NOW,
        )

    result = invoke_cli_tool(
        recorder=None,
        spec=AUDIO_TRANSCRIBE_SPEC,
        handler=handler,
        context=context,
        arguments={"source": "direct"},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert calls == 1
    assert not tuple(tmp_path.iterdir())


def test_global_trace_store_is_composed_only_for_supported_tool_commands(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    trace_store = tmp_path / "state" / "nested" / "trace.sqlite3"
    observed: list[TraceRecorder] = []

    def fake_image_run(
        arguments: object,
        *,
        recorder: TraceRecorder | None = None,
    ) -> int:
        del arguments
        assert recorder is not None
        observed.append(recorder)
        return 0

    monkeypatch.setattr("capicove.cli.app.run_image", fake_image_run)
    assert (
        main(
            [
                "--trace-store",
                str(trace_store),
                "image",
                "run",
                "--tool",
                "image_shape_filter",
                "--input",
                "image.png",
            ]
        )
        == 0
    )
    assert observed
    assert trace_store.is_file()

    excluded_store = tmp_path / "excluded.sqlite3"

    def fake_deduplicate(arguments: object) -> int:
        del arguments
        return 0

    monkeypatch.setattr("capicove.cli.app.run_video_deduplicate", fake_deduplicate)
    assert (
        main(
            [
                "--trace-store",
                str(excluded_store),
                "video",
                "deduplicate",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert not excluded_store.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_trace_store_failure_reports_the_stable_store_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    images = tmp_path / "images"
    images.mkdir()

    with pytest.raises(SystemExit) as captured:
        main(
            [
                "--trace-store",
                str(parent / "traces.sqlite3"),
                "image",
                "run",
                "--tool",
                "image_shape_filter",
                "--input",
                str(images),
            ]
        )

    assert captured.value.code == 2
    error = capsys.readouterr().err
    assert "trace store could not be initialized" in error
    assert "trace store parent permissions must be owner-only" in error
