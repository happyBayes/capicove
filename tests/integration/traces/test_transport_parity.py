"""Integration tests for explicit CLI/MCP one-step trace parity."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from capicove.cli.app import _build_parser
from capicove.cli.commands.audio_transcribe import run_transcribe
from capicove.cli.commands.image import run as run_image
from capicove.cli.commands.video_filter import run as run_video_filter
from capicove.cli.commands.video_split import run as run_video_split
from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolSpec, ToolStatus
from capicove.mcp.audio_transcribe import register_audio_transcribe_mcp
from capicove.mcp.image import register_image_mcp_tools
from capicove.mcp.video_filter import register_video_filter_mcp
from capicove.mcp.video_split import register_video_split_mcp
from capicove.tools.audio_transcribe import AUDIO_TRANSCRIBE_SPEC
from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
from capicove.tools.video_filter import VIDEO_FILTER_SPEC
from capicove.tools.video_split import VIDEO_SPLIT_SPEC
from capicove.traces import EpisodeCompleted, EpisodeStatus, TraceRecorder, TraceSanitizer
from capicove.traces._adapters.sqlite import SqliteTraceStore


_NOW = datetime(2026, 8, 18, 12, tzinfo=UTC)
_McpCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


class _Server:
    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[_McpCallback], _McpCallback]:
        del name, description, input_schema

        def register(callback: _McpCallback) -> _McpCallback:
            return callback

        return register


class _CliCommand(Protocol):
    def __call__(
        self,
        arguments: argparse.Namespace,
        *,
        handler: ToolHandler | None = None,
        recorder: TraceRecorder | None = None,
    ) -> int: ...


class _CanonicalHandler:
    def __init__(self, spec: ToolSpec, *, status: ToolStatus = ToolStatus.SUCCEEDED) -> None:
        self.spec = spec
        self.status = status
        self.contexts: list[InvocationContext] = []

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        del arguments
        self.contexts.append(context)
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=self.spec.tool_id,
            tool_version=self.spec.contract_version,
            status=self.status,
            result={} if self.status is ToolStatus.SUCCEEDED else None,
            error=None,
            started_at=_NOW,
            completed_at=_NOW,
        )


def _recorder(path: Path) -> TraceRecorder:
    return TraceRecorder(
        SqliteTraceStore(path.resolve()),
        sanitizer=TraceSanitizer(),
        clock=lambda: _NOW,
    )


def _assert_one_step(path: Path, trace_id: str) -> tuple[object, ...]:
    records = SqliteTraceStore(path.resolve()).get_episode(trace_id)
    assert [record.record_type.value for record in records] == [
        "episode.started",
        "tool.call",
        "tool.result",
        "episode.completed",
    ]
    completion = records[-1].payload
    assert isinstance(completion, EpisodeCompleted)
    assert completion.status is EpisodeStatus.SUCCEEDED
    return tuple(record.payload for record in records)


def test_all_cli_tool_paths_record_exactly_one_opt_in_episode(tmp_path: Path) -> None:
    cases: list[tuple[ToolSpec, _CanonicalHandler, _CliCommand, argparse.Namespace]] = []

    audio_handler = _CanonicalHandler(AUDIO_TRANSCRIBE_SPEC)
    audio = _build_parser().parse_args(
        [
            "audio",
            "transcribe",
            "--input",
            str(tmp_path / "audio.wav"),
            "--model",
            str(tmp_path / "model.bin"),
            "--vad-model",
            str(tmp_path / "vad.bin"),
        ]
    )
    cases.append((AUDIO_TRANSCRIBE_SPEC, audio_handler, run_transcribe, audio))

    image_name = "image_shape_filter"
    image_spec = IMAGE_TOOL_SPECS_BY_NAME[image_name]
    image_handler = _CanonicalHandler(image_spec)
    image = _build_parser().parse_args(
        ["image", "run", "--tool", image_name, "--input", str(tmp_path / "image.png")]
    )
    cases.append((image_spec, image_handler, run_image, image))

    criteria = tmp_path / "criteria.json"
    criteria.write_text(json.dumps([{"type": "duration", "min_seconds": 1}]))
    filter_handler = _CanonicalHandler(VIDEO_FILTER_SPEC)
    video_filter = _build_parser().parse_args(
        [
            "video",
            "filter",
            "--input",
            str(tmp_path / "video.mp4"),
            "--criteria",
            str(criteria),
        ]
    )
    cases.append((VIDEO_FILTER_SPEC, filter_handler, run_video_filter, video_filter))

    split_handler = _CanonicalHandler(VIDEO_SPLIT_SPEC)
    video_split = _build_parser().parse_args(
        [
            "video",
            "split",
            "--artifact-ref",
            "artifact:video",
            "--artifact-sha256",
            "a" * 64,
            "--strategy",
            "duration",
        ]
    )
    cases.append((VIDEO_SPLIT_SPEC, split_handler, run_video_split, video_split))

    for index, (_, handler, command, arguments) in enumerate(cases):
        path = tmp_path / f"cli-{index}.sqlite3"
        assert command(arguments, handler=handler, recorder=_recorder(path)) == 0
        context = handler.contexts[-1]
        _assert_one_step(path, context.trace_id)


def test_all_mcp_tool_paths_record_exactly_one_opt_in_episode(tmp_path: Path) -> None:
    server = _Server()
    cases: list[tuple[Path, _CanonicalHandler, _McpCallback, Mapping[str, object]]] = []

    audio_path = tmp_path / "mcp-audio.sqlite3"
    audio_handler = _CanonicalHandler(AUDIO_TRANSCRIBE_SPEC)
    audio_callback = register_audio_transcribe_mcp(
        server, audio_handler, recorder=_recorder(audio_path)
    )
    cases.append(
        (
            audio_path,
            audio_handler,
            audio_callback,
            {"source": {"type": "path", "path": "audio.wav"}, "language": "auto"},
        )
    )

    image_path = tmp_path / "mcp-image.sqlite3"
    image_handlers = {
        name: _CanonicalHandler(spec) for name, spec in IMAGE_TOOL_SPECS_BY_NAME.items()
    }
    image_callbacks = register_image_mcp_tools(
        server, image_handlers, recorder=_recorder(image_path)
    )
    image_name = "image_shape_filter"
    cases.append(
        (
            image_path,
            image_handlers[image_name],
            image_callbacks[image_name],
            {"sources": [{"path": "image.png"}]},
        )
    )

    filter_path = tmp_path / "mcp-filter.sqlite3"
    filter_handler = _CanonicalHandler(VIDEO_FILTER_SPEC)
    filter_callback = register_video_filter_mcp(
        server, filter_handler, recorder=_recorder(filter_path)
    )
    cases.append(
        (
            filter_path,
            filter_handler,
            filter_callback,
            {
                "sources": [{"item_id": "video-1", "source": {"type": "path", "path": "v.mp4"}}],
                "criteria": [{"type": "duration", "min_seconds": 1}],
                "aggregation": "all",
            },
        )
    )

    split_path = tmp_path / "mcp-split.sqlite3"
    split_handler = _CanonicalHandler(VIDEO_SPLIT_SPEC)
    split_callback = register_video_split_mcp(server, split_handler, recorder=_recorder(split_path))
    cases.append(
        (
            split_path,
            split_handler,
            split_callback,
            {
                "source": {"type": "artifact", "ref": "artifact:video", "sha256": "a" * 64},
                "strategy": "duration",
                "config": {"split_duration": 60.0},
                "output_format": "bytes",
                "backend": "auto",
                "write_mode": "accurate",
            },
        )
    )

    for path, handler, callback, arguments in cases:
        callback(arguments)
        _assert_one_step(path, handler.contexts[-1].trace_id)


def test_audio_cli_and_mcp_record_the_same_tool_semantics(tmp_path: Path) -> None:
    source = "audio.wav"
    cli_path = tmp_path / "parity-cli.sqlite3"
    cli_handler = _CanonicalHandler(AUDIO_TRANSCRIBE_SPEC)
    cli_arguments = _build_parser().parse_args(
        [
            "audio",
            "transcribe",
            "--input",
            source,
            "--model",
            "model.bin",
            "--vad-model",
            "vad.bin",
        ]
    )
    run_transcribe(cli_arguments, handler=cli_handler, recorder=_recorder(cli_path))

    mcp_path = tmp_path / "parity-mcp.sqlite3"
    mcp_handler = _CanonicalHandler(AUDIO_TRANSCRIBE_SPEC)
    callback = register_audio_transcribe_mcp(_Server(), mcp_handler, recorder=_recorder(mcp_path))
    callback({"source": {"type": "path", "path": source}, "language": "auto"})

    cli_payloads = _assert_one_step(cli_path, cli_handler.contexts[-1].trace_id)
    mcp_payloads = _assert_one_step(mcp_path, mcp_handler.contexts[-1].trace_id)
    assert cli_payloads[1:] == mcp_payloads[1:]
    serialized = repr(cli_payloads)
    assert "base64" not in serialized
    assert "secret" not in serialized
