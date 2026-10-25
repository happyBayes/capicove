import argparse
import base64
import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pytest import CaptureFixture, MonkeyPatch

from capicove.cli.app import _build_parser, main
from capicove.cli.commands.video_split import run
from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolStatus,
)


class FakeHandler:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls: list[tuple[InvocationContext, Mapping[str, object]]] = []

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        self.calls.append((context, arguments))
        return self.result


def make_result(
    status: ToolStatus = ToolStatus.SUCCEEDED,
    *,
    degraded: bool = False,
) -> ToolResult:
    error = (
        ToolError(ToolErrorCode.INTERNAL, "failed", False)
        if status is not ToolStatus.SUCCEEDED
        else None
    )
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.split",
        tool_version=1,
        status=status,
        result={
            "status": "succeeded" if status is ToolStatus.SUCCEEDED else "failed",
            "original": {"type": "bytes", "sha256": "a" * 64},
            "segments": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "type": "path",
                    "path": "/tmp/segment-000.mp4",
                }
            ],
            "warnings": ([{"code": "segment_failed"}] if degraded else []),
            "degraded": degraded,
            "lineage": {"trace_id": "trace-1"},
            "failure": None if error is None else {"code": "internal"},
        },
        error=error,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def parse(arguments: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(["video", "split", *arguments])


def test_stdin_bytes_map_to_canonical_tool_arguments(
    capsys: CaptureFixture[str],
) -> None:
    handler = FakeHandler(make_result())
    arguments = parse(
        [
            "--input",
            "-",
            "--strategy",
            "duration",
            "--split-duration",
            "5",
            "--output-format",
            "bytes",
            "--output",
            "json",
        ]
    )

    exit_code = run(arguments, handler=handler, stdin=io.BytesIO(b"video"))

    assert exit_code == 0
    canonical = handler.calls[0][1]
    assert canonical["source"] == {
        "type": "bytes",
        "base64": base64.b64encode(b"video").decode(),
    }
    assert canonical["strategy"] == "duration"
    assert canonical["config"] == {
        "split_duration": 5.0,
        "overlap_duration": 0.0,
        "min_last_split_duration": 0.0,
    }
    assert canonical["write_mode"] == "accurate"
    assert json.loads(capsys.readouterr().out)["trace_id"] == "trace-1"


def test_path_source_and_scene_options_map_without_backend_flags() -> None:
    handler = FakeHandler(make_result())
    arguments = parse(
        [
            "--input",
            "/tmp/video.mp4",
            "--strategy",
            "scene",
            "--detector",
            "content",
            "--scene-option",
            "threshold=27.0",
            "--output-format",
            "path",
            "--output-directory",
            "/tmp/segments",
            "--backend",
            "ffmpeg",
        ]
    )

    assert run(arguments, handler=handler) == 0

    canonical = handler.calls[0][1]
    assert canonical["source"] == {"type": "path", "path": "/tmp/video.mp4"}
    assert canonical["config"] == {
        "detector": "content",
        "options": {"threshold": 27.0},
    }
    assert "ffmpeg_flags" not in canonical


def test_keyframe_minimum_duration_maps_to_canonical_config() -> None:
    handler = FakeHandler(make_result())
    arguments = parse(
        [
            "--input",
            "/tmp/video.mp4",
            "--strategy",
            "keyframe",
            "--min-keyframe-segment-duration",
            "6",
        ]
    )

    assert run(arguments, handler=handler) == 0
    assert handler.calls[0][1]["config"] == {"min_segment_duration": 6.0}


def test_degraded_human_output_keeps_success_exit_code(
    capsys: CaptureFixture[str],
) -> None:
    handler = FakeHandler(make_result(degraded=True))

    exit_code = run(
        parse(["--input", "-", "--strategy", "keyframe"]),
        handler=handler,
        stdin=io.BytesIO(b"video"),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "segments: 1" in output
    assert "segment 0: 0.000-5.000s -> /tmp/segment-000.mp4" in output
    assert "segment_failed" in output


def test_failed_tool_result_returns_exit_one(capsys: CaptureFixture[str]) -> None:
    handler = FakeHandler(make_result(ToolStatus.FAILED))

    assert (
        run(
            parse(["--input", "-", "--strategy", "keyframe", "--output", "json"]),
            handler=handler,
            stdin=io.BytesIO(b"video"),
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "internal"


def test_parser_rejects_conflicting_source_representations() -> None:
    with pytest.raises(SystemExit) as captured:
        parse(
            [
                "--input",
                "video.mp4",
                "--artifact-ref",
                "artifact:video:1",
                "--artifact-sha256",
                "a" * 64,
            ]
        )
    assert captured.value.code == 2


def test_app_dispatches_nested_video_split_command(monkeypatch: MonkeyPatch) -> None:
    observed: list[argparse.Namespace] = []

    def fake_run(namespace: argparse.Namespace) -> int:
        observed.append(namespace)
        return 0

    monkeypatch.setattr("capicove.cli.app.run_video_split", fake_run)

    assert main(["video", "split", "--input", "-", "--strategy", "keyframe"]) == 0
    assert observed[0].video_command == "split"


def test_output_directory_selects_path_output_by_default() -> None:
    handler = FakeHandler(make_result())
    arguments = parse(
        [
            "--input",
            "/tmp/video.mp4",
            "--strategy",
            "duration",
            "--output-directory",
            "/tmp/segments",
        ]
    )

    assert run(arguments, handler=handler) == 0
    canonical = handler.calls[0][1]
    assert canonical["output_format"] == "path"
    assert canonical["output_directory"] == "/tmp/segments"


def test_output_format_defaults_to_bytes_without_output_directory() -> None:
    handler = FakeHandler(make_result())

    assert run(parse(["--input", "/tmp/video.mp4", "--strategy", "keyframe"]), handler=handler) == 0
    assert handler.calls[0][1]["output_format"] == "bytes"


def test_byte_output_with_output_directory_is_a_usage_error(
    capsys: CaptureFixture[str],
) -> None:
    handler = FakeHandler(make_result())
    arguments = parse(
        [
            "--input",
            "/tmp/video.mp4",
            "--strategy",
            "keyframe",
            "--output-format",
            "bytes",
            "--output-directory",
            "/tmp/segments",
        ]
    )

    assert run(arguments, handler=handler) == 2
    assert handler.calls == []
    assert "--output-directory cannot be used with --output-format bytes" in (
        capsys.readouterr().err
    )
