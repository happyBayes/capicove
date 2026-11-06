"""Quality host configuration and transport contracts, without claiming Linux execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import _build_parser
from capicove.cli.commands.video_quality import run
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.video_mcp import VideoDependencyError, video_settings


@pytest.fixture
def host(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    programs = tmp_path / "programs"
    programs.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        program = programs / name
        program.write_text("#!/bin/sh\nexit 99\n")
        program.chmod(0o700)
    value = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "quality",
        "actor_id": "tester",
        "environment_id": "local",
        "workspace_root": str(workspace),
        "profile": "read",
        "permissions": ["video:read", "subprocess:execute"],
        "enabled_tools": ["capicove.video.quality"],
        "trace_store": None,
        "capabilities": {
            "video": {
                "schema_version": 1,
                "read_roots": [str(clips)],
                "staging_root": str(workspace / "staging"),
                "backend": "ffmpeg",
                "ffmpeg": str(programs / "ffmpeg"),
                "ffprobe": str(programs / "ffprobe"),
                "max_input_bytes": 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    path = workspace / "host.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return path


def settings(path: Path) -> Any:
    config = load_mcp_host_config(path)
    assert isinstance(config, WorkspaceHostConfiguration)
    return video_settings(config)


def test_quality_only_read_profile_uses_host_byte_ceiling(
    host: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_modules(*names: str) -> None:
        raise AssertionError(f"quality must not load optional frame/model modules: {names}")

    monkeypatch.setattr("capicove.runtime.video_mcp._require_modules", no_modules)
    result = settings(host)
    assert result.output_root is None
    assert result.quality_limits.max_input_bytes == 1024 * 1024


@pytest.mark.parametrize("program", ["ffmpeg", "ffprobe"])
def test_quality_requires_both_configured_executables(host: Path, program: str) -> None:
    value = json.loads(host.read_text())
    value["capabilities"]["video"][program] = None
    host.write_text(json.dumps(value))
    with pytest.raises(VideoDependencyError):
        settings(host)


@pytest.mark.parametrize(
    "limits",
    [
        {"threads": 0},
        {"threads": True},
        {"threads": 100},
        {"max_input_bytes": 2 * 1024 * 1024},
        {"root": "/elsewhere"},
        {"max_duration_seconds": float("inf")},
    ],
)
def test_quality_limits_cannot_widen_authority(host: Path, limits: dict[str, Any]) -> None:
    value = json.loads(host.read_text())
    value["capabilities"]["video"]["quality_limits"] = limits
    host.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        settings(host)


def test_quality_limits_can_be_narrowed_by_operator(host: Path) -> None:
    value = json.loads(host.read_text())
    value["capabilities"]["video"]["quality_limits"] = {"threads": 1, "max_events": 10}
    host.write_text(json.dumps(value))
    result = settings(host)
    assert result.quality_limits.threads == 1
    assert result.quality_limits.max_events == 10


def test_configured_cli_forwards_canonical_request_without_local_fallback(
    host: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[Any] = []

    def invoke(config: Path, tool: str, arguments: dict[str, object]) -> dict[str, object]:
        seen.append((config, tool, arguments))
        return {"status": "succeeded", "result": {"verdict": "fail", "checks": []}, "error": None}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("configured CLI must never construct standalone authority")

    monkeypatch.setattr("capicove.cli.configured.invoke_configured", invoke)
    monkeypatch.setattr(
        "capicove.cli.commands.video_quality.build_video_quality_tool_handler", forbidden
    )
    arguments = _build_parser().parse_args(
        [
            "video",
            "quality",
            "--input",
            "/authorized/lecture.mp4",
            "--checks",
            "decode",
            "--runtime-config",
            str(host),
            "--output",
            "json",
        ]
    )
    assert run(arguments) == 1
    assert seen[0][1] == "capicove.video.quality"
    assert seen[0][2] == {"path": "/authorized/lecture.mp4", "checks": ["decode"]}
    assert json.loads(capsys.readouterr().out)["result"]["verdict"] == "fail"


def test_configured_cli_trace_authority_rejected_before_local_store(
    host: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    trace = tmp_path / "unwanted.sqlite"
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--trace-store",
                str(trace),
                "video",
                "quality",
                "--input",
                "/authorized/lecture.mp4",
                "--runtime-config",
                str(host),
            ]
        )
    assert caught.value.code == 2
    assert not trace.exists()
    assert "trace_store from --runtime-config" in capsys.readouterr().err


def test_shared_staging_keeps_legacy_call_signature() -> None:
    from contextlib import nullcontext

    from capicove.contracts import ToolResult, ToolStatus
    from capicove.mcp.bindings import default_context
    from capicove.runtime.video_mcp import _staged
    from capicove.tools.video_quality import VIDEO_QUALITY_SPEC

    class Intake:
        def authorize(self, path: Path) -> None:
            pass

        def stage(self, paths: list[Path]) -> Any:
            return nullcontext(())

    class Pins:
        def open(self) -> Any:
            return nullcontext({})

    context = default_context(VIDEO_QUALITY_SPEC)
    result = ToolResult(
        context.request_id,
        context.trace_id,
        VIDEO_QUALITY_SPEC.tool_id,
        1,
        ToolStatus.SUCCEEDED,
        result=None,
    )
    assert _staged(VIDEO_QUALITY_SPEC, Intake(), Pins(), context, [], lambda *_: result) == result  # type: ignore[arg-type]
