"""Exercise audio clipping through the actual CLI and workspace bindings."""

from __future__ import annotations

import json
import os
import shutil
import wave
from pathlib import Path

import pytest

from capicove.cli.app import main
from capicove.mcp.bindings import invoke_binding
from capicove.runtime.mcp_config import load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host


pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("ffmpeg") is None, reason="requires local FFmpeg"
)


def make_source(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\x10\x01" * 8000)


def configuration(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source = inputs / "input.wav"
    make_source(source)
    output = root / "clips"
    output.mkdir(mode=0o700)
    config = root / "host.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_name": "capicove-local",
                "workspace_id": "workspace",
                "actor_id": "owner",
                "environment_id": "local",
                "workspace_root": str(root),
                "profile": "write",
                "permissions": ["audio:read", "artifact:write", "subprocess:execute"],
                "enabled_tools": ["capicove.audio.clip"],
                "trace_store": None,
                "network": {"enabled": False, "source_hosts": []},
                "capabilities": {
                    "audio": {
                        "schema_version": 2,
                        "read_roots": [str(inputs)],
                        "staging_root": str(root / "staging"),
                        "evidence_root": str(root / "evidence"),
                        "output_root": str(output),
                        "ffmpeg": shutil.which("ffmpeg"),
                        "ffprobe": shutil.which("ffprobe"),
                        "max_input_bytes": 1024 * 1024,
                        "max_duration_seconds": 30,
                    }
                },
            }
        )
    )
    config.chmod(0o600)
    return config, source, output


def test_standalone_cli_creates_verified_clip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "input.wav"
    make_source(source)
    assert (
        main(
            [
                "audio",
                "clip",
                "--input",
                str(source),
                "--destination",
                str(tmp_path / "clip.wav"),
                "--start",
                "0.1",
                "--end",
                "0.3",
                "--store",
                str(tmp_path / "evidence"),
                "--output",
                "json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["tool_id"] == "capicove.audio.clip"
    assert result["result"]["output"]["frame_count"] == 1600
    assert str(tmp_path) not in json.dumps(result)


def test_clip_capabilities_requires_no_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audio", "capabilities", "--component", "clip", "--output", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["available"] and report["component"] == "capicove.audio.clip"
    assert report["network_access"] is False
    assert report["cancellation"] == "not_supported"
    assert report["ffprobe_version"].startswith("ffprobe version")
    assert report["fd_protocols"] == {
        "ffprobe_input": True,
        "ffmpeg_input": True,
        "ffmpeg_output": True,
    }
    assert report["platform"]["python_minimum"] == "3.11"
    assert all(report["filesystem_primitives"].values())


def test_missing_clip_backend_returns_unavailable(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "audio",
                "capabilities",
                "--component",
                "clip",
                "--ffmpeg",
                "/__capicove_missing__/ffmpeg",
                "--output",
                "json",
            ]
        )
        == 1
    )
    assert not json.loads(capsys.readouterr().out)["available"]


def test_host_catalog_creates_no_directories_or_subprocesses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    config, _, output = configuration(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("catalog cannot execute media")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert main(["mcp", "catalog", "--config", str(config)]) == 0
    tools = json.loads(capsys.readouterr().out)["tools"]
    assert tools[0]["name"] == "capicove_audio_clip"
    assert tools[0]["annotations"]["idempotentHint"] is False
    assert not (output.parent / "staging").exists()
    assert not (output.parent / "evidence").exists()


def test_mcp_clips_and_rejects_destinations_outside_authority(tmp_path: Path) -> None:
    config, source, output = configuration(tmp_path)
    arguments = {
        "source": {"type": "path", "path": str(source)},
        "destination": {"type": "path", "path": str(output / "clip.wav")},
        "start_seconds": 0.1,
        "end_seconds": 0.3,
    }
    with compose_mcp_host(load_mcp_host_config(config)) as bindings:
        result = invoke_binding(bindings[0], arguments)
        assert result["status"] == "succeeded"
        body = result["result"]
        assert isinstance(body, dict)
        assert body["output"]["frame_count"] == 1600
        arguments["destination"] = {"type": "path", "path": str(tmp_path / "outside.wav")}
        denied = invoke_binding(bindings[0], arguments)
        error = denied["error"]
        assert isinstance(error, dict)
        assert denied["status"] == "rejected" and error["code"] == "permission_denied"
    assert not (tmp_path / "outside.wav").exists()


def test_configured_cli_uses_host_authority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, source, output = configuration(tmp_path)
    assert (
        main(
            [
                "audio",
                "clip",
                "--runtime-config",
                str(config),
                "--input",
                str(source),
                "--destination",
                str(output / "cli.wav"),
                "--start",
                "0.2",
                "--end",
                "0.4",
                "--output",
                "json",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["result"]["artifact"]["destination_name"] == "cli.wav"
    )


def test_expired_mcp_deadline_starts_no_staging_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta

    config, source, output = configuration(tmp_path)
    arguments = {
        "source": {"type": "path", "path": str(source)},
        "destination": {"type": "path", "path": str(output / "clip.wav")},
        "start_seconds": 0,
        "end_seconds": 0.1,
    }
    with compose_mcp_host(load_mcp_host_config(config)) as bindings:
        binding = bindings[0]
        context = replace(
            binding.context_factory(binding.spec), deadline=datetime.now(UTC) - timedelta(seconds=1)
        )

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("expired invocation started a child process")

        monkeypatch.setattr(subprocess, "Popen", forbidden)
        result = binding.handler(context, arguments)
    assert result.error and result.error.code.value == "timeout"
    assert not (output.parent / "staging").exists()


def test_host_rejects_selected_stream_over_duration_limit(tmp_path: Path) -> None:
    config, source, output = configuration(tmp_path)
    body = json.loads(config.read_text())
    body["capabilities"]["audio"]["max_duration_seconds"] = 1
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\x10\x01" * 16000)
    config.write_text(json.dumps(body))
    with compose_mcp_host(load_mcp_host_config(config)) as bindings:
        result = invoke_binding(
            bindings[0],
            {
                "source": {"type": "path", "path": str(source)},
                "destination": {"type": "path", "path": str(output / "clip.wav")},
                "start_seconds": 0,
                "end_seconds": 0.1,
            },
        )
    assert result["status"] == "failed"
    assert not (output / "clip.wav").exists()
