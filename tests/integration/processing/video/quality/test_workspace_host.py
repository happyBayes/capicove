"""Linux workspace quality integration through real pinned media providers."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import main
from capicove.mcp.bindings import InvalidToolArguments, invoke_binding
from capicove.runtime.executables import fd_paths_supported
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host


pytestmark = pytest.mark.skipif(
    not fd_paths_supported() or not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="requires Linux procfs and real local FFmpeg/FFprobe (ADR-0026)",
)


@pytest.fixture
def host(tmp_path: Path) -> tuple[Path, Path]:
    workspace, clips = tmp_path / "workspace", tmp_path / "clips"
    workspace.mkdir(mode=0o700)
    clips.mkdir()
    media = clips / "lesson.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(media),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    programs = workspace / "programs"
    programs.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        assert executable is not None
        shutil.copyfile(Path(executable).resolve(), programs / name)
        (programs / name).chmod(0o700)
    body = {
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
                "quality_limits": {"threads": 1},
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    config = workspace / "host.json"
    config.write_text(json.dumps(body))
    config.chmod(0o600)
    return config, media


def load(path: Path) -> WorkspaceHostConfiguration:
    result = load_mcp_host_config(path)
    assert isinstance(result, WorkspaceHostConfiguration)
    return result


def test_real_host_report_and_configured_cli_parity(
    host: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, media = host
    with compose_mcp_host(load(config)) as (binding,):
        payload = invoke_binding(binding, {"path": str(media)})
    assert payload["status"] == "succeeded", payload
    report: Any = payload["result"]
    assert report["source"] == {
        "path": str(media),
        "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "size_bytes": media.stat().st_size,
    }
    assert report["coverage"]["video"]["frames"] == 10
    assert report["coverage"]["audio"]["samples"] == 48000
    assert "/proc/" not in json.dumps(payload)
    assert str(config.parent / "staging") not in json.dumps(payload)
    assert list((config.parent / "staging").iterdir()) == []
    assert (
        main(
            [
                "video",
                "quality",
                "--runtime-config",
                str(config),
                "--input",
                str(media),
                "--output",
                "json",
            ]
        )
        == 0
    )
    cli = json.loads(capsys.readouterr().out)
    for key in ("source", "selection", "configuration", "checks", "verdict"):
        assert cli["result"][key] == report[key]


def test_catalog_and_doctor_do_not_decode_or_stage(
    host: tuple[Path, Path], capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = host

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("catalog/readiness must not start a provider")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert main(["mcp", "catalog", "--config", str(config)]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert [t["tool_id"] for t in catalog["tools"]] == ["capicove.video.quality"]
    assert catalog["tools"][0]["annotations"]["readOnlyHint"] is True
    assert main(["setup", "doctor", "--config", str(config)]) == 0
    capsys.readouterr()
    assert not (config.parent / "staging").exists()


def test_authority_schema_and_deadline_fail_closed(host: tuple[Path, Path]) -> None:
    config, media = host
    outside = config.parent / "outside.mkv"
    shutil.copyfile(media, outside)
    link = media.parent / "link.mkv"
    link.symlink_to(outside)
    with compose_mcp_host(load(config)) as (binding,):
        for denied in (outside, link):
            payload = invoke_binding(binding, {"path": str(denied)})
            assert payload["status"] == "failed", payload
            assert payload["error"]["code"] == "permission_denied"  # type: ignore[index]
        for extra in ({"ffmpeg": "/bin/true"}, {"limits": {"threads": 200}}):
            with pytest.raises(InvalidToolArguments):
                invoke_binding(binding, {"path": str(media), **extra})
        context = replace(
            binding.context_factory(binding.spec), deadline=datetime.now(UTC) - timedelta(seconds=1)
        )
        payload = invoke_binding(binding, {"path": str(media)}, context=context)
        assert payload["error"]["code"] == "timeout"  # type: ignore[index]
        assert payload["result"] is None
    assert (
        not (config.parent / "staging").exists()
        or list((config.parent / "staging").iterdir()) == []
    )


def test_replaced_executable_is_never_run(host: tuple[Path, Path]) -> None:
    config, media = host
    marker = config.parent / "executed"
    with compose_mcp_host(load(config)) as (binding,):
        executable = config.parent / "programs" / "ffmpeg"
        executable.unlink()
        executable.write_text(f"#!/bin/sh\ntouch {marker}\n")
        executable.chmod(0o700)
        payload = invoke_binding(binding, {"path": str(media)})
    assert payload["error"]["code"] == "unavailable"  # type: ignore[index]
    assert not marker.exists()


def test_workspace_quality_mcp_stdio(host: tuple[Path, Path]) -> None:
    config, media = host
    mcp = importlib.import_module("mcp")
    stdio_client = importlib.import_module("mcp.client.stdio").stdio_client

    async def exercise() -> None:
        parameters = mcp.StdioServerParameters(
            command=sys.executable,
            args=["-m", "capicove", "mcp", "serve", "--config", str(config)],
            env={"PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        )
        async with (
            stdio_client(parameters) as (reader, writer),
            mcp.ClientSession(
                reader, writer, read_timeout_seconds=timedelta(seconds=30)
            ) as session,
        ):
            await session.initialize()
            listing = await session.list_tools()
            assert [tool.name for tool in listing.tools] == ["capicove_video_quality"]
            response = await session.call_tool(
                "capicove_video_quality", {"path": str(media), "checks": ["decode"]}
            )
            assert response.isError is False
            payload = response.structuredContent
            assert payload is not None and payload["status"] == "succeeded", payload
            assert payload["result"]["coverage"]["video"]["frames"] == 10

    asyncio.run(exercise())


def test_staged_snapshot_survives_source_and_staging_path_replacement(
    host: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.processing.video.quality._adapters.ffmpeg_quality import FfmpegQualityBackend

    config, media = host
    original_sha = hashlib.sha256(media.read_bytes()).hexdigest()
    original_probe = FfmpegQualityBackend.probe
    staging = config.parent / "staging"
    relocated = config.parent / "relocated"
    outside = config.parent / "other"
    outside.mkdir(mode=0o700)

    def probe(self: Any, source: Any, deadline: Any) -> Any:
        # Replace both logical paths after intake but before the real provider runs.
        media.write_bytes(b"replacement, not the authorized snapshot")
        staging.rename(relocated)
        staging.symlink_to(outside, target_is_directory=True)
        return original_probe(self, source, deadline)

    monkeypatch.setattr(FfmpegQualityBackend, "probe", probe)
    with compose_mcp_host(load(config)) as (binding,):
        payload = invoke_binding(binding, {"path": str(media), "checks": ["decode"]})
    assert payload["status"] == "succeeded", payload
    report: Any = payload["result"]
    assert report["source"]["sha256"] == original_sha
    assert report["coverage"]["video"]["frames"] == 10
    assert list(relocated.iterdir()) == []
    assert list(outside.iterdir()) == []


def test_configured_policy_violation_keeps_completed_status(
    host: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, media = host
    policy = config.parent / "policy.json"
    policy.write_text(json.dumps({"min_display_width": 1920}))
    assert (
        main(
            [
                "video",
                "quality",
                "--runtime-config",
                str(config),
                "--input",
                str(media),
                "--policy",
                str(policy),
                "--output",
                "json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded" and payload["result"]["verdict"] == "fail"


def test_configured_trace_uses_host_store_without_staging_locator(
    host: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    config, media = host
    body = json.loads(config.read_text())
    trace = config.parent / "trace.sqlite"
    body["trace_store"] = str(trace)
    config.write_text(json.dumps(body))
    assert (
        main(
            [
                "video",
                "quality",
                "--runtime-config",
                str(config),
                "--input",
                str(media),
                "--checks",
                "decode",
                "--output",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded"
    assert trace.is_file()
    assert b"/proc/" not in trace.read_bytes()
    assert str(config.parent / "staging").encode() not in trace.read_bytes()
