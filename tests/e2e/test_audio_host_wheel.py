"""RFC-0022 acceptance: audio import and transcription over MCP from a clean wheel.

The journey test uses fake Whisper executables and qualifies authority and
transport only. ASR quality requires the real-asset test below, which runs only
when an operator provisions pinned local binaries and models through
CAPICOVE_QUALIFY_* variables; without them it reports a blocked qualification.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.e2e.test_retrieval_wheel import install_wheel
from tests.e2e.wheel_cache import WheelCache
from tests.unit.runtime.test_audio_mcp import FAKE_VAD, FAKE_WHISPER


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.name != "posix", reason="host media handoff is POSIX-only (ADR-0026)"),
    pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs local FFmpeg"),
]
_REAL_ASSETS = (
    "CAPICOVE_QUALIFY_WHISPER_CLI",
    "CAPICOVE_QUALIFY_WHISPER_VAD_CLI",
    "CAPICOVE_QUALIFY_ASR_MODEL",
    "CAPICOVE_QUALIFY_VAD_MODEL",
    "CAPICOVE_QUALIFY_SPEECH_WAV",
    "CAPICOVE_QUALIFY_EXPECTED_WORDS",
)


def config_body(
    workspace: Path, clips: Path, whisper: Path, vad: Path, model: Path, vad_model: Path
) -> dict[str, object]:
    def pinned(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    return {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(workspace),
        "profile": "write",
        "permissions": ["artifact:write", "audio:read", "audio:write", "subprocess:execute"],
        "enabled_tools": ["capicove.audio.import", "capicove.audio.transcribe"],
        "trace_store": None,
        "capabilities": {
            "audio": {
                "schema_version": 1,
                "read_roots": [str(clips)],
                "staging_root": str(workspace / "staging"),
                "evidence_root": str(workspace / "evidence"),
                "ffmpeg": FFMPEG,
                "ffprobe": FFPROBE,
                "whisper_cli": str(whisper),
                "whisper_vad_cli": str(vad),
                "model": pinned(model),
                "vad_model": pinned(vad_model),
                "max_input_bytes": 64 * 1024 * 1024,
                "max_duration_seconds": 600,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }


async def run_journey(python: str, config: Path, source: Path) -> dict[str, dict[str, Any]]:
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=300)
        ) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert sorted(tool.name for tool in listing.tools) == [
                "capicove_audio_import",
                "capicove_audio_transcribe",
            ]
            results: dict[str, dict[str, Any]] = {}
            for label, name, arguments in (
                (
                    "import",
                    "capicove_audio_import",
                    {"source": {"type": "path", "path": str(source)}},
                ),
                (
                    "again",
                    "capicove_audio_import",
                    {"source": {"type": "path", "path": str(source)}},
                ),
                (
                    "transcribe",
                    "capicove_audio_transcribe",
                    {"source": {"type": "path", "path": str(source)}, "language": "auto"},
                ),
            ):
                response = await session.call_tool(name, arguments)
                assert response.structuredContent is not None, label
                results[label] = response.structuredContent
            return results


def prepare(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    return workspace, clips


@pytest.mark.slow
def test_audio_host_journey_with_fake_backends(tmp_path: Path, wheel_cache: WheelCache) -> None:
    python, _ = install_wheel(wheel_cache, extra="mcp")
    workspace, clips = prepare(tmp_path)
    assert FFMPEG is not None
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(clips / "tone.wav"),
        ],
        check=True,
        timeout=60,
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls.log"
    for name, template in (("whisper-cli", FAKE_WHISPER), ("whisper-vad", FAKE_VAD)):
        script = tools / name
        script.write_text(textwrap.dedent(template).format(python=sys.executable, calls=str(calls)))
        script.chmod(0o755)
    (tools / "model.bin").write_bytes(b"fake model" * 32)
    (tools / "vad.bin").write_bytes(b"fake vad" * 32)
    config = workspace / "host.json"
    config.write_text(
        json.dumps(
            config_body(
                workspace,
                clips,
                tools / "whisper-cli",
                tools / "whisper-vad",
                tools / "model.bin",
                tools / "vad.bin",
            )
        )
    )
    config.chmod(0o600)

    results = asyncio.run(run_journey(python, config, clips / "tone.wav"))

    digest = hashlib.sha256((clips / "tone.wav").read_bytes()).hexdigest()
    assert results["import"]["result"]["evidence"]["evidence_id"] == f"sha256:{digest}"
    assert results["import"]["result"]["reused"] is False
    assert results["again"]["result"]["reused"] is True
    assert results["transcribe"]["status"] == "succeeded"
    assert results["transcribe"]["result"]["evidence"]["source_name"] == "tone.wav"
    assert str(workspace / "staging") not in json.dumps(results)


@pytest.mark.slow
def test_real_asr_assets_qualification(tmp_path: Path, wheel_cache: WheelCache) -> None:
    missing = [name for name in _REAL_ASSETS if not os.environ.get(name)]
    if missing:
        pytest.skip(
            "BLOCKED: real ASR qualification needs locally provisioned assets: "
            + ", ".join(missing)
        )
    python, _ = install_wheel(wheel_cache, extra="mcp")
    workspace, clips = prepare(tmp_path)
    speech = Path(os.environ["CAPICOVE_QUALIFY_SPEECH_WAV"])
    shutil.copyfile(speech, clips / speech.name)
    config = workspace / "host.json"
    config.write_text(
        json.dumps(
            config_body(
                workspace,
                clips,
                Path(os.environ["CAPICOVE_QUALIFY_WHISPER_CLI"]),
                Path(os.environ["CAPICOVE_QUALIFY_WHISPER_VAD_CLI"]),
                Path(os.environ["CAPICOVE_QUALIFY_ASR_MODEL"]),
                Path(os.environ["CAPICOVE_QUALIFY_VAD_MODEL"]),
            )
        )
    )
    config.chmod(0o600)

    results = asyncio.run(run_journey(python, config, clips / speech.name))

    transcript = results["transcribe"]["result"]
    assert results["transcribe"]["status"] == "succeeded", results["transcribe"]
    text = transcript["text"].casefold()
    for word in os.environ["CAPICOVE_QUALIFY_EXPECTED_WORDS"].split(","):
        assert word.strip().casefold() in text
    starts = [segment["start_seconds"] for segment in transcript["segments"]]
    assert starts == sorted(starts)
    assert transcript["lineage"]["vad_model_sha256"]
