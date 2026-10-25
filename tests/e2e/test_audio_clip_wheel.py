"""Qualify installed-wheel clipping through both CLI and real MCP stdio."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.e2e.test_retrieval_wheel import install_wheel
from tests.e2e.wheel_cache import WheelCache
from tests.unit.cli.test_audio_clip import configuration


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.name != "posix" or shutil.which("ffmpeg") is None,
        reason="requires Linux media execution",
    ),
]


async def _clip_over_stdio(
    python: str, config: Path, source: Path, destination: Path
) -> dict[str, object]:
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with (
        stdio_client(parameters) as (reader, writer),
        ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=60)) as session,
    ):
        await session.initialize()
        listing = await session.list_tools()
        assert [tool.name for tool in listing.tools] == ["capicove_audio_clip"]
        assert listing.tools[0].inputSchema["additionalProperties"] is False
        result = await session.call_tool(
            "capicove_audio_clip",
            {
                "source": {"type": "path", "path": str(source)},
                "destination": {"type": "path", "path": str(destination)},
                "start_seconds": 0.1,
                "end_seconds": 0.3,
            },
        )
        assert result.structuredContent is not None
        return result.structuredContent


def test_installed_wheel_clips_over_cli_and_mcp(tmp_path: Path, wheel_cache: WheelCache) -> None:
    python, executable = install_wheel(wheel_cache)
    config, source, output = configuration(tmp_path)
    operator = tmp_path / "operator"
    operator.mkdir()
    result = subprocess.run(
        [
            executable,
            "audio",
            "clip",
            "--input",
            str(source),
            "--destination",
            str(operator / "clip.wav"),
            "--start",
            "0.1",
            "--end",
            "0.3",
            "--store",
            str(operator / "evidence"),
            "--output",
            "json",
        ],
        cwd=operator,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    cli = json.loads(result.stdout)
    mcp = asyncio.run(_clip_over_stdio(python, config, source, output / "mcp.wav"))
    assert mcp["status"] == "succeeded"
    body = mcp["result"]
    assert isinstance(body, dict)
    assert body["output"] == cli["result"]["output"]
    assert body["artifact"]["sha256"] == cli["result"]["artifact"]["sha256"]
    assert (output / "mcp.wav").read_bytes() == (operator / "clip.wav").read_bytes()
