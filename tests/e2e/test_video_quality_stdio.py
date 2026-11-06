"""Exercise the actual standalone media quality MCP protocol without the host."""

import asyncio
import importlib
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

_SERVER = """
import sys
from pathlib import Path
from capicove.tools import build_video_quality_tool_handler
from capicove.mcp import create_video_quality_mcp_server
from capicove.mcp.stdio import run_stdio
handler = build_video_quality_tool_handler('standalone-test', (Path(sys.argv[1]),))
run_stdio(create_video_quality_mcp_server(handler))
"""


def test_standalone_mcp_exposes_and_executes_only_quality(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("local media providers required")
    media = tmp_path / "example.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=10:d=1",
            "-c:v",
            "ffv1",
            str(media),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    mcp = importlib.import_module("mcp")
    stdio_client = importlib.import_module("mcp.client.stdio").stdio_client

    async def exercise() -> None:
        parameters = mcp.StdioServerParameters(
            command=sys.executable, args=["-c", _SERVER, str(tmp_path)]
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
            assert payload is not None
            assert payload["status"] == "succeeded"
            assert payload["result"]["coverage"]["video"]["frames"] == 10
            assert len(payload["result"]["coverage"]["passes"]) == 2

    asyncio.run(exercise())
