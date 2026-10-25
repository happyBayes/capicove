"""RFC-0021 acceptance: selected video Tools over MCP stdio from a clean wheel."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.e2e.test_retrieval_wheel import install_wheel
from tests.e2e.wheel_cache import WheelCache


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.name != "posix", reason="host media handoff is POSIX-only (ADR-0026)"),
    pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs local FFmpeg"),
]


def render(target: Path, source: str) -> None:
    assert FFMPEG is not None
    subprocess.run(
        [
            *(FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source),
            *("-c:v", "libx264", "-g", "10", "-pix_fmt", "yuv420p", str(target)),
        ],
        check=True,
        timeout=60,
    )


async def exercise(python: str, config: Path, clips: Path) -> dict[str, dict[str, Any]]:
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=120)
        ) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert sorted(tool.name for tool in listing.tools) == [
                "capicove_video_deduplicate",
                "capicove_video_filter",
                "capicove_video_split",
            ]
            calls: dict[str, tuple[str, dict[str, Any]]] = {
                "split": (
                    "capicove_video_split",
                    {
                        "source": {"type": "path", "path": str(clips / "moving.mp4")},
                        "strategy": "duration",
                        "config": {
                            "split_duration": 2.0,
                            "overlap_duration": 0.0,
                            "min_last_split_duration": 0.0,
                        },
                        "output_format": "path",
                        "backend": "auto",
                    },
                ),
                "filter": (
                    "capicove_video_filter",
                    {
                        "sources": [
                            {
                                "item_id": "moving",
                                "source": {"type": "path", "path": str(clips / "moving.mp4")},
                            },
                            {
                                "item_id": "still",
                                "source": {"type": "path", "path": str(clips / "still.mp4")},
                            },
                        ],
                        "criteria": [
                            {
                                "type": "motion",
                                "criterion_id": "moves",
                                "min_score": 0.05,
                                "videos_aggregation": "any",
                            }
                        ],
                    },
                ),
                "dedup": (
                    "capicove_video_deduplicate",
                    {
                        "paths": [
                            str(clips / name) for name in ("moving.mp4", "copy.mp4", "still.mp4")
                        ]
                    },
                ),
                "denied": (
                    "capicove_video_deduplicate",
                    {"paths": [str(config)]},
                ),
            }
            results: dict[str, dict[str, Any]] = {}
            for label, (name, arguments) in calls.items():
                response = await session.call_tool(name, arguments)
                assert response.structuredContent is not None, label
                results[label] = response.structuredContent
            return results


@pytest.mark.slow
def test_video_tools_from_installed_wheel(tmp_path: Path, wheel_cache: WheelCache) -> None:
    python, _ = install_wheel(wheel_cache, extra="mcp,video,video-filter,video-dedup")
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    render(clips / "moving.mp4", "testsrc=duration=4:size=64x48:rate=10")
    render(clips / "still.mp4", "color=c=gray:duration=4:size=64x48:rate=10")
    shutil.copyfile(clips / "moving.mp4", clips / "copy.mp4")
    before = {path.name: path.read_bytes() for path in clips.iterdir()}
    config = workspace / "host.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_name": "capicove-local",
                "workspace_id": "workspace",
                "actor_id": "owner",
                "environment_id": "local",
                "workspace_root": str(workspace),
                "profile": "write",
                "permissions": ["artifact:write", "subprocess:execute", "video:read"],
                "enabled_tools": [
                    "capicove.video.deduplicate",
                    "capicove.video.filter",
                    "capicove.video.split",
                ],
                "trace_store": None,
                "capabilities": {
                    "video": {
                        "schema_version": 1,
                        "read_roots": [str(clips)],
                        "staging_root": str(workspace / "staging"),
                        "output_root": str(workspace / "output"),
                        "backend": "ffmpeg",
                        "ffmpeg": FFMPEG,
                        "ffprobe": FFPROBE,
                        "max_input_bytes": 64 * 1024 * 1024,
                    }
                },
                "network": {"enabled": False, "source_hosts": []},
            }
        )
    )
    config.chmod(0o600)

    results = asyncio.run(exercise(python, config, clips))

    split = results["split"]["result"]
    assert results["split"]["status"] == "succeeded"
    assert split["original"]["sha256"] == hashlib.sha256(before["moving.mp4"]).hexdigest()
    assert [round(item["end_seconds"]) for item in split["segments"]] == [2, 4]
    assert all(Path(item["path"]).parent == workspace / "output" for item in split["segments"])
    motion = results["filter"]["result"]["criteria"][0]["video_results"]
    assert {item["item_id"]: item["passed"] for item in motion} == {"moving": True, "still": False}
    groups = results["dedup"]["result"]["groups"]
    assert [group["paths"] for group in groups] == [
        [str(clips / "copy.mp4"), str(clips / "moving.mp4")]
    ]
    assert results["denied"]["error"]["code"] == "permission_denied"
    assert {path.name: path.read_bytes() for path in clips.iterdir()} == before
    assert list((workspace / "staging").iterdir()) == []
    assert str(workspace / "staging") not in json.dumps(results)
