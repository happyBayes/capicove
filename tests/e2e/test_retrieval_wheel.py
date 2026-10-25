"""RFC-0018 acceptance: the offline source-to-Search-to-Evidence journey from a clean wheel."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.e2e.wheel_cache import WheelCache
from tests.unit.runtime.test_local_retrieval import PNG_BYTES, retrieval_config_body


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.name != "posix", reason="retrieval is POSIX-only (ADR-0024)"),
]


def install_wheel(wheel_cache: WheelCache, extra: str = "mcp") -> tuple[str, str]:
    """Return the shared environment with comma-separated extras installed."""
    installed = wheel_cache.install(tuple(extra.split(",")))
    return installed.python, installed.executable


def run(executable: str, cwd: Path, *args: str) -> dict[str, object]:
    completed = subprocess.run(
        [executable, *args], cwd=cwd, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, f"{args}\n{completed.stdout}\n{completed.stderr}"
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


async def replay_over_mcp(
    python: str, config: Path, stale_evidence: str
) -> tuple[dict[str, object], dict[str, object]]:
    parameters = StdioServerParameters(
        command=python, args=["-m", "capicove", "mcp", "serve", "--config", str(config)]
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert sorted(tool.name for tool in listing.tools) == [
                "capicove_get_evidence",
                "capicove_search",
            ]
            found = await session.call_tool("capicove_search", {"query": "Lyon"})
            assert found.structuredContent is not None
            hit = found.structuredContent["result"]["hits"][0]
            opened = await session.call_tool(
                "capicove_get_evidence", {"evidence_id": hit["evidence_id"]}
            )
            stale = await session.call_tool(
                "capicove_get_evidence", {"evidence_id": stale_evidence}
            )
            assert stale.structuredContent is not None
            assert stale.structuredContent["status"] == "failed"
            assert opened.structuredContent is not None
            return found.structuredContent, opened.structuredContent


@pytest.mark.slow
def test_offline_retrieval_journey_from_installed_wheel(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    python, capicove = install_wheel(wheel_cache)
    operator = tmp_path / "operator"
    operator.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    notes = tmp_path / "notes"
    notes.mkdir()
    markdown = notes / "travel.md"
    markdown.write_text("# Travel\n\nAlice moved to Paris on January 1.\n", encoding="utf-8")
    (notes / "pixel.png").write_bytes(PNG_BYTES)
    config = workspace / "host.json"
    config.write_text(json.dumps(retrieval_config_body(workspace, notes, workspace / "index")))
    config.chmod(0o600)
    runtime = ("--runtime-config", str(config))
    source_bytes = {path.name: path.read_bytes() for path in notes.iterdir()}

    for name in ("travel.md", "pixel.png"):
        run(
            capicove,
            operator,
            "source",
            "import",
            *runtime,
            "--root",
            "notes",
            "--relative-path",
            name,
        )
    generation_a = run(capicove, operator, "index", "build", *runtime)
    found_a = run(capicove, operator, "search", "Paris", *runtime)
    evidence_a = found_a["result"]["hits"][0]["evidence_id"]  # type: ignore[index]
    opened_a = run(capicove, operator, "get-evidence", evidence_a, *runtime)
    assert "Alice moved to Paris" in opened_a["result"]["metadata"]["text"]  # type: ignore[index]
    assert {path.name: path.read_bytes() for path in notes.iterdir()} == source_bytes

    markdown.write_text("# Travel\n\nAlice moved to Lyon in March.\n", encoding="utf-8")
    run(
        capicove,
        operator,
        "source",
        "import",
        *runtime,
        "--root",
        "notes",
        "--relative-path",
        "travel.md",
    )
    generation_b = run(capicove, operator, "index", "build", *runtime)
    assert generation_b["generation_id"] != generation_a["generation_id"]

    found, opened = asyncio.run(replay_over_mcp(python, config, str(evidence_a)))
    assert found["status"] == "succeeded"
    assert "Lyon" in opened["result"]["metadata"]["text"]  # type: ignore[index]
    assert opened["result"]["metadata"]["generation_id"] == generation_b["generation_id"]  # type: ignore[index]
    cli_found = run(capicove, operator, "search", "Lyon", *runtime)
    assert cli_found["evidence_refs"] == found["evidence_refs"]

    # RFC-0018 amendment 1: retire every source, build an empty generation.
    unsafe = subprocess.run(
        [capicove, "source", "forget", *runtime, "--root", "notes", "--relative-path", "../x.md"],
        cwd=operator,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert unsafe.returncode == 2 and unsafe.stdout == ""
    for name in ("travel.md", "pixel.png"):
        retired = run(
            capicove,
            operator,
            "source",
            "forget",
            *runtime,
            "--root",
            "notes",
            "--relative-path",
            name,
        )
        assert retired["forgotten"] is True
    empty = run(capicove, operator, "index", "build", *runtime)
    assert empty["sources"] == 0 and empty["text_units"] == 0
    assert run(capicove, operator, "search", "Lyon", *runtime)["result"]["hits"] == []  # type: ignore[index]
    stale = subprocess.run(
        [capicove, "get-evidence", cli_found["evidence_refs"][0], *runtime],  # type: ignore[index]
        cwd=operator,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert stale.returncode == 1
    assert markdown.read_text(encoding="utf-8") == "# Travel\n\nAlice moved to Lyon in March.\n"
