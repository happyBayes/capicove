"""The installed-style stdio host serves selected scoped Memory Tools."""

import asyncio
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


pytestmark = pytest.mark.e2e


async def _roundtrip(config: Path) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "capicove.cli", "mcp", "serve", "--config", str(config)],
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {item.name for item in tools.tools} == {
                "capicove_memory_remember_episode",
                "capicove_memory_recall",
            }
            remembered = await session.call_tool(
                "capicove_memory_remember_episode",
                {
                    "scope": "notes",
                    "content": "Alice moved to Paris.",
                    "event_time": "2026-01-01T00:00:00+00:00",
                    "idempotency_key": "alice-paris-1",
                },
            )
            assert remembered.structuredContent is not None
            assert remembered.structuredContent["status"] == "succeeded"
            recalled = await session.call_tool(
                "capicove_memory_recall", {"scope": "notes", "text": "Paris"}
            )
            assert recalled.structuredContent is not None
            assert recalled.structuredContent["status"] == "succeeded"
            denied = await session.call_tool(
                "capicove_memory_recall", {"scope": "foreign", "text": "Paris"}
            )
            assert denied.structuredContent is not None
            assert denied.structuredContent["error"]["code"] == "permission_denied"
            erased = await session.call_tool("capicove_memory_erase_scope", {"scope": "notes"})
            assert erased.isError


@pytest.mark.slow
def test_memory_stdio_roundtrip(tmp_path: Path) -> None:
    config = tmp_path / "host.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "server_name": "capicove-memory",
                "capability": "memory",
                "workspace_id": "workspace-a",
                "actor_id": "owner",
                "environment_id": "local",
                "permissions": ["memory:read", "memory:write"],
                "enabled_tools": [
                    "capicove.memory.remember_episode",
                    "capicove.memory.recall",
                ],
                "trace_store": None,
                "capability_config": {
                    "root": str(tmp_path / "memory"),
                    "scopes": ["notes"],
                },
            }
        )
    )

    asyncio.run(_roundtrip(config))
