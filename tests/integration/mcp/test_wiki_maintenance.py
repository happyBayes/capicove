"""Default local MCP registration exposes only canonical read operations."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.contracts import InvocationContext
from tests.integration.wiki.test_maintenance_publication import prepared


def test_registration_uses_host_context_and_read_only_catalog(tmp_path: Path) -> None:
    from capicove.mcp.wiki_maintenance import register_wiki_maintenance_mcp

    service, _, _ = prepared(tmp_path)

    class Server:
        def tool(self, **kwargs: Any) -> Any:
            return lambda callback: callback

    context = InvocationContext(
        "r",
        "t",
        "local",
        "operator",
        "mcp",
        datetime.now(UTC) + timedelta(seconds=60),
        frozenset({"wiki:read"}),
    )
    callbacks = register_wiki_maintenance_mcp(
        Server(), {"local": service}, context_factory=lambda: context
    )
    assert "review" not in callbacks and "apply" not in callbacks
    result = callbacks["status"]({"workspace_id": "local"})
    assert result["status"] == "succeeded"
    assert callbacks["status"]({"workspace_id": "other"})["status"] == "rejected"


def test_sdk_client_session_uses_canonical_host_binding(tmp_path: Path) -> None:
    import asyncio

    from mcp.shared.memory import create_connected_server_and_client_session

    from capicove.mcp.wiki_maintenance import create_wiki_maintenance_mcp_server

    service, _, _ = prepared(tmp_path)
    context = InvocationContext(
        "r",
        "t",
        "local",
        "operator",
        "mcp",
        datetime.now(UTC) + timedelta(seconds=60),
        frozenset({"wiki:read"}),
    )
    server = create_wiki_maintenance_mcp_server({"local": service}, context_factory=lambda: context)

    async def run() -> None:
        async with create_connected_server_and_client_session(server) as client:
            catalog = await client.list_tools()
            names = {tool.name for tool in catalog.tools}
            assert "capicove_wiki_maintenance_apply" not in names
            result = await client.call_tool(
                "capicove_wiki_maintenance_status", {"workspace_id": "local"}
            )
            assert not result.isError
            assert result.structuredContent is not None
            assert result.structuredContent["status"] == "succeeded"
            denied = await client.call_tool(
                "capicove_wiki_maintenance_status", {"workspace_id": "other"}
            )
            assert denied.structuredContent is not None
            assert denied.structuredContent["status"] == "rejected"
            assert denied.structuredContent["error"]["code"] == "permission_denied"

    asyncio.run(run())
