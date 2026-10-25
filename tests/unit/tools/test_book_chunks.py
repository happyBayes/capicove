import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.books import BookService, LocalBookWorkspace
from capicove.contracts import InvocationContext, ToolStatus
from tests.unit.cli.test_book import _Parser


def _prepared(tmp_path: Path) -> tuple[LocalBookWorkspace, str]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    return workspace, BookService(workspace).prepare(source, _Parser()).book_id


def test_read_only_preview_tool_uses_same_resolved_configuration(tmp_path: Path) -> None:
    from capicove.tools.book_chunks import CHUNK_TOOL_SPECS, BookChunkToolHandler

    workspace, book_id = _prepared(tmp_path)
    spec = CHUNK_TOOL_SPECS["chunk-preview"]
    context = InvocationContext(
        "request",
        "trace",
        "workspace",
        "agent",
        "local",
        datetime.now(UTC) + timedelta(minutes=1),
        spec.required_permissions,
        spec.privacy_class,
    )
    result = BookChunkToolHandler("chunk-preview")(
        context,
        {
            "workspace": str(workspace.root),
            "book_id": book_id,
            "chunking": {"preset": "technical-code", "target_core_tokens": 700},
        },
    )
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    resolved = result.result["resolved_config"]
    assert isinstance(resolved, dict)
    assert resolved["target_core_tokens"] == 700
    assert "filesystem:write" not in spec.required_permissions
    assert not list(workspace.root.rglob("run.json"))
    invalid = BookChunkToolHandler("chunk-preview")(
        context,
        {
            "workspace": str(workspace.root),
            "book_id": book_id,
            "config_file": "/private/secret.toml",
        },
    )
    assert invalid.status is ToolStatus.FAILED


def test_real_mcp_server_lists_and_executes_chunk_preview(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from mcp import types

    from capicove.mcp.book_chunks import create_book_chunk_mcp_server

    workspace, book_id = _prepared(tmp_path)
    server = create_book_chunk_mcp_server(workspace.root)

    async def run() -> Any:
        listing = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest())
        assert "capicove_book_chunk_preview" in {tool.name for tool in listing.root.tools}
        return await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="capicove_book_chunk_preview",
                    arguments={
                        "workspace": str(workspace.root),
                        "book_id": book_id,
                        "chunking": {"target_core_tokens": 700, "front_matter_policy": "preserve"},
                    },
                )
            )
        )

    response = asyncio.run(run())
    assert response.root.isError is False
    assert response.root.structuredContent["result"]["resolved_config"]["target_core_tokens"] == 700
    assert (
        response.root.structuredContent["result"]["resolved_config"]["front_matter_policy"]
        == "preserve"
    )


def test_canonical_chunk_tools_create_read_and_enforce_permissions(tmp_path: Path) -> None:
    from dataclasses import replace

    from capicove.tools.book_chunks import (
        CHUNK_TOOL_SPECS,
        BookChunkToolHandler,
        register_book_chunk_tools,
    )
    from capicove.tools.registry import ToolRegistry

    workspace, book_id = _prepared(tmp_path)
    spec = CHUNK_TOOL_SPECS["chunk"]
    context = InvocationContext(
        "request",
        "trace",
        "workspace",
        "agent",
        "local",
        datetime.now(UTC) + timedelta(minutes=1),
        spec.required_permissions,
        spec.privacy_class,
    )
    args: dict[str, object] = {"workspace": str(workspace.root), "book_id": book_id}
    result = BookChunkToolHandler("chunk")(context, args)
    assert result.status is ToolStatus.SUCCEEDED and result.result is not None
    args["run_id"] = result.result["run_id"]
    chunks = BookChunkToolHandler("chunks")(context, {**args, "limit": 1})
    assert chunks.status is ToolStatus.SUCCEEDED and chunks.result is not None
    items = chunks.result["items"]
    assert isinstance(items, list)
    parent = BookChunkToolHandler("context")(
        context, {**args, "chunk_id": items[0]["chunk_id"], "max_context_read_tokens": 10}
    )
    assert parent.status is ToolStatus.SUCCEEDED and parent.result is not None
    assert parent.result["max_tokens"] == 10
    denied = BookChunkToolHandler("chunk")(replace(context, permissions=frozenset()), args)
    assert denied.status is ToolStatus.FAILED
    for operation, invalid in (
        ("chunks", {**args, "tasks": "yes"}),
        ("chunk", {"workspace": str(workspace.root), "book_id": book_id, "chunking": []}),
        ("chunks", {**args, "workspace": ""}),
    ):
        assert BookChunkToolHandler(operation)(context, invalid).status is ToolStatus.FAILED
    registry = ToolRegistry()
    register_book_chunk_tools(registry)
    with pytest.raises(ValueError, match="unknown"):
        BookChunkToolHandler("bad")


def test_workspace_bound_tools_reject_external_roots_and_tokenizer_paths(tmp_path: Path) -> None:
    from capicove.tools.book_chunks import CHUNK_TOOL_SPECS, BookChunkToolHandler

    workspace, book_id = _prepared(tmp_path)
    spec = CHUNK_TOOL_SPECS["chunk-preview"]
    context = InvocationContext(
        "request",
        "trace",
        "workspace",
        "agent",
        "local",
        datetime.now(UTC) + timedelta(minutes=1),
        spec.required_permissions,
        spec.privacy_class,
    )
    handler = BookChunkToolHandler("chunk-preview", workspace=workspace.root)
    for args in (
        {"workspace": str(tmp_path), "book_id": book_id},
        {
            "workspace": str(workspace.root),
            "book_id": book_id,
            "chunking": {"tokenizer": "local:" + str(tmp_path / "outside.json")},
        },
    ):
        result = handler(context, args)
        assert result.status is ToolStatus.FAILED
        assert result.error is not None and "authorized" in result.error.message


@pytest.mark.slow
def test_mcp_stdio_roundtrip_uses_workspace_bound_cli(tmp_path: Path) -> None:
    import sys

    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    workspace, book_id = _prepared(tmp_path)

    async def run() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "capicove", "book", "chunk-mcp", str(workspace.root)],
        )
        async with (
            asyncio.timeout(20),
            stdio_client(parameters) as (reader, writer),
            ClientSession(reader, writer) as session,
        ):
            await session.initialize()
            listing = await session.list_tools()
            assert {tool.name for tool in listing.tools} == {
                "capicove_book_chunk",
                "capicove_book_chunk_preview",
                "capicove_book_chunks",
                "capicove_book_context",
                "capicove_book_chunk_tasks",
                "capicove_book_chunk_status",
            }
            result = await session.call_tool(
                "capicove_book_chunk_preview",
                {"workspace": str(workspace.root), "book_id": book_id},
            )
            assert result.isError is False and result.structuredContent is not None
            assert result.structuredContent["result"]["dry_run"] is True
            denied = await session.call_tool(
                "capicove_book_chunk_preview", {"workspace": str(tmp_path), "book_id": book_id}
            )
            assert denied.structuredContent is not None
            assert denied.structuredContent["status"] == "failed"

    asyncio.run(run())
