"""Lazy, workspace-bound MCP stdio transport for configurable book chunking."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.books.workspace import LocalBookWorkspace
from capicove.contracts import InvocationContext, ToolSpec
from capicove.mcp.video_split import encode_mcp_result
from capicove.tools.book_chunk_results import BOOK_RESULT_TOOL_SPECS, BookChunkResultToolHandler
from capicove.tools.book_chunks import CHUNK_TOOL_SPECS, BookChunkToolHandler


def create_book_chunk_mcp_server(
    workspace: Path, *, enabled_result_tools: frozenset[str] = frozenset()
) -> Any:
    """Expose exact canonical schemas and reject access outside the host root."""
    from capicove.mcp.bindings import McpToolBinding
    from capicove.mcp.server import create_bound_mcp_server

    writable_ids = frozenset({"capicove.book.chunk-submit", "capicove.book.chunk-publish"})
    if enabled_result_tools - writable_ids:
        raise ValueError("unknown Book result Tool selection")
    if enabled_result_tools:
        selected_path = workspace.expanduser().absolute()
        if any(part.is_symlink() for part in (selected_path, *selected_path.parents)):
            raise ValueError("writable Book workspace must not use symlinks")
        if os.name == "posix":
            if selected_path.stat().st_mode & 0o077:
                raise ValueError("writable Book workspace must be owner-private")
        elif os.name == "nt":
            from capicove._windows_files import validate_windows_path

            validate_windows_path(selected_path, directory=True, private_acl=True)
        else:
            raise ValueError("writable Book workspace requires secure file APIs")
    book_workspace = LocalBookWorkspace.open(workspace)
    root = book_workspace.root
    wiki_service = None
    if "capicove.book.chunk-publish" in enabled_result_tools:
        from capicove.wiki import WikiService

        wiki_service = WikiService(root)
        wiki_service.require_chunk_publication_ready()

    def context_factory(spec: ToolSpec) -> InvocationContext:
        return InvocationContext(
            str(uuid.uuid4()),
            str(uuid.uuid4()),
            "book-mcp",
            "mcp-client",
            "local",
            datetime.now(UTC) + timedelta(seconds=spec.timeout_seconds),
            spec.required_permissions,
            spec.privacy_class,
        )

    bindings = [
        McpToolBinding(
            "capicove_book_" + operation.replace("-", "_"),
            spec,
            BookChunkToolHandler(operation, workspace=root),
            context_factory,
            encode_result=encode_mcp_result,
            advertise_output_schema=False,
        )
        for operation, spec in CHUNK_TOOL_SPECS.items()
    ]
    for operation in ("chunk-tasks", "chunk-status"):
        bindings.append(
            McpToolBinding(
                "capicove_book_" + operation.replace("-", "_"),
                BOOK_RESULT_TOOL_SPECS[operation],
                BookChunkResultToolHandler(operation, workspace=book_workspace),
                context_factory,
                encode_result=encode_mcp_result,
                advertise_output_schema=False,
            )
        )
    for tool_id in sorted(enabled_result_tools):
        operation = tool_id.rsplit(".", 1)[1]
        bindings.append(
            McpToolBinding(
                "capicove_book_" + operation.replace("-", "_"),
                BOOK_RESULT_TOOL_SPECS[operation],
                BookChunkResultToolHandler(
                    operation, workspace=book_workspace, wiki_service=wiki_service
                ),
                context_factory,
                encode_result=encode_mcp_result,
                advertise_output_schema=False,
            )
        )
    return create_bound_mcp_server(tuple(bindings), server_name="capicove-book-chunks")


def run_stdio(workspace: Path, *, enabled_result_tools: frozenset[str] = frozenset()) -> None:
    """Run the optional SDK using stdout exclusively for protocol messages."""
    from capicove.mcp.stdio import run_stdio as serve_stdio

    serve_stdio(create_book_chunk_mcp_server(workspace, enabled_result_tools=enabled_result_tools))
