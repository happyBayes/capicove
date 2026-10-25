"""MCP transport registration for canonical book Tools."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Protocol

from capicove.contracts import InvocationContext, ToolHandler, ToolResult
from capicove.tools.book import BOOK_TOOL_SPECS


BOOK_MCP_NAMES = {
    "prepare": "capicove_book_prepare",
    "next-task": "capicove_book_next_task",
    "submit": "capicove_book_submit",
    "status": "capicove_book_status",
    "publish": "capicove_book_publish",
}
McpCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


class BookMcpRegistrationServer(Protocol):
    """Narrow MCP registration surface used by the book transport."""

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[McpCallback], McpCallback]: ...


def register_book_mcp(
    server: BookMcpRegistrationServer,
    handlers: Mapping[str, ToolHandler],
) -> dict[str, McpCallback]:
    """Register all five canonical book operations on one MCP server."""
    if set(handlers) != set(BOOK_MCP_NAMES):
        raise ValueError("book MCP requires exactly five canonical handlers")
    callbacks: dict[str, McpCallback] = {}
    for operation, name in BOOK_MCP_NAMES.items():
        spec = BOOK_TOOL_SPECS[operation]
        handler = handlers[operation]

        def invoke(
            arguments: Mapping[str, object],
            *,
            bound_handler: ToolHandler = handler,
            bound_operation: str = operation,
        ) -> Mapping[str, object]:
            bound_spec = BOOK_TOOL_SPECS[bound_operation]
            context = InvocationContext(
                str(uuid.uuid4()),
                str(uuid.uuid4()),
                "mcp-server",
                "mcp-client",
                "mcp-server",
                datetime.now(UTC) + timedelta(seconds=bound_spec.timeout_seconds),
                bound_spec.required_permissions,
                bound_spec.privacy_class,
            )
            return encode_book_tool_result(bound_handler(context, arguments))

        callback = server.tool(
            name=name,
            description=spec.description,
            input_schema=spec.input_schema,
        )(invoke)
        callbacks[operation] = callback
    return callbacks


def encode_book_tool_result(result: ToolResult) -> dict[str, object]:
    """Preserve the canonical Tool envelope across MCP transport."""
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": None if result.result is None else dict(result.result),
        "evidence_refs": list(result.evidence_refs),
        "error": None
        if result.error is None
        else {
            "code": result.error.code.value,
            "message": result.error.message,
            "retryable": result.error.retryable,
        },
    }


__all__ = ["BOOK_MCP_NAMES", "encode_book_tool_result", "register_book_mcp"]
