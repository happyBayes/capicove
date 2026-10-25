"""MCP mapping for canonical text search."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Protocol

from capicove.contracts import InvocationContext, PrivacyClass, ToolHandler, ToolResult
from capicove.tools.search import CAPICOVE_SEARCH_SPEC


MCP_SEARCH_NAME = "capicove_search"
McpCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


class McpRegistrationServer(Protocol):
    """Narrow MCP registration boundary."""

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[McpCallback], McpCallback]: ...


def decode_search_mcp_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Preserve provider-neutral search arguments for canonical decoding."""
    return dict(arguments)


def encode_search_mcp_result(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Preserve the canonical Tool payload without transport semantics."""
    return payload


def register_search_mcp(server: McpRegistrationServer, handler: ToolHandler) -> McpCallback:
    """Register search with the exact canonical input schema."""

    @server.tool(
        name=MCP_SEARCH_NAME,
        description=CAPICOVE_SEARCH_SPEC.description,
        input_schema=CAPICOVE_SEARCH_SPEC.input_schema,
    )
    def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
        now = datetime.now(UTC)
        result = handler(
            InvocationContext(
                request_id=f"mcp-{now.timestamp()}",
                trace_id=f"mcp-trace-{now.timestamp()}",
                workspace_id="mcp-server",
                actor_id="mcp-client",
                environment_id="mcp-server",
                deadline=now + timedelta(seconds=CAPICOVE_SEARCH_SPEC.timeout_seconds),
                permissions=CAPICOVE_SEARCH_SPEC.required_permissions,
                privacy_class=PrivacyClass.CONFIDENTIAL,
            ),
            decode_search_mcp_arguments(arguments),
        )
        return _encode_tool_result(result)

    return invoke


def _encode_tool_result(result: ToolResult) -> dict[str, object]:
    error = result.error
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": None if result.result is None else dict(result.result),
        "evidence_refs": list(result.evidence_refs),
        "error": None
        if error is None
        else {"code": error.code.value, "message": error.message, "retryable": error.retryable},
    }


__all__ = [
    "MCP_SEARCH_NAME",
    "decode_search_mcp_arguments",
    "encode_search_mcp_result",
    "register_search_mcp",
]
