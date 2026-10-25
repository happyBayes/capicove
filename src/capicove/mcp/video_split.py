"""Lazy MCP registration and transport mapping for video splitting."""

from __future__ import annotations

import base64
import binascii
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Protocol

from capicove.contracts import InvocationContext, ToolHandler, ToolResult
from capicove.tools.tracing import invoke_one_step_tool
from capicove.tools.video_split import VIDEO_SPLIT_SPEC
from capicove.traces import TraceRecorder


MCP_VIDEO_SPLIT_NAME = "capicove_video_split"
MAX_INLINE_VIDEO_BYTES = 4 * 1024 * 1024
MCP_VIDEO_SPLIT_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "request_id",
        "trace_id",
        "tool_id",
        "tool_version",
        "status",
        "result",
        "evidence_refs",
        "error",
    ],
    "properties": {
        "request_id": {"type": "string"},
        "trace_id": {"type": "string"},
        "tool_id": {"const": "capicove.video.split"},
        "tool_version": {"const": 1},
        "status": {"enum": ["succeeded", "failed", "rejected"]},
        "result": {"anyOf": [dict(VIDEO_SPLIT_SPEC.output_schema), {"type": "null"}]},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "error": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "message", "retryable"],
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "retryable": {"type": "boolean"},
                    },
                },
                {"type": "null"},
            ]
        },
    },
}
McpCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


class McpUnavailableError(RuntimeError):
    """Report that the optional MCP SDK is unavailable."""


class McpRegistrationServer(Protocol):
    """Narrow registration surface consumed by the MCP mapper."""

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[McpCallback], McpCallback]: ...


def decode_mcp_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Validate MCP payload limits while preserving canonical Tool arguments."""
    decoded = dict(arguments)
    source = arguments.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("MCP source must be an object")
    if source.get("type") == "bytes":
        encoded = source.get("base64")
        if not isinstance(encoded, str):
            raise ValueError("MCP inline video must be valid base64")
        if len(encoded) > ((MAX_INLINE_VIDEO_BYTES + 2) // 3) * 4:
            raise ValueError("Use an Artifact reference for video payloads above the inline limit")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("MCP inline video must be valid base64") from error
        if len(payload) > MAX_INLINE_VIDEO_BYTES:
            raise ValueError("Use an Artifact reference for video payloads above the inline limit")
    decoded["source"] = dict(source)
    return decoded


def encode_mcp_result(result: ToolResult) -> dict[str, object]:
    """Encode a canonical ToolResult without changing result semantics."""
    encoded: dict[str, object] = {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": dict(result.result) if result.result is not None else None,
        "evidence_refs": list(result.evidence_refs),
        "error": None,
    }
    if result.error is not None:
        encoded["error"] = {
            "code": result.error.code.value,
            "message": result.error.message,
            "retryable": result.error.retryable,
        }
    return encoded


def register_video_split_mcp(
    server: McpRegistrationServer,
    handler: ToolHandler,
    *,
    recorder: TraceRecorder | None = None,
) -> McpCallback:
    """Register the canonical Tool schema and one transport-only callback."""

    @server.tool(
        name=MCP_VIDEO_SPLIT_NAME,
        description=VIDEO_SPLIT_SPEC.description,
        input_schema=VIDEO_SPLIT_SPEC.input_schema,
    )
    def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return _invoke_handler(handler, arguments, recorder=recorder)

    return invoke


def create_mcp_server(
    handler: ToolHandler,
    *,
    recorder: TraceRecorder | None = None,
) -> object:
    """Create a low-level MCP server with exact canonical JSON schemas."""
    from capicove.mcp.bindings import McpToolBinding, default_context
    from capicove.mcp.server import create_bound_mcp_server

    try:
        return create_bound_mcp_server(
            (
                McpToolBinding(
                    MCP_VIDEO_SPLIT_NAME,
                    VIDEO_SPLIT_SPEC,
                    handler,
                    default_context,
                    decode_mcp_arguments,
                    encode_mcp_result,
                    recorder=recorder,
                ),
            ),
            server_name="capicove",
        )
    except RuntimeError as error:
        raise McpUnavailableError("install capicove[mcp] to create an MCP server") from error


def _invoke_handler(
    handler: ToolHandler,
    arguments: Mapping[str, object],
    *,
    recorder: TraceRecorder | None,
) -> dict[str, object]:
    request_id = str(uuid.uuid4())
    context = InvocationContext(
        request_id=request_id,
        trace_id=str(uuid.uuid4()),
        workspace_id="mcp-server",
        actor_id="mcp-client",
        environment_id="mcp-server",
        deadline=datetime.now(UTC) + timedelta(seconds=VIDEO_SPLIT_SPEC.timeout_seconds),
        permissions=VIDEO_SPLIT_SPEC.required_permissions,
        privacy_class=VIDEO_SPLIT_SPEC.privacy_class,
    )
    result = invoke_one_step_tool(
        recorder=recorder,
        spec=VIDEO_SPLIT_SPEC,
        handler=handler,
        context=context,
        arguments=decode_mcp_arguments(arguments),
    )
    return encode_mcp_result(result)
