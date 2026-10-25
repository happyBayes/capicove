"""MCP transport mapping for local audio transcription."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Protocol

from capicove.cli.commands.audio_transcribe import encode_audio_result
from capicove.contracts import InvocationContext, ToolHandler, ToolResult
from capicove.tools.audio_transcribe import AUDIO_TRANSCRIBE_SPEC
from capicove.tools.tracing import invoke_one_step_tool
from capicove.traces import TraceRecorder


MCP_AUDIO_TRANSCRIBE_NAME = "capicove_audio_transcribe"
MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA: dict[str, object] = {
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
        "tool_id": {"const": "capicove.audio.transcribe"},
        "tool_version": {"const": 2},
        "status": {"enum": ["succeeded", "failed", "rejected"]},
        "result": {"anyOf": [dict(AUDIO_TRANSCRIBE_SPEC.output_schema), {"type": "null"}]},
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


class McpRegistrationServer(Protocol):
    """Narrow MCP registration surface consumed by the transport mapper."""

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[McpCallback], McpCallback]: ...


def decode_audio_transcribe_mcp_arguments(
    arguments: Mapping[str, object],
) -> dict[str, object]:
    """Accept only authorized local resource paths; reject inline and remote media."""
    source = arguments.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("type") != "path"
        or set(source) != {"type", "path"}
    ):
        raise ValueError("Use an authorized resource path for audio transcription")
    return dict(arguments)


def encode_audio_transcribe_mcp_result(result: ToolResult) -> dict[str, object]:
    """Preserve the canonical Tool result without transport-specific semantics."""
    return encode_audio_result(result)


def register_audio_transcribe_mcp(
    server: McpRegistrationServer,
    handler: ToolHandler,
    *,
    recorder: TraceRecorder | None = None,
) -> McpCallback:
    """Register the exact canonical schema and one transport-only callback."""

    @server.tool(
        name=MCP_AUDIO_TRANSCRIBE_NAME,
        description=AUDIO_TRANSCRIBE_SPEC.description,
        input_schema=AUDIO_TRANSCRIBE_SPEC.input_schema,
    )
    def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return _invoke_handler(handler, arguments, recorder=recorder)

    return invoke


def create_audio_transcribe_mcp_server(
    handler: ToolHandler,
    *,
    recorder: TraceRecorder | None = None,
) -> object:
    """Create a low-level MCP server with the exact audio Tool schema."""
    from capicove.mcp.bindings import McpToolBinding, default_context
    from capicove.mcp.server import create_bound_mcp_server

    return create_bound_mcp_server(
        (
            McpToolBinding(
                MCP_AUDIO_TRANSCRIBE_NAME,
                AUDIO_TRANSCRIBE_SPEC,
                handler,
                default_context,
                decode_audio_transcribe_mcp_arguments,
                encode_audio_transcribe_mcp_result,
                recorder=recorder,
            ),
        ),
        server_name="capicove",
    )


def _invoke_handler(
    handler: ToolHandler,
    arguments: Mapping[str, object],
    *,
    recorder: TraceRecorder | None,
) -> dict[str, object]:
    context = InvocationContext(
        request_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        workspace_id="mcp-server",
        actor_id="mcp-client",
        environment_id="mcp-server",
        deadline=datetime.now(UTC) + timedelta(seconds=AUDIO_TRANSCRIBE_SPEC.timeout_seconds),
        permissions=AUDIO_TRANSCRIBE_SPEC.required_permissions,
        privacy_class=AUDIO_TRANSCRIBE_SPEC.privacy_class,
    )
    result = invoke_one_step_tool(
        recorder=recorder,
        spec=AUDIO_TRANSCRIBE_SPEC,
        handler=handler,
        context=context,
        arguments=decode_audio_transcribe_mcp_arguments(arguments),
    )
    return encode_audio_transcribe_mcp_result(result)
