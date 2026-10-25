"""MCP transport mapping for the canonical image Tools."""

from __future__ import annotations

import importlib
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from capicove.contracts import InvocationContext, ToolHandler, ToolResult
from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
from capicove.tools.tracing import invoke_one_step_tool
from capicove.traces import TraceRecorder


MCP_IMAGE_TOOL_NAMES = {
    name: f"capicove_{spec.tool_id.removeprefix('capicove.').replace('.', '_').replace('-', '_')}"
    for name, spec in IMAGE_TOOL_SPECS_BY_NAME.items()
}
McpCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


def image_mcp_output_schema(tool_name: str) -> dict[str, object]:
    """Build the standard MCP envelope for one catalog image Tool."""
    spec = IMAGE_TOOL_SPECS_BY_NAME[tool_name]
    return {
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
            "tool_id": {"const": spec.tool_id},
            "tool_version": {"const": spec.contract_version},
            "status": {"enum": ["succeeded", "failed", "rejected"]},
            "result": {"anyOf": [dict(spec.output_schema), {"type": "null"}]},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
            "error": {"type": ["object", "null"]},
        },
    }


class McpRegistrationServer(Protocol):
    """Narrow registration surface consumed by the image MCP mapper."""

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[McpCallback], McpCallback]: ...


def encode_image_mcp_result(result: ToolResult) -> dict[str, object]:
    """Encode a canonical ToolResult without changing image semantics."""
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": dict(result.result) if result.result is not None else None,
        "evidence_refs": list(result.evidence_refs),
        "error": (
            {
                "code": result.error.code.value,
                "message": result.error.message,
                "retryable": result.error.retryable,
            }
            if result.error is not None
            else None
        ),
    }


def register_image_mcp_tools(
    server: McpRegistrationServer,
    handlers: Mapping[str, ToolHandler],
    *,
    recorder: TraceRecorder | None = None,
) -> dict[str, McpCallback]:
    """Register all catalog image tools with explicitly supplied handlers."""
    missing = set(IMAGE_TOOL_SPECS_BY_NAME).difference(handlers)
    if missing:
        raise ValueError(f"missing image MCP handlers: {sorted(missing)}")
    callbacks: dict[str, McpCallback] = {}
    for name, spec in IMAGE_TOOL_SPECS_BY_NAME.items():
        handler = handlers[name]

        @server.tool(
            name=MCP_IMAGE_TOOL_NAMES[name],
            description=spec.description,
            input_schema=spec.input_schema,
        )
        def invoke(
            arguments: Mapping[str, object],
            handler: ToolHandler = handler,
            spec: Any = spec,
        ) -> Mapping[str, object]:
            context = InvocationContext(
                request_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                workspace_id="mcp-server",
                actor_id="mcp-client",
                environment_id="mcp-server",
                deadline=datetime.now(UTC) + timedelta(seconds=spec.timeout_seconds),
                permissions=spec.required_permissions,
                privacy_class=spec.privacy_class,
            )
            result = invoke_one_step_tool(
                recorder=recorder,
                spec=spec,
                handler=handler,
                context=context,
                arguments=dict(arguments),
            )
            return encode_image_mcp_result(result)

        callbacks[name] = invoke
    return callbacks


def create_image_mcp_server(
    handlers: Mapping[str, ToolHandler],
    *,
    recorder: TraceRecorder | None = None,
) -> object:
    """Create a low-level MCP server exposing the complete image catalog."""
    try:
        server_module = importlib.import_module("mcp.server")
        types_module = importlib.import_module("mcp.types")
    except ImportError as error:
        raise RuntimeError("install capicove[mcp] to create an MCP server") from error
    server: Any = server_module.Server("capicove")
    callbacks = register_image_mcp_tools(server, handlers, recorder=recorder)

    async def list_tools() -> list[Any]:
        return [
            types_module.Tool(
                name=MCP_IMAGE_TOOL_NAMES[name],
                description=spec.description,
                inputSchema=dict(spec.input_schema),
                outputSchema=image_mcp_output_schema(name),
            )
            for name, spec in IMAGE_TOOL_SPECS_BY_NAME.items()
        ]

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, object]:
        for catalog_name, mcp_name in MCP_IMAGE_TOOL_NAMES.items():
            if name == mcp_name:
                return dict(callbacks[catalog_name](arguments))
        raise ValueError(f"Unknown MCP tool: {name}")

    server.list_tools()(list_tools)
    server.call_tool()(call_tool)
    return server
