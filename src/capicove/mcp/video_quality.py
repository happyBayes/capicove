"""Lazy standalone MCP translation for the canonical video quality Tool."""

from collections.abc import Callable, Mapping
from typing import Protocol

from capicove.contracts import ToolHandler
from capicove.mcp.bindings import McpToolBinding, default_context, encode_result, output_schema
from capicove.tools.tracing import invoke_one_step_tool
from capicove.tools.video_quality import VIDEO_QUALITY_SPEC
from capicove.traces import TraceRecorder


MCP_VIDEO_QUALITY_NAME = "capicove_video_quality"
MCP_VIDEO_QUALITY_OUTPUT_SCHEMA = output_schema(VIDEO_QUALITY_SPEC)
QualityCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


class QualityRegistrationServer(Protocol):
    """Small registration interface independent from an optional MCP SDK."""

    def tool(
        self, *, name: str, description: str, input_schema: Mapping[str, object]
    ) -> Callable[[QualityCallback], QualityCallback]: ...


def register_video_quality_mcp(
    server: QualityRegistrationServer,
    handler: ToolHandler,
    *,
    recorder: TraceRecorder | None = None,
) -> QualityCallback:
    """Register one explicitly composed handler, preserving canonical JSON."""

    @server.tool(
        name=MCP_VIDEO_QUALITY_NAME,
        description=VIDEO_QUALITY_SPEC.description,
        input_schema=VIDEO_QUALITY_SPEC.input_schema,
    )
    def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return encode_result(
            invoke_one_step_tool(
                recorder=recorder,
                spec=VIDEO_QUALITY_SPEC,
                handler=handler,
                context=default_context(VIDEO_QUALITY_SPEC),
                arguments=arguments,
            )
        )

    return invoke


def create_video_quality_mcp_server(
    handler: ToolHandler, *, recorder: TraceRecorder | None = None
) -> object:
    """Create an S1 standalone server without altering the workspace host catalog."""
    from capicove.mcp.server import create_bound_mcp_server

    return create_bound_mcp_server(
        (
            McpToolBinding(
                MCP_VIDEO_QUALITY_NAME,
                VIDEO_QUALITY_SPEC,
                handler,
                default_context,
                recorder=recorder,
            ),
        ),
        server_name="capicove",
    )
