"""The standalone MCP quality binding preserves the canonical Tool contract."""

from collections.abc import Callable, Mapping
from typing import Any

from capicove.contracts import InvocationContext, ToolResult, ToolStatus
from capicove.mcp.video_quality import MCP_VIDEO_QUALITY_NAME, register_video_quality_mcp
from capicove.tools.video_quality import VIDEO_QUALITY_SPEC


class Server:
    def tool(self, **kwargs: Any) -> Callable[..., Any]:
        self.options = kwargs
        return lambda callback: callback


class Handler:
    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        self.arguments = arguments
        return ToolResult(
            context.request_id,
            context.trace_id,
            VIDEO_QUALITY_SPEC.tool_id,
            1,
            ToolStatus.SUCCEEDED,
            result={"verdict": "observed"},
        )


def test_registration_and_invocation_do_not_reinterpret_scores() -> None:
    server, handler = Server(), Handler()
    callback = register_video_quality_mcp(server, handler)
    result = callback({"path": "/authorized/video.mp4", "checks": ["streams"]})
    assert server.options["name"] == MCP_VIDEO_QUALITY_NAME
    assert server.options["input_schema"] == VIDEO_QUALITY_SPEC.input_schema
    assert result["result"] == {"verdict": "observed"}
    assert handler.arguments["checks"] == ["streams"]
