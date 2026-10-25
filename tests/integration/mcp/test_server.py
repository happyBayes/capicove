import asyncio
from collections.abc import Awaitable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from capicove.contracts import InvocationContext, ToolResult, ToolStatus
from capicove.mcp.video_split import MCP_VIDEO_SPLIT_OUTPUT_SCHEMA, create_mcp_server
from capicove.tools.video_split import VIDEO_SPLIT_SPEC


pytestmark = pytest.mark.integration


async def resolve(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


class FakeHandler:
    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id="capicove.video.split",
            tool_version=1,
            status=ToolStatus.SUCCEEDED,
            result={
                "status": "succeeded",
                "original": {"type": "bytes", "sha256": "a" * 64},
                "segments": [],
                "warnings": [],
                "degraded": False,
                "lineage": {"trace_id": context.trace_id},
                "failure": None,
            },
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )


def test_real_mcp_server_lists_the_canonical_tool_schema() -> None:
    pytest.importorskip("mcp", reason="install capicove[mcp]")
    from mcp import types
    from mcp.server import Server

    server = create_mcp_server(FakeHandler())

    assert isinstance(server, Server)
    list_handler = server.request_handlers[types.ListToolsRequest]
    response: Any = asyncio.run(resolve(list_handler(types.ListToolsRequest())))
    listed = response.root.tools
    assert len(listed) == 1
    assert listed[0].name == "capicove_video_split"
    assert listed[0].inputSchema == dict(VIDEO_SPLIT_SPEC.input_schema)
    assert listed[0].outputSchema == MCP_VIDEO_SPLIT_OUTPUT_SCHEMA

    call_handler = server.request_handlers[types.CallToolRequest]
    call_response: Any = asyncio.run(
        resolve(
            call_handler(
                types.CallToolRequest(
                    params=types.CallToolRequestParams(
                        name="capicove_video_split",
                        arguments={
                            "source": {"type": "bytes", "base64": "dmlkZW8="},
                            "strategy": "keyframe",
                            "config": {},
                            "output_format": "bytes",
                            "backend": "auto",
                        },
                    )
                )
            )
        )
    )
    assert call_response.root.isError is False
    assert call_response.root.structuredContent["status"] == "succeeded"
