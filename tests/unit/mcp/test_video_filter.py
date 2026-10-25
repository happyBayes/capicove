import base64
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

import pytest

from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolResult, ToolStatus
from capicove.mcp.video_filter import (
    MAX_INLINE_VIDEO_BYTES,
    MCP_VIDEO_FILTER_NAME,
    decode_mcp_arguments,
    encode_mcp_result,
    register_video_filter_mcp,
)
from capicove.tools.video_filter import VIDEO_FILTER_SPEC


class FakeHandler:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls: list[tuple[InvocationContext, Mapping[str, object]]] = []

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        self.calls.append((context, arguments))
        return self.result


class FakeServer:
    def __init__(self) -> None:
        self.name: str | None = None
        self.input_schema: Mapping[str, object] | None = None
        self.callback: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[
        [Callable[[Mapping[str, object]], Mapping[str, object]]],
        Callable[[Mapping[str, object]], Mapping[str, object]],
    ]:
        del description
        self.name = name
        self.input_schema = input_schema

        def decorator(
            callback: Callable[[Mapping[str, object]], Mapping[str, object]],
        ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
            self.callback = callback
            return callback

        return decorator


def _arguments(payload: bytes = b"video") -> dict[str, object]:
    return {
        "sources": [
            {
                "item_id": "item-1",
                "source": {
                    "type": "bytes",
                    "base64": base64.b64encode(payload).decode("ascii"),
                },
            }
        ],
        "criteria": [
            {
                "type": "metadata",
                "criterion_id": "metadata",
                "min_duration_seconds": 1.0,
            }
        ],
    }


def _result(status: ToolStatus = ToolStatus.SUCCEEDED) -> ToolResult:
    error = (
        ToolError(ToolErrorCode.UNAVAILABLE, "backend unavailable", False)
        if status is ToolStatus.FAILED
        else None
    )
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.filter",
        tool_version=1,
        status=status,
        result={
            "status": status.value,
            "keep": True if status is ToolStatus.SUCCEEDED else None,
            "items": [],
            "criteria": [],
            "warnings": [],
            "failures": [],
            "aggregation": "all",
            "lineage": {"trace_id": "trace-1"},
        },
        evidence_refs=("source-1",),
        error=error,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def test_decode_mcp_arguments_validates_each_inline_source() -> None:
    arguments = _arguments()

    assert decode_mcp_arguments(arguments) == arguments

    with pytest.raises(ValueError, match="Artifact"):
        decode_mcp_arguments(_arguments(b"x" * (MAX_INLINE_VIDEO_BYTES + 1)))


def test_decode_mcp_arguments_rejects_malformed_base64_without_echo() -> None:
    arguments = _arguments()
    sources = arguments["sources"]
    assert isinstance(sources, list)
    sources[0]["source"] = {"type": "bytes", "base64": "sensitive%%%"}

    with pytest.raises(ValueError) as captured:
        decode_mcp_arguments(arguments)

    assert "sensitive" not in str(captured.value)


def test_encode_mcp_result_preserves_canonical_filter_envelope() -> None:
    encoded = encode_mcp_result(_result())

    assert encoded["tool_id"] == "capicove.video.filter"
    assert encoded["status"] == "succeeded"
    assert encoded["evidence_refs"] == ["source-1"]


def test_registration_uses_canonical_name_schema_and_handler() -> None:
    server = FakeServer()
    handler = FakeHandler(_result())

    register_video_filter_mcp(server, handler)

    assert server.name == MCP_VIDEO_FILTER_NAME == "capicove_video_filter"
    assert server.input_schema == VIDEO_FILTER_SPEC.input_schema
    assert server.callback is not None
    assert server.callback(_arguments())["status"] == "succeeded"
    assert len(handler.calls) == 1
