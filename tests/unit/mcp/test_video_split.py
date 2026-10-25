import base64
import importlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

import pytest

from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolResult, ToolStatus
from capicove.mcp.video_split import (
    MAX_INLINE_VIDEO_BYTES,
    MCP_VIDEO_SPLIT_NAME,
    McpUnavailableError,
    create_mcp_server,
    decode_mcp_arguments,
    encode_mcp_result,
    register_video_split_mcp,
)
from capicove.tools.video_split import VIDEO_SPLIT_SPEC


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
        self.description: str | None = None
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
        self.name = name
        self.description = description
        self.input_schema = input_schema

        def decorator(
            callback: Callable[[Mapping[str, object]], Mapping[str, object]],
        ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
            self.callback = callback
            return callback

        return decorator


def make_result(status: ToolStatus = ToolStatus.SUCCEEDED) -> ToolResult:
    error = (
        ToolError(ToolErrorCode.UNAVAILABLE, "backend unavailable", True)
        if status is ToolStatus.FAILED
        else None
    )
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.split",
        tool_version=1,
        status=status,
        result={
            "status": status.value,
            "original": {"sha256": "a" * 64},
            "segments": [],
            "warnings": [],
            "degraded": status is ToolStatus.FAILED,
            "lineage": {"trace_id": "trace-1"},
            "failure": None,
        },
        evidence_refs=("source-1",),
        error=error,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def make_arguments(payload: bytes = b"video") -> dict[str, object]:
    return {
        "source": {"type": "bytes", "base64": base64.b64encode(payload).decode()},
        "strategy": "keyframe",
        "config": {},
        "output_format": "bytes",
        "backend": "auto",
    }


def test_decode_mcp_arguments_preserves_bounded_base64() -> None:
    arguments = make_arguments()

    decoded = decode_mcp_arguments(arguments)

    assert decoded == arguments


def test_decode_mcp_arguments_rejects_oversized_inline_payload() -> None:
    with pytest.raises(ValueError, match="Artifact"):
        decode_mcp_arguments(make_arguments(b"x" * (MAX_INLINE_VIDEO_BYTES + 1)))


def test_decode_mcp_arguments_rejects_malformed_base64_without_echo() -> None:
    arguments = make_arguments()
    arguments["source"] = {"type": "bytes", "base64": "sensitive%%%"}

    with pytest.raises(ValueError) as captured:
        decode_mcp_arguments(arguments)

    assert "sensitive" not in str(captured.value)


def test_decode_mcp_arguments_passes_artifact_reference_unchanged() -> None:
    arguments = make_arguments()
    arguments["source"] = {"type": "artifact", "ref": "artifact:1", "sha256": "a" * 64}

    assert decode_mcp_arguments(arguments) == arguments


def test_encode_mcp_result_preserves_canonical_envelope() -> None:
    encoded = encode_mcp_result(make_result())

    assert encoded["trace_id"] == "trace-1"
    assert encoded["status"] == "succeeded"
    assert encoded["evidence_refs"] == ["source-1"]
    result = encoded["result"]
    assert isinstance(result, Mapping)
    original = result["original"]
    assert isinstance(original, Mapping)
    assert original["sha256"] == "a" * 64


def test_registration_uses_canonical_name_schema_and_handler() -> None:
    server = FakeServer()
    handler = FakeHandler(make_result())

    register_video_split_mcp(server, handler)

    assert server.name == MCP_VIDEO_SPLIT_NAME == "capicove_video_split"
    assert server.input_schema == VIDEO_SPLIT_SPEC.input_schema
    assert server.callback is not None
    encoded = server.callback(make_arguments())
    assert encoded["status"] == "succeeded"
    assert len(handler.calls) == 1


def test_real_mcp_composition_reports_actionable_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None) -> object:
        if name.split(".", maxsplit=1)[0] == "mcp":
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    with pytest.raises(McpUnavailableError, match=r"capicove\[mcp\]"):
        create_mcp_server(FakeHandler(make_result()))
