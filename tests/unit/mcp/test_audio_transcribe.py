from __future__ import annotations

import asyncio
import importlib
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from capicove.contracts import InvocationContext, ToolResult, ToolStatus
from capicove.mcp.audio_transcribe import (
    MCP_AUDIO_TRANSCRIBE_NAME,
    MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA,
    create_audio_transcribe_mcp_server,
    decode_audio_transcribe_mcp_arguments,
    encode_audio_transcribe_mcp_result,
    register_audio_transcribe_mcp,
)
from capicove.tools.audio_transcribe import AUDIO_TRANSCRIBE_SPEC


class _Handler:
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


class _Server:
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


async def _resolve(awaitable: Awaitable[Any]) -> Any:
    return await awaitable


def _result() -> ToolResult:
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.audio.transcribe",
        tool_version=2,
        status=ToolStatus.SUCCEEDED,
        result={
            "status": "succeeded",
            "text": "hello",
            "language": "en",
            "segments": [
                {
                    "index": 0,
                    "start_seconds": 0.0,
                    "end_seconds": 1.0,
                    "text": "hello",
                }
            ],
            "warnings": ["UNSELECTED_AUDIO_STREAMS_IGNORED"],
            "evidence": {
                "evidence_id": "sha256:" + "b" * 64,
                "sha256": "b" * 64,
                "size_bytes": 10,
                "media_type": "audio/wav",
                "source_name": "meeting.wav",
            },
            "lineage": {
                "backend": "whisper.cpp",
                "backend_version": "1.7.6",
                "model": "tiny",
                "model_sha256": "a" * 64,
                "vad_model": "silero-v6.2.0",
                "vad_model_sha256": "c" * 64,
                "device": "cpu",
                "precision": "q5_1",
                "audio_stream_selection": {
                    "schema_version": 1,
                    "selected_audio_stream_index": 1,
                    "index_semantics": "zero_based_audio_stream_ordinal",
                    "selection_method": "explicit_index",
                    "preflight_audio_stream_count": 2,
                    "validation_scope": "selected_audio_stream_only",
                },
            },
        },
        evidence_refs=("sha256:" + "b" * 64,),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def test_mcp_accepts_only_authorized_path_representation() -> None:
    arguments = {
        "source": {"type": "path", "path": "/audio/meeting.wav"},
        "language": "auto",
        "audio_stream_index": 1,
    }
    assert decode_audio_transcribe_mcp_arguments(arguments) == arguments

    for invalid in (
        {"source": {"type": "bytes", "base64": "c2VjcmV0"}},
        {"source": {"type": "url", "url": "https://private.example/audio"}},
    ):
        try:
            decode_audio_transcribe_mcp_arguments(invalid)
        except ValueError as error:
            assert "resource path" in str(error)
        else:
            raise AssertionError("inline and remote audio must be rejected")


def test_mcp_registration_and_encoding_preserve_canonical_result() -> None:
    server = _Server()
    handler = _Handler(_result())

    register_audio_transcribe_mcp(server, handler)

    assert server.name == MCP_AUDIO_TRANSCRIBE_NAME == "capicove_audio_transcribe"
    assert server.input_schema == AUDIO_TRANSCRIBE_SPEC.input_schema
    assert server.callback is not None
    encoded = server.callback(
        {
            "source": {"type": "path", "path": "/audio/meeting.wav"},
            "language": "auto",
            "audio_stream_index": 1,
        }
    )
    assert encoded == encode_audio_transcribe_mcp_result(_result())
    assert handler.calls[0][0].permissions == AUDIO_TRANSCRIBE_SPEC.required_permissions
    assert handler.calls[0][1]["audio_stream_index"] == 1


def test_real_mcp_server_lists_the_canonical_audio_tool_schema() -> None:
    pytest.importorskip("mcp", reason="install capicove[mcp]")
    types = importlib.import_module("mcp.types")
    server_module = importlib.import_module("mcp.server")

    server = create_audio_transcribe_mcp_server(_Handler(_result()))

    assert isinstance(server, server_module.Server)
    list_handler = server.request_handlers[types.ListToolsRequest]
    response: Any = asyncio.run(_resolve(list_handler(types.ListToolsRequest())))
    listed = response.root.tools
    assert len(listed) == 1
    assert listed[0].name == MCP_AUDIO_TRANSCRIBE_NAME
    assert listed[0].inputSchema == dict(AUDIO_TRANSCRIBE_SPEC.input_schema)
    assert listed[0].outputSchema == MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA
    properties = MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA["properties"]
    assert isinstance(properties, Mapping)
    assert properties["tool_version"] == {"const": 2}
    Draft202012Validator.check_schema(MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA)
    Draft202012Validator(MCP_AUDIO_TRANSCRIBE_OUTPUT_SCHEMA).validate(
        encode_audio_transcribe_mcp_result(_result())
    )
