"""Lazy MCP transport mapping for the canonical search Tool."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import BoundedSemaphore
from typing import Any, Protocol

from capicove.contracts import (
    MAX_REF_BYTES,
    InvocationContext,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolStatus,
)
from capicove.mcp.video_split import McpUnavailableError
from capicove.tools.image_search import (
    MAX_ERROR_MESSAGE_BYTES,
    MAX_EVIDENCE_REFS,
    SEARCH_OUTPUT_SCHEMA,
    SEARCH_SPEC,
    validate_serialized_search_output,
)
from capicove.tools.tracing import invoke_one_step_tool
from capicove.traces import TraceRecorder


MCP_SEARCH_NAME = "capicove_search_image"
MCP_SEARCH_OUTPUT_SCHEMA: dict[str, object] = {
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
        "request_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "trace_id": {"type": "string", "minLength": 1, "maxLength": MAX_REF_BYTES},
        "tool_id": {"const": "capicove.search.image"},
        "tool_version": {"const": 1},
        "status": {"enum": ["succeeded", "failed", "rejected"]},
        "result": {"anyOf": [SEARCH_OUTPUT_SCHEMA, {"type": "null"}]},
        "evidence_refs": {
            "type": "array",
            "maxItems": MAX_EVIDENCE_REFS,
            "uniqueItems": True,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_REF_BYTES,
            },
        },
        "error": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "message", "retryable"],
                    "properties": {
                        "code": {"enum": [code.value for code in ToolErrorCode]},
                        "message": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_ERROR_MESSAGE_BYTES,
                        },
                        "retryable": {"type": "boolean"},
                    },
                },
                {"type": "null"},
            ]
        },
    },
}
McpCallback = Callable[[Mapping[str, object]], Mapping[str, object]]


class McpInvocationContextFactory(Protocol):
    """Create authenticated invocation context from startup-owned state."""

    def create(
        self,
        *,
        tool_id: str,
        deadline_cap: datetime,
    ) -> InvocationContext:
        """Return a context whose candidate deadline is no later than the cap."""


class McpRegistrationServer(Protocol):
    """Narrow repository-compatible MCP registration surface."""

    def tool(
        self,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, object],
    ) -> Callable[[McpCallback], McpCallback]: ...


def register_search_mcp(
    server: McpRegistrationServer,
    handler: ToolHandler,
    *,
    context_factory: McpInvocationContextFactory,
    recorder: TraceRecorder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> McpCallback:
    """Register one transport-only translation of ``capicove.search``."""
    effective_clock = clock or _utc_now

    @server.tool(
        name=MCP_SEARCH_NAME,
        description=SEARCH_SPEC.description,
        input_schema=SEARCH_SPEC.input_schema,
    )
    def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
        return _invoke_search(
            handler,
            arguments,
            context_factory=context_factory,
            recorder=recorder,
            clock=effective_clock,
        )

    return invoke


def create_search_mcp_server(
    handler: ToolHandler,
    *,
    context_factory: McpInvocationContextFactory,
    recorder: TraceRecorder | None = None,
    clock: Callable[[], datetime] | None = None,
) -> object:
    """Lazily create a low-level MCP server for canonical search."""
    try:
        server_module = importlib.import_module("mcp.server")
        types_module = importlib.import_module("mcp.types")
    except ImportError as error:
        raise McpUnavailableError("install capicove[mcp] to create an MCP server") from error
    effective_clock = clock or _utc_now
    server: Any = server_module.Server("capicove")
    admission = BoundedSemaphore(4)
    pending: set[asyncio.Task[dict[str, object]]] = set()

    def invoke_worker(arguments: Mapping[str, object]) -> dict[str, object]:
        try:
            return _invoke_search(
                handler,
                arguments,
                context_factory=context_factory,
                recorder=recorder,
                clock=effective_clock,
            )
        finally:
            admission.release()

    def worker_done(task: asyncio.Task[dict[str, object]]) -> None:
        pending.discard(task)
        if not task.cancelled():
            task.exception()

    async def list_tools() -> list[Any]:
        return [
            types_module.Tool(
                name=MCP_SEARCH_NAME,
                description=SEARCH_SPEC.description,
                inputSchema=dict(SEARCH_SPEC.input_schema),
                outputSchema=MCP_SEARCH_OUTPUT_SCHEMA,
            )
        ]

    async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, object]:
        if name != MCP_SEARCH_NAME:
            raise ValueError(f"Unknown MCP tool: {name}")
        if not admission.acquire(blocking=False):
            raise RuntimeError("Image search concurrency limit reached; retry later.")
        task = asyncio.create_task(asyncio.to_thread(invoke_worker, dict(arguments)))
        pending.add(task)
        task.add_done_callback(worker_done)
        # Cancellation stops waiting, not the synchronous operation. Keep its
        # admission slot and leases until it exits under its existing deadline.
        return await asyncio.shield(task)

    server.list_tools()(list_tools)
    server.call_tool()(call_tool)
    return server


async def serve_search_stdio(
    handler: ToolHandler,
    *,
    context_factory: McpInvocationContextFactory,
) -> None:
    """Serve one startup-authorized local principal, with no network listener."""
    try:
        stdio_module = importlib.import_module("mcp.server.stdio")
    except ImportError as error:
        raise McpUnavailableError("install capicove[mcp] to serve image search") from error
    server: Any = create_search_mcp_server(handler, context_factory=context_factory)
    async with stdio_module.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def encode_search_mcp_result(
    result: ToolResult,
    *,
    expected_context: InvocationContext | None = None,
) -> dict[str, object]:
    """Hand-serialize one bounded canonical Tool envelope."""
    if not isinstance(result, ToolResult):
        raise TypeError("result must be a ToolResult")
    if result.tool_id != SEARCH_SPEC.tool_id or result.tool_version != SEARCH_SPEC.contract_version:
        raise ValueError("Tool result identity does not match canonical search")
    if not isinstance(result.status, ToolStatus):
        raise ValueError("Tool result status is not canonical")
    if expected_context is not None and (
        result.request_id != expected_context.request_id
        or result.trace_id != expected_context.trace_id
    ):
        raise ValueError("Tool result invocation identity does not match MCP context")
    request_id = _bounded_string(result.request_id, "request_id", MAX_REF_BYTES)
    trace_id = _bounded_string(result.trace_id, "trace_id", MAX_REF_BYTES)
    if len(result.evidence_refs) > MAX_EVIDENCE_REFS:
        raise ValueError("evidence_refs exceeds its limit")
    evidence_refs = [
        _bounded_string(value, "evidence_ref", MAX_REF_BYTES) for value in result.evidence_refs
    ]
    if len(set(evidence_refs)) != len(evidence_refs):
        raise ValueError("evidence_refs must not contain duplicates")

    serialized_result: object = None
    if result.result is not None:
        serialized_result = validate_serialized_search_output(result.result)

    serialized_error: dict[str, object] | None = None
    if result.error is not None:
        if not isinstance(result.error.code, ToolErrorCode):
            raise ValueError("Tool error code is not canonical")
        serialized_error = {
            "code": result.error.code.value,
            "message": _bounded_string(
                result.error.message,
                "error message",
                MAX_ERROR_MESSAGE_BYTES,
            ),
            "retryable": result.error.retryable,
        }

    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": serialized_result,
        "evidence_refs": evidence_refs,
        "error": serialized_error,
    }


def _invoke_search(
    handler: ToolHandler,
    arguments: Mapping[str, object],
    *,
    context_factory: McpInvocationContextFactory,
    recorder: TraceRecorder | None,
    clock: Callable[[], datetime],
) -> dict[str, object]:
    if not isinstance(arguments, Mapping):
        raise TypeError("MCP search arguments must be an object")
    now = clock()
    _require_aware("clock", now)
    deadline_cap = now + timedelta(seconds=SEARCH_SPEC.timeout_seconds)
    context = context_factory.create(
        tool_id=SEARCH_SPEC.tool_id,
        deadline_cap=deadline_cap,
    )
    if not isinstance(context, InvocationContext):
        raise TypeError("context factory must return InvocationContext")
    _require_aware("deadline", context.deadline)
    if context.deadline <= now:
        raise ValueError("context deadline must be in the future")
    if context.deadline > deadline_cap:
        context = replace(context, deadline=deadline_cap)
    result = invoke_one_step_tool(
        recorder=recorder,
        spec=SEARCH_SPEC,
        handler=handler,
        context=context,
        arguments=arguments,
    )
    return encode_search_mcp_result(result, expected_context=context)


def _bounded_string(value: object, name: str, maximum_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    if len(value) > maximum_bytes or len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its byte limit")
    return value


def _require_aware(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "MCP_SEARCH_NAME",
    "MCP_SEARCH_OUTPUT_SCHEMA",
    "McpInvocationContextFactory",
    "McpRegistrationServer",
    "create_search_mcp_server",
    "encode_search_mcp_result",
    "register_search_mcp",
]
