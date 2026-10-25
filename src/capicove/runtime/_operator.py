"""Shared result envelopes for operator Tools (RFC-0016 amendment 1)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import NoReturn

from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolSpec,
    ToolStatus,
)


class Rejected(Exception):
    """A sanitized, non-retryable Tool failure raised by an operator handler."""

    def __init__(self, code: ToolErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def operator_handler(
    spec: ToolSpec, body: Callable[[InvocationContext, Mapping[str, object]], dict[str, object]]
) -> ToolHandler:
    """Wrap a body with the permission check and canonical result envelope."""

    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        started = datetime.now(UTC)
        if spec.required_permissions.difference(context.permissions):
            error = Rejected(
                ToolErrorCode.PERMISSION_DENIED,
                "The request is outside the configured operator authority.",
            )
            return _result(spec, context, started, error=error)
        try:
            payload = body(context, arguments)
        except Rejected as error:
            return _result(spec, context, started, error=error)
        return _result(spec, context, started, payload=payload)

    return handler


def _result(
    spec: ToolSpec,
    context: InvocationContext,
    started: datetime,
    *,
    payload: dict[str, object] | None = None,
    error: Rejected | None = None,
) -> ToolResult:
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=spec.tool_id,
        tool_version=spec.contract_version,
        status=ToolStatus.SUCCEEDED if error is None else ToolStatus.FAILED,
        result=payload,
        error=None if error is None else ToolError(error.code, error.message, retryable=False),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


def catalog_only_handler(
    context: InvocationContext, arguments: Mapping[str, object], /
) -> NoReturn:
    """Stand in for a handler in catalog-only composition."""
    raise RuntimeError("catalog-only operator bindings cannot be invoked")
