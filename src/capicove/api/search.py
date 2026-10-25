"""Optional HTTP boundary for canonical text search."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI

from capicove.contracts import InvocationContext, PrivacyClass, ToolHandler
from capicove.tools.search import CAPICOVE_SEARCH_SPEC, SearchToolHandler, TextSearchOperation


def create_search_app(
    operation: TextSearchOperation,
    *,
    evidence_handler: ToolHandler | None = None,
    evidence_context_factory: Callable[[], InvocationContext] | None = None,
) -> FastAPI:
    """Create an HTTP application over the same canonical search Tool handler."""
    app = FastAPI(title="Capicove Text Search API", version="1.0.0")
    handler = SearchToolHandler(operation)
    if (evidence_handler is None) != (evidence_context_factory is None):
        raise ValueError("evidence handler requires an explicit authenticated context factory")

    if evidence_handler is not None and evidence_context_factory is not None:
        configured_handler = evidence_handler
        configured_context = evidence_context_factory

        @app.post("/v1/evidence")
        def evidence(arguments: dict[str, Any]) -> dict[str, object]:
            result = configured_handler(configured_context(), arguments)
            error = result.error
            return {
                "request_id": result.request_id,
                "trace_id": result.trace_id,
                "tool_id": result.tool_id,
                "tool_version": result.tool_version,
                "status": result.status.value,
                "result": None if result.result is None else dict(result.result),
                "evidence_refs": list(result.evidence_refs),
                "error": None
                if error is None
                else {
                    "code": error.code.value,
                    "message": error.message,
                    "retryable": error.retryable,
                },
            }

    @app.post("/v1/search")
    def search(arguments: dict[str, Any]) -> dict[str, object]:
        now = datetime.now(UTC)
        result = handler(
            InvocationContext(
                request_id=str(uuid.uuid4()),
                trace_id=str(uuid.uuid4()),
                workspace_id="http-api",
                actor_id="http-client",
                environment_id="http-api",
                deadline=now + timedelta(seconds=CAPICOVE_SEARCH_SPEC.timeout_seconds),
                permissions=CAPICOVE_SEARCH_SPEC.required_permissions,
                privacy_class=PrivacyClass.CONFIDENTIAL,
            ),
            arguments,
        )
        error = result.error
        return {
            "request_id": result.request_id,
            "trace_id": result.trace_id,
            "tool_id": result.tool_id,
            "tool_version": result.tool_version,
            "status": result.status.value,
            "result": None if result.result is None else dict(result.result),
            "evidence_refs": list(result.evidence_refs),
            "error": None
            if error is None
            else {
                "code": error.code.value,
                "message": error.message,
                "retryable": error.retryable,
            },
        }

    return app


__all__ = ["create_search_app"]
