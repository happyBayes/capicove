"""Explicit modality entry points over independent canonical Tool handlers.

Text keeps the stable dev interface. Image requires startup-owned authorization.
Video is not implemented here and never falls back to another modality.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolResult, ToolStatus


if TYPE_CHECKING:
    from capicove.tools.image_search import SearchAuthorizationFactory, SearchServicePort
    from capicove.tools.search import TextSearchOperation


def search_text(
    context: InvocationContext,
    arguments: Mapping[str, object],
    *,
    operation: TextSearchOperation,
) -> ToolResult:
    """Invoke text search without changing its schema or caller permissions."""
    from capicove.tools.search import SearchToolHandler

    return SearchToolHandler(operation)(context, arguments)


def search_image(
    context: InvocationContext,
    arguments: Mapping[str, object],
    *,
    service: SearchServicePort,
    authorization_factory: SearchAuthorizationFactory,
) -> ToolResult:
    """Invoke image search through its mandatory authorization boundary."""
    from capicove.tools.image_search import SearchToolHandler

    return SearchToolHandler(service, authorization_factory)(context, arguments)


def search_video(
    context: InvocationContext,
    arguments: Mapping[str, object],
) -> ToolResult:
    """Reject unsupported video retrieval without initializing a backend."""
    now = datetime.now(UTC)
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id="capicove.search.video",
        tool_version=1,
        status=ToolStatus.REJECTED,
        error=ToolError(ToolErrorCode.UNAVAILABLE, "Video search is not implemented.", False),
        started_at=now,
        completed_at=now,
    )


__all__ = ["search_image", "search_text", "search_video"]
