from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from capicove.contracts import ToolResult, ToolStatus
from capicove.mcp.book import BOOK_MCP_NAMES, register_book_mcp


class _Server:
    def __init__(self) -> None:
        self.callbacks: dict[str, Callable[[Mapping[str, object]], Mapping[str, object]]] = {}

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
        del description, input_schema

        def decorate(
            callback: Callable[[Mapping[str, object]], Mapping[str, object]],
        ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
            self.callbacks[name] = callback
            return callback

        return decorate


class _Handler:
    def __call__(self, context: object, arguments: object, /) -> ToolResult:
        del context, arguments
        now = datetime.now(UTC)
        return ToolResult(
            "request",
            "trace",
            "capicove.book.status",
            1,
            ToolStatus.SUCCEEDED,
            {"publishable": False},
            started_at=now,
            completed_at=now,
        )


def test_mcp_registers_five_book_operations_with_canonical_names() -> None:
    server = _Server()

    callbacks = register_book_mcp(
        server,
        {name: _Handler() for name in ("prepare", "next-task", "submit", "status", "publish")},
    )

    assert set(server.callbacks) == set(BOOK_MCP_NAMES.values())
    encoded = callbacks["status"]({"workspace": "/authorized/books", "book_id": "book:" + "1" * 24})
    assert encoded["tool_id"] == "capicove.book.status"
    assert encoded["result"] == {"publishable": False}
