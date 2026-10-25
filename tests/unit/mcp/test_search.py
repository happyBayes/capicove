"""Tests for search MCP transport parity."""

from collections.abc import Callable, Mapping

from capicove.mcp.search import (
    decode_search_mcp_arguments,
    encode_search_mcp_result,
    register_search_mcp,
)
from capicove.tools.search import SearchToolHandler, decode_search_arguments
from tests.unit.tools.test_search import SearchOperation


class Server:
    """Record one MCP registration and expose its callback."""

    callback: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None

    def tool(self, **values: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorate(callback: Callable[..., object]) -> Callable[..., object]:
            self.callback = callback  # type: ignore[assignment]
            return callback

        return decorate


def test_mcp_uses_the_same_closed_arguments_and_result_payload() -> None:
    arguments = {"query": "matching", "top_k": 5}

    assert decode_search_mcp_arguments(arguments) == arguments
    assert decode_search_arguments(
        decode_search_mcp_arguments(arguments)
    ) == decode_search_arguments(arguments)
    payload = {"status": "succeeded", "hits": []}
    assert encode_search_mcp_result(payload) is payload


def test_mcp_registration_invokes_the_canonical_search_handler() -> None:
    server = Server()
    register_search_mcp(server, SearchToolHandler(SearchOperation()))  # type: ignore[arg-type]

    assert server.callback is not None
    result = server.callback({"query": "matching"})
    assert result["tool_id"] == "capicove.search"
    assert result["result"]["query_id"] == "query-1"  # type: ignore[index]
