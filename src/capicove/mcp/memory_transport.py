"""MCP adapters use host-authenticated contexts and canonical memory Tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from capicove.contracts import InvocationContext, ToolHandler
from capicove.tools.memory_tools import MEMORY_SPECS, encode_memory_tool_result


if TYPE_CHECKING:
    from capicove.mcp.search import McpCallback, McpRegistrationServer


def register_memory_mcp(
    server: McpRegistrationServer,
    handlers: Mapping[str, ToolHandler],
    *,
    context_factory: Callable[[], InvocationContext],
) -> tuple[McpCallback, ...]:
    callbacks = []

    def register(name: str, handler: ToolHandler) -> McpCallback:
        spec = MEMORY_SPECS[name]

        @server.tool(
            name="capicove_memory_" + name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
            return encode_memory_tool_result(handler(context_factory(), arguments))

        return invoke

    for name, handler in handlers.items():
        callbacks.append(register(name, handler))
    return tuple(callbacks)
