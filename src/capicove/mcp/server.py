"""A closed MCP host that invokes canonical Tools outside the event loop."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from capicove.mcp.bindings import (
    McpToolBinding,
    invoke_binding,
    output_schema,
    validate_bindings,
)


if TYPE_CHECKING:
    from mcp.server import Server


def create_bound_mcp_server(
    bindings: tuple[McpToolBinding, ...], *, server_name: str
) -> Server[Any]:
    """Register exactly the successfully composed Tools, with lazy SDK imports."""
    validate_bindings(bindings)
    if not server_name.strip():
        raise ValueError("MCP server name must not be blank")
    try:
        sdk = importlib.import_module("mcp.server")
        types = importlib.import_module("mcp.types")
        anyio = importlib.import_module("anyio")
        importlib.import_module("jsonschema")
    except ImportError as error:
        raise RuntimeError("install capicove[mcp] to create an MCP server") from error
    server: Any = sdk.Server(server_name)
    catalog = {item.name: item for item in sorted(bindings, key=lambda item: item.name)}
    limiter = anyio.CapacityLimiter(4)

    async def list_tools() -> list[Any]:
        return [
            types.Tool(
                name=item.name,
                description=item.spec.description,
                inputSchema=dict(item.spec.input_schema),
                outputSchema=output_schema(item.spec) if item.advertise_output_schema else None,
                annotations=types.ToolAnnotations(**item.annotations()),
            )
            for item in catalog.values()
        ]

    def invoke(item: McpToolBinding, arguments: Mapping[str, object]) -> dict[str, object]:
        return invoke_binding(item, arguments)

    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        item = catalog.get(name)
        if item is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Unknown MCP Tool")], isError=True
            )
        try:
            # A cancelled request must not abandon a still-running handler before
            # its owned resources can be closed. Tool deadlines remain cooperative.
            with anyio.CancelScope(shield=True):
                result = await anyio.to_thread.run_sync(
                    invoke, item, arguments, limiter=limiter, abandon_on_cancel=False
                )
            return result
        except Exception:
            # SDK defaults include raw validation/exception values in error text.
            # Return a sanitized result before those defaults can expose inputs.
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="MCP Tool invocation failed")],
                isError=True,
            )

    server.list_tools()(list_tools)
    server.call_tool(validate_input=False)(call_tool)
    return cast("Server[Any]", server)
