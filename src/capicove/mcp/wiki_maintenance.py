"""Explicit local MCP binding; the host supplies authenticated context."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from capicove.contracts import InvocationContext
from capicove.tools.wiki_maintenance import MaintenanceToolHandler, encode_result, maintenance_specs
from capicove.wiki.maintenance.service import MaintenanceService


Callback = Callable[[Mapping[str, object]], Mapping[str, object]]


class RegistrationServer(Protocol):
    """Minimal registration surface independent of a specific MCP SDK."""

    def tool(
        self, *, name: str, description: str, input_schema: Mapping[str, object]
    ) -> Callable[[Callback], Callback]: ...


def register_wiki_maintenance_mcp(
    server: RegistrationServer,
    workspaces: Mapping[str, MaintenanceService],
    *,
    context_factory: Callable[[], InvocationContext],
    enable_mutations: bool = False,
) -> dict[str, Callback]:
    """Register an opt-in local catalog without deriving grants from arguments."""
    callbacks: dict[str, Callback] = {}
    for spec in maintenance_specs(enable_mutations=enable_mutations):
        operation = spec.tool_id.removeprefix("capicove.wiki.").removeprefix("maintenance-")
        handler = MaintenanceToolHandler(operation, workspaces, enable_mutations=enable_mutations)

        def invoke(
            arguments: Mapping[str, object], *, bound: MaintenanceToolHandler = handler
        ) -> dict[str, Any]:
            return encode_result(bound(context_factory(), arguments))

        callbacks[operation] = server.tool(
            name=spec.tool_id.replace(".", "_").replace("-", "_"),
            description=spec.description,
            input_schema=spec.input_schema,
        )(invoke)
    return callbacks


def create_wiki_maintenance_mcp_server(
    workspaces: Mapping[str, MaintenanceService],
    *,
    context_factory: Callable[[], InvocationContext],
    enable_mutations: bool = False,
) -> Any:
    """Compose the real optional SDK host with explicit authenticated grants."""
    from capicove.mcp.bindings import McpToolBinding
    from capicove.mcp.server import create_bound_mcp_server

    bindings = []
    for spec in maintenance_specs(enable_mutations=enable_mutations):
        operation = spec.tool_id.removeprefix("capicove.wiki.").removeprefix("maintenance-")
        bindings.append(
            McpToolBinding(
                name=spec.tool_id.replace(".", "_").replace("-", "_"),
                spec=spec,
                handler=MaintenanceToolHandler(
                    operation, workspaces, enable_mutations=enable_mutations
                ),
                context_factory=lambda _: context_factory(),
                encode_result=encode_result,
            )
        )
    return create_bound_mcp_server(tuple(bindings), server_name="capicove-wiki-maintenance")
