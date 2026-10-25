"""Deterministic registration and resolution of transport-neutral tools."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.contracts import ToolHandler, ToolSpec
from capicove.tools.errors import DuplicateToolError, UnknownToolError


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Bind immutable Tool metadata to its runtime handler."""

    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """Store Tool handlers by canonical ID and contract version."""

    def __init__(self) -> None:
        self._tools: dict[tuple[str, int], RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register one Tool contract without invoking its handler."""
        if spec.registry_key in self._tools:
            raise DuplicateToolError(spec.tool_id, spec.contract_version)
        self._tools[spec.registry_key] = RegisteredTool(spec=spec, handler=handler)

    def resolve(self, tool_id: str, contract_version: int | None = None) -> RegisteredTool:
        """Resolve an exact version or the highest registered contract version."""
        if contract_version is not None:
            return self._resolve_exact(tool_id, contract_version)

        matches = [tool for key, tool in self._tools.items() if key[0] == tool_id]
        if not matches:
            raise UnknownToolError(tool_id)
        return max(matches, key=lambda tool: tool.spec.contract_version)

    def list_specs(self) -> tuple[ToolSpec, ...]:
        """List registered metadata in stable canonical order."""
        registered = sorted(self._tools.values(), key=lambda item: item.spec.registry_key)
        return tuple(tool.spec for tool in registered)

    def _resolve_exact(self, tool_id: str, contract_version: int) -> RegisteredTool:
        try:
            return self._tools[(tool_id, contract_version)]
        except KeyError as error:
            raise UnknownToolError(tool_id, contract_version) from error
