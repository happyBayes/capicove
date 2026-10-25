"""Errors raised by the Capicove Tool Registry."""

from __future__ import annotations


def _tool_reference(tool_id: str, contract_version: int | None) -> str:
    if contract_version is None:
        return tool_id
    return f"{tool_id}@{contract_version}"


class DuplicateToolError(ValueError):
    """Report an attempt to register an existing Tool contract key."""

    def __init__(self, tool_id: str, contract_version: int) -> None:
        super().__init__(f"tool already registered: {_tool_reference(tool_id, contract_version)}")


class UnknownToolError(LookupError):
    """Report a Tool contract that is not present in a registry."""

    def __init__(self, tool_id: str, contract_version: int | None = None) -> None:
        super().__init__(f"unknown tool: {_tool_reference(tool_id, contract_version)}")
