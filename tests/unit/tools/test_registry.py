from collections.abc import Mapping

import pytest

from capicove.contracts import InvocationContext, ToolResult, ToolSpec
from capicove.tools import DuplicateToolError, ToolRegistry, UnknownToolError


def make_spec(tool_id: str = "capicove.search", contract_version: int = 1) -> ToolSpec:
    return ToolSpec(
        tool_id=tool_id,
        contract_version=contract_version,
        description="Search indexed evidence.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


def handler(context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
    raise AssertionError("resolving metadata must not invoke a handler")


def test_registry_resolves_the_latest_registered_contract() -> None:
    registry = ToolRegistry()
    version_one = make_spec(contract_version=1)
    version_two = make_spec(contract_version=2)
    registry.register(version_two, handler)
    registry.register(version_one, handler)

    registered = registry.resolve("capicove.search")

    assert registered.spec == version_two
    assert registered.handler is handler


def test_registry_resolves_an_exact_contract_version() -> None:
    registry = ToolRegistry()
    spec = make_spec(contract_version=1)
    registry.register(spec, handler)

    assert registry.resolve("capicove.search", 1).spec == spec


def test_registry_rejects_duplicate_keys() -> None:
    registry = ToolRegistry()
    spec = make_spec(contract_version=1)
    registry.register(spec, handler)

    with pytest.raises(DuplicateToolError, match=r"capicove\.search@1"):
        registry.register(spec, handler)


@pytest.mark.parametrize("contract_version", [None, 1])
def test_registry_reports_an_unknown_tool(contract_version: int | None) -> None:
    expected = "capicove.search" if contract_version is None else "capicove.search@1"
    with pytest.raises(UnknownToolError, match=expected):
        ToolRegistry().resolve("capicove.search", contract_version)


def test_registry_lists_specs_in_canonical_order() -> None:
    registry = ToolRegistry()
    registry.register(make_spec("capicove.search", 2), handler)
    registry.register(make_spec("capicove.ingest", 1), handler)
    registry.register(make_spec("capicove.search", 1), handler)

    keys = tuple(spec.registry_key for spec in registry.list_specs())

    assert keys == (
        ("capicove.ingest", 1),
        ("capicove.search", 1),
        ("capicove.search", 2),
    )
