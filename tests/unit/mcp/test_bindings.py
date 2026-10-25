"""The common binding preserves canonical contracts and explicit annotations."""

from collections.abc import Mapping
from dataclasses import replace

import pytest

from capicove.contracts import InvocationContext, SideEffect, ToolResult, ToolSpec, ToolStatus
from capicove.mcp.bindings import McpToolBinding, default_context, output_schema, validate_bindings


SPEC = ToolSpec(
    "capicove.test.echo",
    1,
    "Return a deterministic fixture.",
    {"type": "object", "properties": {}, "additionalProperties": False},
    {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
)


def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
    return ToolResult(
        context.request_id,
        context.trace_id,
        SPEC.tool_id,
        1,
        ToolStatus.SUCCEEDED,
        result={"ok": True},
    )


def binding() -> McpToolBinding:
    return McpToolBinding("capicove_test_echo", SPEC, handler, default_context)


def test_schema_is_canonical_and_annotations_are_explicit() -> None:
    item = binding()
    schema = output_schema(SPEC)
    assert schema["properties"]["tool_id"] == {"const": SPEC.tool_id}  # type: ignore[index]
    assert item.annotations()["readOnlyHint"] is True
    write = replace(item, spec=replace(SPEC, side_effect=SideEffect.WRITE))
    assert write.annotations()["readOnlyHint"] is False
    assert write.annotations()["destructiveHint"] is False
    assert replace(write, destructive=True).annotations()["destructiveHint"] is True


def test_catalog_rejects_empty_and_duplicate_identity() -> None:
    item = binding()
    for items in ((), (item, item), (item, replace(item, name="capicove_other"))):
        with pytest.raises(ValueError):
            validate_bindings(items)
    with pytest.raises(ValueError):
        replace(item, name="foreign")
