from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import cast

from capicove.contracts import ToolResult, ToolStatus
from capicove.mcp.image import (
    MCP_IMAGE_TOOL_NAMES,
    encode_image_mcp_result,
    image_mcp_output_schema,
    register_image_mcp_tools,
)
from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME


class FakeServer:
    def __init__(self) -> None:
        self.tools: dict[str, Callable[..., object]] = {}

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

        def decorator(
            callback: Callable[[Mapping[str, object]], Mapping[str, object]],
        ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
            self.tools[name] = callback
            return callback

        return decorator


def _result(tool_id: str) -> ToolResult:
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id=tool_id,
        tool_version=1,
        status=ToolStatus.SUCCEEDED,
        result={},
        evidence_refs=(),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def test_register_image_mcp_tools_registers_every_catalog_tool() -> None:
    server = FakeServer()
    handlers = {
        name: lambda context, arguments, spec=spec: _result(spec.tool_id)
        for name, spec in IMAGE_TOOL_SPECS_BY_NAME.items()
    }

    callbacks = register_image_mcp_tools(server, handlers)

    assert set(callbacks) == set(IMAGE_TOOL_SPECS_BY_NAME)
    assert set(server.tools) == set(MCP_IMAGE_TOOL_NAMES.values())


def test_image_mcp_result_and_schema_preserve_tool_identity() -> None:
    result = encode_image_mcp_result(_result("capicove.image.ocr"))
    schema = cast(Mapping[str, object], image_mcp_output_schema("image_ocr"))
    properties = cast(Mapping[str, object], schema["properties"])

    assert result["tool_id"] == "capicove.image.ocr"
    assert properties["tool_id"] == {"const": "capicove.image.ocr"}


def test_image_mcp_exposes_public_phash_and_lsh_names() -> None:
    assert MCP_IMAGE_TOOL_NAMES["image_phash"] == "capicove_image_phash"
    assert MCP_IMAGE_TOOL_NAMES["image_lsh_search"] == "capicove_image_lsh_search"
    phash_schema = image_mcp_output_schema("image_phash")
    search_schema = image_mcp_output_schema("image_lsh_search")

    assert phash_schema["properties"]["result"]["anyOf"][0] == dict(  # type: ignore[index]
        IMAGE_TOOL_SPECS_BY_NAME["image_phash"].output_schema
    )
    assert search_schema["properties"]["result"]["anyOf"][0] == dict(  # type: ignore[index]
        IMAGE_TOOL_SPECS_BY_NAME["image_lsh_search"].output_schema
    )


def test_register_image_mcp_tools_rejects_missing_handlers() -> None:
    server = FakeServer()

    try:
        register_image_mcp_tools(server, {})
    except ValueError as error:
        assert "missing image MCP handlers" in str(error)
    else:
        raise AssertionError("missing handlers must be rejected")
