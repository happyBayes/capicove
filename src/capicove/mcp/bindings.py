"""SDK-independent bindings for explicitly composed canonical Tools."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from capicove.contracts import InvocationContext, SideEffect, ToolHandler, ToolResult, ToolSpec
from capicove.traces import TraceRecorder


if TYPE_CHECKING:
    from capicove.runtime.mcp_config import McpHostConfiguration, WorkspaceHostConfiguration


def encode_result(result: ToolResult) -> dict[str, object]:
    """Project the canonical result without transport-specific business logic."""
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": dict(result.result) if result.result is not None else None,
        "evidence_refs": list(result.evidence_refs),
        "error": (
            {
                "code": result.error.code.value,
                "message": result.error.message,
                "retryable": result.error.retryable,
            }
            if result.error is not None
            else None
        ),
    }


def output_schema(spec: ToolSpec) -> dict[str, object]:
    """Describe the established canonical result envelope for one Tool."""
    properties: dict[str, object] = {
        "request_id": {"type": "string"},
        "trace_id": {"type": "string"},
        "tool_id": {"const": spec.tool_id},
        "tool_version": {"const": spec.contract_version},
        "status": {"enum": ["succeeded", "failed", "rejected"]},
        "result": {"anyOf": [dict(spec.output_schema), {"type": "null"}]},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "error": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "message", "retryable"],
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "retryable": {"type": "boolean"},
                    },
                },
                {"type": "null"},
            ]
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def default_context(spec: ToolSpec) -> InvocationContext:
    """Preserve the identities of legacy, explicitly composed server creators."""
    return InvocationContext(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        "mcp-server",
        "mcp-client",
        "mcp-server",
        datetime.now(UTC) + timedelta(seconds=spec.timeout_seconds),
        spec.required_permissions,
        spec.privacy_class,
    )


@dataclass(frozen=True, slots=True)
class McpToolBinding:
    """Bind a Tool to startup-owned policy, decoding, encoding, and tracing."""

    name: str
    spec: ToolSpec
    handler: ToolHandler
    context_factory: Callable[[ToolSpec], InvocationContext]
    decode_arguments: Callable[[Mapping[str, object]], dict[str, object]] = dict
    encode_result: Callable[[ToolResult], dict[str, object]] = encode_result
    destructive: bool = False
    open_world: bool = False
    recorder: TraceRecorder | None = None
    advertise_output_schema: bool = True

    def __post_init__(self) -> None:
        if re.fullmatch(r"capicove_[a-z0-9_]+", self.name) is None:
            raise ValueError("MCP names must use the capicove_ prefix and lowercase identifiers")
        if not all(
            callable(value)
            for value in (
                self.handler,
                self.context_factory,
                self.decode_arguments,
                self.encode_result,
            )
        ):
            raise ValueError("MCP bindings require callable adapters")

    def annotations(self) -> dict[str, bool]:
        """Expose explicit side-effect hints; WRITE alone does not imply deletion."""
        return {
            "readOnlyHint": self.spec.side_effect in (SideEffect.NONE, SideEffect.READ),
            "destructiveHint": self.destructive,
            "idempotentHint": self.spec.idempotent,
            "openWorldHint": self.open_world or self.spec.network_access,
        }


def catalog_payload(
    config: McpHostConfiguration | WorkspaceHostConfiguration,
    bindings: tuple[McpToolBinding, ...],
) -> dict[str, object]:
    """Describe a composed catalog exactly as `capicove mcp catalog` prints it."""
    payload: dict[str, object] = {
        "schema_version": config.schema_version,
        "server_name": config.server_name,
        "tools": [
            {
                "name": item.name,
                "tool_id": item.spec.tool_id,
                "tool_version": item.spec.contract_version,
                "input_schema": dict(item.spec.input_schema),
                "output_schema": output_schema(item.spec),
                "annotations": item.annotations(),
            }
            for item in bindings
        ],
    }
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration

    if isinstance(config, WorkspaceHostConfiguration):
        payload["capabilities"] = sorted(config.capabilities)
    else:
        payload["capability"] = config.capability
    return payload


class InvalidToolArguments(ValueError):
    """Report arguments rejected by a Tool's input schema or transport decoder."""


def invoke_binding(
    item: McpToolBinding,
    arguments: Mapping[str, object],
    *,
    context: InvocationContext | None = None,
) -> dict[str, object]:
    """Validate, invoke, trace, and encode one call; shared by every host transport."""
    import importlib

    # Loaded dynamically: jsonschema ships without type stubs in this environment.
    jsonschema: Any = importlib.import_module("jsonschema")

    from capicove.tools.tracing import invoke_one_step_tool

    try:
        jsonschema.validate(dict(arguments), dict(item.spec.input_schema))
        decoded = item.decode_arguments(arguments)
    except (ValueError, TypeError, jsonschema.ValidationError) as error:
        raise InvalidToolArguments("Invalid Tool arguments") from error
    if context is None:
        context = item.context_factory(item.spec)
    result = invoke_one_step_tool(
        recorder=item.recorder,
        spec=item.spec,
        handler=item.handler,
        context=context,
        arguments=decoded,
    )
    if (
        result.request_id != context.request_id
        or result.trace_id != context.trace_id
        or (result.tool_id, result.tool_version) != item.spec.registry_key
    ):
        raise ValueError("Tool result does not match its invocation")
    encoded = item.encode_result(result)
    # Legacy Book bindings deliberately preserve their pre-schema contract:
    # identity is checked above, but no transport output schema is imposed.
    if item.advertise_output_schema:
        jsonschema.validate(encoded, output_schema(item.spec))
    return encoded


def validate_bindings(bindings: tuple[McpToolBinding, ...]) -> None:
    """Reject an empty or ambiguous catalog before importing the optional SDK."""
    if not bindings:
        raise ValueError("MCP catalog must contain at least one configured Tool")
    if len({item.name for item in bindings}) != len(bindings):
        raise ValueError("duplicate MCP name")
    if len({item.spec.tool_id for item in bindings}) != len(bindings):
        raise ValueError("duplicate canonical Tool ID")
