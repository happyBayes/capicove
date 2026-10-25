"""Expose native knowledge queries through the existing HTTP/MCP host."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.runtime.native_knowledge import KnowledgeBase, KnowledgeConfiguration


def specs(*, generation: bool) -> dict[str, ToolSpec]:
    """Declare bounded query-only tools with evidence permissions and redaction."""
    result = {}
    for name in ("status", "search", "ask", "evidence"):
        field = "evidence_id" if name == "evidence" else "question"
        properties = (
            {}
            if name == "status"
            else {field: {"type": "string", "minLength": 1, "maxLength": 4000}}
        )
        result[name] = ToolSpec(
            tool_id=f"capicove.knowledge.{name}",
            contract_version=1,
            description=f"{name.capitalize()} the configured native knowledge base with live evidence checks.",
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            required_permissions=frozenset({"evidence:read", "evidence:search"}),
            privacy_class=PrivacyClass.RESTRICTED,
            network_access=name == "ask" and generation,
            deterministic=name != "ask",
            timeout_seconds=180,
            redacted_fields=("question", "result", "evidence_id"),
        )
    return result


@contextmanager
def compose_knowledge_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Bind operator-owned corpus authority; clients cannot select roots or actors."""
    raw = config.capabilities["knowledge"]
    if set(raw) != {"schema_version", "config"} or not isinstance(raw["config"], str):
        raise ValueError("invalid_knowledge_host_configuration")
    path = Path(raw["config"])
    if not path.is_absolute() or not path.is_relative_to(config.workspace_root):
        raise ValueError("knowledge_configuration_must_be_inside_workspace")
    selected = KnowledgeConfiguration.load(path)
    if selected.workspace_id != config.workspace_id or not selected.storage_root.is_relative_to(
        config.workspace_root
    ):
        raise ValueError("knowledge_workspace_mismatch")
    generation_identity = None
    if selected.generation:
        from capicove.graph_build.remote_generation import load_remote_generation_profile

        profile = load_remote_generation_profile(selected.generation)
        generation_identity = profile.provider_id
        if not config.network["enabled"] or urlsplit(profile.endpoint).hostname not in cast(
            tuple[str, ...], config.network["source_hosts"]
        ):
            raise ValueError("knowledge_generation_host_not_authorized")
    tools = specs(generation=selected.generation is not None)
    enabled = {f"capicove.knowledge.{name}": name for name in tools}
    if any(identity not in enabled for identity in config.enabled_tools):
        raise ValueError("knowledge_tool_unavailable")
    if any(
        not tools[enabled[identity]].required_permissions <= config.permissions
        for identity in config.enabled_tools
    ):
        raise ValueError("knowledge_permissions_missing")

    def handler_for(name: str) -> ToolHandler:
        spec = tools[name]

        def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
            started = datetime.now(UTC)
            code = ToolErrorCode.UNAVAILABLE
            try:
                if catalog_only:
                    raise ValueError("catalog_only")
                if (
                    context.workspace_id != config.workspace_id
                    or context.actor_id != config.actor_id
                    or context.environment_id != config.environment_id
                    or not spec.required_permissions <= context.permissions
                ):
                    code = ToolErrorCode.PERMISSION_DENIED
                    raise PermissionError
                if context.deadline <= started:
                    code = ToolErrorCode.TIMEOUT
                    raise TimeoutError
                expected = (
                    set()
                    if name == "status"
                    else {"evidence_id" if name == "evidence" else "question"}
                )
                if set(arguments) != expected or any(
                    not isinstance(value, str) for value in arguments.values()
                ):
                    code = ToolErrorCode.INVALID_INPUT
                    raise ValueError
                if selected.generation:
                    from capicove.graph_build.remote_generation import (
                        load_remote_generation_profile,
                    )

                    if (
                        load_remote_generation_profile(selected.generation).provider_id
                        != generation_identity
                    ):
                        raise ValueError("generation_configuration_changed_restart_required")
                with KnowledgeBase(selected, read_only=True) as kb:
                    kb.deadline = context.deadline
                    kb.expected_provider = generation_identity
                    output = getattr(kb, name)(**arguments)
                if datetime.now(UTC) >= context.deadline:
                    code = ToolErrorCode.TIMEOUT
                    raise TimeoutError
                return ToolResult(
                    context.request_id,
                    context.trace_id,
                    spec.tool_id,
                    1,
                    ToolStatus.SUCCEEDED,
                    result=output,
                    started_at=started,
                )
            except Exception:
                return ToolResult(
                    context.request_id,
                    context.trace_id,
                    spec.tool_id,
                    1,
                    ToolStatus.REJECTED,
                    error=ToolError(
                        code,
                        "Knowledge request rejected; inspect local readiness and source state.",
                        False,
                    ),
                    started_at=started,
                )

        return handler

    yield tuple(
        McpToolBinding(
            "capicove_knowledge_" + enabled[identity],
            tools[enabled[identity]],
            handler_for(enabled[identity]),
            context_factory,
        )
        for identity in sorted(config.enabled_tools)
    )
