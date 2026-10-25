"""Bind shared-space multimodal Search and Evidence to the version-2 host (RFC-0032)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import NoReturn

from capicove.contracts import InvocationContext, ToolHandler, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.runtime.unified_retrieval import open_unified_retrieval, unified_settings
from capicove.tools.multimodal_search import (
    CAPICOVE_MULTIMODAL_EVIDENCE_SPEC,
    CAPICOVE_MULTIMODAL_SEARCH_SPEC,
)


_TOOLS: Mapping[str, tuple[str, ToolSpec]] = {
    CAPICOVE_MULTIMODAL_SEARCH_SPEC.tool_id: (
        "capicove_search_multimodal",
        CAPICOVE_MULTIMODAL_SEARCH_SPEC,
    ),
    CAPICOVE_MULTIMODAL_EVIDENCE_SPEC.tool_id: (
        "capicove_multimodal_get_evidence",
        CAPICOVE_MULTIMODAL_EVIDENCE_SPEC,
    ),
}


def _catalog_only_handler(
    context: InvocationContext, arguments: Mapping[str, object], /
) -> NoReturn:
    raise RuntimeError("catalog-only multimodal bindings cannot be invoked")


@contextmanager
def compose_multimodal_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate configuration and bind only the selected read-only Tools."""
    unified_settings(config)
    if any(tool_id not in _TOOLS for tool_id in config.enabled_tools):
        raise ValueError("multimodal profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not _TOOLS[tool_id][1].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    handlers: dict[str, ToolHandler] = {}
    runtime = None
    if not catalog_only:
        runtime = open_unified_retrieval(config)
        handlers = {
            CAPICOVE_MULTIMODAL_SEARCH_SPEC.tool_id: runtime.search_handler,
            CAPICOVE_MULTIMODAL_EVIDENCE_SPEC.tool_id: runtime.evidence_handler,
        }
    try:
        yield tuple(
            McpToolBinding(
                _TOOLS[tool_id][0],
                _TOOLS[tool_id][1],
                handlers.get(tool_id, _catalog_only_handler),
                context_factory,
            )
            for tool_id in sorted(config.enabled_tools)
        )
    finally:
        if runtime is not None:
            runtime.close()


__all__ = ["compose_multimodal_mcp"]
