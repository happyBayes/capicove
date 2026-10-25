"""Bind offline workspace Search and Evidence to the version-2 host (ADR-0024)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import NoReturn

from capicove.contracts import InvocationContext, ToolHandler, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.local_retrieval import (
    LOCAL_EVIDENCE_SPEC,
    LOCAL_SEARCH_SPEC,
    open_local_retrieval,
    retrieval_settings,
)
from capicove.runtime.mcp_config import WorkspaceHostConfiguration


_TOOLS: Mapping[str, tuple[str, ToolSpec]] = {
    LOCAL_SEARCH_SPEC.tool_id: ("capicove_search", LOCAL_SEARCH_SPEC),
    LOCAL_EVIDENCE_SPEC.tool_id: ("capicove_get_evidence", LOCAL_EVIDENCE_SPEC),
}


def _catalog_only_handler(
    context: InvocationContext, arguments: Mapping[str, object], /
) -> NoReturn:
    raise RuntimeError("catalog-only retrieval bindings cannot be invoked")


@contextmanager
def compose_retrieval_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate retrieval authority and bind the selected read-only Tools."""
    from capicove.runtime.code_retrieval import is_code_configuration

    code = is_code_configuration(config)
    if code:
        from capicove.runtime.code_retrieval import code_settings

        code_settings(config)
    else:
        settings = retrieval_settings(config)
    if any(tool_id not in _TOOLS for tool_id in config.enabled_tools):
        raise ValueError("retrieval profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not _TOOLS[tool_id][1].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    handlers: dict[str, ToolHandler] = {}
    closers: list[Callable[[], None]] = []
    if not catalog_only and code:
        from capicove.runtime.code_retrieval import open_code_retrieval
        from capicove.tools.evidence import EvidenceToolHandler
        from capicove.tools.search import SearchToolHandler

        code_runtime = open_code_retrieval(config)
        closers.append(code_runtime.close)
        handlers = {
            LOCAL_SEARCH_SPEC.tool_id: SearchToolHandler(code_runtime.operation),
            LOCAL_EVIDENCE_SPEC.tool_id: EvidenceToolHandler(code_runtime.reader),
        }
    elif not catalog_only:
        runtime = open_local_retrieval(config, settings)
        handlers = {
            LOCAL_SEARCH_SPEC.tool_id: runtime.search_handler,
            LOCAL_EVIDENCE_SPEC.tool_id: runtime.evidence_handler,
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
        for close in closers:
            close()


__all__ = ["compose_retrieval_mcp"]
