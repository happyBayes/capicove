"""Compose selected local Memory Tools with host-owned scope authority."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn, cast

from capicove.contracts import InvocationContext, ToolHandler, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.mcp_config import McpHostConfiguration
from capicove.runtime.memory_config import ontology, validate_config
from capicove.tools.memory_tools import MEMORY_SPECS


def _memory_config(host: McpHostConfiguration) -> dict[str, object]:
    """Validate the bounded Memory subset of the version-1 host configuration."""
    allowed = {"root", "scopes", "ontology_version", "predicates", "extractor"}
    if set(host.capability_config) - allowed:
        raise ValueError("memory capability_config contains unsupported fields")
    raw = host.capability_config
    root = raw.get("root")
    scopes = raw.get("scopes")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("memory root must be an absolute path")
    if (
        not isinstance(scopes, tuple)
        or not scopes
        or any(not isinstance(scope, str) or not scope.strip() for scope in scopes)
        or len(set(scopes)) != len(scopes)
    ):
        raise ValueError("memory scopes must contain unique nonblank strings")
    if os.name == "nt":
        raise ValueError("memory MCP requires Windows storage and ACL qualification")
    root_path = Path(root)
    if any(parent.is_symlink() for parent in (root_path, *root_path.parents)):
        raise ValueError("memory root must not contain symlinks")
    if root_path.exists() and not root_path.is_dir():
        raise ValueError("memory root must be a directory")
    if root_path.is_dir() and root_path.stat().st_mode & 0o077:
        raise ValueError("memory root must be private")
    config: dict[str, object] = {
        "root": root,
        "workspace": host.workspace_id,
        "scopes": list(scopes),
    }
    if "ontology_version" in raw:
        config["ontology_version"] = raw["ontology_version"]
    if "predicates" in raw:
        predicates = raw["predicates"]
        if not isinstance(predicates, tuple):
            raise ValueError("memory predicates must be an array")
        config["predicates"] = [
            dict(predicate) if isinstance(predicate, Mapping) else predicate
            for predicate in predicates
        ]
    if "extractor" in raw:
        config["extractor"] = raw["extractor"]
    validate_config(config)
    ontology(config)
    return config


def _catalog_only_handler(
    context: InvocationContext, arguments: Mapping[str, object], /
) -> NoReturn:
    raise RuntimeError("catalog-only Memory bindings cannot be invoked")


@contextmanager
def compose_memory_mcp(
    host: McpHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Own one Memory repository for the MCP process lifetime."""
    config = _memory_config(host)
    selected = {spec.tool_id: name for name, spec in MEMORY_SPECS.items()}
    if any(tool_id not in selected for tool_id in host.enabled_tools):
        raise ValueError("memory profile includes an unavailable Tool")
    for tool_id in host.enabled_tools:
        if not MEMORY_SPECS[selected[tool_id]].required_permissions.issubset(host.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    handlers: dict[str, ToolHandler] = {}
    if not catalog_only:
        from capicove.runtime.memory_providers import configured_providers
        from capicove.runtime.memory_runtime import build_memory_runtime

        version, predicates = ontology(config)
        providers = configured_providers(config)
        handlers = build_memory_runtime(
            Path(str(config["root"])),
            workspace=host.workspace_id,
            scopes=frozenset(cast(list[str], config["scopes"])),
            ontology_version=version,
            predicates=predicates,
            extractor=providers.extractor,
            dense=providers.dense,
        )
    try:
        bindings = tuple(
            McpToolBinding(
                "capicove_memory_" + selected[tool_id],
                MEMORY_SPECS[selected[tool_id]],
                handlers[selected[tool_id]] if not catalog_only else _catalog_only_handler,
                context_factory,
                destructive=selected[tool_id] == "erase_scope",
            )
            for tool_id in sorted(host.enabled_tools)
        )
        yield bindings
    finally:
        if handlers:
            from capicove.runtime.memory_runtime import close_memory_runtime

            close_memory_runtime(handlers)
