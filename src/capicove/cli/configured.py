"""Run one host-selected Tool from the CLI exactly as MCP runs it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path


def invoke_configured(
    config_path: Path, tool_id: str, arguments: Mapping[str, object]
) -> dict[str, object]:
    """Compose the selected binding, then validate, trace, and invoke one call.

    Startup problems raise ValueError, OSError, or ImportError; schema-invalid
    arguments raise ``InvalidToolArguments`` (a ValueError).
    """
    from capicove.mcp.bindings import invoke_binding
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
    from capicove.runtime.mcp_host import compose_mcp_host

    config = load_mcp_host_config(config_path.absolute())
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ValueError("--runtime-config requires a version-2 workspace config")
    if tool_id not in config.enabled_tools:
        raise ValueError(f"--runtime-config must select {tool_id} in enabled_tools")
    # Validate the complete selected catalog exactly as MCP startup does, so a CLI
    # call cannot succeed with a configuration that MCP would reject.
    with compose_mcp_host(config, catalog_only=True):
        pass
    with compose_mcp_host(replace(config, enabled_tools=(tool_id,))) as bindings:
        (binding,) = bindings
        return invoke_binding(binding, arguments)


__all__ = ["invoke_configured"]
