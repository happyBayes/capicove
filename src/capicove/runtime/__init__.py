"""Runtime composition and diagnostics for Capicove."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from capicove.runtime.diagnostics import DiagnosticReport, collect_diagnostics


if TYPE_CHECKING:
    from capicove.runtime.mcp_config import McpHostConfiguration, load_mcp_host_config
    from capicove.runtime.mcp_host import compose_mcp_host
    from capicove.runtime.reverse_image import (
        ActiveGenerationSnapshot,
        PublishedGenerationRegistration,
        ReverseImageRuntime,
        build_reverse_image_runtime,
    )


def __getattr__(name: str) -> Any:
    """Load runtime composition only on explicit public API access."""
    modules = {
        "ActiveGenerationSnapshot": "reverse_image",
        "McpHostConfiguration": "mcp_config",
        "PublishedGenerationRegistration": "reverse_image",
        "ReverseImageRuntime": "reverse_image",
        "build_reverse_image_runtime": "reverse_image",
        "compose_mcp_host": "mcp_host",
        "load_mcp_host_config": "mcp_config",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"capicove.runtime.{modules[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ActiveGenerationSnapshot",
    "DiagnosticReport",
    "McpHostConfiguration",
    "PublishedGenerationRegistration",
    "ReverseImageRuntime",
    "build_reverse_image_runtime",
    "collect_diagnostics",
    "compose_mcp_host",
    "load_mcp_host_config",
]
