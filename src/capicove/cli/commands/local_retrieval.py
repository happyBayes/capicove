"""Operator CLI for offline workspace retrieval (RFC-0018, ADR-0024)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.contracts import InvocationContext, ToolHandler


if TYPE_CHECKING:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration


def add_parsers(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove source import|forget` and `capicove index build`."""
    source = parent.add_parser("source", help="Import local sources for offline retrieval.")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    importer = source_commands.add_parser(
        "import", help="Record one verified source from a configured root."
    )
    _add_runtime_config(importer, required=True)
    importer.add_argument("--root", required=True, help="Configured source root alias.")
    importer.add_argument(
        "--relative-path", required=True, help="Normalized POSIX path below that root."
    )
    forgetter = source_commands.add_parser(
        "forget", help="Retire one source from the inventory used by the next build."
    )
    _add_runtime_config(forgetter, required=True)
    forgetter.add_argument("--root", required=True, help="Configured source root alias.")
    forgetter.add_argument(
        "--relative-path", required=True, help="Normalized POSIX path below that root."
    )
    index = parent.add_parser("index", help="Build offline retrieval index generations.")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    builder = index_commands.add_parser(
        "build", help="Verify imported sources and activate a new generation."
    )
    _add_runtime_config(builder, required=True)


def add_runtime_config_argument(parser: argparse.ArgumentParser) -> None:
    """Let a read command compose its handler from a version-2 host config."""
    _add_runtime_config(parser, required=False)


def run_source(arguments: argparse.Namespace) -> int:
    """Import or retire one source as an explicit operator write."""
    from capicove.runtime import code_retrieval
    from capicove.runtime.local_retrieval import forget_source, import_source

    operation, label = (
        (import_source, "Import")
        if arguments.source_command == "import"
        else (forget_source, "Forget")
    )
    try:
        config = _load(arguments.runtime_config)
        if code_retrieval.is_code_configuration(config):
            code_operation = (
                code_retrieval.import_file
                if arguments.source_command == "import"
                else code_retrieval.forget_file
            )
            payload = code_operation(config, arguments.root, arguments.relative_path)
        else:
            payload = operation(config, alias=arguments.root, relative_path=arguments.relative_path)
    except (OSError, ValueError, ImportError, PermissionError) as error:
        print(f"{label} failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def run_index(arguments: argparse.Namespace) -> int:
    """Build and activate one generation as an explicit operator write."""
    from capicove.runtime import code_retrieval
    from capicove.runtime.local_retrieval import build_index

    try:
        config = _load(arguments.runtime_config)
        if code_retrieval.is_code_configuration(config):
            payload = code_retrieval.build_index(config)
        else:
            payload = build_index(config)
    except (OSError, ValueError, ImportError, PermissionError) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


@contextmanager
def configured_tool(path: Path, tool_id: str) -> Iterator[tuple[ToolHandler, InvocationContext]]:
    """Compose one selected host Tool exactly as MCP does, with host-owned identity."""
    from capicove.runtime.mcp_host import compose_mcp_host

    config = _load(path)
    if tool_id not in config.enabled_tools:
        raise ValueError(f"--runtime-config must select {tool_id} in enabled_tools")
    with compose_mcp_host(config, catalog_only=True):  # Same complete validation as MCP.
        pass
    with compose_mcp_host(replace(config, enabled_tools=(tool_id,))) as bindings:
        (binding,) = bindings
        yield binding.handler, binding.context_factory(binding.spec)


def _load(path: Path) -> WorkspaceHostConfiguration:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config

    config = load_mcp_host_config(path.absolute())
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ValueError("--runtime-config requires a version-2 workspace config")
    return config


def _add_runtime_config(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=required,
        help="Version-2 workspace host JSON with capabilities.retrieval.",
    )
