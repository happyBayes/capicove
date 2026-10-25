"""Operator CLI for model-free local code search (RFC-0033 S1).

``code import`` discovers one configured repository root through an explicitly
selected Git executable; ``code search`` calls the same Search handler as MCP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``capicove code import|search``."""
    parser = parent.add_parser("code", help="Local code search over a configured repository.")
    commands = parser.add_subparsers(dest="code_command", required=True)
    importer = commands.add_parser("import", help="Discover and record one repository root.")
    importer.add_argument("--runtime-config", type=Path, required=True)
    importer.add_argument("--root", required=True, help="Configured repository root alias.")
    importer.add_argument("--git", required=True, help="Absolute Git executable path.")
    searcher = commands.add_parser("search", help="Search the active code generation.")
    searcher.add_argument("query", help="Identifier, relative path or natural-language query.")
    searcher.add_argument("--runtime-config", type=Path, required=True)
    searcher.add_argument("--top-k", type=int, default=10)


def run(arguments: argparse.Namespace) -> int:
    """Dispatch one subcommand; failures exit 2 with a path-free message."""
    from capicove.cli.commands.local_retrieval import configured_tool
    from capicove.runtime import code_retrieval
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config

    try:
        config = load_mcp_host_config(arguments.runtime_config.absolute())
        if not isinstance(config, WorkspaceHostConfiguration) or not (
            code_retrieval.is_code_configuration(config)
        ):
            raise ValueError("--runtime-config must select capabilities.retrieval schema 2")
        if arguments.code_command == "import":
            payload = code_retrieval.import_repository(config, arguments.root, git=arguments.git)
        else:
            code_retrieval.code_settings(config)
            with configured_tool(arguments.runtime_config, "capicove.search") as (handler, context):
                result = handler(context, {"query": arguments.query, "top_k": arguments.top_k})
            if result.error is not None or result.result is None:
                code = "unknown" if result.error is None else result.error.code.value
                print(f"Code search failed: {code}", file=sys.stderr)
                return 2
            payload = dict(result.result)
    except (OSError, ValueError, ImportError, PermissionError) as error:
        print(f"Code {arguments.code_command} failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload.get("status", "succeeded") == "succeeded" else 2
