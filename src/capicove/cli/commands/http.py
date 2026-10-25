"""`capicove http serve`: authenticated loopback HTTP for selected Tools (RFC-0020)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the HTTP transport command without importing web frameworks."""
    http = parent.add_parser("http", help="Serve selected Tools over loopback HTTP.")
    commands = http.add_subparsers(dest="http_command", required=True)
    serve = commands.add_parser(
        "serve", help="Serve the configured Tools on 127.0.0.1 with a bearer token."
    )
    serve.add_argument("--config", type=Path, required=True, help="Version-2 workspace host JSON.")
    serve.add_argument("--port", type=int, default=0, help="TCP port; 0 picks a free port.")
    serve.add_argument(
        "--token-file",
        type=Path,
        help="Token path inside workspace_root (default: <workspace_root>/http.token).",
    )


def run(arguments: argparse.Namespace) -> int:
    """Run the foreground server; startup failures exit 2 before listening."""
    import importlib.util

    # Check optional dependencies before importing modules that need them.
    if any(
        importlib.util.find_spec(name) is None for name in ("starlette", "uvicorn", "jsonschema")
    ):
        print("HTTP startup failed: install capicove[api] to serve HTTP", file=sys.stderr)
        return 2
    try:
        from capicove.api.workspace_server import ServeError
    except ImportError:
        print("HTTP startup failed: install capicove[api] to serve HTTP", file=sys.stderr)
        return 2
    if arguments.trace_store is not None:
        print("HTTP startup failed: configure trace_store in the host JSON", file=sys.stderr)
        return 2
    try:
        from capicove.api.workspace_server import serve

        return serve(
            arguments.config.absolute(), port=arguments.port, token_file=arguments.token_file
        )
    except (ServeError, ValueError, OSError, ImportError) as error:
        print(f"HTTP startup failed: {error}", file=sys.stderr)
        return 2
