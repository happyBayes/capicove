"""Executable local image-search and stdio MCP entry points."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def add_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add image commands without importing model or MCP runtimes."""
    search = commands.add_parser("search", help="Search an activated personal image index.")
    search.add_argument("--config", type=Path, required=True, help="Trusted local host JSON.")
    search.add_argument("--arguments", type=Path, required=True, help="Canonical image query JSON.")
    serve = commands.add_parser("serve", help="Serve image search over local MCP stdio.")
    serve.add_argument("--config", type=Path, required=True, help="Trusted local host JSON.")


def run(arguments: argparse.Namespace) -> int:
    """Run a local principal; stdout is exclusively JSON or MCP protocol data."""
    from capicove.contracts import ToolStatus
    from capicove.mcp.image_search import encode_search_mcp_result
    from capicove.runtime.image_search_host import load_image_search_host, read_json_object
    from capicove.tools.image_search import MAX_INLINE_BASE64_CHARS, SEARCH_SPEC

    try:
        canonical = (
            read_json_object(arguments.arguments, maximum_bytes=MAX_INLINE_BASE64_CHARS + 65536)
            if arguments.image_command == "search"
            else None
        )
        with load_image_search_host(arguments.config) as host:
            if canonical is not None:
                context = host.create(
                    tool_id=SEARCH_SPEC.tool_id,
                    deadline_cap=datetime.now(UTC) + timedelta(seconds=SEARCH_SPEC.timeout_seconds),
                )
                result = host.runtime.tool_handler(context, canonical)
                print(
                    json.dumps(
                        encode_search_mcp_result(result, expected_context=context),
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                return 0 if result.status is ToolStatus.SUCCEEDED else 1
            from capicove.mcp.image_search import serve_search_stdio

            # asyncio.run drains its default executor before closing the runtime,
            # including admitted synchronous calls whose clients cancelled.
            asyncio.run(serve_search_stdio(host.runtime.tool_handler, context_factory=host))
            return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        # Paths, model contents, keys, and private exception messages stay private.
        print(
            "Image search startup or transport failed; verify local configuration, "
            "provisioning, storage, and optional dependencies. See the image-search guide.",
            file=sys.stderr,
        )
        return 2
