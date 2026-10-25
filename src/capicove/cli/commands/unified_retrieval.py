"""Operator CLI for shared-space multimodal retrieval (RFC-0032).

``import``, ``forget``, ``build`` and ``activate`` are explicit operator writes.
``search`` and ``get-evidence`` use the same handlers as the MCP Tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.contracts import InvocationContext


if TYPE_CHECKING:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``capicove retrieval import|forget|build|activate|search|get-evidence``."""
    parser = parent.add_parser("retrieval", help="Shared-space multimodal retrieval.")
    commands = parser.add_subparsers(dest="retrieval_command", required=True)
    importer = commands.add_parser("import", help="Preserve one source from a configured root.")
    _runtime_config(importer)
    importer.add_argument("--root", required=True, help="Configured source root alias.")
    importer.add_argument("--relative-path", required=True, help="POSIX path below that root.")
    importer.add_argument("--audio-stream-index", type=int, help="Zero-based audio stream.")
    forgetter = commands.add_parser("forget", help="Retire one source for the next build.")
    _runtime_config(forgetter)
    forgetter.add_argument("--source-ref", required=True, help="ALIAS:RELATIVE_PATH.")
    builder = commands.add_parser("build", help="Build and publish one generation.")
    _runtime_config(builder)
    activator = commands.add_parser("activate", help="Activate one published generation.")
    _runtime_config(activator)
    activator.add_argument("--generation", required=True, help="Generation ID from build.")
    searcher = commands.add_parser("search", help="Search the active generation.")
    _runtime_config(searcher)
    query = searcher.add_mutually_exclusive_group(required=True)
    query.add_argument("--query", help="Text query.")
    query.add_argument("--audio-query", type=Path, help="Audio file below a source root.")
    query.add_argument("--image-query", type=Path, help="PNG/JPEG file below a source root.")
    searcher.add_argument("--start-ms", type=int, help="Audio query range start.")
    searcher.add_argument("--end-ms", type=int, help="Audio query range end.")
    searcher.add_argument("--audio-stream-index", type=int, help="Query audio stream.")
    searcher.add_argument("--top-k", type=int, default=10)
    searcher.add_argument(
        "--target",
        action="append",
        dest="targets",
        choices=("text_chunk", "image", "audio_segment", "keyframe"),
        help="Restrict targets (repeatable); defaults to every profile target.",
    )
    searcher.add_argument("--detail", choices=("compact", "full"), default="compact")
    reader = commands.add_parser("get-evidence", help="Expand one returned evidence ID.")
    _runtime_config(reader)
    reader.add_argument("--evidence-id", required=True)
    reader.add_argument(
        "--section", action="append", choices=("metadata", "lineage"), dest="sections"
    )


def _runtime_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="Version-2 workspace host JSON with capabilities.multimodal.",
    )


def _load(path: Path) -> WorkspaceHostConfiguration:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config

    config = load_mcp_host_config(path.absolute())
    if not isinstance(config, WorkspaceHostConfiguration):
        raise ValueError("--runtime-config requires a version-2 workspace config")
    return config


def _context(config: WorkspaceHostConfiguration) -> InvocationContext:
    return InvocationContext(
        request_id=f"cli-{datetime.now(UTC).timestamp():.6f}",
        trace_id="cli",
        workspace_id=config.workspace_id,
        actor_id=config.actor_id,
        environment_id=config.environment_id,
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        permissions=config.permissions,
    )


def _search_arguments(arguments: argparse.Namespace, targets: list[str]) -> dict[str, object]:
    if arguments.query is not None or arguments.image_query is not None:
        if (
            arguments.start_ms is not None
            or arguments.end_ms is not None
            or arguments.audio_stream_index is not None
        ):
            raise ValueError("range and stream options apply only to --audio-query")
        query: dict[str, object] = (
            {"type": "text", "text": arguments.query}
            if arguments.query is not None
            else {"type": "image", "path": str(arguments.image_query.absolute())}
        )
    else:
        query = {"type": "audio_clip", "path": str(arguments.audio_query.absolute())}
        if (arguments.start_ms is None) != (arguments.end_ms is None):
            raise ValueError("--start-ms and --end-ms must be given together")
        if arguments.start_ms is not None:
            query.update(start_ms=arguments.start_ms, end_ms=arguments.end_ms)
        if arguments.audio_stream_index is not None:
            query["stream_index"] = arguments.audio_stream_index
    return {
        "query": query,
        "targets": arguments.targets or targets,
        "top_k": arguments.top_k,
        "detail_level": arguments.detail,
    }


def run(arguments: argparse.Namespace) -> int:
    """Dispatch one subcommand; failures exit 2 with a path-free message."""
    from capicove.runtime import unified_retrieval as runtime

    command = arguments.retrieval_command
    try:
        config = _load(arguments.runtime_config)
        if command == "import":
            payload = runtime.import_media(
                config,
                arguments.root,
                arguments.relative_path,
                audio_stream_index=arguments.audio_stream_index,
            )
        elif command == "forget":
            payload = runtime.forget_media(config, arguments.source_ref)
        elif command == "build":
            payload = runtime.build_generation(config)
        elif command == "activate":
            payload = runtime.activate_generation(config, arguments.generation)
        else:
            service = runtime.open_unified_retrieval(config)
            try:
                if command == "search":
                    targets = sorted(
                        item.value for item in runtime.unified_settings(config).profile.targets
                    )
                    result = service.search_handler(
                        _context(config), _search_arguments(arguments, targets)
                    )
                else:
                    result = service.evidence_handler(
                        _context(config),
                        {
                            "evidence_id": arguments.evidence_id,
                            "sections": arguments.sections or ["metadata", "lineage"],
                        },
                    )
            finally:
                service.close()
            if result.error is not None or result.result is None:
                code = "unknown" if result.error is None else result.error.code.value
                print(f"Retrieval {command} failed: {code}", file=sys.stderr)
                return 2
            payload = dict(result.result)
    except (OSError, ValueError, ImportError, PermissionError, LookupError) as error:
        print(f"Retrieval {command} failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if command == "search" and payload.get("status") != "succeeded":
        return 2
    return 0
