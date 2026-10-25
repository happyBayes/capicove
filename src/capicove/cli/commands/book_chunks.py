"""CLI adapters for configurable offline book chunk projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capicove.books.chunk_config import PRESETS, load_chunk_config, resolve_chunk_config
from capicove.books.chunk_service import BookChunkService
from capicove.books.workspace import LocalBookWorkspace


COMMANDS = frozenset({"chunk", "chunks", "context", "chunk-mcp"})


def add_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register immutable chunk creation, preview and bounded read commands."""
    chunk = commands.add_parser(
        "chunk", help="Preview or create an immutable chunk run, without OCR."
    )
    chunk.add_argument("workspace", type=Path)
    chunk.add_argument("book_id")
    chunk.add_argument("--chunk-preset", dest="preset", choices=tuple(PRESETS))
    chunk.add_argument("--chunk-config", type=Path)
    for name in (
        "target-core-tokens",
        "max-chunk-tokens",
        "context-before-tokens",
        "context-after-tokens",
        "max-context-read-tokens",
    ):
        chunk.add_argument("--" + name, type=int)
    chunk.add_argument("--boundary-level", choices=("deepest", "section", "chapter"))
    chunk.add_argument("--tokenizer")
    chunk.add_argument(
        "--protected-types", nargs="*", help="Protected content types; no values clears the list."
    )
    chunk.add_argument("--oversize-policy", choices=("split", "error"))
    chunk.add_argument("--page-boundary", choices=("strict", "allow_continuations"))
    chunk.add_argument("--front-matter-policy", choices=("navigation-v1", "preserve"))
    chunk.add_argument("--merge-small-chunks", action=argparse.BooleanOptionalAction)
    chunk.add_argument("--dry-run", action="store_true")
    chunk.add_argument("--compare-run")
    chunks = commands.add_parser(
        "chunks", help="Read a page of chunks or reference-only summary tasks."
    )
    context = commands.add_parser("context", help="Read bounded original Parent context.")
    for parser in (chunks, context):
        parser.add_argument("workspace", type=Path)
        parser.add_argument("book_id")
        parser.add_argument("run_id")
    chunks.add_argument("--offset", type=int, default=0)
    chunks.add_argument("--limit", type=int, default=20)
    chunks.add_argument("--tasks", action="store_true")
    context.add_argument("chunk_id")
    context.add_argument("--max-context-read-tokens", type=int)
    context.add_argument("--read-policy-id")
    mcp = commands.add_parser("chunk-mcp", help="Serve chunk Tools over workspace-bound MCP stdio.")
    mcp.add_argument("workspace", type=Path)
    mcp.add_argument(
        "--enable-result-tool",
        action="append",
        choices=("capicove.book.chunk-submit", "capicove.book.chunk-publish"),
        help="Explicitly enable an accepted Book result write Tool at startup.",
    )


def run(arguments: argparse.Namespace) -> int:
    """Resolve caller configuration before opening source evidence."""
    if arguments.book_command == "chunk-mcp":
        from capicove.mcp.book_chunks import run_stdio

        run_stdio(
            arguments.workspace,
            enabled_result_tools=frozenset(arguments.enable_result_tool or ()),
        )
        return 0
    config = None
    if arguments.book_command == "chunk":
        fields = (
            "preset",
            "target_core_tokens",
            "max_chunk_tokens",
            "context_before_tokens",
            "context_after_tokens",
            "max_context_read_tokens",
            "boundary_level",
            "tokenizer",
            "protected_types",
            "oversize_policy",
            "page_boundary",
            "front_matter_policy",
            "merge_small_chunks",
        )
        config = resolve_chunk_config(
            load_chunk_config(arguments.chunk_config) if arguments.chunk_config else None,
            {
                name: getattr(arguments, name)
                for name in fields
                if getattr(arguments, name) is not None
            },
        )
    service = BookChunkService(LocalBookWorkspace.open(arguments.workspace))
    if config is not None:
        payload = service.create(
            arguments.book_id, config, dry_run=arguments.dry_run, compare_run=arguments.compare_run
        )
    elif arguments.book_command == "chunks":
        payload = service.read_chunks(
            arguments.book_id,
            arguments.run_id,
            offset=arguments.offset,
            limit=arguments.limit,
            tasks=arguments.tasks,
        )
    else:
        payload = service.read_context(
            arguments.book_id,
            arguments.run_id,
            arguments.chunk_id,
            max_tokens=arguments.max_context_read_tokens,
            read_policy_id=arguments.read_policy_id,
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0
