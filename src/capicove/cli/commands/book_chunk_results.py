"""CLI projections of startup-bound Book chunk-result Tools."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from capicove.books.workspace import LocalBookWorkspace
from capicove.contracts import InvocationContext, ToolSpec, ToolStatus
from capicove.tools.book_chunk_results import BOOK_RESULT_TOOL_SPECS, BookChunkResultToolHandler


COMMANDS = frozenset({"chunk-tasks", "chunk-status", "chunk-submit", "chunk-publish"})


def add_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register bounded result reads without reusing legacy summary commands."""
    for operation in ("chunk-tasks", "chunk-status"):
        parser = commands.add_parser(operation, help=BOOK_RESULT_TOOL_SPECS[operation].description)
        parser.add_argument("workspace", type=Path)
        parser.add_argument("book_id")
        parser.add_argument("run_id")
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--limit", type=int, default=20)
    submit = commands.add_parser(
        "chunk-submit", help=BOOK_RESULT_TOOL_SPECS["chunk-submit"].description
    )
    submit.add_argument("workspace", type=Path)
    submit.add_argument("--input", type=Path, required=True)
    publish = commands.add_parser(
        "chunk-publish", help=BOOK_RESULT_TOOL_SPECS["chunk-publish"].description
    )
    publish.add_argument("workspace", type=Path)
    publish.add_argument("book_id")
    publish.add_argument("run_id")


def run(arguments: argparse.Namespace) -> int:
    """Invoke the canonical Tool after binding the command's local workspace."""
    operation = arguments.book_command
    spec = BOOK_RESULT_TOOL_SPECS[operation]
    workspace = LocalBookWorkspace.open(arguments.workspace)
    if operation == "chunk-submit":
        input_path: Path = arguments.input
        if input_path.is_symlink() or not input_path.is_file():
            raise ValueError("invalid_submission")
        raw = input_path.read_bytes()
        if len(raw) > 1024 * 1024:
            raise ValueError("invalid_submission")
        payload = json.loads(raw)
    else:
        payload = {
            "book_id": arguments.book_id,
            "run_id": arguments.run_id,
        }
        if operation in {"chunk-tasks", "chunk-status"}:
            payload.update({"offset": arguments.offset, "limit": arguments.limit})
    wiki_service = None
    if operation == "chunk-publish":
        from capicove.wiki import WikiService

        wiki_service = WikiService(workspace.root)
    result = BookChunkResultToolHandler(operation, workspace=workspace, wiki_service=wiki_service)(
        _context(spec),
        payload,
    )
    if result.status is ToolStatus.SUCCEEDED:
        print(json.dumps(dict(result.result or {}), ensure_ascii=False, sort_keys=True))
        return 0
    assert result.error is not None
    print(json.dumps({"error": {"code": result.error.message}}, sort_keys=True))
    return 2


def _context(spec: ToolSpec) -> InvocationContext:
    return InvocationContext(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        "book-cli",
        "local-user",
        "local",
        datetime.now(UTC) + timedelta(seconds=spec.timeout_seconds),
        spec.required_permissions,
        spec.privacy_class,
    )
