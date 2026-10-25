"""Public Book result Tools are bound to a startup-selected workspace."""

import asyncio
import os
import shutil
import sys
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from capicove.books import BookChunkResultService, BookService, LocalBookWorkspace
from capicove.books.chunk_config import resolve_chunk_config
from capicove.books.chunk_service import BookChunkService
from capicove.contracts import InvocationContext, ToolErrorCode, ToolStatus
from capicove.wiki import WikiDependencyError, WikiService, WikiUnsafePathError, migrate_to_v2
from tests.unit.cli.test_book import _Parser


def _prepared(tmp_path: Path) -> tuple[LocalBookWorkspace, str, str]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = BookService(workspace).prepare(source, _Parser()).book_id
    run_id = str(BookChunkService(workspace).create(book_id, resolve_chunk_config())["run_id"])
    return workspace, book_id, run_id


def _context(permissions: frozenset[str]) -> InvocationContext:
    return InvocationContext(
        "request",
        "trace",
        "workspace",
        "agent",
        "local",
        datetime.now(UTC) + timedelta(minutes=1),
        permissions,
    )


def test_result_task_tool_uses_bound_workspace_and_bounded_page(tmp_path: Path) -> None:
    """A caller-selected root must never redirect task reads to another Book workspace."""
    from capicove.tools.book_chunk_results import (
        BOOK_RESULT_TOOL_SPECS,
        BookChunkResultToolHandler,
    )

    workspace, book_id, run_id = _prepared(tmp_path)
    spec = BOOK_RESULT_TOOL_SPECS["chunk-tasks"]
    handler = BookChunkResultToolHandler("chunk-tasks", workspace=workspace)
    arguments = {"book_id": book_id, "run_id": run_id, "limit": 1}

    result = handler(_context(frozenset({"book:read"})), arguments)

    assert spec.tool_id == "capicove.book.chunk-tasks"
    assert spec.required_permissions == frozenset({"book:read"})
    assert "workspace" not in cast(dict[str, object], spec.input_schema["properties"])
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert len(cast(list[object], result.result["items"])) == 1
    assert result.result["next_offset"] == 1

    denied = handler(_context(frozenset()), arguments)
    assert denied.status is ToolStatus.FAILED
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.PERMISSION_DENIED

    foreign = handler(
        _context(frozenset({"book:read"})),
        {**arguments, "workspace": str(tmp_path / "foreign")},
    )
    assert foreign.status is ToolStatus.FAILED
    assert foreign.error is not None
    assert foreign.error.code is ToolErrorCode.INVALID_INPUT


def test_result_status_tool_reads_the_same_bound_run(tmp_path: Path) -> None:
    """Status should expose the saved DAG without opening a caller-supplied root."""
    from capicove.tools.book_chunk_results import (
        BOOK_RESULT_TOOL_SPECS,
        BookChunkResultToolHandler,
    )

    workspace, book_id, run_id = _prepared(tmp_path)
    handler = BookChunkResultToolHandler("chunk-status", workspace=workspace)
    context = _context(frozenset({"book:read"}))
    result = handler(context, {"book_id": book_id, "run_id": run_id})

    assert BOOK_RESULT_TOOL_SPECS["chunk-status"].tool_id == "capicove.book.chunk-status"
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["total"] == 5
    assert result.result["publishable"] is False
    assert all(
        item["state"] in {"ready", "blocked"}
        for item in cast(list[dict[str, object]], result.result["items"])
    )

    denied = handler(
        replace(context, permissions=frozenset()), {"book_id": book_id, "run_id": run_id}
    )
    assert denied.status is ToolStatus.FAILED


def test_result_submit_tool_appends_a_cited_version_without_caller_root(tmp_path: Path) -> None:
    """A write-permitted caller can submit a cited leaf but cannot select another root."""
    from capicove.books import BookChunkResultService
    from capicove.tools.book_chunk_results import (
        BOOK_RESULT_TOOL_SPECS,
        BookChunkResultToolHandler,
    )

    workspace, book_id, run_id = _prepared(tmp_path)
    service = BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = BookChunkService(workspace)._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    submission: dict[str, object] = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": task["task_id"],
        "idempotency_key": "first",
        "dependency_result_ids": [],
        "abstract": "A cited summary",
        "key_points": [],
        "claims": [{"text": "A source fact", "citations": [asdict(chunk.core_spans[0])]}],
    }
    spec = BOOK_RESULT_TOOL_SPECS["chunk-submit"]
    handler = BookChunkResultToolHandler("chunk-submit", workspace=workspace)

    denied = handler(_context(frozenset({"book:read"})), submission)
    accepted = handler(_context(frozenset({"book:write"})), submission)
    repeated = handler(_context(frozenset({"book:write"})), submission)
    foreign = handler(
        _context(frozenset({"book:write"})),
        {**submission, "workspace": str(tmp_path / "foreign")},
    )

    assert spec.required_permissions == frozenset({"book:write"})
    assert "workspace" not in cast(dict[str, object], spec.input_schema["properties"])
    assert denied.status is ToolStatus.FAILED
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.PERMISSION_DENIED
    assert accepted.status is ToolStatus.SUCCEEDED
    assert accepted.result is not None
    assert accepted.result["version"] == 1
    assert repeated.result == accepted.result
    assert foreign.status is ToolStatus.FAILED
    assert service.status(book_id, run_id)["items"][0]["current_result_id"] is not None


def test_result_tools_return_stable_error_codes_without_echoing_submission(tmp_path: Path) -> None:
    """Bad citations and incomplete publication must fail without revealing prose."""
    from capicove.books import BookChunkResultService
    from capicove.tools.book_chunk_results import BookChunkResultToolHandler

    workspace, book_id, run_id = _prepared(tmp_path)
    service = BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = BookChunkService(workspace)._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])
    citation["text_sha256"] = "0" * 64
    submit = BookChunkResultToolHandler("chunk-submit", workspace=workspace)
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    baseline_citations = (wiki.wiki_path / "graph" / "citations.jsonl").read_bytes()
    publish = BookChunkResultToolHandler("chunk-publish", workspace=workspace, wiki_service=wiki)

    bad = submit(
        _context(frozenset({"book:write"})),
        {
            "schema_version": 1,
            "book_id": book_id,
            "run_id": run_id,
            "task_id": task["task_id"],
            "idempotency_key": "bad",
            "dependency_result_ids": [],
            "abstract": "PRIVATE ABSTRACT",
            "key_points": [],
            "claims": [{"text": "PRIVATE CLAIM", "citations": [citation]}],
        },
    )
    incomplete = publish(
        _context(frozenset({"book:write", "wiki:write"})),
        {"book_id": book_id, "run_id": run_id},
    )

    assert bad.status is ToolStatus.FAILED
    assert bad.error is not None
    assert bad.error.message == "citation_invalid"
    assert incomplete.status is ToolStatus.FAILED
    assert incomplete.error is not None
    assert incomplete.error.message == "publication_incomplete"
    assert service.status(book_id, run_id)["publishable"] is False
    assert (wiki.wiki_path / "graph" / "citations.jsonl").read_bytes() == baseline_citations


def test_book_chunk_mcp_reads_result_status_without_a_workspace_argument(tmp_path: Path) -> None:
    """The real MCP catalog should bind the read-only result handler at startup."""
    pytest.importorskip("mcp")
    from mcp import types

    from capicove.mcp.book_chunks import create_book_chunk_mcp_server

    workspace, book_id, run_id = _prepared(tmp_path)
    server = create_book_chunk_mcp_server(workspace.root)

    async def exercise() -> tuple[set[str], Any]:
        listed = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest())
        called = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="capicove_book_chunk_status",
                    arguments={"book_id": book_id, "run_id": run_id},
                )
            )
        )
        return {tool.name for tool in listed.root.tools}, called.root

    names, result = asyncio.run(exercise())

    assert "capicove_book_chunk_tasks" in names
    assert "capicove_book_chunk_status" in names
    assert "capicove_book_chunk_submit" not in names
    assert "capicove_book_chunk_publish" not in names
    assert result.isError is False
    assert result.structuredContent["result"]["total"] == 5


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-root startup check")
def test_book_chunk_mcp_submits_only_when_explicitly_enabled(tmp_path: Path) -> None:
    """The write result Tool should require opt-in and a private startup root."""
    pytest.importorskip("mcp")
    from mcp import types

    from capicove.mcp.book_chunks import create_book_chunk_mcp_server

    workspace, book_id, run_id = _prepared(tmp_path)
    task = next(
        row
        for row in BookChunkResultService(workspace).list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = BookChunkService(workspace)._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    submission = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": task["task_id"],
        "idempotency_key": "mcp-leaf",
        "dependency_result_ids": [],
        "abstract": "Cited chapter",
        "key_points": [],
        "claims": [{"text": "A source fact", "citations": [asdict(chunk.core_spans[0])]}],
    }
    selected = frozenset({"capicove.book.chunk-submit"})

    with pytest.raises(ValueError, match="private"):
        create_book_chunk_mcp_server(workspace.root, enabled_result_tools=selected)
    workspace.root.chmod(0o700)
    server = create_book_chunk_mcp_server(workspace.root, enabled_result_tools=selected)

    async def exercise() -> tuple[set[str], Any]:
        listed = await server.request_handlers[types.ListToolsRequest](types.ListToolsRequest())
        called = await server.request_handlers[types.CallToolRequest](
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="capicove_book_chunk_submit",
                    arguments=submission,
                )
            )
        )
        return {tool.name for tool in listed.root.tools}, called.root

    names, result = asyncio.run(exercise())

    assert "capicove_book_chunk_submit" in names
    assert "capicove_book_chunk_publish" not in names
    assert result.isError is False
    assert result.structuredContent["result"]["version"] == 1


def test_book_chunk_mcp_publish_requires_yaml_before_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selected publisher must not announce a Tool it cannot execute."""
    pytest.importorskip("mcp")
    from capicove.mcp.book_chunks import create_book_chunk_mcp_server

    workspace, _, _ = _prepared(tmp_path)
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 30))
    migrate_to_v2(wiki.wiki_path, preview=False)
    if os.name == "posix":
        workspace.root.chmod(0o700)
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(WikiDependencyError, match="capicove\\[wiki\\]"):
        create_book_chunk_mcp_server(
            workspace.root,
            enabled_result_tools=frozenset({"capicove.book.chunk-publish"}),
        )

    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication startup path race")
def test_book_chunk_mcp_rejects_graph_swap_before_ontology_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("mcp")
    from capicove.mcp.book_chunks import create_book_chunk_mcp_server

    workspace, _, _ = _prepared(tmp_path)
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 30))
    migrate_to_v2(wiki.wiki_path, preview=False)
    workspace.root.chmod(0o700)
    graph = wiki.wiki_path / "graph"
    foreign = tmp_path / "foreign-graph"
    shutil.copytree(graph, foreign)
    (foreign / "ontology.yaml").write_text("[private content", encoding="utf-8")
    original_scan = WikiService.scan

    def swap_after_scan(service: WikiService) -> Any:
        inventory = original_scan(service)
        graph.rename(wiki.wiki_path / "graph-held")
        graph.symlink_to(foreign, target_is_directory=True)
        return inventory

    monkeypatch.setattr(WikiService, "scan", swap_after_scan)

    with pytest.raises(WikiUnsafePathError):
        create_book_chunk_mcp_server(
            workspace.root,
            enabled_result_tools=frozenset({"capicove.book.chunk-publish"}),
        )

    assert (foreign / "ontology.yaml").read_text(encoding="utf-8") == "[private content"
    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()
