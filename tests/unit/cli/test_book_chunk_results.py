"""CLI projections for startup-bound Book chunk-result reads."""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from pytest import CaptureFixture

from capicove.books import BookChunkResultService, BookService, LocalBookWorkspace
from capicove.books.chunk_config import resolve_chunk_config
from capicove.books.chunk_service import BookChunkService
from capicove.cli import main
from capicove.wiki import WikiService, migrate_to_v2
from tests.unit.cli.test_book import _Parser


def test_book_result_cli_reads_bounded_tasks_and_status(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """The CLI should expose the accepted result view for one explicit local workspace."""
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = BookService(workspace).prepare(source, _Parser()).book_id
    run_id = str(BookChunkService(workspace).create(book_id, resolve_chunk_config())["run_id"])

    assert main(["book", "chunk-tasks", str(workspace.root), book_id, run_id, "--limit", "1"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert len(tasks["items"]) == 1
    assert tasks["next_offset"] == 1
    assert tasks["items"][0]["submission_contract"] == "chunk-result-v1"

    assert main(["book", "chunk-status", str(workspace.root), book_id, run_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["total"] == 5
    assert status["publishable"] is False


def test_book_chunk_submit_cli_uses_canonical_cited_result_handler(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """A local submission file should append one accepted result version."""
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    task = next(
        row
        for row in BookChunkResultService(workspace).list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    submission = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": task["task_id"],
        "idempotency_key": "cli-leaf",
        "dependency_result_ids": [],
        "abstract": "Cited chapter",
        "key_points": [],
        "claims": [{"text": "A source fact", "citations": [asdict(chunk.core_spans[0])]}],
    }
    input_path = tmp_path / "submission.json"
    input_path.write_text(json.dumps(submission), encoding="utf-8")

    assert main(["book", "chunk-submit", str(workspace.root), "--input", str(input_path)]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["version"] == 1
    assert accepted["task_id"] == task["task_id"]
    assert (
        BookChunkResultService(workspace).status(book_id, run_id)["items"][0]["current_result_id"]
        == accepted["result_id"]
    )


def test_book_chunk_publish_cli_writes_cited_pages_to_bound_wiki(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """The explicit local CLI root should bind both Book and Wiki publication."""
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, _, _ = chunks._load(book_id, run_id)
    by_chunk = {chunk.chunk_id: chunk for chunk in run.chunks}
    accepted: dict[str, dict[str, Any]] = {}
    citations: dict[str, dict[str, Any]] = {}
    for number, task in enumerate(tasks):
        dependencies = task["dependency_task_ids"]
        citation = (
            asdict(by_chunk[task["chunk_ids"][0]].core_spans[0])
            if task["task_type"] == "chunk_summary"
            else citations[dependencies[0]]
        )
        accepted[task["task_id"]] = service.submit(
            {
                "schema_version": 1,
                "book_id": book_id,
                "run_id": run_id,
                "task_id": task["task_id"],
                "idempotency_key": f"cli-{number}",
                "dependency_result_ids": [
                    accepted[identifier]["result_id"] for identifier in dependencies
                ],
                "abstract": "Cited chapter",
                "key_points": [],
                "claims": [{"text": "A cited fact", "citations": [citation]}],
            }
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)

    assert main(["book", "chunk-publish", str(workspace.root), book_id, run_id]) == 0
    published = json.loads(capsys.readouterr().out)
    assert (
        published["publication_id"]
        == service.publication_manifest(book_id, run_id)["publication_id"]
    )
    assert len(wiki.query("A cited fact").hits) >= 2
