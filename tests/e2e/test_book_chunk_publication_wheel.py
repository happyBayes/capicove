"""Exercise installed Book result transports over a seeded local workspace."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from capicove.books import BookService, LocalBookWorkspace
from capicove.books.chunk_config import resolve_chunk_config
from capicove.books.chunk_service import BookChunkService
from capicove.wiki import WikiService, migrate_to_v2
from tests.e2e.wheel_cache import WheelCache, WheelInstallError
from tests.integration.books.test_service import _Parser


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def _run_cli(executable: str, *arguments: str) -> dict[str, Any]:
    result = subprocess.run([executable, *arguments], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"{result.args}\n{result.stdout}\n{result.stderr}"
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


async def _check_mcp(python: str, workspace: Path, book_id: str, run_id: str) -> None:
    parameters = StdioServerParameters(
        command=python,
        args=["-m", "capicove", "book", "chunk-mcp", str(workspace)],
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert "capicove_book_chunk_status" in names
            assert "capicove_book_chunk_submit" not in names
            status = await session.call_tool(
                "capicove_book_chunk_status", {"book_id": book_id, "run_id": run_id}
            )
            assert status.isError is False
            assert status.structuredContent is not None
            assert status.structuredContent["result"]["publishable"] is True


async def _submit_over_mcp(
    python: str, workspace: Path, submission: dict[str, Any]
) -> dict[str, Any]:
    parameters = StdioServerParameters(
        command=python,
        args=[
            "-m",
            "capicove",
            "book",
            "chunk-mcp",
            str(workspace),
            "--enable-result-tool",
            "capicove.book.chunk-submit",
        ],
    )
    async with stdio_client(parameters) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            assert "capicove_book_chunk_submit" in names
            assert "capicove_book_chunk_publish" not in names
            submitted = await session.call_tool("capicove_book_chunk_submit", submission)
            assert submitted.isError is False
            assert submitted.structuredContent is not None
            assert submitted.structuredContent["status"] == "succeeded"
            result = submitted.structuredContent["result"]
            assert isinstance(result, dict)
            return cast(dict[str, Any], result)


@pytest.mark.parametrize("with_wiki", [False, True], ids=["missing-wiki", "with-wiki"])
def test_installed_book_result_cli_and_read_only_mcp(
    tmp_path: Path, wheel_cache: WheelCache, *, with_wiki: bool
) -> None:
    """A wheel can resume a seeded DAG without an editable-install transport."""
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-seeded-transport-fixture")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    run, _, _ = chunks._load(book_id, run_id)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in run.chunks}
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 30))
    migrate_to_v2(wiki.wiki_path, preview=False)
    if os.name == "posix":
        workspace.root.chmod(0o700)

    try:
        installed = wheel_cache.install(("mcp", "wiki") if with_wiki else ("mcp",))
    except WheelInstallError as error:
        if (
            with_wiki
            and os.environ.get("UV_OFFLINE") == "1"
            and "pyyaml was not found in the cache" in str(error).lower()
        ):
            pytest.skip("the locked PyYAML wheel is unavailable in the offline package cache")
        raise
    python, executable = installed.python, installed.executable

    task_page = _run_cli(executable, "book", "chunk-tasks", str(workspace.root), book_id, run_id)
    assert task_page["total"] == 5
    accepted: dict[str, str] = {}
    citations: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(task_page["items"]):
        dependencies = task["dependency_task_ids"]
        citation = (
            asdict(chunk_by_id[task["chunk_ids"][0]].core_spans[0])
            if task["task_type"] == "chunk_summary"
            else citations[dependencies[0]]
        )
        submission = {
            "schema_version": 1,
            "book_id": book_id,
            "run_id": run_id,
            "task_id": task["task_id"],
            "idempotency_key": f"wheel-{index}",
            "dependency_result_ids": [accepted[item] for item in dependencies],
            "abstract": "Grounded summary",
            "key_points": [],
            "claims": [{"text": "Grounded wheel claim", "citations": [citation]}],
        }
        if index == len(task_page["items"]) - 1:
            result = asyncio.run(_submit_over_mcp(python, workspace.root, submission))
        else:
            input_path = tmp_path / f"submission-{index}.json"
            input_path.write_text(json.dumps(submission), encoding="utf-8")
            result = _run_cli(
                executable,
                "book",
                "chunk-submit",
                str(workspace.root),
                "--input",
                str(input_path),
            )
        accepted[task["task_id"]] = result["result_id"]
        citations[task["task_id"]] = citation

    status = _run_cli(executable, "book", "chunk-status", str(workspace.root), book_id, run_id)
    assert status["publishable"] is True
    if not with_wiki:
        denied = subprocess.run(
            [executable, "book", "chunk-publish", str(workspace.root), book_id, run_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert denied.returncode == 2
        assert json.loads(denied.stdout) == {"error": {"code": "invalid_input"}}
        assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()
        assert not tuple((wiki.wiki_path / "pages" / "books").rglob("*.md"))
        asyncio.run(_check_mcp(python, workspace.root, book_id, run_id))
        return
    published = _run_cli(executable, "book", "chunk-publish", str(workspace.root), book_id, run_id)
    assert published["publication_id"].startswith("chunk-publication:")
    queried = _run_cli(executable, "wiki", "query", str(workspace.root), "Grounded wheel claim")
    assert len(queried["hits"]) >= 2
    assert all(hit["source_ids"] for hit in queried["hits"])
    asyncio.run(_check_mcp(python, workspace.root, book_id, run_id))
