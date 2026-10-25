"""Accepted chunk-result tasks remain separate from reference-only task rows."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

import capicove.books as books
import capicove.books.chunk_result_ledger as result_ledger
import capicove.books.workspace as book_workspace
import capicove.wiki.chunk_publication as chunk_publication
from capicove.books.chunk_config import resolve_chunk_config
from capicove.books.chunk_models import digest
from capicove.books.chunk_result_submission import canonical_json
from capicove.books.chunk_service import BookChunkService
from capicove.books.workspace import _atomic_write, _reject_symlinks
from capicove.wiki import (
    ChunkPublicationCitation,
    ChunkPublicationClaim,
    ChunkPublicationPacket,
    ChunkPublicationPage,
    WikiDependencyError,
    WikiIndexUnavailableError,
    WikiService,
    WikiUnsafePathError,
    WikiValidationError,
    migrate_to_v2,
)
from capicove.wiki._adapters.sqlite_index import rebuild_index as wiki_rebuild_index
from capicove.wiki.filesystem import atomic_write as wiki_atomic_write
from capicove.wiki.filesystem import scan as wiki_scan
from capicove.wiki.indexing import load_documents as wiki_load_documents

from .test_service import _Parser


def test_chunk_result_task_page_preserves_dag_ids_and_authorized_core_scope(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    reference = chunks.read_chunks(book_id, run_id, tasks=True)["items"]

    page = books.BookChunkResultService(workspace).list_tasks(book_id, run_id)

    assert page["total"] == 5
    assert page["next_offset"] is None
    assert [row["task_id"] for row in page["items"]] == [row["task_id"] for row in reference]
    assert all(row["submission_contract"] is None for row in reference)
    assert all(row["book_id"] == book_id and row["run_id"] == run_id for row in page["items"])
    assert all(row["submission_contract"] == "chunk-result-v1" for row in page["items"])
    assert all(row["max_result_bytes"] == 1024 * 1024 for row in page["items"])
    leaves = [row for row in page["items"] if row["task_type"] == "chunk_summary"]
    reductions = [row for row in page["items"] if row["task_type"] != "chunk_summary"]
    assert len(leaves) == 2
    assert len(reductions) == 3
    assert all(row["citation_policy"] == "chunk_core" for row in leaves)
    assert all(row["citation_policy"] == "dependency_claims" for row in reductions)
    assert len({row["citation_scope_sha256"] for row in leaves}) == 2
    assert all(len(row["citation_scope_sha256"]) == 64 for row in page["items"])


def test_leaf_submission_is_durable_and_identical_retry_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    task = next(
        row
        for row in books.BookChunkResultService(workspace).list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    submission = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": task["task_id"],
        "idempotency_key": "leaf-1",
        "dependency_result_ids": [],
        "abstract": "First chapter summary",
        "key_points": ["A cited fact"],
        "claims": [{"text": "A cited fact", "citations": [asdict(chunk.core_spans[0])]}],
    }

    first = books.BookChunkResultService(workspace).submit(submission)
    reopened = books.LocalBookWorkspace.open(workspace.root)
    repeated = books.BookChunkResultService(reopened).submit(submission)

    assert first == repeated
    assert first["version"] == 1
    assert first["task_id"] == task["task_id"]
    assert first["result_id"].startswith("chunk-result:")


def test_submission_with_unencodable_text_writes_no_result(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])

    with pytest.raises(ValueError, match="invalid abstract"):
        service.submit(
            {
                "schema_version": 1,
                "book_id": book_id,
                "run_id": run_id,
                "task_id": task["task_id"],
                "idempotency_key": "invalid-unicode",
                "dependency_result_ids": [],
                "abstract": "\ud800",
                "key_points": [],
                "claims": [
                    {
                        "text": "Cited fact",
                        "citations": [asdict(chunk.core_spans[0])],
                    }
                ],
            }
        )

    assert not (chunks._directory(book_id, run_id) / "chunk-results").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ledger write race")
def test_result_receipt_write_rejects_parent_swapped_after_symlink_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        item
        for item in service.list_tasks(book_id, run_id)["items"]
        if item["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    receipt = chunks._directory(book_id, run_id) / "chunk-results/v1/receipts.json"
    foreign = tmp_path / "foreign-ledger"
    foreign.mkdir()
    original_reject = book_workspace._reject_symlinks
    swapped = False

    def swap_after_receipt_check(path: Path) -> None:
        nonlocal swapped
        original_reject(path)
        if path == receipt and not swapped:
            receipt.parent.rename(receipt.parent.with_name("v1-held"))
            receipt.parent.symlink_to(foreign, target_is_directory=True)
            swapped = True

    monkeypatch.setattr(book_workspace, "_reject_symlinks", swap_after_receipt_check)

    with pytest.raises((OSError, ValueError)):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "receipt-parent-race",
        )

    assert swapped
    assert not (foreign / "receipts.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ledger write race")
def test_result_receipt_write_does_not_report_success_after_parent_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        item
        for item in service.list_tasks(book_id, run_id)["items"]
        if item["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    receipt = chunks._directory(book_id, run_id) / "chunk-results/v1/receipts.json"
    foreign = tmp_path / "foreign-ledger"
    foreign.mkdir()
    original_replace = os.replace
    swapped = False

    def swap_before_receipt_replace(source: Any, target: Any, **kwargs: Any) -> None:
        nonlocal swapped
        if target == "receipts.json" and not swapped:
            receipt.parent.rename(receipt.parent.with_name("v1-held"))
            receipt.parent.symlink_to(foreign, target_is_directory=True)
            swapped = True
        original_replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", swap_before_receipt_replace)

    with pytest.raises((OSError, ValueError)):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "receipt-moved-parent",
        )

    assert swapped
    assert not (foreign / "receipts.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ledger file race")
@pytest.mark.parametrize("target", ["receipt", "record", "sidecar"])
def test_result_ledger_rejects_file_swapped_after_symlink_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        item
        for item in service.list_tasks(book_id, run_id)["items"]
        if item["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(
        service,
        book_id,
        run_id,
        task,
        asdict(chunk.core_spans[0]),
        "accepted-leaf",
    )
    receipts = chunks._directory(book_id, run_id) / "chunk-results/v1/receipts.json"
    record = receipts.parent / "records/00000001/record.json"
    sidecar = record.parent / "sha256"
    checked_path = {"receipt": receipts, "record": record, "sidecar": sidecar}[target]
    foreign = tmp_path / f"foreign-{target}"
    foreign.write_bytes(checked_path.read_bytes())
    expected_bytes = foreign.read_bytes()
    original_reject = _reject_symlinks
    swapped = False

    def swap_after_file_check(path: Path) -> None:
        nonlocal swapped
        original_reject(path)
        if path == checked_path and not swapped:
            checked_path.rename(checked_path.with_name(f"{checked_path.name}.held"))
            checked_path.symlink_to(foreign)
            swapped = True

    monkeypatch.setattr(result_ledger, "_reject_symlinks", swap_after_file_check)

    with pytest.raises(ValueError):
        result_ledger.ChunkResultLedger(
            chunks._directory(book_id, run_id), run_id, book_id=book_id
        )._read_receipts()

    assert swapped
    assert foreign.read_bytes() == expected_bytes


def test_reduction_uses_accepted_child_claim_citations(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    parent = next(row for row in tasks if row["task_type"] == "section_summary")
    child = next(row for row in tasks if row["task_id"] == parent["dependency_task_ids"][0])
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == child["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])
    child_result = service.submit(
        {
            "schema_version": 1,
            "book_id": book_id,
            "run_id": run_id,
            "task_id": child["task_id"],
            "idempotency_key": "child",
            "dependency_result_ids": [],
            "abstract": "Cited chapter",
            "key_points": [],
            "claims": [{"text": "Chapter fact", "citations": [citation]}],
        }
    )
    updated_parent = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_id"] == parent["task_id"]
    )
    submission = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": parent["task_id"],
        "idempotency_key": "parent",
        "dependency_result_ids": [child_result["result_id"]],
        "abstract": "Reduced chapter",
        "key_points": [],
        "claims": [{"text": "Chapter fact", "citations": [citation]}],
    }

    parent_result = service.submit(submission)

    assert updated_parent["citation_scope_sha256"] != parent["citation_scope_sha256"]
    assert parent_result["version"] == 1
    assert parent_result["task_id"] == parent["task_id"]


def test_status_distinguishes_ready_complete_and_blocked_tasks(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    first_leaf = next(row for row in tasks if row["task_type"] == "chunk_summary")
    run, _, _ = chunks._load(book_id, run_id)
    first_chunk = next(item for item in run.chunks if item.chunk_id == first_leaf["chunk_ids"][0])
    service.submit(
        {
            "schema_version": 1,
            "book_id": book_id,
            "run_id": run_id,
            "task_id": first_leaf["task_id"],
            "idempotency_key": "first-leaf",
            "dependency_result_ids": [],
            "abstract": "Cited chapter",
            "key_points": [],
            "claims": [{"text": "Chapter fact", "citations": [asdict(first_chunk.core_spans[0])]}],
        }
    )

    status = service.status(book_id, run_id)
    states = {item["task_id"]: item["state"] for item in status["items"]}

    assert status["publishable"] is False
    assert status["total"] == 5
    assert states[first_leaf["task_id"]] == "complete"
    assert sum(state == "ready" for state in states.values()) == 2
    assert sum(state == "blocked" for state in states.values()) == 2


def test_retry_recovers_complete_record_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    submission = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": task["task_id"],
        "idempotency_key": "interrupted-leaf",
        "dependency_result_ids": [],
        "abstract": "Cited chapter",
        "key_points": [],
        "claims": [{"text": "Chapter fact", "citations": [asdict(chunk.core_spans[0])]}],
    }

    def fail_receipt(path: Path, content: bytes) -> None:
        if path.name == "receipts.json":
            raise OSError("interrupted before receipt")
        _atomic_write(path, content)

    with monkeypatch.context() as scoped:
        scoped.setattr(result_ledger, "_atomic_write", fail_receipt)
        with pytest.raises(OSError, match="interrupted before receipt"):
            service.submit(submission)

    assert service.status(book_id, run_id)["publishable"] is False
    first = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root)).submit(
        submission
    )
    repeated = service.submit(submission)

    assert first == repeated
    assert first["version"] == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX orphan read race")
def test_orphan_recovery_never_opens_external_record_after_symlink_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])

    def fail_receipt(path: Path, content: bytes) -> None:
        if path.name == "receipts.json":
            raise OSError("interrupted before receipt")
        _atomic_write(path, content)

    with monkeypatch.context() as scoped:
        scoped.setattr(result_ledger, "_atomic_write", fail_receipt)
        with pytest.raises(OSError, match="interrupted before receipt"):
            _submit_claim(service, book_id, run_id, task, citation, "orphan")

    record = chunks._directory(book_id, run_id) / "chunk-results/v1/records/00000001/record.json"
    foreign = tmp_path / "foreign-record.json"
    foreign.write_bytes(record.read_bytes())
    original_reject = _reject_symlinks
    original_read = Path.read_bytes
    swapped = False

    def swap_after_record_check(path: Path) -> None:
        nonlocal swapped
        original_reject(path)
        if path == record and not swapped:
            record.rename(record.with_name("record-held.json"))
            record.symlink_to(foreign)
            swapped = True

    def forbid_foreign_path_read(path: Path) -> bytes:
        if path == record and path.is_symlink():
            raise AssertionError("external record path was opened")
        return original_read(path)

    monkeypatch.setattr(result_ledger, "_reject_symlinks", swap_after_record_check)
    monkeypatch.setattr(Path, "read_bytes", forbid_foreign_path_read)

    with pytest.raises(ValueError, match="unrecovered chunk result record"):
        _submit_claim(service, book_id, run_id, task, citation, "new-version")

    assert swapped
    assert not (record.parent.parent.parent / "receipts.json").exists()


def test_retry_discards_uncommitted_result_staging_directory(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    records_root = chunks._directory(book_id, run_id) / "chunk-results/v1/records"
    crashed_staging = records_root / ".result-crashed"
    crashed_staging.mkdir(parents=True)
    (crashed_staging / "record.json").write_bytes(b'{"incomplete": true}')

    accepted = _submit_claim(
        service,
        book_id,
        run_id,
        task,
        asdict(chunk.core_spans[0]),
        "after-staging-crash",
    )

    assert accepted["version"] == 1
    assert not crashed_staging.exists()
    states = {row["task_id"]: row["state"] for row in service.status(book_id, run_id)["items"]}
    assert states[task["task_id"]] == "complete"


@pytest.mark.skipif(os.name != "posix", reason="POSIX staging symlink boundary")
def test_retry_rejects_symlinked_staging_without_touching_external(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    records_root = chunks._directory(book_id, run_id) / "chunk-results/v1/records"
    records_root.mkdir(parents=True)
    foreign = tmp_path / "foreign-staging"
    foreign.mkdir()
    marker = foreign / "do-not-delete"
    marker.write_bytes(b"private")
    (records_root / ".result-linked").symlink_to(foreign, target_is_directory=True)

    with pytest.raises(ValueError, match="staging"):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "linked-staging",
        )

    assert marker.read_bytes() == b"private"
    assert not (records_root.parent / "receipts.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX result staging race")
def test_result_staging_never_uses_records_root_swapped_after_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    records_root = chunks._directory(book_id, run_id) / "chunk-results/v1/records"
    records_root.mkdir(parents=True)
    foreign = tmp_path / "foreign-records"
    foreign.mkdir()
    record_directory = records_root / "00000001"
    original_reject = _reject_symlinks
    original_mkdtemp = tempfile.mkdtemp
    swapped = False
    checks = 0

    def swap_after_record_check(path: Path) -> None:
        nonlocal checks, swapped
        original_reject(path)
        if path == record_directory:
            checks += 1
        if path == record_directory and checks == 2 and not swapped:
            records_root.rename(records_root.with_name("records-held"))
            records_root.symlink_to(foreign, target_is_directory=True)
            swapped = True

    def forbid_external_staging(*args: Any, **kwargs: Any) -> str:
        if kwargs.get("dir") == records_root and records_root.is_symlink():
            raise AssertionError("external staging directory creation attempted")
        return cast(str, original_mkdtemp(*args, **kwargs))

    monkeypatch.setattr(result_ledger, "_reject_symlinks", swap_after_record_check)
    monkeypatch.setattr(tempfile, "mkdtemp", forbid_external_staging)

    with pytest.raises((OSError, ValueError)):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "records-parent-race",
        )

    assert swapped
    assert list(foreign.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX result staging race")
def test_result_staging_writes_stay_pinned_after_records_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    records_root = chunks._directory(book_id, run_id) / "chunk-results/v1/records"
    replacement = tmp_path / "foreign-records"
    replacement.mkdir()
    original_mkdir = os.mkdir
    swapped = False
    staging_name = ""

    def replace_after_staging_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped, staging_name
        original_mkdir(path, mode, dir_fd=dir_fd)
        if isinstance(path, str) and path.startswith(".result-") and dir_fd is not None:
            staging_name = path
            records_root.rename(records_root.with_name("records-held"))
            replacement.rename(records_root)
            original_mkdir(records_root / staging_name)
            swapped = True

    monkeypatch.setattr(os, "mkdir", replace_after_staging_mkdir)

    with pytest.raises((OSError, ValueError)):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "staging-write-parent-race",
        )

    assert swapped
    assert list((records_root / staging_name).iterdir()) == []
    assert not (records_root.parent / "receipts.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX result staging race")
def test_result_submit_rejects_staging_replacement_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    records_root = chunks._directory(book_id, run_id) / "chunk-results/v1/records"
    original_rename = os.rename
    swapped = False

    def replace_stage_before_install(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if (
            isinstance(src, str)
            and src.startswith(".result-")
            and dst == "00000001"
            and src_dir_fd is not None
            and dst_dir_fd is not None
        ):
            original_rename(src, ".result-held", src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            replacement = records_root / src
            replacement.mkdir()
            (replacement / "record.json").write_bytes(b"not an accepted result")
            (replacement / "sha256").write_bytes(b"not a digest")
            swapped = True
        original_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "rename", replace_stage_before_install)

    with pytest.raises(ValueError, match="staging"):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "staging-install-race",
        )

    assert swapped
    assert not (records_root.parent / "receipts.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX result staging race")
def test_result_submit_rejects_staged_record_tampering_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    records_root = chunks._directory(book_id, run_id) / "chunk-results/v1/records"
    original_rename = os.rename
    tampered = False

    def tamper_stage_before_install(
        src: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        dst: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal tampered
        if (
            isinstance(src, str)
            and src.startswith(".result-")
            and dst == "00000001"
            and src_dir_fd is not None
            and dst_dir_fd is not None
        ):
            (records_root / src / "record.json").write_bytes(b"tampered")
            tampered = True
        original_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    monkeypatch.setattr(os, "rename", tamper_stage_before_install)

    with pytest.raises(ValueError, match="hash mismatch"):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "staged-record-tamper",
        )

    assert tampered
    assert not (records_root.parent / "receipts.json").exists()


def test_interrupted_sidecar_write_can_retry_without_partial_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    submission = {
        "schema_version": 1,
        "book_id": book_id,
        "run_id": run_id,
        "task_id": task["task_id"],
        "idempotency_key": "sidecar-interrupted",
        "dependency_result_ids": [],
        "abstract": "Cited chapter",
        "key_points": [],
        "claims": [{"text": "Chapter fact", "citations": [asdict(chunk.core_spans[0])]}],
    }

    def fail_sidecar(path: Path, content: bytes) -> None:
        if path.suffix == ".sha256" or path.name == "sha256":
            raise OSError("interrupted before sidecar")
        _atomic_write(path, content)

    original_staged_write = result_ledger._write_staged_file_posix

    def fail_staged_sidecar(staging_fd: int, name: str, content: bytes) -> None:
        if name == "sha256":
            raise OSError("interrupted before sidecar")
        original_staged_write(staging_fd, name, content)

    with monkeypatch.context() as scoped:
        if os.name == "posix":
            scoped.setattr(result_ledger, "_write_staged_file_posix", fail_staged_sidecar)
        else:
            scoped.setattr(result_ledger, "_atomic_write", fail_sidecar)
        with pytest.raises(OSError, match="interrupted before sidecar"):
            service.submit(submission)

    assert service.status(book_id, run_id)["publishable"] is False
    accepted = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root)).submit(
        submission
    )
    assert accepted["version"] == 1


def test_new_submission_recovers_prior_complete_orphan_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])

    def fail_receipt(path: Path, content: bytes) -> None:
        if path.name == "receipts.json":
            raise OSError("receipt interrupted")
        _atomic_write(path, content)

    with monkeypatch.context() as scoped:
        scoped.setattr(result_ledger, "_atomic_write", fail_receipt)
        with pytest.raises(OSError, match="receipt interrupted"):
            _submit_claim(service, book_id, run_id, task, citation, "orphan")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    second = _submit_claim(reopened, book_id, run_id, task, citation, "new-version")
    first = _submit_claim(reopened, book_id, run_id, task, citation, "orphan")

    assert first["version"] == 1
    assert second["version"] == 2
    assert first["result_id"] != second["result_id"]
    assert reopened.status(book_id, run_id)["items"][0]["version"] == 2


def test_parent_submission_recovers_complete_child_orphan_before_dependency_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    parent = next(row for row in tasks if row["task_type"] == "section_summary")
    child = next(row for row in tasks if row["task_id"] == parent["dependency_task_ids"][0])
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == child["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])

    def fail_receipt(path: Path, content: bytes) -> None:
        if path.name == "receipts.json":
            raise OSError("receipt interrupted")
        _atomic_write(path, content)

    with monkeypatch.context() as scoped:
        scoped.setattr(result_ledger, "_atomic_write", fail_receipt)
        with pytest.raises(OSError, match="receipt interrupted"):
            _submit_claim(service, book_id, run_id, child, citation, "orphan-child")

    record_path = (
        chunks._directory(book_id, run_id) / "chunk-results/v1/records/00000001/record.json"
    )
    orphan_result_id = json.loads(record_path.read_text(encoding="utf-8"))["result_id"]
    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))

    accepted = _submit_claim(
        reopened,
        book_id,
        run_id,
        parent,
        citation,
        "parent-after-interruption",
        [orphan_result_id],
    )

    assert accepted["version"] == 1
    states = {row["task_id"]: row for row in reopened.status(book_id, run_id)["items"]}
    assert states[child["task_id"]]["state"] == "complete"
    assert states[parent["task_id"]]["state"] == "complete"


def test_tampered_orphan_citation_cannot_be_receipted_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])

    def fail_receipt(path: Path, content: bytes) -> None:
        if path.name == "receipts.json":
            raise OSError("receipt interrupted")
        _atomic_write(path, content)

    with monkeypatch.context() as scoped:
        scoped.setattr(result_ledger, "_atomic_write", fail_receipt)
        with pytest.raises(OSError, match="receipt interrupted"):
            _submit_claim(service, book_id, run_id, task, citation, "orphan")

    record_dir = (
        chunks._directory(book_id, run_id) / "chunk-results" / "v1" / "records" / "00000001"
    )
    record_path = record_dir / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["submission"]["claims"][0]["citations"][0]["text_sha256"] = "0" * 64
    identity = [run_id, task["task_id"], 1, record["submission"]]
    canonical_identity = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    record["result_id"] = "chunk-result:" + hashlib.sha256(canonical_identity).hexdigest()[:24]
    canonical_record = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record_path.write_bytes(canonical_record)
    (record_dir / "sha256").write_text(
        hashlib.sha256(canonical_record).hexdigest(), encoding="ascii"
    )

    with pytest.raises(ValueError, match="unrecovered chunk result record"):
        _submit_claim(service, book_id, run_id, task, citation, "new-version")

    assert not (record_dir.parent.parent / "receipts.json").exists()
    assert len(list(record_dir.parent.iterdir())) == 1


def test_stale_reduction_cannot_be_used_for_book_result(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    by_id = {task["task_id"]: task for task in tasks}
    book_task = next(task for task in tasks if task["task_type"] == "book_summary")
    run, _, _ = chunks._load(book_id, run_id)
    parents: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for number, parent_id in enumerate(book_task["dependency_task_ids"]):
        parent = by_id[parent_id]
        child = by_id[parent["dependency_task_ids"][0]]
        chunk = next(item for item in run.chunks if item.chunk_id == child["chunk_ids"][0])
        citation = asdict(chunk.core_spans[0])
        child_result = _submit_claim(service, book_id, run_id, child, citation, f"child-{number}")
        parent_result = _submit_claim(
            service,
            book_id,
            run_id,
            parent,
            citation,
            f"parent-{number}",
            [child_result["result_id"]],
        )
        parents.append(parent_result)
        citations.append(citation)
    first_parent = by_id[book_task["dependency_task_ids"][0]]
    first_child = by_id[first_parent["dependency_task_ids"][0]]
    _submit_claim(service, book_id, run_id, first_child, citations[0], "replacement")

    status = service.status(book_id, run_id)
    states = {item["task_id"]: item["state"] for item in status["items"]}
    assert states[first_parent["task_id"]] == "stale"
    with pytest.raises(ValueError, match="dependency_stale"):
        _submit_claim(
            service,
            book_id,
            run_id,
            book_task,
            citations[0],
            "book",
            [parent["result_id"] for parent in parents],
        )


def test_identical_parent_retry_returns_history_after_child_superseded(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    parent = next(task for task in tasks if task["task_type"] == "section_summary")
    child = next(task for task in tasks if task["task_id"] == parent["dependency_task_ids"][0])
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == child["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])
    child_result = _submit_claim(service, book_id, run_id, child, citation, "original-child")
    parent_result = _submit_claim(
        service, book_id, run_id, parent, citation, "original-parent", [child_result["result_id"]]
    )
    _submit_claim(service, book_id, run_id, child, citation, "replacement-child")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    repeated = _submit_claim(
        reopened, book_id, run_id, parent, citation, "original-parent", [child_result["result_id"]]
    )

    assert repeated == parent_result
    states = {item["task_id"]: item for item in reopened.status(book_id, run_id)["items"]}
    assert states[parent["task_id"]]["state"] == "stale"
    assert states[parent["task_id"]]["version"] == 1


def test_boolean_receipt_version_is_rejected_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(service, book_id, run_id, task, asdict(chunk.core_spans[0]), "leaf")
    receipt_path = (
        workspace.root
        / "books"
        / book_id.replace(":", "-")
        / "chunk-runs"
        / run_id.replace(":", "-")
        / "chunk-results"
        / "v1"
        / "receipts.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["records"][0]["version"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="invalid result receipt index"):
        reopened.status(book_id, run_id)


def test_unencodable_receipt_key_is_rejected_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(service, book_id, run_id, task, asdict(chunk.core_spans[0]), "leaf")
    receipt_path = chunks._directory(book_id, run_id) / "chunk-results/v1/receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["records"][0]["idempotency_key"] = "\ud800"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="invalid result receipt index"):
        reopened.status(book_id, run_id)


def test_duplicate_receipt_index_key_is_rejected_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(service, book_id, run_id, task, asdict(chunk.core_spans[0]), "leaf")
    receipt_path = chunks._directory(book_id, run_id) / "chunk-results/v1/receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = json.dumps(receipt["records"], separators=(",", ":"))
    receipt_path.write_text(
        '{"schema_version":1,"records":' + records + ',"records":' + records + "}",
        encoding="utf-8",
    )

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="invalid result receipt index"):
        reopened.status(book_id, run_id)


def test_receipt_key_must_match_accepted_result_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(service, book_id, run_id, task, asdict(chunk.core_spans[0]), "leaf")
    receipt_path = chunks._directory(book_id, run_id) / "chunk-results/v1/receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["records"][0]["idempotency_key"] = "different-valid-key"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="invalid result receipt index"):
        reopened.status(book_id, run_id)


def test_result_record_must_match_the_selected_book_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(service, book_id, run_id, task, asdict(chunk.core_spans[0]), "leaf")
    ledger = chunks._directory(book_id, run_id) / "chunk-results/v1"
    receipt_path = ledger / "receipts.json"
    record_path = ledger / "records/00000001/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["submission"]["book_id"] = "book:" + "f" * 24
    record["result_id"] = (
        "chunk-result:"
        + digest([run_id, task["task_id"], record["version"], record["submission"]])[:24]
    )
    raw = canonical_json(record)
    record_path.write_bytes(raw)
    receipt_hash = hashlib.sha256(raw).hexdigest()
    (record_path.parent / "sha256").write_text(receipt_hash)
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipts["records"][0]["result_id"] = record["result_id"]
    receipts["records"][0]["sha256"] = receipt_hash
    receipt_path.write_text(json.dumps(receipts), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="invalid chunk result identity"):
        reopened.status(book_id, run_id)


def test_unencodable_result_record_is_rejected_after_restart(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    _submit_claim(service, book_id, run_id, task, asdict(chunk.core_spans[0]), "leaf")
    ledger = chunks._directory(book_id, run_id) / "chunk-results/v1"
    receipt_path = ledger / "receipts.json"
    record_path = ledger / "records/00000001/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["submission"]["abstract"] = "\ud800"
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("ascii")
    record_path.write_bytes(raw)
    (record_path.parent / "sha256").write_text(hashlib.sha256(raw).hexdigest())
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipts["records"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    receipt_path.write_text(json.dumps(receipts), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="invalid chunk result record"):
        reopened.status(book_id, run_id)


def test_wrong_source_slice_hash_rejects_without_creating_result_area(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    bad_citation = asdict(chunk.core_spans[0])
    bad_citation["text_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="citation_invalid"):
        _submit_claim(service, book_id, run_id, task, bad_citation, "wrong-slice")

    run_directory = (
        workspace.root
        / "books"
        / book_id.replace(":", "-")
        / "chunk-runs"
        / run_id.replace(":", "-")
    )
    assert not (run_directory / "chunk-results").exists()


def test_foreign_or_out_of_core_citation_writes_no_result(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    leaves = [
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    ]
    assert len(leaves) == 2
    run, _, _ = chunks._load(book_id, run_id)
    foreign_chunk = next(item for item in run.chunks if item.chunk_id == leaves[1]["chunk_ids"][0])

    with pytest.raises(ValueError, match="citation_invalid"):
        _submit_claim(
            service,
            book_id,
            run_id,
            leaves[0],
            asdict(foreign_chunk.core_spans[0]),
            "foreign-core",
        )

    run_directory = (
        workspace.root
        / "books"
        / book_id.replace(":", "-")
        / "chunk-runs"
        / run_id.replace(":", "-")
    )
    assert not (run_directory / "chunk-results").exists()


def test_changed_pdf_rejects_submission_without_record(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, parsed, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    (workspace.root / "raw" / f"{parsed.source_sha256}.pdf").write_bytes(b"%PDF-changed")

    with pytest.raises(ValueError, match="source_changed"):
        _submit_claim(
            service,
            book_id,
            run_id,
            task,
            asdict(chunk.core_spans[0]),
            "changed-pdf",
        )

    run_directory = (
        workspace.root
        / "books"
        / book_id.replace(":", "-")
        / "chunk-runs"
        / run_id.replace(":", "-")
    )
    assert not (run_directory / "chunk-results").exists()


def test_new_result_version_preserves_old_record_and_supersedes_identity(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    task = next(
        row
        for row in service.list_tasks(book_id, run_id)["items"]
        if row["task_type"] == "chunk_summary"
    )
    run, _, _ = chunks._load(book_id, run_id)
    chunk = next(item for item in run.chunks if item.chunk_id == task["chunk_ids"][0])
    citation = asdict(chunk.core_spans[0])
    first = _submit_claim(service, book_id, run_id, task, citation, "first")
    records = (
        workspace.root
        / "books"
        / book_id.replace(":", "-")
        / "chunk-runs"
        / run_id.replace(":", "-")
        / "chunk-results"
        / "v1"
        / "records"
    )
    original_bytes = (records / "00000001" / "record.json").read_bytes()

    second = _submit_claim(service, book_id, run_id, task, citation, "second")

    first_record = json.loads(original_bytes)
    second_record = json.loads((records / "00000002" / "record.json").read_bytes())
    assert (records / "00000001" / "record.json").read_bytes() == original_bytes
    assert first_record["supersedes_result_id"] is None
    assert second_record["supersedes_result_id"] == first["result_id"]
    assert second_record["result_id"] == second["result_id"]


def test_unknown_chunk_result_version_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    versions = (
        workspace.root
        / "books"
        / book_id.replace(":", "-")
        / "chunk-runs"
        / run_id.replace(":", "-")
        / "chunk-results"
    )
    (versions / "v2").mkdir(parents=True)

    with pytest.raises(ValueError, match="unknown chunk result version"):
        books.BookChunkResultService(workspace).list_tasks(book_id, run_id)


def test_complete_dag_has_stable_source_pinned_publication_manifest(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, result, _ = chunks._load(book_id, run_id)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"task-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation

    manifest = service.publication_manifest(book_id, run_id)
    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))

    assert manifest == reopened.publication_manifest(book_id, run_id)
    assert manifest["publication_id"].startswith("chunk-publication:")
    assert manifest["book_id"] == book_id
    assert manifest["run_id"] == run_id
    assert manifest["result_ids"] == [accepted[task["task_id"]]["result_id"] for task in tasks]
    assert manifest["source_sha256"] == result.source_sha256
    assert manifest["representation_sha256"] == result.representation_sha256
    assert manifest["wiki_schema_version"] == 2
    assert manifest["render_policy_version"] == 1


@pytest.mark.parametrize("operation", ["status", "publication_manifest"])
def test_complete_result_consumer_rechecks_receipted_citations(
    tmp_path: Path, operation: str
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"verify-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    assert (
        service.publication_manifest(book_id, run_id)["result_ids"][-1]
        == accepted[tasks[-1]["task_id"]]["result_id"]
    )

    ledger = chunks._directory(book_id, run_id) / "chunk-results/v1"
    receipt_path = ledger / "receipts.json"
    record_path = ledger / f"records/{len(tasks):08d}/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["submission"]["claims"][0]["citations"][0]["text_sha256"] = "0" * 64
    record["result_id"] = (
        "chunk-result:"
        + digest([run_id, tasks[-1]["task_id"], record["version"], record["submission"]])[:24]
    )
    raw = canonical_json(record)
    record_path.write_bytes(raw)
    receipt_hash = hashlib.sha256(raw).hexdigest()
    (record_path.parent / "sha256").write_text(receipt_hash)
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipts["records"][-1]["result_id"] = record["result_id"]
    receipts["records"][-1]["sha256"] = receipt_hash
    receipt_path.write_text(json.dumps(receipts), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="citation_invalid"):
        getattr(reopened, operation)(book_id, run_id)


def test_reduction_rejects_semantically_tampered_receipted_child(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    parent = next(task for task in tasks if task["task_type"] == "section_summary")
    child = next(task for task in tasks if task["task_id"] == parent["dependency_task_ids"][0])
    foreign_leaf = next(
        task
        for task in tasks
        if task["task_type"] == "chunk_summary" and task["task_id"] != child["task_id"]
    )
    run, _, _ = chunks._load(book_id, run_id)
    by_chunk = {chunk.chunk_id: chunk for chunk in run.chunks}
    child_citation = asdict(by_chunk[child["chunk_ids"][0]].core_spans[0])
    foreign_citation = asdict(by_chunk[foreign_leaf["chunk_ids"][0]].core_spans[0])
    assert foreign_citation != child_citation
    _submit_claim(service, book_id, run_id, child, child_citation, "child")

    ledger = chunks._directory(book_id, run_id) / "chunk-results/v1"
    record_path = ledger / "records/00000001/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["submission"]["claims"][0]["citations"] = [foreign_citation]
    record["result_id"] = (
        "chunk-result:"
        + digest([run_id, child["task_id"], record["version"], record["submission"]])[:24]
    )
    raw = canonical_json(record)
    record_path.write_bytes(raw)
    receipt_hash = hashlib.sha256(raw).hexdigest()
    (record_path.parent / "sha256").write_text(receipt_hash)
    receipt_path = ledger / "receipts.json"
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipts["records"][0]["result_id"] = record["result_id"]
    receipts["records"][0]["sha256"] = receipt_hash
    receipt_path.write_text(json.dumps(receipts), encoding="utf-8")

    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    with pytest.raises(ValueError, match="citation_invalid"):
        _submit_claim(
            reopened,
            book_id,
            run_id,
            parent,
            foreign_citation,
            "parent",
            [record["result_id"]],
        )

    assert not (ledger / "records/00000002").exists()
    assert len(json.loads(receipt_path.read_text(encoding="utf-8"))["records"]) == 1


def test_book_result_rechecks_semantically_tampered_grandchild(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    book_task = next(task for task in tasks if task["task_type"] == "book_summary")
    first_parent = next(
        task
        for task in tasks
        if task["task_type"] == "section_summary"
        and task["task_id"] in book_task["dependency_task_ids"]
    )
    first_leaf = next(
        task for task in tasks if task["task_id"] == first_parent["dependency_task_ids"][0]
    )
    other_leaf = next(
        task
        for task in tasks
        if task["task_type"] == "chunk_summary" and task["task_id"] != first_leaf["task_id"]
    )
    run, _, _ = chunks._load(book_id, run_id)
    by_chunk = {chunk.chunk_id: chunk for chunk in run.chunks}
    foreign_citation = asdict(by_chunk[other_leaf["chunk_ids"][0]].core_spans[0])
    accepted: dict[str, dict[str, Any]] = {}
    citations: dict[str, dict[str, Any]] = {}
    positions: dict[str, int] = {}
    for position, task in enumerate(tasks[:-1], 1):
        dependencies = task["dependency_task_ids"]
        citation = (
            asdict(by_chunk[task["chunk_ids"][0]].core_spans[0])
            if task["task_type"] == "chunk_summary"
            else citations[dependencies[0]]
        )
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"accepted-{position}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
        positions[task["task_id"]] = position

    ledger = chunks._directory(book_id, run_id) / "chunk-results/v1"
    receipt_path = ledger / "receipts.json"

    def tamper(task: dict[str, Any], *, dependency_ids: list[str] | None = None) -> str:
        position = positions[task["task_id"]]
        record_path = ledger / f"records/{position:08d}/record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["submission"]["claims"][0]["citations"] = [foreign_citation]
        if dependency_ids is not None:
            record["submission"]["dependency_result_ids"] = dependency_ids
        record["result_id"] = (
            "chunk-result:"
            + digest([run_id, task["task_id"], record["version"], record["submission"]])[:24]
        )
        raw = canonical_json(record)
        record_path.write_bytes(raw)
        receipt_hash = hashlib.sha256(raw).hexdigest()
        (record_path.parent / "sha256").write_text(receipt_hash)
        receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipts["records"][position - 1]["result_id"] = record["result_id"]
        receipts["records"][position - 1]["sha256"] = receipt_hash
        receipt_path.write_text(json.dumps(receipts), encoding="utf-8")
        return str(record["result_id"])

    new_leaf_id = tamper(first_leaf)
    new_parent_id = tamper(first_parent, dependency_ids=[new_leaf_id])
    book_dependencies = [
        new_parent_id
        if identifier == first_parent["task_id"]
        else accepted[identifier]["result_id"]
        for identifier in book_task["dependency_task_ids"]
    ]
    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))

    with pytest.raises(ValueError, match="citation_invalid"):
        _submit_claim(
            reopened,
            book_id,
            run_id,
            book_task,
            foreign_citation,
            "book",
            book_dependencies,
        )

    assert not (ledger / "records/00000005").exists()
    assert len(json.loads(receipt_path.read_text(encoding="utf-8"))["records"]) == 4


def test_complete_chunk_results_publish_searchable_cited_wiki(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, result, _ = chunks._load(book_id, run_id)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"publish-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)

    published = service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 2))
    hits = wiki.query("Cited fact").hits
    citation_lines = (
        (wiki.wiki_path / "graph" / "citations.jsonl").read_text(encoding="utf-8").splitlines()
    )
    ledger = [json.loads(line) for line in citation_lines]

    assert (
        published["publication_id"]
        == service.publication_manifest(book_id, run_id)["publication_id"]
    )
    assert published["wiki_schema_version"] == 2
    assert len(published["page_paths"]) >= 2
    assert len(hits) >= 2
    assert any("Cited fact" in hit.snippet for hit in hits)
    assert published["claim_ids"]
    assert all(record["book_id"] == book_id for record in ledger)
    assert all(record["source_sha256"] == result.source_sha256 for record in ledger)
    assert all(record["representation_sha256"] == result.representation_sha256 for record in ledger)
    assert {record["page_index"] for record in ledger} == {0, 1}
    assert all(record["block_ids"] for record in ledger)


def test_public_chunk_publish_tool_requires_bound_wiki_and_write_authority(tmp_path: Path) -> None:
    """The Tool should publish the bound Wiki and reject caller-selected destinations."""
    from datetime import UTC, datetime, timedelta

    from capicove.contracts import InvocationContext, ToolErrorCode, ToolStatus
    from capicove.tools.book_chunk_results import (
        BOOK_RESULT_TOOL_SPECS,
        BookChunkResultToolHandler,
    )

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"tool-publish-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    spec = BOOK_RESULT_TOOL_SPECS["chunk-publish"]
    handler = BookChunkResultToolHandler("chunk-publish", workspace=workspace, wiki_service=wiki)
    arguments = {"book_id": book_id, "run_id": run_id}

    def context(permissions: frozenset[str]) -> InvocationContext:
        return InvocationContext(
            "request",
            "trace",
            "workspace",
            "agent",
            "local",
            datetime.now(UTC) + timedelta(minutes=1),
            permissions,
        )

    denied = handler(context(frozenset({"book:write"})), arguments)
    foreign = handler(
        context(frozenset({"book:write", "wiki:write"})),
        {**arguments, "wiki_root": str(tmp_path / "foreign")},
    )
    published = handler(context(frozenset({"book:write", "wiki:write"})), arguments)

    assert spec.required_permissions == frozenset({"book:write", "wiki:write"})
    assert "wiki_root" not in cast(dict[str, object], spec.input_schema["properties"])
    assert denied.status is ToolStatus.FAILED
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.PERMISSION_DENIED
    assert foreign.status is ToolStatus.FAILED
    assert published.status is ToolStatus.SUCCEEDED
    assert published.result is not None
    assert (
        published.result["publication_id"]
        == service.publication_manifest(book_id, run_id)["publication_id"]
    )
    assert len(wiki.query("Cited fact").hits) >= 2


@pytest.mark.parametrize("artifact", ["raw", "representations"])
def test_repeated_publication_rejects_tampered_wiki_evidence(tmp_path: Path, artifact: str) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, parsed, _ = chunks._load(book_id, run_id)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"tamper-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    published = service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 2))
    assert service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 3)) == published
    if artifact == "raw":
        target = wiki.wiki_path / "raw" / f"{parsed.source_sha256}.pdf"
    else:
        target = wiki.wiki_path / "representations" / f"{parsed.representation_sha256}.json"
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="publication_conflict"):
        service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 3))

    assert (
        published["publication_id"]
        == service.publication_manifest(book_id, run_id)["publication_id"]
    )
    assert target.read_bytes() == b"tampered"


def test_wiki_publisher_rejects_packet_path_escape_before_any_write(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    malicious_book_id = "../../../escape"
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": malicious_book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    packet = ChunkPublicationPacket(
        "chunk-publication:" + hashlib.sha256(canonical).hexdigest()[:24],
        malicious_book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)

    with pytest.raises(ValueError, match="publication_conflict"):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert not (workspace.root / "escape").exists()
    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()
    assert not (wiki.wiki_path / "raw" / f"{parsed.source_sha256}.pdf").exists()


def test_wiki_publisher_missing_yaml_dependency_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    packet = ChunkPublicationPacket(
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24],
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    monkeypatch.setitem(sys.modules, "yaml", None)

    with pytest.raises(WikiDependencyError, match="capicove\\[wiki\\]"):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()
    assert not (wiki.wiki_path / "raw" / f"{parsed.source_sha256}.pdf").exists()
    assert not tuple((wiki.wiki_path / "pages" / "books").rglob("*.md"))


@pytest.mark.parametrize(
    ("record_kind", "overrides"),
    (
        ("pending", {"schema_version": True}),
        ("pending", {"published_on": "private-pending-value"}),
        ("receipt", {"published_on": "private-receipt-value"}),
        ("pending_duplicate", {}),
    ),
)
def test_wiki_publisher_rejects_malformed_publication_metadata(
    tmp_path: Path, record_kind: str, overrides: dict[str, object]
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    publication_id = "chunk-publication:" + hashlib.sha256(canonical).hexdigest()[:24]
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    path = (
        wiki.wiki_path / "_migration" / "chunk-publication.pending"
        if record_kind.startswith("pending")
        else wiki.wiki_path
        / "_migration"
        / "chunk-publications"
        / (publication_id.replace(":", "-") + ".json")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "publication_id": publication_id,
        "published_on": "2026-09-02",
        **overrides,
    }
    raw_metadata = json.dumps(metadata)
    if record_kind == "pending_duplicate":
        raw_metadata = raw_metadata[:-1] + f', "publication_id": "{publication_id}"' + "}"
    path.write_text(raw_metadata, encoding="utf-8")

    with pytest.raises(ValueError, match="publication_conflict") as captured:
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 3))

    assert "private-" not in str(captured.value)
    assert path.exists()
    assert not (wiki.wiki_path / "raw" / f"{parsed.source_sha256}.pdf").exists()


def test_wiki_publisher_renders_claim_text_without_raw_html(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    block = parsed.pages[0].blocks[0]
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    citation = ChunkPublicationCitation(
        block.block_id,
        block.page_index,
        0,
        len(block.exact_text),
        hashlib.sha256(block.exact_text.encode()).hexdigest(),
    )
    packet = ChunkPublicationPacket(
        "chunk-publication:" + hashlib.sha256(canonical).hexdigest()[:24],
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (
            ChunkPublicationPage(
                "book",
                "Overview",
                (
                    ChunkPublicationClaim(
                        "<script>alert(1)</script>", (citation,), result_ids[0], 1
                    ),
                ),
            ),
        ),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)

    published = wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))
    paths = published["page_paths"]
    assert isinstance(paths, list) and paths and isinstance(paths[0], str)
    page = (wiki.wiki_path / paths[0]).read_text(encoding="utf-8")

    assert "<script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_receipted_publication_retry_restores_missing_search_index(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    packet = ChunkPublicationPacket(
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24],
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    published = wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))
    assert wiki.query("Overview").hits
    (wiki.wiki_path / "_derived" / "search.sqlite3").unlink()
    with pytest.raises(WikiIndexUnavailableError):
        wiki.query("Overview")

    retried = wiki.publish_chunk_results(packet, published_on=date(2026, 9, 3))

    assert retried == published
    assert wiki.query("Overview").hits


def test_receipted_publication_retry_refuses_invalid_page_before_index_rebuild(
    tmp_path: Path,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    packet = ChunkPublicationPacket(
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24],
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))
    (wiki.wiki_path / "pages" / "invalid.md").write_text(
        "---\nid: invalid\ntitle: Invalid\nkind: summary\ncreated: 2026-09-02\n"
        "updated: 2026-09-02\nsources: []\n---\nInvalid page.\n",
        encoding="utf-8",
    )
    index = wiki.wiki_path / "_derived" / "search.sqlite3"
    index.unlink()
    assert not wiki.lint().valid

    with pytest.raises(WikiValidationError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 3))

    assert not index.exists()


@pytest.mark.parametrize(
    "interrupt_stage",
    (
        "raw",
        "representations",
        "pages",
        "citations",
        "search_index",
        "index_page",
        "receipt",
        "marker_cleanup",
    ),
)
def test_interrupted_publication_is_not_query_visible_and_same_id_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt_stage: str
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"interrupted-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    prior_source = tmp_path / "prior.md"
    prior_source.write_text("# Prior page\n\nExisting local evidence remains intact.\n")
    prior = wiki.ingest(prior_source, ingested_on=date(2026, 9, 1))
    wiki.build(built_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    prior_page = wiki.wiki_path / prior.page_path
    prior_bytes = prior_page.read_bytes()
    original_write = wiki_atomic_write
    original_posix_write = chunk_publication._atomic_write_posix
    original_ensure = chunk_publication._ensure_content
    original_posix_rebuild = chunk_publication._rebuild_index_posix
    original_rebuild = wiki_rebuild_index
    original_unlink = chunk_publication._unlink_marker

    def should_interrupt(relative: Path) -> bool:
        if interrupt_stage == "receipt":
            return relative.parent.name == "chunk-publications"
        if interrupt_stage == "citations":
            return relative == Path("graph/citations.jsonl")
        if interrupt_stage == "index_page":
            return relative == Path("index.md")
        return relative.parts[0] == interrupt_stage

    def interrupt_receipt(path: Path, content: bytes | str) -> None:
        if should_interrupt(path.relative_to(wiki.wiki_path)):
            raise OSError("simulated interruption during publication")
        original_write(path, content)

    def interrupt_posix_receipt(root: Path, relative: Path, content: bytes) -> None:
        if should_interrupt(relative):
            raise OSError("simulated interruption during publication")
        original_posix_write(root, relative, content)

    def interrupt_content(path: Path, content: bytes, *, wiki: Path) -> None:
        if should_interrupt(path.relative_to(wiki)):
            raise OSError("simulated interruption during publication")
        original_ensure(path, content, wiki=wiki)

    def interrupt_posix_index(root: Path, documents: Any) -> None:
        if interrupt_stage == "search_index":
            raise OSError("simulated interruption during publication")
        original_posix_rebuild(root, documents)

    def interrupt_index(path: Path, documents: Any) -> None:
        if interrupt_stage == "search_index":
            raise OSError("simulated interruption during publication")
        original_rebuild(path, documents)

    def interrupt_marker(root: Path, marker: Path) -> None:
        if interrupt_stage == "marker_cleanup":
            raise OSError("simulated interruption during publication")
        original_unlink(root, marker)

    monkeypatch.setattr(chunk_publication, "atomic_write", interrupt_receipt)
    monkeypatch.setattr(chunk_publication, "_atomic_write_posix", interrupt_posix_receipt)
    monkeypatch.setattr(chunk_publication, "_ensure_content", interrupt_content)
    monkeypatch.setattr(chunk_publication, "_rebuild_index_posix", interrupt_posix_index)
    monkeypatch.setattr(chunk_publication, "rebuild_index", interrupt_index)
    monkeypatch.setattr(chunk_publication, "_unlink_marker", interrupt_marker)
    with pytest.raises(OSError, match="simulated interruption"):
        service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 2))
    marker = wiki.wiki_path / "_migration" / "chunk-publication.pending"
    assert marker.exists()
    assert prior_page.read_bytes() == prior_bytes
    with pytest.raises(WikiIndexUnavailableError):
        wiki.query("Cited fact")

    monkeypatch.undo()
    reopened = books.BookChunkResultService(books.LocalBookWorkspace.open(workspace.root))
    recovered_wiki = WikiService(workspace.root)
    result = reopened.publish(book_id, run_id, recovered_wiki, published_on=date(2026, 9, 3))

    assert (
        result["publication_id"] == reopened.publication_manifest(book_id, run_id)["publication_id"]
    )
    assert not marker.exists()
    assert recovered_wiki.query("Cited fact").hits
    assert recovered_wiki.query("Existing local evidence").hits
    assert prior_page.read_bytes() == prior_bytes


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink race requires native POSIX handles")
def test_wiki_publisher_rejects_source_swap_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    source_path = workspace.root / "raw" / f"{parsed.source_sha256}.pdf"
    packet = ChunkPublicationPacket(
        "chunk-publication:" + hashlib.sha256(canonical).hexdigest()[:24],
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        source_path,
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    foreign = tmp_path / "foreign.pdf"
    foreign.write_bytes(source_path.read_bytes())
    original_lstat = Path.lstat
    swapped = False

    def swap_after_lstat(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = original_lstat(path)
        if path == source_path and not swapped:
            swapped = True
            source_path.rename(tmp_path / "held.pdf")
            source_path.symlink_to(foreign)
        return observed

    monkeypatch.setattr(Path, "lstat", swap_after_lstat)
    with pytest.raises(ValueError, match="source_changed"):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert swapped
    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()
    assert not (wiki.wiki_path / "raw" / f"{parsed.source_sha256}.pdf").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_writes_through_swapped_page_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    books_directory = wiki.wiki_path / "pages" / "books"
    original_prepare = chunk_publication._prepare

    def swap_after_preparation(value: ChunkPublicationPacket, published_on: date) -> Any:
        prepared = original_prepare(value, published_on)
        books_directory.rename(wiki.wiki_path / "pages" / "books-held")
        books_directory.symlink_to(foreign, target_is_directory=True)
        return prepared

    monkeypatch.setattr(chunk_publication, "_prepare", swap_after_preparation)

    with pytest.raises((WikiUnsafePathError, WikiValidationError)):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert not list(foreign.rglob("*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_replaces_citations_through_swapped_graph_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    graph_directory = wiki.wiki_path / "graph"
    foreign = tmp_path / "foreign-graph"
    shutil.copytree(graph_directory, foreign)
    foreign_citations = foreign / "citations.jsonl"
    foreign_citations.write_bytes(b"\n")
    original_prepare = chunk_publication._prepare

    def swap_after_preparation(value: ChunkPublicationPacket, published_on: date) -> Any:
        prepared = original_prepare(value, published_on)
        graph_directory.rename(wiki.wiki_path / "graph-held")
        graph_directory.symlink_to(foreign, target_is_directory=True)
        return prepared

    monkeypatch.setattr(chunk_publication, "_prepare", swap_after_preparation)

    with pytest.raises((WikiUnsafePathError, WikiValidationError)):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert foreign_citations.read_bytes() == b"\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_rejects_graph_swap_before_ontology_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    graph = wiki.wiki_path / "graph"
    foreign = tmp_path / "foreign-graph"
    shutil.copytree(graph, foreign)
    (foreign / "ontology.yaml").write_text("[private content", encoding="utf-8")
    original_scan = wiki_scan

    def swap_after_scan(path: Path) -> Any:
        inventory = original_scan(path)
        graph.rename(wiki.wiki_path / "graph-held")
        graph.symlink_to(foreign, target_is_directory=True)
        return inventory

    monkeypatch.setattr(chunk_publication, "scan", swap_after_scan)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert (foreign / "ontology.yaml").read_text(encoding="utf-8") == "[private content"
    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_rejects_swapped_graph_before_citation_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    graph_directory = wiki.wiki_path / "graph"
    foreign = tmp_path / "foreign-graph"
    shutil.copytree(graph_directory, foreign)
    (foreign / "citations.jsonl").unlink()
    (foreign / "citations.jsonl").mkdir()
    original_prepare = chunk_publication._prepare

    def swap_after_preparation(value: ChunkPublicationPacket, published_on: date) -> Any:
        prepared = original_prepare(value, published_on)
        graph_directory.rename(wiki.wiki_path / "graph-held")
        graph_directory.symlink_to(foreign, target_is_directory=True)
        return prepared

    monkeypatch.setattr(chunk_publication, "_prepare", swap_after_preparation)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_rejects_swapped_migration_before_marker_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    migration_directory = wiki.wiki_path / "_migration"
    foreign = tmp_path / "foreign-migration"
    shutil.copytree(migration_directory, foreign)
    (foreign / "chunk-publication.pending").write_bytes(
        json.dumps(
            {
                "schema_version": 1,
                "publication_id": publication_id,
                "published_on": "not-a-date",
            }
        ).encode()
    )

    def swap_after_scan(path: Path) -> Any:
        inventory = wiki_scan(path)
        migration_directory.rename(wiki.wiki_path / "migration-held")
        migration_directory.symlink_to(foreign, target_is_directory=True)
        return inventory

    monkeypatch.setattr(chunk_publication, "scan", swap_after_scan)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_rejects_swapped_receipt_parent_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))
    receipt_parent = wiki.wiki_path / "_migration" / "chunk-publications"
    receipt = receipt_parent / f"{publication_id.replace(':', '-')}.json"
    foreign = tmp_path / "foreign-receipts"
    shutil.copytree(receipt_parent, foreign)
    foreign_receipt = foreign / receipt.name
    expected_bytes = foreign_receipt.read_bytes()
    original_read_json = chunk_publication._read_json
    swapped = False

    def swap_after_receipt_read(path: Path, *, wiki: Path) -> dict[str, Any]:
        nonlocal swapped
        value = original_read_json(path, wiki=wiki)
        if path == receipt and not swapped:
            receipt_parent.rename(wiki / "_migration" / "chunk-publications-held")
            receipt_parent.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return value

    monkeypatch.setattr(chunk_publication, "_read_json", swap_after_receipt_read)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 3))

    assert swapped
    assert foreign_receipt.read_bytes() == expected_bytes


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_rejects_swapped_raw_parent_on_receipted_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))
    raw_directory = wiki.wiki_path / "raw"
    foreign = tmp_path / "foreign-raw"
    shutil.copytree(raw_directory, foreign)
    foreign_pdf = foreign / f"{parsed.source_sha256}.pdf"
    expected_bytes = foreign_pdf.read_bytes()
    original_read = chunk_publication._read_wiki_bytes_posix
    swapped = False

    def swap_after_citation_read(
        root: Path, relative: Path, *, max_bytes: int | None = None
    ) -> bytes:
        nonlocal swapped
        content = original_read(root, relative, max_bytes=max_bytes)
        if relative == Path("graph/citations.jsonl") and not swapped:
            raw_directory.rename(wiki.wiki_path / "raw-held")
            raw_directory.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return content

    monkeypatch.setattr(chunk_publication, "_read_wiki_bytes_posix", swap_after_citation_read)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 3))

    assert swapped
    assert foreign_pdf.read_bytes() == expected_bytes


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_rejects_swapped_raw_parent_before_pending_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    raw_directory = wiki.wiki_path / "raw"
    foreign = tmp_path / "foreign-raw"
    foreign.mkdir()
    (foreign / f"{parsed.source_sha256}.pdf").mkdir()
    original_prepare = chunk_publication._prepare
    swapped = False

    def swap_after_preparation(value: ChunkPublicationPacket, published_on: date) -> Any:
        nonlocal swapped
        prepared = original_prepare(value, published_on)
        raw_directory.rename(wiki.wiki_path / "raw-held")
        raw_directory.symlink_to(foreign, target_is_directory=True)
        swapped = True
        return prepared

    monkeypatch.setattr(chunk_publication, "_prepare", swap_after_preparation)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert swapped
    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_writes_marker_through_swapped_migration_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    migration_directory = wiki.wiki_path / "_migration"
    foreign = tmp_path / "foreign-migration"
    shutil.copytree(migration_directory, foreign)
    original_prepare = chunk_publication._prepare

    def swap_after_preparation(value: ChunkPublicationPacket, published_on: date) -> Any:
        prepared = original_prepare(value, published_on)
        migration_directory.rename(wiki.wiki_path / "migration-held")
        migration_directory.symlink_to(foreign, target_is_directory=True)
        return prepared

    monkeypatch.setattr(chunk_publication, "_prepare", swap_after_preparation)

    with pytest.raises((WikiUnsafePathError, WikiValidationError)):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert not (foreign / "chunk-publication.pending").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_writes_receipt_through_swapped_migration_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    migration_directory = wiki.wiki_path / "_migration"
    foreign = tmp_path / "foreign-migration"
    original_rebuild_index = wiki_rebuild_index

    def swap_after_index(path: Path, documents: Any) -> None:
        original_rebuild_index(path, documents)
        shutil.copytree(migration_directory, foreign)
        migration_directory.rename(wiki.wiki_path / "migration-held")
        migration_directory.symlink_to(foreign, target_is_directory=True)

    monkeypatch.setattr(chunk_publication, "rebuild_index", swap_after_index)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    receipt_name = publication_id.replace(":", "-") + ".json"
    assert not (foreign / "chunk-publications" / receipt_name).exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_builds_index_through_swapped_derived_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    derived_directory = wiki.wiki_path / "_derived"
    foreign = tmp_path / "foreign-derived"
    shutil.copytree(derived_directory, foreign)
    original_load_documents = wiki_load_documents

    def swap_after_loading(path: Path) -> Any:
        documents = original_load_documents(path)
        derived_directory.rename(wiki.wiki_path / "derived-held")
        derived_directory.symlink_to(foreign, target_is_directory=True)
        return documents

    monkeypatch.setattr(chunk_publication, "load_documents", swap_after_loading)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert not list(foreign.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_replaces_index_page_through_swapped_wiki_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    wiki_root = wiki.wiki_path
    foreign = tmp_path / "foreign-wiki"
    original_rebuild_index = chunk_publication._rebuild_index_posix

    def swap_after_index(path: Path, documents: Any) -> None:
        original_rebuild_index(path, documents)
        shutil.copytree(wiki_root, foreign)
        (foreign / "index.md").write_bytes(b"untouched external index")
        wiki_root.rename(workspace.root / "wiki-held")
        wiki_root.symlink_to(foreign, target_is_directory=True)

    monkeypatch.setattr(chunk_publication, "_rebuild_index_posix", swap_after_index)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert (foreign / "index.md").read_bytes() == b"untouched external index"


@pytest.mark.skipif(os.name != "posix", reason="POSIX publication path race")
def test_wiki_publisher_never_unlinks_external_pending_marker_after_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    _, parsed, _ = chunks._load(book_id, run_id)
    result_ids = ("chunk-result:" + "0" * 24,)
    manifest = {
        "book_id": book_id,
        "run_id": run_id,
        "result_ids": list(result_ids),
        "source_sha256": parsed.source_sha256,
        "representation_sha256": parsed.representation_sha256,
        "wiki_schema_version": 2,
        "render_policy_version": 1,
    }
    publication_id = (
        "chunk-publication:"
        + hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
    )
    packet = ChunkPublicationPacket(
        publication_id,
        book_id,
        run_id,
        result_ids,
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    migration_directory = wiki.wiki_path / "_migration"
    foreign = tmp_path / "foreign-migration"
    original_write = chunk_publication._atomic_write_posix

    def swap_after_receipt(root: Path, relative: Path, content: bytes) -> None:
        original_write(root, relative, content)
        if relative.parent.name == "chunk-publications":
            shutil.copytree(migration_directory, foreign)
            migration_directory.rename(wiki.wiki_path / "migration-held")
            migration_directory.symlink_to(foreign, target_is_directory=True)

    monkeypatch.setattr(chunk_publication, "_atomic_write_posix", swap_after_receipt)

    with pytest.raises(WikiUnsafePathError):
        wiki.publish_chunk_results(packet, published_on=date(2026, 9, 2))

    assert (foreign / "chunk-publication.pending").exists()


def test_new_publication_preserves_old_version(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, _, _ = chunks._load(book_id, run_id)
    by_chunk = {chunk.chunk_id: chunk for chunk in run.chunks}
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    citations: dict[str, dict[str, Any]] = {}
    publications = []
    for generation in (1, 2):
        accepted: dict[str, dict[str, Any]] = {}
        for number, task in enumerate(tasks):
            dependencies = task["dependency_task_ids"]
            citation = (
                asdict(by_chunk[task["chunk_ids"][0]].core_spans[0])
                if task["task_type"] == "chunk_summary"
                else citations[dependencies[0]]
            )
            accepted[task["task_id"]] = _submit_claim(
                service,
                book_id,
                run_id,
                task,
                citation,
                f"generation-{generation}-{number}",
                [accepted[identifier]["result_id"] for identifier in dependencies],
            )
            citations[task["task_id"]] = citation
        publications.append(
            service.publish(book_id, run_id, wiki, published_on=date(2026, 9, generation + 1))
        )
        if generation == 1:
            old_pages = {
                path: (wiki.wiki_path / path).read_bytes() for path in publications[0]["page_paths"]
            }
            old_citations = {
                json.loads(line)["citation_id"]
                for line in (wiki.wiki_path / "graph" / "citations.jsonl").read_text().splitlines()
            }

    assert publications[0]["publication_id"] != publications[1]["publication_id"]
    assert all(
        (wiki.wiki_path / path).read_bytes() == content for path, content in old_pages.items()
    )
    assert old_citations <= {
        json.loads(line)["citation_id"]
        for line in (wiki.wiki_path / "graph" / "citations.jsonl").read_text().splitlines()
    }
    assert len(wiki.query("Cited fact").hits) >= 4


def test_conflicting_publication_page_preserves_prior_wiki(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = books.LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = books.BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = books.BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, parsed, _ = chunks._load(book_id, run_id)
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
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"conflict-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    publication_id = service.publication_manifest(book_id, run_id)["publication_id"]
    section = next(task for task in tasks if task["task_type"] == "section_summary")
    section_id_json = json.dumps(section["parent_id"], separators=(",", ":")).encode()
    section_key = hashlib.sha256(section_id_json).hexdigest()[:24]
    occupied = (
        wiki.wiki_path
        / "pages"
        / "books"
        / book_id.replace(":", "-")
        / publication_id.replace(":", "-")
        / f"section-{section_key}.md"
    )
    occupied.parent.mkdir(parents=True)
    old_page = (
        b"---\n"
        b"id: page:CONFLICT0000\n"
        b'title: "Occupied evidence"\n'
        b"kind: summary\n"
        b"created: 2026-09-01\n"
        b"updated: 2026-09-01\n"
        b"sources: []\n"
        b"---\n\n"
        b"# Occupied evidence\n"
    )
    occupied.write_bytes(old_page)
    wiki.build(built_on=date(2026, 9, 1))
    citations_before = (wiki.wiki_path / "graph" / "citations.jsonl").read_bytes()
    assert wiki.query("Occupied evidence").hits

    with pytest.raises(ValueError, match="publication_conflict"):
        service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 2))

    assert occupied.read_bytes() == old_page
    assert (wiki.wiki_path / "graph" / "citations.jsonl").read_bytes() == citations_before
    assert not (wiki.wiki_path / "_migration" / "chunk-publication.pending").exists()
    assert not (wiki.wiki_path / "raw" / f"{parsed.source_sha256}.pdf").exists()
    assert wiki.query("Occupied evidence").hits


def _submit_claim(
    service: books.BookChunkResultService,
    book_id: str,
    run_id: str,
    task: dict[str, Any],
    citation: dict[str, Any],
    key: str,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    return service.submit(
        {
            "schema_version": 1,
            "book_id": book_id,
            "run_id": run_id,
            "task_id": task["task_id"],
            "idempotency_key": key,
            "dependency_result_ids": dependencies or [],
            "abstract": "Cited result",
            "key_points": [],
            "claims": [{"text": "Cited fact", "citations": [citation]}],
        }
    )
