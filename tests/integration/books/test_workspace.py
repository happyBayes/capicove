from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from capicove.books import LocalBookWorkspace


def test_workspace_preserves_content_addressed_source_idempotently(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nbook")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")

    first = workspace.preserve_source(source)
    second = workspace.preserve_source(source)

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert first == second == (tmp_path / "workspace" / "raw" / f"{digest}.pdf")
    assert first.read_bytes() == source.read_bytes()


def test_workspace_rejects_source_changed_after_identity_was_computed(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"new bytes")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")

    with pytest.raises(ValueError, match="changed"):
        workspace.preserve_source(source, expected_sha256=hashlib.sha256(b"old bytes").hexdigest())

    assert tuple((workspace.root / "raw").iterdir()) == ()


def test_workspace_rejects_symlinked_managed_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace = LocalBookWorkspace.create(workspace_root)
    raw = workspace_root / "raw"
    raw.rename(workspace_root / "real-raw")
    raw.symlink_to(workspace_root / "real-raw", target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        LocalBookWorkspace.open(workspace.root)


@pytest.mark.parametrize("managed", ["book", "representation.json", "tasks.jsonl", "submissions"])
def test_workspace_rejects_nested_symlinks_on_reads(tmp_path: Path, managed: str) -> None:
    from capicove.books import BookService

    from .test_service import _Parser

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-synthetic")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    prepared = BookService(workspace).prepare(source, _Parser())
    book = workspace.root / "books" / prepared.book_id.replace(":", "-")
    target = book if managed == "book" else book / managed
    external = tmp_path / "external"
    target.rename(external)
    target.symlink_to(external, target_is_directory=external.is_dir())

    with pytest.raises(ValueError, match="symbolic link"):
        if managed == "tasks.jsonl":
            workspace.read_tasks(prepared.book_id)
        elif managed == "submissions":
            workspace.latest_submissions(prepared.book_id)
        else:
            workspace.read_result(prepared.book_id)


def test_workspace_does_not_write_through_submission_symlink(tmp_path: Path) -> None:
    from capicove.books import AtomicClaim, BookService, SummarySubmission

    from .test_service import _Parser

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-synthetic")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)
    prepared = service.prepare(source, _Parser())
    task = service.tasks(prepared.book_id)[0]
    submission = SummarySubmission(
        task.task_id,
        task.idempotency_key,
        "Synthetic summary",
        ("Synthetic point",),
        (AtomicClaim("claim:test", "Synthetic fact", (task.allowed_citation_ids[0],)),),
        "test-agent",
        "test-model",
    )
    submissions = workspace.root / "books" / prepared.book_id.replace(":", "-") / "submissions"
    external = tmp_path / "external"
    submissions.rename(external)
    submissions.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        workspace.append_submission(prepared.book_id, submission)
    assert list(external.iterdir()) == []
