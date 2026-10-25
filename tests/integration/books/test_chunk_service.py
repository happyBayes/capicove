import os
from pathlib import Path

import pytest

from capicove.books import BookService, LocalBookWorkspace
from capicove.books.chunk_config import resolve_chunk_config

from .test_service import _Parser


def _prepared(tmp_path: Path) -> tuple[LocalBookWorkspace, str]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-test")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    prepared = BookService(workspace).prepare(source, _Parser())
    return workspace, prepared.book_id


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


def test_preview_is_read_only_and_versions_preserve_existing_results(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    before = _snapshot(workspace.root)
    service = BookChunkService(workspace)
    config = resolve_chunk_config()
    preview = service.create(book_id, config, dry_run=True)
    assert _snapshot(workspace.root) == before
    first = service.create(book_id, config)
    assert first["run_id"] == preview["run_id"]
    assert service.create(book_id, config)["created"] is False
    changed = service.create(
        book_id,
        resolve_chunk_config(overrides={"target_core_tokens": 700}),
        compare_run=str(first["run_id"]),
    )
    assert changed["run_id"] != first["run_id"]
    assert changed["comparison"]["baseline_run_id"] == first["run_id"]
    after = _snapshot(workspace.root)
    assert all(after[key] == value for key, value in before.items())
    assert len(service.read_chunks(book_id, str(first["run_id"]), limit=1)["items"]) == 1


def test_context_reads_and_reference_tasks_are_bounded(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    created = service.create(book_id, resolve_chunk_config())
    run_id = str(created["run_id"])
    chunks = service.read_chunks(book_id, run_id)
    chunk_id = chunks["items"][0]["chunk_id"]
    context = service.read_context(book_id, run_id, chunk_id, max_tokens=60)
    assert context["tokens"] <= 60
    assert context["spans"]
    tasks = service.read_chunks(book_id, run_id, tasks=True)["items"]
    seen: set[str] = set()
    for task in tasks:
        assert set(task["dependency_task_ids"]) <= seen
        seen.add(task["task_id"])
    assert tasks[-1]["task_type"] == "book_summary"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow source handles")
def test_chunk_read_rejects_saved_pdf_swapped_for_identical_external_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-hash symlink swap must not authorize reading a foreign file."""
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    run_id = str(service.create(book_id, resolve_chunk_config())["run_id"])
    preserved = next((workspace.root / "raw").glob("*.pdf"))
    foreign = tmp_path / "foreign.pdf"
    foreign.write_bytes(preserved.read_bytes())
    original_lstat = Path.lstat
    swapped = False

    def swap_after_check(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = original_lstat(path)
        if path == preserved and not swapped:
            swapped = True
            preserved.rename(tmp_path / "held.pdf")
            preserved.symlink_to(foreign)
        return observed

    monkeypatch.setattr(Path, "lstat", swap_after_check)

    with pytest.raises(ValueError, match="source_changed"):
        service.read_chunks(book_id, run_id, limit=1)

    assert swapped


def test_invalid_ids_and_symlinks_are_rejected(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    with pytest.raises(ValueError, match="book"):
        service.create("../../outside", resolve_chunk_config())
    run_root = workspace.root / "books" / book_id.replace(":", "-") / "chunk-runs"
    other = tmp_path / "other"
    other.mkdir()
    run_root.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        service.create(book_id, resolve_chunk_config())


def test_changed_read_budget_preserves_chunk_and_task_ids(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    first = service.create(book_id, resolve_chunk_config())
    second = service.create(
        book_id, resolve_chunk_config(overrides={"max_context_read_tokens": 3000})
    )
    assert first["run_id"] == second["run_id"]
    assert second["created"] is False
    assert second["resolved_config"]["max_context_read_tokens"] == 3000


def test_saved_read_policy_limits_reads_and_can_be_explicitly_selected(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    first = service.create(book_id, resolve_chunk_config(overrides={"max_context_read_tokens": 10}))
    chunk_id = service.read_chunks(book_id, first["run_id"])["items"][0]["chunk_id"]
    assert service.read_context(book_id, first["run_id"], chunk_id, max_tokens=2000)["tokens"] <= 10
    second = service.create(
        book_id, resolve_chunk_config(overrides={"max_context_read_tokens": 3000})
    )
    assert first["run_id"] == second["run_id"]
    assert first["read_policy_id"] != second["read_policy_id"]
    selected = service.read_context(
        book_id, first["run_id"], chunk_id, read_policy_id=second["read_policy_id"]
    )
    assert selected["max_tokens"] == 3000
    assert selected["read_policy_id"] == second["read_policy_id"]


@pytest.mark.parametrize("target", ["source", "run", "policy"])
def test_tampered_durable_evidence_is_rejected(tmp_path: Path, target: str) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    created = service.create(book_id, resolve_chunk_config())
    run_id = created["run_id"]
    chunk_id = service.read_chunks(book_id, run_id)["items"][0]["chunk_id"]
    if target == "source":
        path = next((workspace.root / "raw").glob("*.pdf"))
    elif target == "run":
        path = next(workspace.root.rglob("run.json"))
    else:
        path = next(workspace.root.rglob("read-policy-*.json"))
    path.write_text("{}")
    expected_error = "source_changed" if target == "source" else r"hash|identity"
    with pytest.raises(ValueError, match=expected_error):
        service.read_context(book_id, run_id, chunk_id)


def test_pending_review_blocks_chunking_and_pagination_is_validated(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    from .test_service import _LowConfidenceParser

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book = BookService(workspace).prepare(source, _LowConfidenceParser(), evidence_only=True)
    service = BookChunkService(workspace)
    with pytest.raises(ValueError, match="review"):
        service.create(book.book_id, resolve_chunk_config())
    with pytest.raises(ValueError, match="limit"):
        service.read_chunks(book.book_id, "invalid", limit=51)


def test_context_rejects_unknown_chunk_and_policy_and_bounds_empty_paging(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    run_id = service.create(book_id, resolve_chunk_config())["run_id"]
    assert service.read_chunks(book_id, run_id, offset=100)["items"] == []
    with pytest.raises(ValueError, match="chunk ID"):
        service.read_context(book_id, run_id, "bad")
    with pytest.raises(ValueError, match="policy ID"):
        service.read_context(book_id, run_id, "bad", read_policy_id="../bad")
    with pytest.raises(ValueError, match="run ID"):
        service.read_chunks(book_id, "../bad")


def test_rehashed_run_cannot_declare_arbitrary_source_exclusions(tmp_path: Path) -> None:
    import hashlib
    import json

    from capicove.books.chunk_service import BookChunkService

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    run_id = service.create(book_id, resolve_chunk_config())["run_id"]
    path = next(workspace.root.rglob("run.json"))
    value = json.loads(path.read_bytes())
    removed = value["run"]["chunks"].pop(0)
    value["run"]["exclusions"].extend(
        [[span["block_id"], "empty"] for span in removed["core_spans"]]
    )
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    path.with_name("sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="exclusion"):
        service.read_chunks(book_id, run_id)
