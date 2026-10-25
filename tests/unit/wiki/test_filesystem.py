import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Barrier

import pytest

import capicove.wiki.filesystem as wiki_filesystem
from capicove.wiki import (
    WikiError,
    WikiExistsError,
    WikiInventory,
    WikiService,
    WikiUnsafePathError,
)


def _writer_lock_process(wiki: str, connection: Connection, *, hold: bool) -> None:
    connection.send("attempting")
    with wiki_filesystem.wiki_write_lock(Path(wiki)):
        connection.send("acquired")
        if hold:
            connection.recv()
    connection.close()


def test_initialize_creates_the_minimal_standalone_layout(tmp_path: Path) -> None:
    service = WikiService(tmp_path)

    inventory = service.initialize(created_on=date(2026, 9, 1))

    wiki = tmp_path / "wiki"
    assert inventory.schema_version == 1
    assert inventory.authority.value == "standalone"
    assert inventory.pages == 0
    assert inventory.sources == 0
    assert inventory.facts == 0
    assert sorted(path.name for path in wiki.iterdir()) == [
        "SCHEMA.md",
        "_derived",
        "_migration",
        "graph",
        "index.md",
        "log.md",
        "pages",
        "raw",
        "wiki.toml",
    ]
    assert (wiki / "wiki.toml").read_text(encoding="utf-8") == (
        'schema_version = 1\nauthority = "standalone"\n'
    )
    assert (wiki / "graph" / "facts.jsonl").read_bytes() == b""
    assert (wiki / "graph" / "identity-events.jsonl").read_bytes() == b""


def test_initialize_refuses_to_overwrite_an_existing_wiki(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))

    with pytest.raises(WikiExistsError, match="already exists"):
        service.initialize(created_on=date(2026, 9, 1))


@pytest.mark.parametrize("failed_write", range(1, 9))
def test_initialize_failure_leaves_no_partial_wiki_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_write: int
) -> None:
    service = WikiService(tmp_path)
    original_atomic_write = wiki_filesystem._atomic_write
    writes = 0

    def fail_selected_write(path: Path, content: str | bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == failed_write:
            raise OSError("injected initialization failure")
        original_atomic_write(path, content)

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(wiki_filesystem, "_atomic_write", fail_selected_write)
        with pytest.raises(WikiError, match="initialization failed"):
            service.initialize(created_on=date(2026, 9, 1))

    assert not service.wiki_path.exists()
    assert list(tmp_path.glob(".wiki.init-*")) == []
    inventory = service.initialize(created_on=date(2026, 9, 1))
    assert inventory.pages == 0


def test_initialize_publish_failure_cleans_staging_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)

    def fail_publish(source: Path, target: str | Path) -> Path:
        raise OSError("injected publish failure")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(Path, "rename", fail_publish)
        with pytest.raises(WikiError, match="initialization failed"):
            service.initialize(created_on=date(2026, 9, 1))

    assert not service.wiki_path.exists()
    assert list(tmp_path.glob(".wiki.init-*")) == []
    inventory = service.initialize(created_on=date(2026, 9, 1))
    assert inventory.pages == 0


def test_concurrent_initializers_publish_exactly_one_complete_wiki(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    barrier = Barrier(2)
    original_scan = wiki_filesystem.scan

    def scan_staging_together(path: Path) -> WikiInventory:
        inventory = original_scan(path)
        if path.parent.name.startswith(".wiki.init-"):
            barrier.wait(timeout=5)
        return inventory

    def attempt_initialize() -> str:
        try:
            WikiService(tmp_path).initialize(created_on=date(2026, 9, 1))
        except WikiExistsError:
            return "exists"
        return "created"

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(wiki_filesystem, "scan", scan_staging_together)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(lambda _: attempt_initialize(), range(2)))

    assert sorted(results) == ["created", "exists"]
    assert WikiService(tmp_path).scan().pages == 0
    assert list(tmp_path.glob(".wiki.init-*")) == []


@pytest.mark.slow
def test_writer_lock_serializes_separate_processes(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    context = multiprocessing.get_context("spawn")
    first_parent, first_child = context.Pipe()
    second_parent, second_child = context.Pipe()
    first = context.Process(
        target=_writer_lock_process,
        args=(str(service.wiki_path), first_child),
        kwargs={"hold": True},
    )
    second = context.Process(
        target=_writer_lock_process,
        args=(str(service.wiki_path), second_child),
        kwargs={"hold": False},
    )
    processes = (first, second)
    try:
        first.start()
        assert first_parent.recv() == "attempting"
        assert first_parent.recv() == "acquired"
        second.start()
        assert second_parent.recv() == "attempting"
        assert second_parent.poll(0.25) is False

        first_parent.send("release")

        assert second_parent.poll(5)
        assert second_parent.recv() == "acquired"
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        first_parent.close()
        second_parent.close()
        first_child.close()
        second_child.close()


def test_scan_rejects_a_symlinked_managed_root(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "wiki" / "pages").rmdir()
    (tmp_path / "wiki" / "pages").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WikiUnsafePathError, match="symbolic link"):
        service.scan()


def test_scan_rejects_a_symlinked_managed_file(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    outside = tmp_path / "outside.toml"
    outside.write_text('schema_version = 1\nauthority = "standalone"\n', encoding="utf-8")
    manifest = tmp_path / "wiki" / "wiki.toml"
    manifest.unlink()
    manifest.symlink_to(outside)

    with pytest.raises(WikiUnsafePathError, match="symbolic link"):
        service.scan()
