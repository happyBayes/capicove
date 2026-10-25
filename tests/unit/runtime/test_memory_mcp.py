"""Memory MCP composition keeps selected tools and scope authority host-owned."""

import os
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import capicove.runtime.memory_mcp as memory_mcp
import capicove.runtime.memory_runtime as memory_runtime
from capicove.knowledge import SQLiteMemoryRepository
from capicove.runtime.mcp_config import McpHostConfiguration
from capicove.runtime.mcp_host import compose_mcp_host
from capicove.source.errors import EvidenceUnavailable


def _configuration(root: Path) -> McpHostConfiguration:
    return McpHostConfiguration(
        1,
        "capicove-memory",
        "memory",
        "workspace-a",
        "owner",
        "local",
        frozenset({"memory:read"}),
        ("capicove.memory.recall", "capicove.memory.doctor"),
        None,
        {"root": str(root), "scopes": ["notes"]},
    )


def test_memory_catalog_is_selected_without_creating_storage(tmp_path: Path) -> None:
    root = tmp_path / "memory"

    with compose_mcp_host(_configuration(root), catalog_only=True) as bindings:
        assert {item.name for item in bindings} == {
            "capicove_memory_recall",
            "capicove_memory_doctor",
        }
        assert not root.exists()


def test_memory_catalog_rejects_file_as_storage_root(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.write_text("not a directory")

    with pytest.raises(ValueError, match="directory"):
        with compose_mcp_host(_configuration(root), catalog_only=True):
            pytest.fail("catalog must not mark unusable storage as ready")


@pytest.mark.skipif(os.name == "nt", reason="Memory MCP awaits native Windows qualification")
def test_memory_catalog_rejects_nonprivate_nonempty_storage(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "existing.txt").write_text("existing state")
    root.chmod(0o755)

    with pytest.raises(ValueError, match="private"):
        with compose_mcp_host(_configuration(root), catalog_only=True):
            pytest.fail("catalog must not advertise storage that startup rejects")


def test_memory_host_rejects_foreign_scope_and_closes_repository(tmp_path: Path) -> None:
    root = tmp_path / "memory"

    with compose_mcp_host(_configuration(root)) as bindings:
        recall = next(item for item in bindings if item.name == "capicove_memory_recall")
        context = recall.context_factory(recall.spec)
        assert context.workspace_id == "workspace-a"
        allowed = recall.handler(context, {"scope": "notes", "text": "hello"})
        denied = recall.handler(context, {"scope": "foreign", "text": "hello"})
        assert allowed.status.value == "succeeded"
        assert denied.error is not None
        assert denied.error.code.value == "permission_denied"
        repository = recall.handler.service.knowledge.repository  # type: ignore[attr-defined]

    assert root.is_dir()
    assert repository._db is not None
    try:
        repository._db.execute("SELECT 1")
    except Exception as error:
        assert "closed" in str(error)
    else:
        raise AssertionError("memory repository remains open after MCP shutdown")


def test_memory_runtime_closes_repository_when_evidence_store_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "memory"
    root.mkdir(mode=0o700)
    (root / "evidence.sqlite3").mkdir()
    repositories: list[SQLiteMemoryRepository] = []

    def capture_repository(path: Path) -> SQLiteMemoryRepository:
        repository = SQLiteMemoryRepository(path)
        repositories.append(repository)
        return repository

    monkeypatch.setattr(memory_runtime, "SQLiteMemoryRepository", capture_repository)

    with pytest.raises(EvidenceUnavailable, match="privacy_authority_unavailable"):
        memory_runtime.build_memory_runtime(
            root,
            workspace="workspace-a",
            scopes=frozenset({"notes"}),
            ontology_version="v1",
            predicates=(),
        )

    assert len(repositories) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        repositories[0]._db.execute("SELECT 1")


def test_memory_host_closes_repository_if_binding_creation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repositories: list[SQLiteMemoryRepository] = []

    def capture_repository(path: Path) -> SQLiteMemoryRepository:
        repository = SQLiteMemoryRepository(path)
        repositories.append(repository)
        return repository

    def fail_binding(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated binding failure")

    monkeypatch.setattr(memory_runtime, "SQLiteMemoryRepository", capture_repository)
    monkeypatch.setattr(memory_mcp, "McpToolBinding", fail_binding)

    with pytest.raises(RuntimeError, match="simulated binding failure"):
        with compose_mcp_host(_configuration(tmp_path / "memory")):
            pytest.fail("a partially composed host must not be exposed")

    assert len(repositories) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        repositories[0]._db.execute("SELECT 1")


def test_memory_host_rejects_unselected_authority_and_unsupported_tracing(tmp_path: Path) -> None:
    config = _configuration(tmp_path / "memory")
    with pytest.raises(ValueError, match="permissions"):
        with compose_mcp_host(
            replace(config, enabled_tools=("capicove.memory.remember_episode",)),
            catalog_only=True,
        ):
            pytest.fail("write Tool must not be advertised without write permission")
    with pytest.raises(ValueError, match="tracing"):
        with compose_mcp_host(replace(config, trace_store=tmp_path / "trace.db")):
            pytest.fail("memory content must not enter generic tracing")


def test_memory_host_rejects_symlinked_root_and_unknown_config(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlinks"):
        with compose_mcp_host(_configuration(alias / "memory"), catalog_only=True):
            pytest.fail("linked root must not be selected")
    config = _configuration(root)
    with pytest.raises(ValueError, match="unsupported fields"):
        with compose_mcp_host(
            replace(config, capability_config={"root": str(root), "scopes": ["notes"], "other": 1}),
            catalog_only=True,
        ):
            pytest.fail("unknown startup authority must fail closed")


def test_memory_profile_fails_closed_until_windows_storage_is_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_mcp, "os", SimpleNamespace(name="nt"), raising=False)

    with pytest.raises(ValueError, match=r"Windows.*qualification"):
        with compose_mcp_host(_configuration(tmp_path / "memory"), catalog_only=True):
            pytest.fail("unqualified Windows memory profile must not be advertised")
