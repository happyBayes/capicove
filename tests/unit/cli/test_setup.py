"""A clean install can create and inspect an offline starter workspace."""

import json
import os
import stat
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace

import pytest

from capicove.cli.app import main
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config


def test_setup_init_creates_explicit_read_only_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"

    assert main(["setup", "init", str(workspace)]) == 0

    config_path = workspace / "host.json"
    result = json.loads(capsys.readouterr().out)
    assert result["next_commands"] == [
        ["capicove", "setup", "doctor", "--config", str(config_path)],
        ["capicove", "mcp", "catalog", "--config", str(config_path)],
    ]
    body = json.loads(config_path.read_text())
    assert body["schema_version"] == 2
    assert body["workspace_root"] == str(workspace)
    assert body["profile"] == "read"
    assert body["network"] == {"enabled": False, "source_hosts": []}
    assert body["enabled_tools"] == ["capicove.image.filter.shape"]
    assert (workspace / "inputs").is_dir()
    assert load_mcp_host_config(config_path).schema_version == 2
    if os.name == "posix":
        assert workspace.stat().st_mode & 0o077 == 0
        assert config_path.stat().st_mode & 0o077 == 0


def test_setup_init_does_not_overwrite_existing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "keep.txt"
    marker.write_text("keep")

    assert main(["setup", "init", str(workspace)]) == 2

    assert marker.read_text() == "keep"
    assert not (workspace / "host.json").exists()


def test_setup_init_checks_windows_parent_before_creating_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected junction parent must not receive even a temporary directory."""
    import capicove.cli.commands.setup as setup_command
    from capicove import _windows_files

    workspace = tmp_path / "workspace"
    created = False
    original_mkdir = Path.mkdir

    def observe_mkdir(
        path: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
    ) -> None:
        nonlocal created
        if path == workspace:
            created = True
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    def reject_parent(path: Path, *, include_self: bool = True) -> None:
        assert path == workspace
        assert include_self is False
        raise ValueError("simulated junction parent")

    monkeypatch.setattr(Path, "mkdir", observe_mkdir)
    monkeypatch.setattr(setup_command, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(_windows_files, "reject_windows_reparse_components", reject_parent)

    assert main(["setup", "init", str(workspace)]) == 2
    assert not created
    assert not workspace.exists()


def test_setup_init_rejects_unusable_workspace_identity_before_writing(tmp_path: Path) -> None:
    workspace = tmp_path / ("w" * 129)

    assert main(["setup", "init", str(workspace)]) == 2

    assert not workspace.exists()


def test_setup_init_does_not_expose_partial_config_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"

    def fail_fsync(descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    assert main(["setup", "init", str(workspace)]) == 2

    assert not (workspace / "host.json").exists()


def test_setup_init_removes_new_empty_directories_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed create-only init must not leave an unusable new workspace."""
    workspace = tmp_path / "new-workspace"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    assert main(["setup", "init", str(workspace)]) == 2
    assert not workspace.exists()


def test_setup_init_preserves_existing_empty_root_after_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback may remove only the inputs directory it created."""
    workspace = tmp_path / "existing-workspace"
    workspace.mkdir(mode=0o700)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    assert main(["setup", "init", str(workspace), "--use-existing"]) == 2
    assert workspace.is_dir()
    assert not (workspace / "inputs").exists()
    assert not (workspace / "host.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory fsync rollback")
@pytest.mark.parametrize("use_existing", [False, True])
def test_setup_init_rolls_back_linked_config_after_directory_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_existing: bool
) -> None:
    """An interrupted directory commit must not publish a partial workspace."""
    workspace = tmp_path / "workspace"
    if use_existing:
        workspace.mkdir(mode=0o700)
    original_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated interrupted directory commit")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    args = ["setup", "init", str(workspace)]
    if use_existing:
        args.append("--use-existing")
    assert main(args) == 2

    assert workspace.is_dir() is use_existing
    assert not (workspace / "host.json").exists()
    assert not (workspace / "inputs").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory quarantine")
def test_setup_init_rollback_never_removes_public_workspace_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent replacement at the public name must not be deleted."""
    workspace = tmp_path / "workspace"
    original_rmdir = os.rmdir
    removed_public_name = False

    def observe_rmdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal removed_public_name
        if path == workspace.name and dir_fd is not None:
            removed_public_name = True
        original_rmdir(path, dir_fd=dir_fd)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "rmdir", observe_rmdir)
    monkeypatch.setattr(os, "fsync", fail_fsync)

    assert main(["setup", "init", str(workspace)]) == 2
    assert not removed_public_name
    assert not workspace.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory quarantine")
def test_setup_init_restores_foreign_root_swapped_before_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quarantine must check its moved inode before deleting anything."""
    workspace = tmp_path / "workspace"
    held = tmp_path / "workspace-held"
    original_rename = os.rename
    swapped = False

    def swap_before_quarantine(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if source == workspace.name and str(target).startswith(".workspace-rollback-"):
            original_rename(workspace, held)
            workspace.mkdir(mode=0o700)
            swapped = True
        original_rename(source, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "rename", swap_before_quarantine)
    monkeypatch.setattr(os, "fsync", fail_fsync)

    assert main(["setup", "init", str(workspace)]) == 2
    assert swapped
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []
    assert held.is_dir()
    assert list(held.iterdir()) == []
    assert not list(tmp_path.glob(".workspace-rollback-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory quarantine")
def test_setup_init_restores_owned_root_when_quarantine_removal_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup error must not strand an owned workspace under a hidden name."""
    workspace = tmp_path / "workspace"
    original_rmdir = os.rmdir

    def fail_quarantine_rmdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        if str(path).startswith(".workspace-rollback-"):
            raise OSError("simulated quarantine removal failure")
        original_rmdir(path, dir_fd=dir_fd)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "rmdir", fail_quarantine_rmdir)
    monkeypatch.setattr(os, "fsync", fail_fsync)

    assert main(["setup", "init", str(workspace)]) == 2
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []
    assert not list(tmp_path.glob(".workspace-rollback-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory quarantine")
def test_setup_init_preserves_existing_quarantine_name_on_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup must never replace an unrelated sibling at its chosen name."""
    workspace = tmp_path / "workspace"
    token = "f" * 32
    collision = tmp_path / f".workspace-rollback-{token}"
    collision.mkdir(mode=0o700)
    marker = collision / "keep.txt"
    marker.write_text("keep")
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex=token))

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    assert main(["setup", "init", str(workspace)]) == 2
    assert marker.read_text() == "keep"
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory handle race")
def test_setup_init_never_creates_workspace_through_swapped_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "authorized"
    parent.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    workspace = parent / "workspace"
    original_mkdir = os.mkdir
    swapped = False

    def swap_before_workspace_mkdir(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if not swapped and (path == workspace or (path == workspace.name and dir_fd is not None)):
            parent.rename(tmp_path / "authorized-held")
            parent.symlink_to(foreign, target_is_directory=True)
            swapped = True
        original_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", swap_before_workspace_mkdir)

    result = main(["setup", "init", str(workspace)])

    assert swapped
    assert not (foreign / "workspace").exists()
    assert result == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory identity check")
def test_setup_init_rejects_replaced_parent_without_publishing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed directory inode must not receive a starter config."""
    import capicove.cli.commands.setup as setup_command

    parent = tmp_path / "authorized"
    parent.mkdir(mode=0o700)
    held = tmp_path / "authorized-held"
    workspace = parent / "workspace"
    original_verify = setup_command._verify_posix_workspace_path
    swapped = False

    def swap_before_verify(root: Path, parent_fd: int, root_fd: int) -> None:
        nonlocal swapped
        if not swapped:
            parent.rename(held)
            parent.mkdir(mode=0o700)
            swapped = True
        original_verify(root, parent_fd, root_fd)

    monkeypatch.setattr(setup_command, "_verify_posix_workspace_path", swap_before_verify)

    assert main(["setup", "init", str(workspace)]) == 2
    assert swapped
    assert list(parent.iterdir()) == []
    assert list(held.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory identity check")
def test_setup_init_does_not_remove_foreign_replacement_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rollback may remove only the directory created by this initialization."""
    import capicove.cli.commands.setup as setup_command

    workspace = tmp_path / "workspace"
    held = tmp_path / "workspace-held"
    original_verify = setup_command._verify_posix_workspace_path
    swapped = False

    def swap_before_verify(root: Path, parent_fd: int, root_fd: int) -> None:
        nonlocal swapped
        if not swapped:
            workspace.rename(held)
            workspace.mkdir(mode=0o700)
            swapped = True
        original_verify(root, parent_fd, root_fd)

    monkeypatch.setattr(setup_command, "_verify_posix_workspace_path", swap_before_verify)

    assert main(["setup", "init", str(workspace)]) == 2
    assert swapped
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []
    assert held.is_dir()
    assert list(held.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory identity check")
def test_setup_init_rolls_back_linked_config_after_workspace_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-link path swap must not remove the replacement directory."""
    import capicove.cli.commands.setup as setup_command

    workspace = tmp_path / "workspace"
    held = tmp_path / "workspace-held"
    original_verify = setup_command._verify_posix_workspace_path
    verification_count = 0

    def swap_after_link(root: Path, parent_fd: int, root_fd: int) -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 3:
            workspace.rename(held)
            workspace.mkdir(mode=0o700)
        original_verify(root, parent_fd, root_fd)

    monkeypatch.setattr(setup_command, "_verify_posix_workspace_path", swap_after_link)

    assert main(["setup", "init", str(workspace)]) == 2
    assert verification_count == 3
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []
    assert held.is_dir()
    assert list(held.iterdir()) == []


def test_setup_doctor_reports_starter_tool_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(workspace / "host.json")]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["ready"] is True
    assert report["tools"] == [{"tool_id": "capicove.image.filter.shape", "status": "ready"}]


def test_setup_doctor_preserves_unrelated_ready_tool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["capabilities"]["memory"] = {
        "schema_version": 1,
        "root": str(workspace / "memory"),
        "scopes": ["notes"],
    }
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.unknown")
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.memory.unknown", "status": "unavailable"},
    ]


def test_setup_doctor_keeps_ready_tool_when_sibling_tool_is_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["enabled_tools"].append("capicove.image.unknown")
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.image.unknown", "status": "unavailable"},
    ]


def test_setup_doctor_identifies_unsafe_image_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["capabilities"]["image"]["read_roots"] = [str(workspace / "missing")]
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["diagnostics"][0]["code"] == "unsafe_path"
    assert "read_roots" in report["diagnostics"][0]["hint"]
    assert report["tools"] == [{"tool_id": "capicove.image.filter.shape", "status": "unsafe_path"}]


def test_setup_doctor_identifies_unsafe_trace_store_without_creating_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["trace_store"] = str(workspace / "missing" / "traces.sqlite")
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["tools"] == [{"tool_id": "capicove.image.filter.shape", "status": "unsafe_path"}]
    assert report["diagnostics"] == [
        {
            "capability": "image",
            "code": "unsafe_path",
            "hint": "Configure trace_store below an existing owner-private local directory.",
        }
    ]
    assert not (workspace / "missing").exists()


def test_setup_doctor_distinguishes_missing_image_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    del body["capabilities"]["image"]["read_roots"]
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "not_configured"}
    ]
    assert report["diagnostics"][0]["code"] == "not_configured"
    assert "read_roots" in report["diagnostics"][0]["hint"]


def test_setup_doctor_distinguishes_missing_memory_root_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["capabilities"]["memory"] = {"schema_version": 1, "scopes": ["notes"]}
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.recall")
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.memory.recall", "status": "not_configured"},
    ]
    assert report["diagnostics"][0]["code"] == "not_configured"
    assert "capabilities.memory.root" in report["diagnostics"][0]["hint"]


def test_setup_doctor_distinguishes_missing_memory_scopes_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["capabilities"]["memory"] = {
        "schema_version": 1,
        "root": str(workspace / "memory"),
    }
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.recall")
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.memory.recall", "status": "not_configured"},
    ]
    assert report["diagnostics"][0]["code"] == "not_configured"
    assert "capabilities.memory.scopes" in report["diagnostics"][0]["hint"]


def test_setup_doctor_keeps_unrelated_tool_ready_when_dependency_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Readiness output must not disclose an optional import exception."""
    import capicove.runtime.mcp_host as mcp_host

    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["capabilities"]["memory"] = {
        "schema_version": 1,
        "root": str(workspace / "memory"),
        "scopes": ["notes"],
    }
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.recall")
    config_path.write_text(json.dumps(body))
    original_compose = mcp_host.compose_mcp_host

    def compose_with_missing_dependency(
        config: WorkspaceHostConfiguration, *, catalog_only: bool = False
    ) -> AbstractContextManager[tuple[McpToolBinding, ...]]:
        if config.enabled_tools == ("capicove.memory.recall",):
            raise ImportError("private-provider-token")
        return original_compose(config, catalog_only=catalog_only)

    monkeypatch.setattr(mcp_host, "compose_mcp_host", compose_with_missing_dependency)
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.memory.recall", "status": "missing_dependency"},
    ]
    assert report["diagnostics"][0]["code"] == "missing_dependency"
    assert "private-provider-token" not in captured.out + captured.err
    assert not (workspace / "memory").exists()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyError])
def test_setup_doctor_sanitizes_unavailable_runtime_without_losing_ready_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    import capicove.runtime.mcp_host as mcp_host

    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    body = json.loads(config_path.read_text())
    body["capabilities"]["memory"] = {
        "schema_version": 1,
        "root": str(workspace / "memory"),
        "scopes": ["notes"],
    }
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.recall")
    config_path.write_text(json.dumps(body))
    original_compose = mcp_host.compose_mcp_host

    def unavailable_runtime(
        config: WorkspaceHostConfiguration, *, catalog_only: bool = False
    ) -> AbstractContextManager[tuple[McpToolBinding, ...]]:
        if config.enabled_tools == ("capicove.memory.recall",):
            raise error_type("private trace body /outside/workspace")
        return original_compose(config, catalog_only=catalog_only)

    monkeypatch.setattr(mcp_host, "compose_mcp_host", unavailable_runtime)
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.memory.recall", "status": "unavailable"},
    ]
    assert report["diagnostics"][0]["code"] == "unavailable"
    assert "private trace body" not in captured.out + captured.err
    assert "/outside/workspace" not in captured.out + captured.err


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner-private directory mode")
def test_setup_doctor_rejects_nonprivate_empty_memory_root_without_changing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    assert main(["setup", "init", str(workspace)]) == 0
    config_path = workspace / "host.json"
    memory_root = workspace / "memory"
    memory_root.mkdir(mode=0o700)
    memory_root.chmod(0o755)
    body = json.loads(config_path.read_text())
    body["capabilities"]["memory"] = {
        "schema_version": 1,
        "root": str(memory_root),
        "scopes": ["notes"],
    }
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.recall")
    config_path.write_text(json.dumps(body))
    capsys.readouterr()

    assert main(["setup", "doctor", "--config", str(config_path)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert report["tools"] == [
        {"tool_id": "capicove.image.filter.shape", "status": "ready"},
        {"tool_id": "capicove.memory.recall", "status": "unsafe_path"},
    ]
    assert memory_root.stat().st_mode & 0o777 == 0o755
    assert list(memory_root.iterdir()) == []


def test_setup_accepts_relative_workspace_and_config_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["setup", "init", "workspace"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["config"] == str(tmp_path / "workspace" / "host.json")

    assert main(["setup", "doctor", "--config", "workspace/host.json"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


# RFC-0017 amendment 1: an optional offline search starter.


def _notes(tmp_path: Path) -> Path:
    notes = tmp_path / "notes"
    (notes / "travel").mkdir(parents=True)
    (notes / "travel" / "paris.md").write_text("# Paris\n\nThe best bakery sells croissants.\n")
    return notes


@pytest.mark.skipif(os.name != "posix", reason="retrieval requires POSIX no-follow file APIs")
def test_setup_init_with_source_root_enables_offline_search(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    before = (notes / "travel" / "paris.md").stat()
    workspace = tmp_path / "workspace"

    assert main(["setup", "init", str(workspace), "--source-root", f"notes={notes}"]) == 0

    config_path = workspace / "host.json"
    config = str(config_path)
    result = json.loads(capsys.readouterr().out)
    assert result["next_commands"] == [
        ["capicove", "setup", "doctor", "--config", config],
        [
            *("capicove", "source", "import", "--runtime-config", config),
            *("--root", "notes", "--relative-path", "RELATIVE_PATH"),
        ],
        ["capicove", "index", "build", "--runtime-config", config],
        ["capicove", "search", "QUERY", "--runtime-config", config],
    ]
    body = json.loads(config_path.read_text())
    assert body["profile"] == "write"
    assert body["network"] == {"enabled": False, "source_hosts": []}
    assert body["permissions"] == ["evidence:read", "evidence:search", "image:read"]
    assert body["enabled_tools"] == [
        "capicove.get_evidence",
        "capicove.image.filter.shape",
        "capicove.search",
    ]
    assert body["capabilities"]["retrieval"] == {
        "schema_version": 1,
        "source_roots": {"notes": str(notes)},
        "index_root": str(workspace / "index"),
        "corpus_ref": "corpus:workspace",
        "retrieval_profile_ref": "local-default",
        "max_source_bytes": 1048576,
    }
    # init never builds an index or touches the sources.
    assert not (workspace / "index").exists()
    assert (notes / "travel" / "paris.md").stat().st_mtime_ns == before.st_mtime_ns

    assert main(["setup", "doctor", "--config", config]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True
    import_command = ["source", "import", "--runtime-config", config]
    assert main([*import_command, "--root", "notes", "--relative-path", "travel/paris.md"]) == 0
    assert main(["index", "build", "--runtime-config", config]) == 0
    capsys.readouterr()
    assert main(["search", "bakery", "--runtime-config", config]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["hits"]


@pytest.mark.parametrize(
    "value",
    [
        "Notes={notes}",
        "notes",
        "notes=",
        "notes={missing}",
        "notes={workspace}/index",
    ],
)
def test_setup_init_rejects_invalid_source_root_without_creating_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    notes = _notes(tmp_path)
    workspace = tmp_path / "workspace"
    argument = value.format(notes=notes, missing=tmp_path / "missing", workspace=workspace)

    assert main(["setup", "init", str(workspace), "--source-root", argument]) == 2

    assert "Setup failed" in capsys.readouterr().err
    assert not workspace.exists()


def test_setup_init_rejects_duplicate_source_aliases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    notes = _notes(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    workspace = tmp_path / "workspace"

    command = ["setup", "init", str(workspace), "--source-root", f"notes={notes}"]
    assert main([*command, "--source-root", f"notes={other}"]) == 2

    assert "source root aliases must be unique" in capsys.readouterr().err
    assert not workspace.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX existing-root reuse")
def test_setup_init_failed_source_root_preserves_existing_empty_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    command = ["setup", "init", str(workspace), "--use-existing"]

    assert main([*command, "--source-root", f"notes={tmp_path / 'missing'}"]) == 2

    capsys.readouterr()
    assert workspace.is_dir()
    assert list(workspace.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="retrieval fails earlier off POSIX")
def test_setup_init_reports_missing_fts5_without_creating_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.retrieval._adapters import sqlite_fts

    notes = _notes(tmp_path)
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(sqlite_fts, "fts5_available", lambda: False)

    assert main(["setup", "init", str(workspace), "--source-root", f"notes={notes}"]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Setup failed: retrieval requires SQLite with FTS5" in captured.err
    assert not workspace.exists()


def test_windows_image_policy_error_has_path_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from capicove._windows_files import WindowsPathPolicyError
    from capicove.runtime import image_mcp

    root = tmp_path / "workspace"
    assert main(["setup", "init", str(root)]) == 0
    capsys.readouterr()
    monkeypatch.setattr(image_mcp, "os", SimpleNamespace(name="nt"))

    def reject(*args: object, **kwargs: object) -> None:
        raise WindowsPathPolicyError("authored native path rejection")

    monkeypatch.setattr("capicove._windows_files.validate_windows_path", reject)
    assert main(["setup", "doctor", "--config", str(root / "host.json")]) == 2
    assert json.loads(capsys.readouterr().out)["diagnostics"][0]["code"] == "unsafe_path"


def test_windows_trace_policy_error_has_path_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from capicove._windows_files import WindowsPathPolicyError
    from capicove.runtime import mcp_host

    root = tmp_path / "workspace"
    assert main(["setup", "init", str(root)]) == 0
    path = root / "host.json"
    body = json.loads(path.read_text())
    body["trace_store"] = str(root / "missing" / "trace.sqlite")
    path.write_text(json.dumps(body))
    capsys.readouterr()
    monkeypatch.setattr(mcp_host, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr("capicove._windows_files.validate_windows_path", lambda *a, **k: None)

    def reject(*args: object, **kwargs: object) -> None:
        raise WindowsPathPolicyError("authored native path rejection")

    monkeypatch.setattr("capicove._windows_files.validate_windows_file_target", reject)
    assert main(["setup", "doctor", "--config", str(path)]) == 2
    assert json.loads(capsys.readouterr().out)["diagnostics"][0]["code"] == "unsafe_path"
    assert not (root / "missing").exists()


@pytest.mark.parametrize("field", ["root", "scopes"])
def test_windows_missing_memory_fields_are_reported_before_platform_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    from types import SimpleNamespace

    from capicove.runtime import memory_mcp

    root = tmp_path / "workspace"
    assert main(["setup", "init", str(root)]) == 0
    path = root / "host.json"
    body = json.loads(path.read_text())
    memory = {"schema_version": 1, "root": str(root / "memory"), "scopes": ["notes"]}
    del memory[field]
    body["capabilities"]["memory"] = memory
    body["permissions"].append("memory:read")
    body["enabled_tools"].append("capicove.memory.recall")
    path.write_text(json.dumps(body))
    capsys.readouterr()
    monkeypatch.setattr(memory_mcp, "os", SimpleNamespace(name="nt"))
    assert main(["setup", "doctor", "--config", str(path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["diagnostics"][0]["code"] == "not_configured"
    assert not (root / "memory").exists()
