"""Composed profiles cannot expand operator-selected authority."""

import ctypes
import json
import os
import sqlite3
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import capicove.runtime.mcp_host as mcp_host
from capicove import _windows_files
from capicove.contracts import InvocationContext, ToolErrorCode, ToolSpec, ToolStatus
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime import mcp_config
from capicove.runtime.mcp_config import McpHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host
from tests.unit.runtime.test_mcp_config import config_body, workspace_config_body


@pytest.mark.skipif(os.name == "nt", reason="Memory fails closed on Windows")
def test_workspace_catalog_composes_selected_capabilities_without_storage(
    tmp_path: Path,
) -> None:
    body = workspace_config_body(tmp_path)
    body["capabilities"] = {
        "image": {"schema_version": 1, "read_roots": [str(tmp_path)]},
        "memory": {
            "schema_version": 1,
            "root": str(tmp_path / "memory"),
            "scopes": ["personal"],
        },
    }
    body["permissions"] = ["image:read", "memory:read"]
    body["enabled_tools"] = ["capicove.image.filter.shape", "capicove.memory.recall"]
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))

    with compose_mcp_host(load_mcp_host_config(path), catalog_only=True) as bindings:
        assert [item.spec.tool_id for item in bindings] == [
            "capicove.image.filter.shape",
            "capicove.memory.recall",
        ]
    assert not (tmp_path / "memory").exists()


def test_workspace_host_closes_earlier_factory_when_later_factory_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = workspace_config_body(tmp_path)
    invalid_memory_root = tmp_path / "memory-file"
    invalid_memory_root.write_text("not a directory")
    body["capabilities"] = {
        "image": {"schema_version": 1, "read_roots": [str(tmp_path)]},
        "memory": {
            "schema_version": 1,
            "root": str(invalid_memory_root),
            "scopes": ["personal"],
        },
    }
    body["permissions"] = ["image:read", "memory:read"]
    body["enabled_tools"] = ["capicove.image.filter.shape", "capicove.memory.recall"]
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    original_compose = mcp_host._compose_one
    connections: list[sqlite3.Connection] = []

    @contextmanager
    def compose_with_resource(
        capability_config: McpHostConfiguration,
        context_factory: Callable[[ToolSpec], InvocationContext],
        *,
        catalog_only: bool,
    ) -> Iterator[tuple[McpToolBinding, ...]]:
        with original_compose(
            capability_config, context_factory, catalog_only=catalog_only
        ) as bindings:
            if capability_config.capability != "image":
                yield bindings
                return
            connection = sqlite3.connect(":memory:")
            connections.append(connection)
            try:
                yield bindings
            finally:
                connection.close()

    monkeypatch.setattr(mcp_host, "_compose_one", compose_with_resource)

    # Memory fails closed on Windows before it inspects the root.
    expected = "Windows" if os.name == "nt" else "directory"
    with pytest.raises(ValueError, match=expected):
        with compose_mcp_host(config):
            pytest.fail("a partial workspace catalog must not be exposed")

    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")


@pytest.mark.skipif(os.name != "nt", reason="Windows-only fail-closed behavior")
def test_workspace_memory_fails_closed_on_windows_without_storage(tmp_path: Path) -> None:
    body = workspace_config_body(tmp_path)
    body["capabilities"] = {
        "memory": {
            "schema_version": 1,
            "root": str(tmp_path / "memory"),
            "scopes": ["personal"],
        },
    }
    body["permissions"] = ["memory:read"]
    body["enabled_tools"] = ["capicove.memory.recall"]
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))

    with pytest.raises(ValueError, match="Windows storage and ACL qualification"):
        with compose_mcp_host(load_mcp_host_config(path), catalog_only=True):
            pytest.fail("Memory must not be exposed on Windows")
    assert not (tmp_path / "memory").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX workspace-root identity")
def test_workspace_host_rejects_root_replacement_before_publishing_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replaced workspace path must not authorize a different directory."""
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    (root / "inputs").mkdir(mode=0o700)
    body = workspace_config_body(root)
    body["capabilities"] = {"image": {"schema_version": 1, "read_roots": [str(root / "inputs")]}}
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    original_compose = mcp_host._compose_one
    moved = tmp_path / "original-workspace"

    @contextmanager
    def swap_before_factory(
        capability_config: McpHostConfiguration,
        context_factory: Callable[[ToolSpec], InvocationContext],
        *,
        catalog_only: bool,
    ) -> Iterator[tuple[McpToolBinding, ...]]:
        root.rename(moved)
        root.mkdir(mode=0o700)
        (root / "inputs").mkdir(mode=0o700)
        with original_compose(
            capability_config, context_factory, catalog_only=catalog_only
        ) as bindings:
            yield bindings

    monkeypatch.setattr(mcp_host, "_compose_one", swap_before_factory)

    with pytest.raises(ValueError, match="workspace_root changed during startup"):
        with compose_mcp_host(config, catalog_only=True):
            pytest.fail("replacement workspace must not be published")

    assert moved.is_dir()
    assert root.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="POSIX workspace-root identity")
def test_workspace_host_denies_tool_after_root_replacement(tmp_path: Path) -> None:
    """A live selected Tool must not read a replacement workspace root."""
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    (root / "inputs").mkdir(mode=0o700)
    body = workspace_config_body(root)
    body["capabilities"] = {"image": {"schema_version": 1, "read_roots": [str(root / "inputs")]}}
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)

    with compose_mcp_host(config) as bindings:
        root.rename(tmp_path / "original-workspace")
        root.mkdir(mode=0o700)
        (root / "inputs").mkdir(mode=0o700)
        foreign = root / "inputs" / "foreign.png"
        Image.new("RGB", (2, 2), "red").save(foreign)
        binding = bindings[0]
        result = binding.handler(
            binding.context_factory(binding.spec), {"sources": [{"path": str(foreign)}]}
        )

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.result is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX pinned image root")
def test_workspace_host_image_read_rejects_root_swapped_after_call_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root swap after the host check must not reach image source bytes."""
    import capicove.tools.image as image_tools

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    (root / "inputs").mkdir(mode=0o700)
    body = workspace_config_body(root)
    body["capabilities"] = {"image": {"schema_version": 1, "read_roots": [str(root / "inputs")]}}
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    foreign = root / "inputs" / "foreign.png"
    reads = 0
    original_reader = image_tools._read_bounded_regular_fd

    def observed_read(descriptor: int, maximum_bytes: int) -> bytes:
        nonlocal reads
        reads += 1
        return original_reader(descriptor, maximum_bytes)

    with compose_mcp_host(config) as bindings:
        original_check = mcp_host._require_current_posix_workspace_root

        def swap_after_check(workspace: Path, pinned: int) -> None:
            original_check(workspace, pinned)
            root.rename(tmp_path / "original-workspace")
            root.mkdir(mode=0o700)
            (root / "inputs").mkdir(mode=0o700)
            Image.new("RGB", (2, 2), "red").save(foreign)

        monkeypatch.setattr(mcp_host, "_require_current_posix_workspace_root", swap_after_check)
        monkeypatch.setattr(image_tools, "_read_bounded_regular_fd", observed_read)
        binding = bindings[0]
        result = binding.handler(
            binding.context_factory(binding.spec), {"sources": [{"path": str(foreign)}]}
        )

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert reads == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX pinned image root")
def test_workspace_host_rejects_image_root_replaced_during_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Image read root must not change between validation and binding."""
    import capicove.tools.image as image_tools

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    body = workspace_config_body(root)
    body["capabilities"] = {"image": {"schema_version": 1, "read_roots": [str(inputs)]}}
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    original_builder = image_tools.build_image_tool_handler

    def swap_before_handler(
        name: str,
        capability: object,
        *,
        read_roots: tuple[Path, ...],
        write_roots: tuple[Path, ...] = (),
        read_root_identities: Mapping[Path, tuple[int, int]] | None = None,
    ) -> image_tools.ImageToolHandler:
        inputs.rename(root / "original-inputs")
        inputs.mkdir(mode=0o700)
        return original_builder(
            name,
            capability,
            read_roots=read_roots,
            write_roots=write_roots,
            read_root_identities=read_root_identities,
        )

    monkeypatch.setattr(image_tools, "build_image_tool_handler", swap_before_handler)

    with pytest.raises(ValueError, match="image read_roots changed during startup"):
        with compose_mcp_host(config, catalog_only=True):
            pytest.fail("replacement Image read root must not be advertised")

    assert (root / "original-inputs").is_dir()
    assert inputs.is_dir()


@pytest.mark.skipif(os.name != "posix", reason="POSIX image-root startup classification")
def test_image_root_anchor_open_failure_has_unsafe_path_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An anchor failure must not leak its OS detail or lose remediation."""
    import capicove.runtime.image_mcp as image_mcp
    from capicove.cli.commands.setup import _readiness_diagnostic

    def fail_anchor(*args: object, **kwargs: object) -> int:
        raise OSError("private image root failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "open", fail_anchor)
        with pytest.raises(ValueError, match="image read_roots") as captured:
            image_mcp._open_posix_image_root(Path("/"))

    code, hint = _readiness_diagnostic("image", {"read_roots": ("/",)}, captured.value)
    assert code == "unsafe_path"
    assert "read_roots" in hint
    assert "private image root failure" not in str(captured.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX workspace-root startup classification")
def test_workspace_host_rejects_anchor_open_failure_without_private_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected host must reject an unreadable root before publishing Tools."""
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(workspace_config_body(tmp_path)))
    config = load_mcp_host_config(path)
    original_open = os.open

    def fail_anchor(
        file: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if file == tmp_path.anchor:
            raise OSError("private workspace root failure")
        return original_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_anchor)
    with pytest.raises(ValueError, match="workspace_root") as captured:
        with compose_mcp_host(config, catalog_only=True):
            pytest.fail("a failed workspace root must not publish a Tool catalog")
    assert "private workspace root failure" not in str(captured.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX image-root handle lifetime")
def test_workspace_host_keeps_image_root_pinned_until_host_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected factory owns its root lease for the entire host lifetime."""
    import capicove.runtime.image_mcp as image_mcp

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    body = workspace_config_body(root)
    body["capabilities"] = {"image": {"schema_version": 1, "read_roots": [str(inputs)]}}
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    original_open = image_mcp._open_posix_image_root
    opened: list[int] = []

    def observe_open(selected: Path) -> int:
        descriptor = original_open(selected)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(image_mcp, "_open_posix_image_root", observe_open)

    with compose_mcp_host(config) as bindings:
        assert len(bindings) == 1
        assert opened
        pinned = opened[0]
        assert os.fstat(pinned).st_ino == inputs.stat().st_ino

    with pytest.raises(OSError):
        os.fstat(pinned)


@pytest.mark.skipif(os.name != "posix", reason="POSIX pinned image root")
def test_workspace_host_rejects_image_root_restored_after_handler_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient replacement must not become the handler's authority."""
    import capicove.tools.image as image_tools

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    inputs = root / "inputs"
    inputs.mkdir(mode=0o700)
    body = workspace_config_body(root)
    body["capabilities"] = {"image": {"schema_version": 1, "read_roots": [str(inputs)]}}
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    original_builder = image_tools.build_image_tool_handler
    original_inputs = root / "original-inputs"
    alternate_inputs = root / "alternate-inputs"

    def capture_replacement(
        name: str,
        capability: object,
        *,
        read_roots: tuple[Path, ...],
        write_roots: tuple[Path, ...] = (),
        read_root_identities: Mapping[Path, tuple[int, int]] | None = None,
    ) -> image_tools.ImageToolHandler:
        inputs.rename(original_inputs)
        inputs.mkdir(mode=0o700)
        handler = original_builder(
            name,
            capability,
            read_roots=read_roots,
            write_roots=write_roots,
            read_root_identities=read_root_identities,
        )
        inputs.rename(alternate_inputs)
        original_inputs.rename(inputs)
        return handler

    monkeypatch.setattr(image_tools, "build_image_tool_handler", capture_replacement)

    with compose_mcp_host(config) as bindings:
        inputs.rename(original_inputs)
        alternate_inputs.rename(inputs)
        foreign = inputs / "foreign.png"
        Image.new("RGB", (2, 2), "red").save(foreign)
        binding = bindings[0]
        result = binding.handler(
            binding.context_factory(binding.spec), {"sources": [{"path": str(foreign)}]}
        )

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED


def test_workspace_read_profile_rejects_selected_write_tool(tmp_path: Path) -> None:
    body = workspace_config_body(tmp_path)
    body["capabilities"] = {
        "memory": {
            "schema_version": 1,
            "root": str(tmp_path / "memory"),
            "scopes": ["personal"],
        }
    }
    body["permissions"] = ["memory:write"]
    body["enabled_tools"] = ["capicove.memory.remember_episode"]
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))

    with pytest.raises(ValueError, match="profile"):
        with compose_mcp_host(load_mcp_host_config(path), catalog_only=True):
            pytest.fail("read profile must not advertise a writing Tool")
    assert not (tmp_path / "memory").exists()


def test_unselected_memory_does_not_disable_image_trace_catalog(tmp_path: Path) -> None:
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir(mode=0o700)
    body = workspace_config_body(tmp_path)
    body["capabilities"] = {
        "image": {"schema_version": 1, "read_roots": [str(tmp_path)]},
        "memory": {
            "schema_version": 1,
            "root": str(tmp_path / "memory"),
            "scopes": ["personal"],
        },
    }
    body["trace_store"] = str(trace_directory / "trace.sqlite3")
    path = tmp_path / "host-v2.json"
    path.write_text(json.dumps(body))

    with compose_mcp_host(load_mcp_host_config(path), catalog_only=True) as bindings:
        assert [item.spec.tool_id for item in bindings] == ["capicove.image.filter.shape"]
    assert not (trace_directory / "trace.sqlite3").exists()


def _make_private_windows_directory(path: Path) -> None:
    kernel32, advapi32 = _windows_files._windows_apis()
    sid = _windows_files._windows_current_sid(kernel32, advapi32)
    for options in (
        ["/grant:r", f"*{sid}:(OI)(CI)F"],
        ["/inheritance:r"],
    ):
        result = subprocess.run(["icacls", str(path), *options], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    _windows_files.validate_windows_path(path, directory=True, private_acl=True)


def test_direct_host_composition_rejects_missing_windows_security_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = McpHostConfiguration(
        1,
        "capicove-local",
        "image",
        "workspace",
        "owner",
        "local",
        frozenset({"image:read"}),
        ("capicove.image.filter.shape",),
        None,
        {"read_roots": [str(tmp_path)]},
    )
    monkeypatch.setattr(mcp_config, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(_windows_files, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(ctypes, "WinDLL", None, raising=False)
    with pytest.raises(ValueError, match="native Windows file security APIs"):
        with compose_mcp_host(config):
            pytest.fail("host must check native file APIs before composing bindings")


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_windows_trace_parent_os_errors_are_actionable_value_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[OSError]
) -> None:
    trace_path = tmp_path / "missing" / "trace.db"
    config = McpHostConfiguration(
        1,
        "capicove-local",
        "image",
        "workspace",
        "owner",
        "local",
        frozenset({"image:read"}),
        ("capicove.image.filter.shape",),
        trace_path,
        {"read_roots": [str(tmp_path)]},
    )

    def fail_target(path: Path) -> None:
        assert path == trace_path
        raise failure("fixture parent failure")

    monkeypatch.setattr(mcp_host, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(mcp_host, "_require_secure_file_apis", lambda: None)
    monkeypatch.setattr(_windows_files, "validate_windows_file_target", fail_target)
    with pytest.raises(ValueError, match="existing private Windows parent directory") as caught:
        with compose_mcp_host(config, catalog_only=True):
            pytest.fail("trace parent failure must stop composition")
    assert isinstance(caught.value.__cause__, failure)
    assert not trace_path.exists()


def test_context_identity_permissions_and_selected_catalog(tmp_path: Path) -> None:
    path = tmp_path / "host.json"
    body = config_body(tmp_path)
    body["enabled_tools"] = ["capicove.image.filter.aspect-ratio", "capicove.image.filter.shape"]
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    with compose_mcp_host(config) as bindings:
        assert len(bindings) == 2
        context = bindings[0].context_factory(bindings[0].spec)
        assert (context.workspace_id, context.actor_id, context.environment_id) == (
            "workspace",
            "owner",
            "local",
        )
        assert context.permissions == frozenset({"image:read"})
        denied = bindings[0].handler(replace(context, permissions=frozenset()), {"sources": []})
        assert denied.error is not None and denied.error.code.value == "permission_denied"


def test_catalog_does_not_create_trace_store(tmp_path: Path) -> None:
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir(mode=0o700)
    if os.name == "nt":
        _make_private_windows_directory(trace_directory)
    path = tmp_path / "host.json"
    body = config_body(tmp_path)
    body["trace_store"] = str(trace_directory / "trace.db")
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)
    with compose_mcp_host(config, catalog_only=True) as bindings:
        assert bindings[0].recorder is None
        assert not config.trace_store.exists()  # type: ignore[union-attr]
    with compose_mcp_host(config) as bindings:
        assert bindings[0].recorder is not None
    assert config.trace_store.is_file()  # type: ignore[union-attr]


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows ACLs")
def test_windows_direct_host_rejects_readable_trace_parent(tmp_path: Path) -> None:
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir()
    _make_private_windows_directory(trace_directory)
    grant = subprocess.run(
        ["icacls", str(trace_directory), "/grant", "*S-1-1-0:(OI)(CI)R"],
        capture_output=True,
        text=True,
    )
    assert grant.returncode == 0, grant.stderr
    config = McpHostConfiguration(
        1,
        "capicove-local",
        "image",
        "workspace",
        "owner",
        "local",
        frozenset({"image:read"}),
        ("capicove.image.filter.shape",),
        trace_directory / "trace.db",
        {"read_roots": [str(tmp_path)]},
    )
    with pytest.raises(ValueError, match="ACL"):
        with compose_mcp_host(config, catalog_only=True):
            pytest.fail("trace parent must be private")


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows ACLs")
def test_windows_trace_parent_requires_private_inheritance_before_creation(tmp_path: Path) -> None:
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir()
    kernel32, advapi32 = _windows_files._windows_apis()
    sid = _windows_files._windows_current_sid(kernel32, advapi32)
    script = """
$ErrorActionPreference = 'Stop'
$path = $env:CAPICOVE_TEST_ACL_PATH
$sid = $env:CAPICOVE_TEST_ACL_SID
$acl = [System.IO.Directory]::GetAccessControl($path)
$acl.SetSecurityDescriptorSddlForm(
    "D:P(A;;FA;;;$sid)", [System.Security.AccessControl.AccessControlSections]::Access)
[System.IO.Directory]::SetAccessControl($path, $acl)
$verified = [System.IO.Directory]::GetAccessControl($path)
$rules = @($verified.GetAccessRules(
    $true, $true, [System.Security.Principal.SecurityIdentifier]))
if ($rules.Count -ne 1 -or $rules[0].IdentityReference.Value -ne $sid -or
    $rules[0].InheritanceFlags -ne 'None' -or $rules[0].IsInherited -or
    $rules[0].AccessControlType -ne 'Allow') {
    $sddl = $verified.GetSecurityDescriptorSddlForm(
        [System.Security.AccessControl.AccessControlSections]::Access)
    throw "Fixture ACL differs from the requested non-inheriting private grant: $sddl"
}
"verified"
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        env={
            **os.environ,
            "CAPICOVE_TEST_ACL_PATH": str(trace_directory),
            "CAPICOVE_TEST_ACL_SID": sid,
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "verified"
    target = trace_directory / "trace.db"
    with pytest.raises(ValueError, match="inherit"):
        _windows_files.validate_windows_file_target(target)
    assert not target.exists()


def test_symlink_root_and_unavailable_tool_never_create_trace_store(tmp_path: Path) -> None:
    path = tmp_path / "host.json"
    body = config_body(tmp_path)
    body["trace_store"] = str(tmp_path / "trace.db")
    link = tmp_path / "alias"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink creation is unavailable")
    body["capability_config"] = {"read_roots": [str(link)]}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match=r"symlink|reparse"):
        with compose_mcp_host(load_mcp_host_config(path)):
            pytest.fail("must reject a symlink root")
    assert not (tmp_path / "trace.db").exists()


def test_foreign_symlink_source_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (3, 4), "red").save(outside)
    alias = root / "alias.png"
    try:
        alias.symlink_to(outside)
    except OSError:
        pytest.skip("Windows symlink creation is unavailable")
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(root)))
    with compose_mcp_host(load_mcp_host_config(path)) as bindings:
        item = bindings[0]
        result = item.handler(item.context_factory(item.spec), {"sources": [{"path": str(alias)}]})
        assert result.error is not None
        assert result.error.code.value == "permission_denied"
