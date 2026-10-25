"""Startup configuration must fail closed before composing any capability."""

import json
import os
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import cast

import pytest

from capicove import _windows_files
from capicove.runtime import mcp_config
from capicove.runtime.mcp_config import (
    McpHostConfiguration,
    WorkspaceHostConfiguration,
    load_mcp_host_config,
)


def config_body(root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "server_name": "capicove-local",
        "capability": "image",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "permissions": ["image:read"],
        "enabled_tools": ["capicove.image.filter.shape"],
        "trace_store": None,
        "capability_config": {"read_roots": [str(root)]},
    }


def workspace_config_body(root: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "read",
        "permissions": ["image:read"],
        "enabled_tools": ["capicove.image.filter.shape"],
        "trace_store": None,
        "capabilities": {"image": {"schema_version": 1, "read_roots": [str(root)]}},
        "network": {"enabled": False, "source_hosts": []},
    }


def test_version_two_workspace_config_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(workspace_config_body(tmp_path)))

    config = load_mcp_host_config(path)

    assert isinstance(config, WorkspaceHostConfiguration)
    assert config.schema_version == 2
    assert config.workspace_root == tmp_path
    assert config.capabilities["image"]["read_roots"] == (str(tmp_path),)
    with pytest.raises(TypeError):
        config.capabilities["image"]["read_roots"] = ()  # type: ignore[index]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX directory descriptors")
def test_config_parent_swap_never_loads_foreign_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trusted = tmp_path / "trusted"
    foreign = tmp_path / "foreign"
    trusted.mkdir()
    foreign.mkdir()
    path = trusted / "host.json"
    path.write_text(json.dumps(workspace_config_body(tmp_path)))
    foreign_body = workspace_config_body(tmp_path)
    foreign_body["actor_id"] = "foreign"
    (foreign / "host.json").write_text(json.dumps(foreign_body))
    original_open = os.open
    swapped = False

    def swap_before_file_open(
        file: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if not swapped and (file == path or (file == path.name and dir_fd is not None)):
            trusted.rename(tmp_path / "moved")
            trusted.symlink_to(foreign, target_is_directory=True)
            swapped = True
        return original_open(file, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_before_file_open)

    config = load_mcp_host_config(path)

    assert swapped
    assert isinstance(config, WorkspaceHostConfiguration)
    assert config.actor_id == "owner"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "superuser"),
        ("profile", []),
        ("network", {"enabled": True, "source_hosts": []}),
        ("capabilities", {"image": {"schema_version": 9, "read_roots": []}}),
        ("workspace_root", "relative"),
        ("enabled_tools", ["capicove.memory.erase_scope"]),
        ("permissions", ["image:read", "memory:erase"]),
    ],
)
def test_version_two_rejects_authority_widening(tmp_path: Path, field: str, value: object) -> None:
    body = workspace_config_body(tmp_path)
    body[field] = value
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body))

    with pytest.raises(ValueError):
        load_mcp_host_config(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX import dependency check")
def test_posix_mcp_runtime_imports_without_ctypes() -> None:
    script = """
import builtins

original_import = builtins.__import__

def reject_ctypes(name, *args, **kwargs):
    if name == "ctypes" or name.startswith("ctypes."):
        raise ImportError("ctypes is unavailable")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_ctypes
import capicove.runtime.mcp_config
import capicove.runtime.mcp_host
import capicove.runtime.image_mcp
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "unsupported", ["unknown_os", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "geteuid"]
)
def test_config_requires_posix_secure_file_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported: str
) -> None:
    def fail_open(*args: object, **kwargs: object) -> int:
        pytest.fail("unsupported platform must not open the configuration")

    platform_os = SimpleNamespace(
        name="posix",
        O_RDONLY=0,
        O_DIRECTORY=4,
        O_NOFOLLOW=1,
        O_NONBLOCK=2,
        geteuid=lambda: 0,
        open=fail_open,
    )
    if unsupported == "unknown_os":
        platform_os.name = "wasi"
    else:
        delattr(platform_os, unsupported)
    monkeypatch.setattr(mcp_config, "os", platform_os)

    message = "POSIX or Windows" if unsupported == "unknown_os" else "POSIX secure file APIs"
    with pytest.raises(ValueError, match=message):
        load_mcp_host_config(tmp_path / "host.json")


@pytest.mark.parametrize(
    "mask",
    [
        0x0002,
        0x0004,
        0x0010,
        0x0040,
        0x0100,
        0x00010000,
        0x00040000,
        0x00080000,
        0x10000000,
        0x40000000,
        0x02000000,
    ],
)
def test_windows_acl_rejects_other_principal_write_rights(mask: int) -> None:
    with pytest.raises(ValueError, match="ACL"):
        _windows_files._validate_windows_acl("S-1-5-21-1", "S-1-5-21-1", [(0, 0, mask, "S-1-1-0")])


def test_windows_acl_allows_other_principal_read_only() -> None:
    _windows_files._validate_windows_acl(
        "S-1-5-21-1",
        "S-1-5-21-1",
        [
            (0, 0, 0x00120089, "S-1-1-0"),
            (0, 0, 0x001F01FF, "S-1-5-18"),
            (0, 0, 0x001F01FF, "S-1-5-32-544"),
        ],
    )


def test_windows_trace_acl_rejects_other_principal_read_only() -> None:
    with pytest.raises(ValueError, match="ACL"):
        _windows_files._validate_windows_acl(
            "S-1-5-21-1",
            "S-1-5-21-1",
            [(0, 0, 0x0001, "S-1-1-0")],
            private_acl=True,
        )


@pytest.mark.parametrize("flags,mask", [(0, 0x001F01FF), (0x02, 0x001F01FF), (0x01, 0x0001)])
def test_windows_trace_parent_requires_inherited_owner_read_write_delete(
    flags: int, mask: int
) -> None:
    with pytest.raises(ValueError, match="inherit"):
        _windows_files._validate_windows_acl(
            "S-1-5-21-1",
            "S-1-5-21-1",
            [(0, flags, mask, "S-1-5-21-1")],
            private_acl=True,
            require_private_inheritance=True,
        )


def test_windows_trace_parent_accepts_private_inherited_owner_rights() -> None:
    _windows_files._validate_windows_acl(
        "S-1-5-21-1",
        "S-1-5-21-1",
        [(0, 0x03, 0x001F01FF, "S-1-5-21-1")],
        private_acl=True,
        require_private_inheritance=True,
    )


def test_windows_trace_parent_accepts_object_only_and_separate_inherited_rights() -> None:
    owner = "S-1-5-21-1"
    for aces in (
        [(0, 0x01, 0x001F01FF, owner)],
        [(0, 0x01, 0x001F01FF, owner), (0, 0x02, 0x001F01FF, owner)],
    ):
        _windows_files._validate_windows_acl(
            owner,
            owner,
            aces,
            private_acl=True,
            require_private_inheritance=True,
        )


def test_windows_trace_parent_does_not_count_container_only_file_rights() -> None:
    owner = "S-1-5-21-1"
    with pytest.raises(ValueError, match="inherit"):
        _windows_files._validate_windows_acl(
            owner,
            owner,
            [(0, 0x01, 0x0001, owner), (0, 0x02, 0x001F01FF, owner)],
            private_acl=True,
            require_private_inheritance=True,
        )


def test_windows_acl_accepts_current_token_administrators_default_owner() -> None:
    _windows_files._validate_windows_acl(
        "S-1-5-32-544",
        "S-1-5-21-1",
        [(0, 0, 0x001F01FF, "S-1-5-21-1")],
        token_owner_sid="S-1-5-32-544",
    )


@pytest.mark.parametrize("token_owner", [None, "S-1-5-21-2"])
def test_windows_acl_rejects_untrusted_or_mismatched_default_owner(
    token_owner: str | None,
) -> None:
    with pytest.raises(ValueError, match="owner"):
        _windows_files._validate_windows_acl(
            "S-1-5-21-2",
            "S-1-5-21-1",
            [],
            token_owner_sid=token_owner,
        )


@pytest.mark.parametrize(
    "owner,aces",
    [
        ("S-1-5-21-2", []),
        ("S-1-5-21-1", [(9, 0, 0x0002, "S-1-1-0")]),
    ],
)
def test_windows_acl_rejects_foreign_owner_or_unknown_ace(
    owner: str, aces: list[tuple[int, int, int, str]]
) -> None:
    with pytest.raises(ValueError, match="ACL"):
        _windows_files._validate_windows_acl(owner, "S-1-5-21-1", aces)


def test_windows_reparse_parent_is_rejected_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "junction"
    path = parent / "nested" / "host.json"
    visited: list[Path] = []

    def fake_lstat(part: Path) -> SimpleNamespace:
        visited.append(part)
        return SimpleNamespace(st_file_attributes=0x0400 if part == parent else 0)

    fake_os = SimpleNamespace(lstat=fake_lstat)
    monkeypatch.setattr(_windows_files, "os", fake_os)
    monkeypatch.setattr(_windows_files, "_require_local_windows_drive", lambda path: None)
    with pytest.raises(ValueError, match="reparse"):
        _windows_files.reject_windows_reparse_components(path, include_self=False)
    expected = list(reversed(path.parents))
    assert visited == expected[: expected.index(parent) + 1]
    assert path.parent not in visited


@pytest.mark.parametrize("drive_type", [0, 1, 2, 4, 5, 6])
def test_windows_remote_or_unknown_drive_is_rejected_before_path_io(
    monkeypatch: pytest.MonkeyPatch, drive_type: int
) -> None:
    visited: list[object] = []
    fake_os = SimpleNamespace(lstat=lambda part: visited.append(part))
    fake_kernel32 = SimpleNamespace(GetDriveTypeW=lambda root: drive_type)
    monkeypatch.setattr(_windows_files, "os", fake_os)
    monkeypatch.setattr(_windows_files, "_windows_apis", lambda: (fake_kernel32, object()))
    path = cast(Path, PureWindowsPath("Z:/mapped/host.json"))
    with pytest.raises(ValueError, match="local drive"):
        _windows_files.reject_windows_reparse_components(path)
    assert visited == []


def test_windows_missing_drive_api_is_rejected_before_path_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[object] = []
    monkeypatch.setattr(
        _windows_files, "os", SimpleNamespace(lstat=lambda part: visited.append(part))
    )
    monkeypatch.setattr(_windows_files, "_windows_apis", lambda: (SimpleNamespace(), object()))
    path = cast(Path, PureWindowsPath("Z:/mapped/host.json"))
    with pytest.raises(ValueError, match="drive type"):
        _windows_files.reject_windows_reparse_components(path)
    assert visited == []


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows drive type API")
def test_windows_native_temp_drive_passes_local_drive_check(tmp_path: Path) -> None:
    _windows_files.reject_windows_reparse_components(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows ACLs")
def test_windows_config_rejects_world_write_acl(tmp_path: Path) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(tmp_path)))
    grant = subprocess.run(
        ["icacls", str(path), "/grant", "*S-1-1-0:(W)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert grant.returncode == 0, grant.stderr
    with pytest.raises(ValueError, match="ACL"):
        load_mcp_host_config(path)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows reparse points")
def test_windows_config_rejects_file_reparse_point(tmp_path: Path) -> None:
    target = tmp_path / "host.json"
    target.write_text(json.dumps(config_body(tmp_path)))
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Windows symlink creation is unavailable")
    with pytest.raises(ValueError, match="reparse"):
        load_mcp_host_config(link)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junctions")
def test_windows_config_rejects_junction_parent(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "host.json").write_text(json.dumps(config_body(tmp_path)))
    junction = tmp_path / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(ValueError, match="reparse"):
        load_mcp_host_config(junction / "host.json")


def test_config_is_deeply_immutable(tmp_path: Path) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(tmp_path)))
    config = load_mcp_host_config(path)
    assert isinstance(config, McpHostConfiguration)
    assert config.permissions == frozenset({"image:read"})
    assert config.capability_config["read_roots"] == (str(tmp_path),)
    with pytest.raises(TypeError):
        config.capability_config["read_roots"] = ()  # type: ignore[index]


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits do not represent Windows ACLs")
@pytest.mark.parametrize("mode", [0o620, 0o602, 0o666])
def test_config_rejects_other_user_write_access(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(tmp_path)))
    path.chmod(mode)
    with pytest.raises(ValueError, match=r"owner|writable"):
        load_mcp_host_config(path)


@pytest.mark.skipif(os.name != "posix", reason="geteuid is POSIX-only")
def test_config_rejects_foreign_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(tmp_path)))
    monkeypatch.setattr(os, "geteuid", lambda: path.stat().st_uid + 1)
    with pytest.raises(ValueError, match="owner"):
        load_mcp_host_config(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits do not represent Windows ACLs")
@pytest.mark.parametrize("mode", [0o400, 0o600, 0o644])
def test_config_allows_owner_controlled_readable_files(tmp_path: Path, mode: int) -> None:
    path = tmp_path / "host.json"
    path.write_text(json.dumps(config_body(tmp_path)))
    path.chmod(mode)
    assert load_mcp_host_config(path).actor_id == "owner"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("server_name", "../secret"),
        ("actor_id", " "),
        ("permissions", ["image:read", "image:read"]),
        ("permissions", "image:read"),
        ("enabled_tools", []),
        ("enabled_tools", ["capicove.image.filter.shape"] * 2),
        ("enabled_tools", ["not-a-tool"]),
        ("trace_store", "relative.db"),
        ("capability_config", []),
        ("endpoint", "secret-value"),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    body = config_body(tmp_path)
    body[field] = value
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError) as error:
        load_mcp_host_config(path)
    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "text",
    ['{"x":1,"x":2}', '{"x":NaN}', "[1]", "{" * 1000, " " * 65537],
    ids=["duplicate-key", "nonfinite", "array", "malformed", "oversized"],
)
def test_malformed_or_oversized_config(tmp_path: Path, text: str) -> None:
    path = tmp_path / "host.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_mcp_host_config(path)


@pytest.mark.skipif(os.name != "posix", reason="mkfifo is POSIX-only")
def test_symlink_directory_and_fifo_rejected(tmp_path: Path) -> None:
    import os

    target = tmp_path / "host.json"
    target.write_text(json.dumps(config_body(tmp_path)))
    link = tmp_path / "link.json"
    link.symlink_to(target)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path in (link, fifo, tmp_path):
        with pytest.raises(ValueError):
            load_mcp_host_config(path)


@pytest.mark.parametrize("value", ["1e999", "[" * 100 + "0" + "]" * 100])
def test_nonfinite_and_deep_values_are_rejected(tmp_path: Path, value: str) -> None:
    body = config_body(tmp_path)
    body["capability_config"] = {"value": "PLACEHOLDER"}
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body).replace('"PLACEHOLDER"', value))
    with pytest.raises(ValueError):
        load_mcp_host_config(path)
