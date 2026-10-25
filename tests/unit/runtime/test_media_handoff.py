"""ADR-0026: host media handoff must import everywhere and deny where unsupported."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

import capicove.runtime.media_handoff as media_handoff


def _load_isolated_copy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load a private copy so other tests keep the shared exception classes."""
    origin = media_handoff.__file__
    assert origin is not None
    spec = importlib.util.spec_from_file_location("_isolated_media_handoff", origin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flag", ["O_DIRECTORY", "O_NOFOLLOW"])
def test_import_succeeds_and_directories_are_denied_without_posix_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """Windows lacks these flags; setup doctor must still import the module."""
    monkeypatch.delattr(os, flag, raising=False)

    module = _load_isolated_copy(monkeypatch)

    with pytest.raises(module.MediaDenied, match="POSIX"):
        module.open_private_directory(tmp_path / "staging")
    assert not (tmp_path / "staging").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow directory handle behavior")
def test_private_directory_creation_tolerates_a_concurrent_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "staging"
    real_mkdir = os.mkdir

    def racing_mkdir(name: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        real_mkdir(name, mode, dir_fd=dir_fd)  # Another call wins the race first.
        raise FileExistsError(name)

    monkeypatch.setattr(os, "mkdir", racing_mkdir)
    descriptor = media_handoff.open_private_directory(target)
    os.close(descriptor)
    assert target.is_dir()
