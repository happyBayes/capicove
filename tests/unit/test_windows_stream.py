"""Windows stream bridging closes duplicated handles on conversion failures."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest

from capicove import _windows_files


def test_null_duplicated_handle_is_rejected_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    kernel = Mock()
    kernel.DuplicateHandle.return_value = True

    @contextmanager
    def checked(*args: object, **kwargs: object) -> Any:
        yield kernel, 17

    monkeypatch.setattr(_windows_files, "_windows_checked_handle", checked)
    monkeypatch.setitem(sys.modules, "msvcrt", ModuleType("msvcrt"))
    with pytest.raises(_windows_files.WindowsPathPolicyError, match="null"):
        with _windows_files.open_windows_secure_stream(Path("authored.bin")):
            pytest.fail("a null duplicate must never be used")
    kernel.CloseHandle.assert_called_once()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows file handles")
def test_native_windows_stream_reads_the_verified_file(tmp_path: Path) -> None:
    from tests.unit.runtime.test_mcp_host import _make_private_windows_directory

    _make_private_windows_directory(tmp_path)
    source = tmp_path / "authored.bin"
    source.write_bytes(b"authored stream content")
    with _windows_files.open_windows_secure_stream(source) as stream:
        assert stream.read() == b"authored stream content"
    assert stream.closed
