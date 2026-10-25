"""Hash a preserved Book source without following filesystem replacements."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def source_sha256(path: Path) -> str:
    """Hash one regular file through a pinned, no-follow local handle."""
    try:
        if not path.is_absolute():
            raise ValueError("source_changed")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("source_changed")
        if os.name == "posix":
            if not all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")):
                raise ValueError("source_changed")
            directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            directory_fd = os.open(path.anchor, directory_flags)
            try:
                for component in path.parts[1:-1]:
                    next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
                    os.close(directory_fd)
                    directory_fd = next_fd
                file_fd = os.open(path.name, file_flags, dir_fd=directory_fd)
            finally:
                os.close(directory_fd)
            with os.fdopen(file_fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                    raise ValueError("source_changed")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
        elif os.name == "nt":
            from capicove._windows_files import read_windows_secure_file

            content = read_windows_secure_file(path, max_bytes=before.st_size)
            digest = hashlib.sha256(content).hexdigest()
            after = path.lstat()
        else:
            raise ValueError("source_changed")
        if _identity(after) != _identity(before) or _identity(path.lstat()) != _identity(before):
            raise ValueError("source_changed")
        return digest
    except (OSError, ValueError) as error:
        raise ValueError("source_changed") from error
