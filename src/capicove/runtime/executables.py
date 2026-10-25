"""Identity-preserving handoff for operator-configured executables (ADR-0026).

At startup each executable is opened through no-follow handles and its identity
and SHA-256 are recorded. Every call reopens it the same way, checks that it is
still the same file with the same bytes, and runs exactly that open file through
its ``/proc/<pid>/fd/<n>`` path. A later replacement of the configured path is
never executed. This needs Linux procfs; other platforms fail closed.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class ExecutableChanged(ValueError):
    """A configured executable is missing, replaced, or modified since startup."""


@dataclass(frozen=True, slots=True)
class _Identity:
    device: int
    inode: int
    size: int
    modified_ns: int
    sha256: str


def fd_paths_supported() -> bool:
    """Return whether this process can name its own descriptors through procfs."""
    return Path(f"/proc/{os.getpid()}/fd").is_dir()


class PinnedExecutables:
    """Snapshot executables at startup and run only those exact files later."""

    def __init__(self, executables: Mapping[str, Path]) -> None:
        if not fd_paths_supported():
            raise ValueError("host media execution requires Linux /proc descriptor paths")
        self._paths = dict(executables)
        self._identities = {name: _snapshot(path) for name, path in self._paths.items()}

    @contextmanager
    def open(self) -> Iterator[Mapping[str, str]]:
        """Yield descriptor paths that stay bound to the verified files until exit."""
        descriptors: dict[str, int] = {}
        try:
            for name, path in self._paths.items():
                descriptor = _open_no_follow(path)
                descriptors[name] = descriptor
                if _identity(descriptor) != self._identities[name]:
                    raise ExecutableChanged(f"{path.name} changed since startup; restart the host")
            pid = os.getpid()
            yield {name: f"/proc/{pid}/fd/{fd}" for name, fd in descriptors.items()}
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)


def _snapshot(path: Path) -> _Identity:
    descriptor = _open_no_follow(path)
    try:
        return _identity(descriptor)
    finally:
        os.close(descriptor)


def _open_no_follow(path: Path) -> int:
    if not path.is_absolute():
        raise ExecutableChanged("executables must be absolute paths")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory = os.open(path.anchor, flags)
    except OSError:
        raise ExecutableChanged(f"{path.name} is unavailable") from None
    try:
        for component in path.parts[1:-1]:
            next_directory = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    except OSError:
        raise ExecutableChanged(f"{path.name} is not an executable file") from None
    finally:
        os.close(directory)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or not metadata.st_mode & 0o111:
        os.close(descriptor)
        raise ExecutableChanged(f"{path.name} is not an executable file")
    return descriptor


def _identity(descriptor: int) -> _Identity:
    before = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ExecutableChanged("executable changed while it was verified")
    return _Identity(
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, digest.hexdigest()
    )


__all__ = ["ExecutableChanged", "PinnedExecutables", "fd_paths_supported"]
