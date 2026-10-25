"""Bounded descriptor-based filesystem access for maintenance records."""

import json
import os
import re
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from capicove.wiki.filesystem import _wiki_filesystem_lock


def read_authorized(path: Path, roots: tuple[Path, ...], limit: int) -> bytes:
    """Read through a no-follow directory chain and the same checked descriptor."""
    path = Path(os.path.abspath(path)) if ".." not in path.parts else path
    if ".." in path.parts or not any(path.is_relative_to(root) for root in roots):
        raise ValueError("evidence_unavailable: unauthorized source")
    descriptor = -1
    try:
        descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        for component in path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        opened = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(opened, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("evidence_unavailable: regular file required")
            if before.st_size > limit:
                raise ValueError("capacity_exceeded")
            data = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            current = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)

            def identity(s: os.stat_result) -> tuple[int, int, int, int, int]:
                return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

            if identity(before) != identity(after) or identity(after) != identity(current):
                raise ValueError("evidence_unavailable: source changed during read")
            if len(data) > limit:
                raise ValueError("capacity_exceeded")
            return data
    except OSError as error:
        raise ValueError("evidence_unavailable: unsafe or unavailable path") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@contextmanager
def maintenance_lock(wiki: Path) -> Iterator[None]:
    """Share the existing writer lock; caller validates experimental v3 state."""
    # Reject redirected workspace directories before acquiring the common lock.
    for path in (wiki, *wiki.parents, wiki / "_migration"):
        if path.is_symlink():
            raise ValueError("evidence_unavailable: unsafe workspace")
    with _wiki_filesystem_lock(wiki):
        yield


def _directory(path: Path, *, create: bool = False) -> int:
    """Open an anchored POSIX directory chain without following any symlink."""
    if os.name != "posix" or not path.is_absolute() or ".." in path.parts:
        raise ValueError("unsupported_filesystem: anchored POSIX paths required")
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            try:
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _verify_directory(path: Path, descriptor: int) -> None:
    current = _directory(path)
    try:
        before, after = os.fstat(descriptor), os.fstat(current)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("recovery_conflict: directory changed")
    finally:
        os.close(current)


def ensure_directory(path: Path, wiki: Path) -> None:
    """Create host-selected directories with descriptor-relative operations."""
    if not path.is_relative_to(wiki):
        raise ValueError("unauthorized maintenance directory")
    descriptor = _directory(path, create=True)
    try:
        _verify_directory(path, descriptor)
    finally:
        os.close(descriptor)


def _store_bytes(path: Path, data: bytes, wiki: Path, *, create_only: bool) -> bool:
    if not path.is_relative_to(wiki) or ".." in path.parts:
        raise ValueError("unauthorized maintenance destination")
    parent = _directory(path.parent, create=True)
    temporary = ".capicove-" + uuid.uuid4().hex
    try:
        try:
            existing = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("recovery_conflict: destination is not a regular file")
        if create_only and existing is not None:
            if read_authorized(path, (wiki,), len(data)) != data:
                raise ValueError("evidence_digest_mismatch: destination collision")
            return False
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_directory(path.parent, parent)
        created = True
        if create_only:
            try:
                os.link(
                    temporary,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                created = False
        else:
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
        _verify_directory(path.parent, parent)
        if read_authorized(path, (wiki,), len(data)) != data:
            raise ValueError("evidence_digest_mismatch: published bytes")
        return created
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass
        os.close(parent)


def create_verified(path: Path, data: bytes, wiki: Path) -> bool:
    """Create immutable bytes without path-based link or copy fallback."""
    return _store_bytes(path, data, wiki, create_only=True)


def write_authorized(path: Path, data: bytes, wiki: Path) -> None:
    """Replace one host-selected file through its verified directory descriptor."""
    _store_bytes(path, data, wiki, create_only=False)


def unlink_authorized(path: Path, wiki: Path) -> None:
    """Remove one verified regular-file destination without following parents."""
    if not path.is_relative_to(wiki) or ".." in path.parts:
        raise ValueError("unauthorized maintenance destination")
    parent = _directory(path.parent)
    try:
        _verify_directory(path.parent, parent)
        try:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("recovery_conflict: destination is not a regular file")
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)
        _verify_directory(path.parent, parent)
    finally:
        os.close(parent)


def canonical_object(value: dict[str, Any]) -> bytes:
    """Encode a bounded internal record with the same canonical JSON rules."""
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    if len(data) > 1048576:
        raise ValueError("capacity_exceeded: persistent record")
    return data


def read_object(path: Path, wiki: Path, fields: set[str]) -> dict[str, Any]:
    """Decode a closed host record; duplicate fields and non-integers fail."""
    from capicove.wiki.maintenance.codec import _forbid_number, _object

    value = json.loads(
        read_authorized(path, (wiki,), 1048576),
        object_pairs_hook=_object,
        parse_float=_forbid_number,
        parse_constant=_forbid_number,
    )
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid maintenance record fields")
    canonical_object(value)
    return value


def record_path(wiki: Path, folder: str, identity: str, prefix: str) -> Path:
    """Resolve only full host-issued identifiers inside one fixed record folder."""
    if re.fullmatch(re.escape(prefix) + r":[0-9a-f]{64}", identity) is None:
        raise ValueError("invalid maintenance identity")
    directory = wiki / "maintenance" / folder
    if directory.is_symlink():
        raise ValueError("unsafe maintenance directory")
    return directory / (identity.split(":", 1)[1] + ".json")


def write_object(path: Path, value: dict[str, Any], wiki: Path) -> None:
    """Publish a fully validated immutable record with create-only semantics."""
    data = canonical_object(value)
    if path.parent.is_symlink():
        raise ValueError("unsafe maintenance directory")
    create_verified(path, data, wiki)
