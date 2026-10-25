"""Safe deterministic discovery for local directory batches."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import BinaryIO

from capicove.batch.models import BatchItem, BatchModality, SourceSnapshot


_HASH_CHUNK_SIZE = 1024 * 1024


class DirectoryDiscoveryError(RuntimeError):
    """Report that a directory or source cannot be discovered safely."""


class SourceChangedDuringDiscoveryError(DirectoryDiscoveryError):
    """Report that a source changed while its identity was being recorded."""


class DirectoryDiscoverer:
    """Discover supported regular files without following symbolic links."""

    def __init__(self, media_types: Mapping[str, str]) -> None:
        normalized: dict[str, str] = {}
        for suffix, media_type in media_types.items():
            normalized_suffix = _normalize_suffix(suffix)
            if not media_type.strip():
                raise ValueError("media_type must not be blank")
            existing = normalized.get(normalized_suffix)
            if existing is not None and existing != media_type:
                raise ValueError("conflicting media types for one normalized suffix")
            normalized[normalized_suffix] = media_type
        if not normalized:
            raise ValueError("at least one media type rule is required")
        self._media_types = MappingProxyType(normalized)

    def discover(self, root: Path, modality: BatchModality) -> tuple[BatchItem, ...]:
        """Return supported files ordered by their root-relative POSIX path."""
        if not isinstance(modality, BatchModality):
            raise ValueError("modality must be a BatchModality")
        expanded_root = root.expanduser()
        if expanded_root.is_symlink():
            raise DirectoryDiscoveryError("batch input root must be a directory, not a symlink")
        try:
            resolved_root = expanded_root.resolve(strict=True)
        except OSError as error:
            raise DirectoryDiscoveryError(
                "batch input root must be a readable directory"
            ) from error
        if not resolved_root.is_dir():
            raise DirectoryDiscoveryError("batch input root must be a directory")

        candidates = sorted(
            self._candidate_paths(resolved_root),
            key=lambda path: path.relative_to(resolved_root).as_posix(),
        )
        items: list[BatchItem] = []
        for source_path in candidates:
            relative_path = PurePosixPath(source_path.relative_to(resolved_root).as_posix())
            observed = _observe_regular_file(source_path)
            if observed is None:
                continue
            sha256, snapshot = observed
            items.append(
                BatchItem(
                    item_id=_item_id(modality, relative_path),
                    modality=modality,
                    source_path=source_path,
                    relative_path=relative_path,
                    media_type=self._media_types[source_path.suffix.lower()],
                    sha256=sha256,
                    snapshot=snapshot,
                )
            )
        return tuple(items)

    def _candidate_paths(self, root: Path) -> list[Path]:
        candidates: list[Path] = []

        def raise_walk_error(error: OSError) -> None:
            raise DirectoryDiscoveryError("batch input directory is not readable") from error

        for directory_name, child_directories, file_names in os.walk(
            root,
            topdown=True,
            onerror=raise_walk_error,
            followlinks=False,
        ):
            directory = Path(directory_name)
            child_directories[:] = sorted(
                name for name in child_directories if not (directory / name).is_symlink()
            )
            for file_name in sorted(file_names):
                source_path = directory / file_name
                if source_path.is_symlink():
                    continue
                if source_path.suffix.lower() in self._media_types:
                    candidates.append(source_path)
        return candidates


def _observe_regular_file(
    path: Path, *, expected_snapshot: SourceSnapshot | None = None
) -> tuple[str, SourceSnapshot] | None:
    try:
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        for parent in path.parents:
            parent_stat = parent.lstat()
            if stat.S_ISLNK(parent_stat.st_mode) or (
                getattr(parent_stat, "st_file_attributes", 0) & reparse_flag
            ):
                raise SourceChangedDuringDiscoveryError(
                    "source parent changed while being discovered"
                )
        path_before = path.lstat()
        if not stat.S_ISREG(path_before.st_mode):
            return None
        if expected_snapshot is not None and _snapshot(path_before) != expected_snapshot:
            raise SourceChangedDuringDiscoveryError("source changed while being discovered")
        digest = hashlib.sha256()
        with _open_regular_file_no_follow(path) as stream:
            opened_before = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_before.st_mode) or _stat_identity(
                path_before
            ) != _stat_identity(opened_before):
                raise SourceChangedDuringDiscoveryError("source changed while being discovered")
            if expected_snapshot is not None and _snapshot(opened_before) != expected_snapshot:
                raise SourceChangedDuringDiscoveryError("source changed while being discovered")
            while chunk := stream.read(_HASH_CHUNK_SIZE):
                digest.update(chunk)
            opened_after = os.fstat(stream.fileno())
            path_after = path.lstat()
    except SourceChangedDuringDiscoveryError:
        raise
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR, errno.ENOENT}:
            raise SourceChangedDuringDiscoveryError(
                "source changed while being discovered"
            ) from error
        raise DirectoryDiscoveryError("batch source is not readable") from error
    if not (
        _stat_identity(opened_before) == _stat_identity(opened_after) == _stat_identity(path_after)
    ):
        raise SourceChangedDuringDiscoveryError("source changed while being discovered")
    return digest.hexdigest(), _snapshot(opened_after)


def _open_regular_file_no_follow(path: Path) -> BinaryIO:
    """Open a POSIX source through pinned, no-follow directory handles."""
    if os.name != "posix":
        return path.open("rb")
    if not path.is_absolute() or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise DirectoryDiscoveryError("batch source requires a safe absolute path")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(path.name, file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        return os.fdopen(file_fd, "rb")
    except Exception:
        os.close(file_fd)
        raise


def _stat_identity(source_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )


def _snapshot(source_stat: os.stat_result) -> SourceSnapshot:
    return SourceSnapshot(
        device=source_stat.st_dev,
        inode=source_stat.st_ino,
        size_bytes=source_stat.st_size,
        modified_ns=source_stat.st_mtime_ns,
        changed_ns=source_stat.st_ctime_ns,
    )


def _item_id(modality: BatchModality, relative_path: PurePosixPath) -> str:
    digest = hashlib.sha256()
    digest.update(b"capicove.batch.item.v1\0")
    digest.update(modality.value.encode("utf-8"))
    digest.update(b"\0")
    digest.update(relative_path.as_posix().encode("utf-8"))
    return f"item:{digest.hexdigest()}"


def _normalize_suffix(value: str) -> str:
    suffix = value.strip().lower()
    if not suffix:
        raise ValueError("media type suffix must not be blank")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    if suffix == "." or "/" in suffix or "\\" in suffix:
        raise ValueError("media type suffix must be a file extension")
    return suffix
