"""Safe local resolution and content-addressed copying of export media."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from capicove.traces import MediaLocatorKind, MediaReference


_COPY_CHUNK_BYTES = 1_048_576
_EXTENSIONS = {
    "audio/wav": ".wav",
    "video/mp4": ".mp4",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


class TraceMediaCopyError(RuntimeError):
    """Report one stable media copy rejection code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ExportMediaResolver(Protocol):
    """Resolve one immutable reference to an explicitly local source path."""

    def resolve(self, reference: MediaReference, /) -> Path | None: ...


def media_reference_identity_sha256(reference: MediaReference, /) -> str:
    """Hash every immutable MediaReference field for conflict detection."""
    locator = reference.locator
    value = {
        "media_id": reference.media_id,
        "modality": reference.modality.value,
        "media_type": reference.media_type,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
        "evidence_ref": reference.evidence_ref,
        "locator": (
            None if locator is None else {"kind": locator.kind.value, "value": locator.value}
        ),
        "start_seconds": reference.start_seconds,
        "end_seconds": reference.end_seconds,
        "frame_indices": list(reference.frame_indices),
        "derived_from": list(reference.derived_from),
    }
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class WorkspaceExportMediaResolver:
    """Resolve only workspace-relative locators below one absolute root."""

    def __init__(self, workspace_root: Path) -> None:
        if not isinstance(workspace_root, Path) or not workspace_root.is_absolute():
            raise ValueError("workspace_root must be absolute")
        try:
            root_lstat = workspace_root.lstat()
        except OSError as error:
            raise ValueError("workspace_root must be an existing directory") from error
        if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
            raise ValueError("workspace_root must be a regular directory, not a symlink")
        self._workspace_root = workspace_root.resolve()

    def resolve(self, reference: MediaReference, /) -> Path | None:
        """Return a safe local candidate or None for an unsupported locator."""
        locator = reference.locator
        if locator is None or locator.kind is not MediaLocatorKind.WORKSPACE_RELATIVE:
            return None
        relative = PurePosixPath(locator.value)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        candidate = self._workspace_root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            return None
        if not resolved.is_relative_to(self._workspace_root):
            return None
        return candidate


@dataclass(frozen=True, slots=True)
class MediaCopyResult:
    """Describe one verified content-addressed copy or deduplication hit."""

    relative_path: PurePosixPath
    size_bytes: int
    sha256: str
    copied: bool


def copy_media_reference(
    reference: MediaReference,
    source: Path,
    export_root: Path,
    /,
) -> MediaCopyResult:
    """Copy one regular local file without following symlinks or changing source."""
    if not isinstance(source, Path) or not source.is_absolute():
        raise TraceMediaCopyError("INVALID_SOURCE", "media source must be absolute")
    if not isinstance(export_root, Path) or not export_root.is_absolute():
        raise TraceMediaCopyError("PATH_ESCAPE", "export root must be absolute")
    try:
        root_stat = export_root.lstat()
    except OSError as error:
        raise TraceMediaCopyError("PATH_ESCAPE", "export root is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise TraceMediaCopyError("PATH_ESCAPE", "export root must be a regular directory")

    extension = _EXTENSIONS.get(reference.media_type, ".bin")
    relative = PurePosixPath(
        "media", "sha256", reference.sha256[:2], f"{reference.sha256}{extension}"
    )
    if os.name == "posix":
        return _copy_media_reference_posix(reference, source, export_root, relative, root_stat)
    target = export_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _verify_existing_target(target, reference)
        return MediaCopyResult(relative, reference.size_bytes, reference.sha256, False)

    source_descriptor = _open_regular_source(source)
    temporary_path: Path | None = None
    try:
        before = os.fstat(source_descriptor)
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{reference.sha256}.tmp-",
            dir=target.parent,
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size_bytes = 0
        with os.fdopen(temporary_descriptor, "wb", closefd=True) as output:
            with os.fdopen(source_descriptor, "rb", closefd=False) as input_stream:
                while True:
                    chunk = input_stream.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_descriptor)
        if _source_identity(before) != _source_identity(after):
            raise TraceMediaCopyError("SOURCE_CHANGED", "media source changed during copy")
        if size_bytes != reference.size_bytes or digest.hexdigest() != reference.sha256:
            raise TraceMediaCopyError(
                "SOURCE_MISMATCH", "media source size or hash does not match its reference"
            )
        try:
            os.link(temporary_path, target)
            copied = True
        except FileExistsError:
            _verify_existing_target(target, reference)
            copied = False
        temporary_path.unlink()
        temporary_path = None
        _sync_directory(target.parent)
        return MediaCopyResult(relative, size_bytes, reference.sha256, copied)
    finally:
        os.close(source_descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _copy_media_reference_posix(
    reference: MediaReference,
    source: Path,
    export_root: Path,
    relative: PurePosixPath,
    root_stat: os.stat_result,
) -> MediaCopyResult:
    """Create and verify a media copy relative to a pinned export directory."""
    parent_fd = _open_export_parent_posix(export_root, relative.parent, root_stat)
    temporary: str | None = None
    source_fd: int | None = None
    try:
        target_name = relative.name
        if _verify_existing_target_posix(parent_fd, target_name, reference):
            return MediaCopyResult(relative, reference.size_bytes, reference.sha256, False)
        source_fd = _open_regular_source(source)
        before = os.fstat(source_fd)
        temporary = f".{reference.sha256}.tmp-{secrets.token_hex(16)}"
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        digest = hashlib.sha256()
        size_bytes = 0
        with os.fdopen(temporary_fd, "wb") as output:
            with os.fdopen(source_fd, "rb", closefd=False) as input_stream:
                while chunk := input_stream.read(_COPY_CHUNK_BYTES):
                    output.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if _source_identity(before) != _source_identity(os.fstat(source_fd)):
            raise TraceMediaCopyError("SOURCE_CHANGED", "media source changed during copy")
        if size_bytes != reference.size_bytes or digest.hexdigest() != reference.sha256:
            raise TraceMediaCopyError(
                "SOURCE_MISMATCH", "media source size or hash does not match its reference"
            )
        try:
            os.link(
                temporary,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            copied = True
        except FileExistsError as error:
            if not _verify_existing_target_posix(parent_fd, target_name, reference):
                raise TraceMediaCopyError(
                    "DESTINATION_CONFLICT", "media destination disappeared during copy"
                ) from error
            copied = False
        os.unlink(temporary, dir_fd=parent_fd)
        temporary = None
        os.fsync(parent_fd)
        return MediaCopyResult(relative, size_bytes, reference.sha256, copied)
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _open_export_parent_posix(
    export_root: Path, relative_parent: PurePosixPath, root_stat: os.stat_result
) -> int:
    """Pin every output ancestor and create only missing owned subdirectories."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise TraceMediaCopyError("PATH_ESCAPE", "secure export directory APIs are unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        parts = export_root.parts
        if len(parts) > 4 and parts[1:3] == ("dev", "fd") and parts[3].isdecimal():
            directory_fd = os.dup(int(parts[3]))
            components = parts[4:]
        else:
            directory_fd = os.open(export_root.anchor, flags)
            components = parts[1:]
    except OSError as error:
        raise TraceMediaCopyError("PATH_ESCAPE", "export directory is unsafe") from error
    try:
        for component in components:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        opened_root = os.fstat(directory_fd)
        if (opened_root.st_dev, opened_root.st_ino) != (root_stat.st_dev, root_stat.st_ino):
            raise TraceMediaCopyError("PATH_ESCAPE", "export root changed while opening")
        for component in relative_parent.parts:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except OSError as error:
        os.close(directory_fd)
        raise TraceMediaCopyError("PATH_ESCAPE", "export directory is unsafe") from error
    except Exception:
        os.close(directory_fd)
        raise


def _verify_existing_target_posix(
    parent_fd: int, target_name: str, reference: MediaReference
) -> bool:
    """Hash a pre-existing copy through the pinned parent without following links."""
    try:
        descriptor = os.open(
            target_name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return False
    except OSError as error:
        raise TraceMediaCopyError(
            "DESTINATION_CONFLICT", "media destination could not be inspected"
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise TraceMediaCopyError(
                "DESTINATION_CONFLICT", "media destination is not a regular file"
            )
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := stream.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
            size_bytes += len(chunk)
        after = os.fstat(stream.fileno())
    if (
        _source_identity(before) != _source_identity(after)
        or size_bytes != reference.size_bytes
        or digest.hexdigest() != reference.sha256
    ):
        raise TraceMediaCopyError(
            "DESTINATION_CONFLICT", "media destination contains different bytes"
        )
    return True


def _open_regular_source(source: Path) -> int:
    try:
        observed = source.lstat()
    except OSError as error:
        raise TraceMediaCopyError("INVALID_SOURCE", "media source is unavailable") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise TraceMediaCopyError("INVALID_SOURCE", "media source must be a regular file")
    try:
        if os.name == "posix":
            descriptor = _open_source_posix(source)
        else:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(source, flags)
    except OSError as error:
        raise TraceMediaCopyError("INVALID_SOURCE", "media source could not be opened") from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise TraceMediaCopyError("INVALID_SOURCE", "media source could not be opened") from error
    if not stat.S_ISREG(opened.st_mode) or (
        observed.st_dev,
        observed.st_ino,
    ) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise TraceMediaCopyError("INVALID_SOURCE", "media source identity changed while opening")
    return descriptor


def _open_source_posix(source: Path) -> int:
    """Open a source beneath pinned, no-follow directory handles."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise TraceMediaCopyError("INVALID_SOURCE", "secure media source APIs are unavailable")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_fd = os.open(source.anchor, directory_flags)
    try:
        for component in source.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(source.name, file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def _verify_existing_target(target: Path, reference: MediaReference) -> None:
    try:
        observed = target.lstat()
    except OSError as error:
        raise TraceMediaCopyError(
            "DESTINATION_CONFLICT", "media destination could not be inspected"
        ) from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise TraceMediaCopyError("DESTINATION_CONFLICT", "media destination is not a regular file")
    digest = hashlib.sha256()
    size_bytes = 0
    with target.open("rb") as stream:
        while True:
            chunk = stream.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
    if size_bytes != reference.size_bytes or digest.hexdigest() != reference.sha256:
        raise TraceMediaCopyError(
            "DESTINATION_CONFLICT", "media destination contains different bytes"
        )


def _source_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
