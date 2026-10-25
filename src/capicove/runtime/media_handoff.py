"""Verified private staging of host media for path-based backends (ADR-0026).

An intake authorizes a caller path against configured read roots, opens it
through pinned no-follow directory handles, and either copies it to a read-only
per-invocation file or hashes it in place. The source identity is checked before
and after the read; any change rejects the call.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


CHUNK_BYTES = 1024 * 1024
_SIGNATURE_BYTES = 192


class MediaDenied(Exception):
    """The path is outside the configured roots, is a link, or does not exist."""


class MediaInvalid(Exception):
    """The file is not an accepted media container or exceeds the size bound."""


class MediaChanged(Exception):
    """The source changed while it was read."""


class AssetIntegrityError(ValueError):
    """A configured model or asset no longer matches its pinned SHA-256."""


@dataclass(frozen=True, slots=True)
class StagedFile:
    """One verified read-only copy and the identity of its original."""

    original: Path
    staged: Path
    sha256: str
    size_bytes: int
    # The same file under its resolved staging path; backends may report either form.
    resolved: str = ""


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Hashes of one source read through pinned handles."""

    size_bytes: int
    sha256: str
    xxh3_128: str | None


def _directory_flags() -> int:
    """Return no-follow directory flags, denying platforms that lack them."""
    if not (hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")):
        raise MediaDenied("media handoff requires POSIX no-follow directory handles")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _read_chunk(descriptor: int, size: int) -> bytes:
    """Read one chunk; a module-level seam so race tests can intervene."""
    return os.read(descriptor, size)


class MediaIntake:
    """Authorize, verify, and stage media for one capability."""

    def __init__(
        self,
        *,
        read_roots: Sequence[Path],
        staging_root: Path,
        max_input_bytes: int,
        accepts: Callable[[Path, bytes], bool],
        reject_hardlinks: bool = False,
    ) -> None:
        self._roots = tuple(read_roots)
        self._staging_root = staging_root
        self._max = max_input_bytes
        self._accepts = accepts
        self._reject_hardlinks = reject_hardlinks

    def authorize(self, path: Path) -> tuple[Path, tuple[str, ...]]:
        """Return the matching read root and the normalized parts below it."""
        if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
            raise MediaDenied("media paths must be absolute and normalized")
        for root in self._roots:
            if path != root and path.is_relative_to(root):
                return root, path.relative_to(root).parts
        raise MediaDenied("media path is outside the configured read roots")

    @contextmanager
    def stage(
        self,
        paths: Sequence[Path],
        assets: Sequence[tuple[Path, str]] = (),
        *,
        check_deadline: Callable[[], None] | None = None,
    ) -> Iterator[tuple[StagedFile, ...]]:
        """Copy sources, then pinned assets, into a private directory removed afterwards.

        Every staging write goes through descriptors pinned from ``staging_root``,
        and staged files are exposed as ``/proc/<pid>/fd/<n>/...`` paths bound to
        that directory, so replacing a staging path cannot redirect reads or writes.
        Staged files keep their original basenames in per-file subdirectories.
        """
        if check_deadline is not None:
            check_deadline()
        authorized = [(path, *self.authorize(path)) for path in paths]
        if not Path(f"/proc/{os.getpid()}/fd").is_dir():
            raise MediaDenied("media staging requires Linux /proc descriptor paths")
        staging = open_private_directory(self._staging_root)
        try:
            _remove_stale_invocations(staging)
            name = f"inv-{os.getpid()}-{secrets.token_hex(8)}"
            os.mkdir(name, 0o700, dir_fd=staging)
            invocation = os.open(name, _directory_flags(), dir_fd=staging)
            try:
                base = Path(f"/proc/{os.getpid()}/fd/{invocation}")
                staged = []
                for index, (path, root, parts) in enumerate(authorized):
                    staged.append(
                        self._copy(
                            path,
                            root,
                            parts,
                            invocation,
                            base,
                            f"{index:04d}",
                            True,
                            check_deadline=check_deadline,
                        )
                    )
                for index, (path, expected) in enumerate(assets):
                    copy = self._copy(
                        path,
                        Path(path.anchor),
                        path.parts[1:],
                        invocation,
                        base,
                        f"asset-{index:02d}",
                        False,
                        limit=None,
                        check_deadline=check_deadline,
                    )
                    if copy.sha256 != expected:
                        raise AssetIntegrityError(f"{path.name} does not match its pinned SHA-256")
                    staged.append(copy)
                yield tuple(staged)
            finally:
                os.close(invocation)
                shutil.rmtree(name, dir_fd=staging, ignore_errors=True)
        finally:
            os.close(staging)

    def digest(self, path: Path, *, fast_hash: Callable[[], object] | None = None) -> FileDigest:
        """Hash one authorized source in place, verifying its identity throughout."""
        root, parts = self.authorize(path)
        descriptor = _open_source(root, parts)
        try:
            before = _require_regular(descriptor, self._max)
            if self._reject_hardlinks and before.st_nlink != 1:
                raise MediaDenied("hard-linked source is not authorized")
            sha = hashlib.sha256()
            fast = fast_hash() if fast_hash is not None else None
            total = 0
            for chunk in _bounded_chunks(descriptor, before.st_size):
                total += len(chunk)
                sha.update(chunk)
                if fast is not None:
                    fast.update(chunk)  # type: ignore[attr-defined]
            _require_unchanged(before, os.fstat(descriptor), total)
        finally:
            os.close(descriptor)
        return FileDigest(
            total,
            sha.hexdigest(),
            fast.hexdigest() if fast is not None else None,  # type: ignore[attr-defined]
        )

    def _copy(
        self,
        original: Path,
        root: Path,
        parts: tuple[str, ...],
        invocation: int,
        base: Path,
        subdirectory: str,
        check_signature: bool,
        *,
        limit: int | None = -1,
        check_deadline: Callable[[], None] | None = None,
    ) -> StagedFile:
        descriptor = _open_source(root, parts)
        try:
            before = _require_regular(descriptor, self._max if limit == -1 else limit)
            if self._reject_hardlinks and before.st_nlink != 1:
                raise MediaDenied("hard-linked source is not authorized")
            if check_signature and not self._accepts(
                original, os.pread(descriptor, _SIGNATURE_BYTES, 0)
            ):
                raise MediaInvalid("the file is not an accepted media container")
            os.mkdir(subdirectory, 0o700, dir_fd=invocation)
            directory = os.open(subdirectory, _directory_flags(), dir_fd=invocation)
            try:
                output = os.open(
                    original.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory,
                )
            finally:
                os.close(directory)
            sha = hashlib.sha256()
            total = 0
            try:
                for chunk in _bounded_chunks(descriptor, before.st_size):
                    if check_deadline is not None:
                        check_deadline()
                    total += len(chunk)
                    sha.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        view = view[os.write(output, view) :]
                os.fsync(output)
                os.fchmod(output, 0o400)
            finally:
                os.close(output)
            _require_unchanged(before, os.fstat(descriptor), total)
        finally:
            os.close(descriptor)
        resolved = os.path.join(os.readlink(base), subdirectory, original.name)
        return StagedFile(
            original, base / subdirectory / original.name, sha.hexdigest(), total, resolved
        )


def open_private_directory(path: Path, *, create: bool = True) -> int:
    """Open an owner-private directory through no-follow handles, creating its leaf."""
    if not path.is_absolute() or ".." in path.parts:
        raise MediaDenied("directories must be absolute and normalized")
    try:
        parent = os.open(path.anchor, _directory_flags())
    except OSError:
        raise MediaDenied("directory is unavailable") from None
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, _directory_flags(), dir_fd=parent)
            os.close(parent)
            parent = child
        try:
            descriptor = os.open(path.name, _directory_flags(), dir_fd=parent)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(path.name, 0o700, dir_fd=parent)
            except FileExistsError:
                pass  # A concurrent call created it; the checks below still apply.
            descriptor = os.open(path.name, _directory_flags(), dir_fd=parent)
    except OSError:
        raise MediaDenied("directory must exist without symlinks") from None
    finally:
        os.close(parent)
    metadata = os.fstat(descriptor)
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        os.close(descriptor)
        raise MediaDenied("directory must be owner-private")
    return descriptor


def restore_originals(value: object, staged: Sequence[StagedFile]) -> object:
    """Replace staged locators with original paths anywhere in a JSON-like value."""
    mapping = {str(item.staged): str(item.original) for item in staged}
    mapping.update({item.resolved: str(item.original) for item in staged if item.resolved})

    def visit(item: object) -> object:
        if isinstance(item, str):
            for locator, original in mapping.items():
                if locator in item:
                    item = item.replace(locator, original)
            return item
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        return item

    return visit(value)


def verify_asset(path: Path, expected_sha256: str) -> None:
    """Hash a configured asset through pinned no-follow handles; no copy is made."""
    if not path.is_absolute():
        raise AssetIntegrityError("asset paths must be absolute")
    try:
        descriptor = _open_source(Path(path.anchor), path.parts[1:])
    except MediaDenied:
        raise AssetIntegrityError(f"{path.name} is not a readable regular file") from None
    try:
        before = _require_regular(descriptor, None)
        digest = hashlib.sha256()
        total = 0
        for chunk in _bounded_chunks(descriptor, before.st_size):
            total += len(chunk)
            digest.update(chunk)
        _require_unchanged(before, os.fstat(descriptor), total)
    except (MediaDenied, MediaChanged):
        raise AssetIntegrityError(f"{path.name} could not be verified") from None
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise AssetIntegrityError(f"{path.name} does not match its pinned SHA-256")


def _mpeg_audio_frame(head: bytes) -> bool:
    return len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0


# Each accepted audio extension must start with its own container signature, so a
# text playlist or other indirect format can never be staged under an audio name.
AUDIO_SIGNATURES: dict[str, Callable[[bytes], bool]] = {
    ".wav": lambda head: head.startswith(b"RIFF") and head[8:12] == b"WAVE",
    ".mp3": lambda head: (
        _mpeg_audio_frame(head) or (head.startswith(b"ID3") and b"#EXT" not in head)
    ),
    ".flac": lambda head: head.startswith(b"fLaC"),
    ".ogg": lambda head: head.startswith(b"OggS"),
    ".opus": lambda head: head.startswith(b"OggS"),
    ".m4a": lambda head: head[4:8] == b"ftyp",
    ".aac": lambda head: (
        head.startswith(b"ADIF") or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0)
    ),
    ".webm": lambda head: head.startswith(b"\x1a\x45\xdf\xa3"),
}


def is_audio_container(path: Path, head: bytes) -> bool:
    """Accept only supported audio extensions whose bytes match that container."""
    check = AUDIO_SIGNATURES.get(path.suffix.lower())
    return check is not None and check(head)


def is_video_container(head: bytes) -> bool:
    """Recognize common video containers by their leading bytes."""
    return (
        head[4:8] == b"ftyp"
        or head.startswith(b"\x1a\x45\xdf\xa3")
        or (head.startswith(b"RIFF") and head[8:12] == b"AVI ")
        or head.startswith(b"FLV")
        or head.startswith(b"\x00\x00\x01\xba")
        or (len(head) > 188 and head[0] == 0x47 and head[188] == 0x47)
    )


def _open_source(root: Path, parts: tuple[str, ...]) -> int:
    flags = _directory_flags()
    try:
        directory = os.open(root.anchor, flags)
    except OSError:
        raise MediaDenied("read root is unavailable") from None
    try:
        for component in (*root.parts[1:], *parts[:-1]):
            next_directory = os.open(component, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except OSError:
        raise MediaDenied("media must be an existing file without symlinks") from None
    finally:
        os.close(directory)


def _bounded_chunks(descriptor: int, size: int) -> Iterator[bytes]:
    """Yield exactly the size observed at open; a short file means it changed."""
    remaining = size
    while remaining:
        chunk = _read_chunk(descriptor, min(CHUNK_BYTES, remaining))
        if not chunk:
            raise MediaChanged("media changed while it was read")
        remaining -= len(chunk)
        yield chunk


def _require_regular(descriptor: int, limit: int | None) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise MediaDenied("media must be a regular file")
    if limit is not None and metadata.st_size > limit:
        raise MediaInvalid("media exceeds max_input_bytes")
    return metadata


def _require_unchanged(before: os.stat_result, after: os.stat_result, total: int) -> None:
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity) or (
        total != after.st_size
    ):
        raise MediaChanged("media changed while it was read")


def _remove_stale_invocations(staging: int) -> None:
    """Remove invocation directories whose owning process no longer exists."""
    for name in os.listdir(staging):
        if not name.startswith("inv-"):
            continue
        pid_text = name.split("-")[1]
        if not pid_text.isdigit():
            continue
        try:
            metadata = os.stat(name, dir_fd=staging, follow_symlinks=False)
        except OSError:
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        try:
            os.kill(int(pid_text), 0)
        except ProcessLookupError:
            shutil.rmtree(name, dir_fd=staging, ignore_errors=True)
        except (PermissionError, OverflowError):
            continue


__all__ = [
    "AssetIntegrityError",
    "FileDigest",
    "MediaChanged",
    "MediaDenied",
    "MediaIntake",
    "MediaInvalid",
    "StagedFile",
    "is_audio_container",
    "is_video_container",
    "open_private_directory",
    "restore_originals",
    "verify_asset",
]
