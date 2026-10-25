"""Validate and import audio as immutable, content-addressed evidence."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from capicove.processing.audio.models import (
    AudioEvidence,
    AudioFailureCode,
    AudioImportResult,
    AudioStreamInventory,
    AudioStreamSelection,
)
from capicove.processing.audio.ports import (
    AudioStreamInspector,
    AudioValidationError,
    AudioValidator,
)


_AUDIO_MEDIA_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
_CHUNK_SIZE = 1024 * 1024
_SIGNATURE_SIZE = 12
_AUDIO_STREAM_INDEX_OMITTED = object()
_SECURE_STORE_PRIMITIVES_AVAILABLE = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.link in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


class AudioImportError(ValueError):
    """Report an audio source that cannot be accepted as evidence."""

    def __init__(
        self,
        code: AudioFailureCode,
        message: str,
        *,
        preflight_audio_stream_count: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.preflight_audio_stream_count = preflight_audio_stream_count


@dataclass(frozen=True, slots=True)
class _FileSignature:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def evidence_exists(store_root: Path, sha256: str) -> bool:
    """Return whether the content-addressed store already holds these bytes."""
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ValueError("sha256 must be a lowercase hex digest")
    blob = store_root / "sha256" / sha256[:2] / sha256 / "blob"
    try:
        return stat.S_ISREG(blob.lstat().st_mode)
    except OSError:
        return False


class AudioProcessor:
    """Validate and preserve audio without modifying the caller's source."""

    def __init__(
        self,
        stream_inspector: AudioStreamInspector,
        validator: AudioValidator,
    ) -> None:
        self._stream_inspector = stream_inspector
        self._validator = validator

    def import_file(
        self,
        source_path: Path,
        store_root: Path,
        *,
        audio_stream_index: object = _AUDIO_STREAM_INDEX_OMITTED,
        deadline: datetime | None = None,
    ) -> AudioEvidence:
        """Validate and copy one audio file into a content-addressed store."""
        return self.import_file_with_selection(
            source_path,
            store_root,
            audio_stream_index=audio_stream_index,
            deadline=deadline,
        ).evidence

    def import_file_with_selection(
        self,
        source_path: Path,
        store_root: Path,
        *,
        audio_stream_index: object = _AUDIO_STREAM_INDEX_OMITTED,
        deadline: datetime | None = None,
    ) -> AudioImportResult:
        """Preserve original bytes and return the selected stream lineage."""
        selector = _validate_audio_stream_index(audio_stream_index)
        _require_remaining_deadline(deadline)
        try:
            source = source_path.expanduser().resolve()
            initial_signature = _file_signature(source)
        except OSError as error:
            raise AudioImportError(
                AudioFailureCode.INVALID_AUDIO,
                "audio source is not readable",
            ) from error
        try:
            root = store_root.expanduser().absolute()
        except OSError as error:
            raise AudioImportError(
                AudioFailureCode.STORE_FAILED,
                "audio evidence could not be stored",
            ) from error
        if not _secure_store_primitives_available():
            raise AudioImportError(
                AudioFailureCode.STORE_FAILED,
                "audio evidence store requires secure filesystem primitives",
            )
        if not source.is_file():
            raise AudioImportError(
                AudioFailureCode.INVALID_AUDIO,
                "audio source is not a regular file",
            )
        if initial_signature.size < 1:
            raise AudioImportError(
                AudioFailureCode.INVALID_AUDIO,
                "audio source must not be empty",
            )

        suffix = source.suffix.lower()
        try:
            media_type = _AUDIO_MEDIA_TYPES[suffix]
        except KeyError as error:
            raise AudioImportError(
                AudioFailureCode.UNSUPPORTED_FORMAT,
                f"unsupported audio extension: {suffix or '<none>'}",
            ) from error
        try:
            _require_remaining_deadline(deadline)
            with source.open("rb") as stream:
                content_matches_extension = _content_matches_extension(suffix, stream)
            _require_remaining_deadline(deadline)
        except OSError as error:
            raise AudioImportError(
                AudioFailureCode.INVALID_AUDIO,
                "audio source is not readable",
            ) from error
        if not content_matches_extension:
            raise AudioImportError(
                AudioFailureCode.INVALID_AUDIO,
                f"audio content does not match extension: {suffix}",
            )

        try:
            _require_remaining_deadline(deadline)
            if deadline is None:
                inventory = self._stream_inspector.inspect(source)
            else:
                inventory = self._stream_inspector.inspect(source, deadline=deadline)
        except AudioValidationError as error:
            raise AudioImportError(error.code, str(error)) from error
        selection = _resolve_audio_stream_selection(inventory, selector)
        try:
            _require_remaining_deadline(deadline)
            if deadline is None:
                self._validator.validate(
                    source,
                    audio_stream_index=selection.selected_audio_stream_index,
                )
            else:
                self._validator.validate(
                    source,
                    audio_stream_index=selection.selected_audio_stream_index,
                    deadline=deadline,
                )
        except AudioValidationError as error:
            raise AudioImportError(
                error.code,
                str(error),
                preflight_audio_stream_count=inventory.audio_stream_count,
            ) from error

        open_directories: list[int] = []
        temporary_name: str | None = None
        temporary_directory: int | None = None
        try:
            sha256 = _hash_file(source, deadline=deadline)
            _require_remaining_deadline(deadline)
            if _file_signature(source) != initial_signature:
                raise AudioImportError(
                    AudioFailureCode.SOURCE_CHANGED,
                    "audio source changed during import",
                )
            stored_path = root / "sha256" / sha256[:2] / sha256 / "blob"
            _require_remaining_deadline(deadline)
            root.mkdir(parents=True, mode=0o700, exist_ok=True)
            root_descriptor, root_identity = _open_store_root(root, deadline)
            open_directories.append(root_descriptor)
            sha_directory = _open_or_create_directory(root_descriptor, "sha256", deadline)
            open_directories.append(sha_directory)
            prefix_directory = _open_or_create_directory(
                sha_directory,
                sha256[:2],
                deadline,
            )
            open_directories.append(prefix_directory)
            hash_directory = _open_or_create_directory(
                prefix_directory,
                sha256,
                deadline,
            )
            open_directories.append(hash_directory)

            existing_hash = _existing_blob_hash(hash_directory, deadline)
            if existing_hash is not None:
                if existing_hash != sha256:
                    raise AudioImportError(
                        AudioFailureCode.STORE_CORRUPTED,
                        "content-addressed evidence path is corrupted",
                    )
            else:
                temporary_directory = _open_or_create_directory(
                    root_descriptor,
                    ".tmp",
                    deadline,
                )
                open_directories.append(temporary_directory)
                descriptor, temporary_name = _create_temporary_file(temporary_directory)
                copied_digest = hashlib.sha256()
                copied_size = 0
                with (
                    os.fdopen(descriptor, "wb") as output_stream,
                    source.open("rb") as input_stream,
                ):
                    while True:
                        _require_remaining_deadline(deadline)
                        chunk = input_stream.read(_CHUNK_SIZE)
                        if not chunk:
                            break
                        _require_remaining_deadline(deadline)
                        output_stream.write(chunk)
                        copied_digest.update(chunk)
                        copied_size += len(chunk)
                    _require_remaining_deadline(deadline)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())

                _require_remaining_deadline(deadline)
                if (
                    _file_signature(source) != initial_signature
                    or copied_size != initial_signature.size
                    or copied_digest.hexdigest() != sha256
                ):
                    raise AudioImportError(
                        AudioFailureCode.SOURCE_CHANGED,
                        "audio source changed during import",
                    )

                try:
                    _require_remaining_deadline(deadline)
                    os.link(
                        temporary_name,
                        "blob",
                        src_dir_fd=temporary_directory,
                        dst_dir_fd=hash_directory,
                        follow_symlinks=False,
                    )
                    os.unlink(temporary_name, dir_fd=temporary_directory)
                    temporary_name = None
                except FileExistsError:
                    if _existing_blob_hash(hash_directory, deadline) != sha256:
                        raise AudioImportError(
                            AudioFailureCode.STORE_CORRUPTED,
                            "content-addressed evidence path is corrupted",
                        ) from None
                _require_remaining_deadline(deadline)
                if _existing_blob_hash(hash_directory, deadline) != sha256:
                    raise AudioImportError(
                        AudioFailureCode.STORE_CORRUPTED,
                        "content-addressed evidence path is corrupted",
                    )
                _require_remaining_deadline(deadline)
                os.fsync(hash_directory)

            _require_remaining_deadline(deadline)
            _require_store_path_unchanged(
                root,
                root_descriptor,
                root_identity,
                (
                    (root_descriptor, "sha256", sha_directory),
                    (sha_directory, sha256[:2], prefix_directory),
                    (prefix_directory, sha256, hash_directory),
                ),
            )
            return AudioImportResult(
                evidence=AudioEvidence(
                    evidence_id=f"sha256:{sha256}",
                    sha256=sha256,
                    size_bytes=initial_signature.size,
                    media_type=media_type,
                    source_name=source.name,
                    stored_path=stored_path,
                ),
                stream_selection=selection,
            )
        except AudioImportError:
            raise
        except OSError as error:
            raise AudioImportError(
                AudioFailureCode.STORE_FAILED,
                "audio evidence could not be stored",
            ) from error
        finally:
            if temporary_name is not None and temporary_directory is not None:
                try:
                    os.unlink(temporary_name, dir_fd=temporary_directory)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            for directory in reversed(open_directories):
                try:
                    os.close(directory)
                except OSError:
                    pass


def _validate_audio_stream_index(value: object) -> int | None:
    if value is _AUDIO_STREAM_INDEX_OMITTED:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AudioImportError(
            AudioFailureCode.INVALID_AUDIO_STREAM_INDEX,
            "audio_stream_index must be a non-negative integer",
        )
    return value


def _resolve_audio_stream_selection(
    inventory: AudioStreamInventory,
    selector: int | None,
) -> AudioStreamSelection:
    count = inventory.audio_stream_count
    if count == 0:
        raise AudioImportError(
            AudioFailureCode.NO_AUDIO_STREAM,
            "audio input contains no audio streams",
            preflight_audio_stream_count=0,
        )
    if selector is None:
        if count != 1:
            raise AudioImportError(
                AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED,
                "multiple audio streams require audio_stream_index",
                preflight_audio_stream_count=count,
            )
        return AudioStreamSelection(
            selected_audio_stream_index=0,
            selection_method="sole_stream_implicit",
            preflight_audio_stream_count=1,
        )
    if selector >= count:
        raise AudioImportError(
            AudioFailureCode.AUDIO_STREAM_INDEX_OUT_OF_RANGE,
            "audio_stream_index is outside the available audio stream range",
            preflight_audio_stream_count=count,
        )
    return AudioStreamSelection(
        selected_audio_stream_index=selector,
        selection_method="explicit_index",
        preflight_audio_stream_count=count,
    )


def _file_signature(path: Path) -> _FileSignature:
    status = path.stat()
    return _FileSignature(
        device=status.st_dev,
        inode=status.st_ino,
        size=status.st_size,
        modified_ns=status.st_mtime_ns,
        changed_ns=status.st_ctime_ns,
    )


def _hash_file(path: Path, *, deadline: datetime | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            _require_remaining_deadline(deadline)
            chunk = stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    _require_remaining_deadline(deadline)
    return digest.hexdigest()


def _require_remaining_deadline(deadline: datetime | None) -> None:
    if deadline is None:
        return
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    if deadline <= datetime.now(UTC):
        raise AudioImportError(
            AudioFailureCode.TIMEOUT,
            "audio import deadline has expired",
        )


def _secure_store_primitives_available() -> bool:
    return _SECURE_STORE_PRIMITIVES_AVAILABLE


def _open_store_root(
    root: Path,
    deadline: datetime | None,
) -> tuple[int, tuple[int, int]]:
    _require_remaining_deadline(deadline)
    try:
        status = root.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise AudioImportError(
                AudioFailureCode.STORE_CORRUPTED,
                "content-addressed evidence path is corrupted",
            )
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened_status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) != (opened_status.st_dev, opened_status.st_ino):
            os.close(descriptor)
            raise AudioImportError(
                AudioFailureCode.STORE_CORRUPTED,
                "content-addressed evidence path is corrupted",
            )
        return descriptor, (status.st_dev, status.st_ino)
    except AudioImportError:
        raise
    except OSError as error:
        raise AudioImportError(
            AudioFailureCode.STORE_FAILED,
            "audio evidence could not be stored",
        ) from error


def _open_or_create_directory(parent: int, name: str, deadline: datetime | None) -> int:
    _require_remaining_deadline(deadline)
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent)
    except FileExistsError:
        pass
    try:
        status = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISDIR(status.st_mode):
            raise OSError("store path component is not a directory")
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent,
        )
        opened_status = os.fstat(descriptor)
        if (status.st_dev, status.st_ino) != (opened_status.st_dev, opened_status.st_ino):
            os.close(descriptor)
            raise OSError("store path component changed while opening")
        return descriptor
    except OSError as error:
        raise AudioImportError(
            AudioFailureCode.STORE_CORRUPTED,
            "content-addressed evidence path is corrupted",
        ) from error


def _create_temporary_file(directory: int) -> tuple[int, str]:
    for _ in range(100):
        name = f"audio-{uuid.uuid4().hex}"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
        except FileExistsError:
            continue
        return descriptor, name
    raise OSError("could not allocate a unique audio temporary file")


def _existing_blob_hash(directory: int, deadline: datetime | None) -> str | None:
    _require_remaining_deadline(deadline)
    try:
        descriptor = os.open(
            "blob",
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory,
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise AudioImportError(
            AudioFailureCode.STORE_CORRUPTED,
            "content-addressed evidence path is corrupted",
        ) from error
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise AudioImportError(
                AudioFailureCode.STORE_CORRUPTED,
                "content-addressed evidence path is corrupted",
            )
        digest = hashlib.sha256()
        while True:
            _require_remaining_deadline(deadline)
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        _require_remaining_deadline(deadline)
        path_status = os.stat("blob", dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(path_status.st_mode) or (status.st_dev, status.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise AudioImportError(
                AudioFailureCode.STORE_CORRUPTED,
                "content-addressed evidence path is corrupted",
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _require_store_path_unchanged(
    root: Path,
    root_descriptor: int,
    root_identity: tuple[int, int],
    components: tuple[tuple[int, str, int], ...],
) -> None:
    try:
        root_status = root.lstat()
        opened_root_status = os.fstat(root_descriptor)
        if (
            stat.S_ISLNK(root_status.st_mode)
            or (
                root_status.st_dev,
                root_status.st_ino,
            )
            != root_identity
            or (
                opened_root_status.st_dev,
                opened_root_status.st_ino,
            )
            != root_identity
        ):
            raise OSError("store root changed during import")
        for parent, name, child in components:
            path_status = os.stat(name, dir_fd=parent, follow_symlinks=False)
            child_status = os.fstat(child)
            if not stat.S_ISDIR(path_status.st_mode) or (
                path_status.st_dev,
                path_status.st_ino,
            ) != (child_status.st_dev, child_status.st_ino):
                raise OSError("store path component changed during import")
    except OSError as error:
        raise AudioImportError(
            AudioFailureCode.STORE_CORRUPTED,
            "content-addressed evidence path is corrupted",
        ) from error


def _is_mpeg_audio_frame_header(header: bytes) -> bool:
    if len(header) < 3 or header[0] != 0xFF or header[1] & 0xE0 != 0xE0:
        return False
    version_bits = header[1] & 0x18
    layer_bits = header[1] & 0x06
    bitrate_index = header[2] >> 4
    sample_rate_index = (header[2] >> 2) & 0x03
    return (
        version_bits != 0x08
        and layer_bits != 0
        and bitrate_index not in {0, 0x0F}
        and sample_rate_index != 0x03
    )


def _content_matches_extension(suffix: str, stream: BinaryIO) -> bool:
    header = stream.read(_SIGNATURE_SIZE)
    if suffix == ".aac":
        return header.startswith(b"ADIF") or (
            len(header) >= 2 and header[0] == 0xFF and header[1] & 0xF6 == 0xF0
        )
    if suffix == ".flac":
        return header.startswith(b"fLaC")
    if suffix == ".m4a":
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if suffix == ".mp3":
        if not header.startswith(b"ID3"):
            return _is_mpeg_audio_frame_header(header)
        if (
            len(header) < 10
            or header[3] not in {2, 3, 4}
            or any(byte & 0x80 for byte in header[6:10])
        ):
            return False
        tag_size = 0
        for byte in header[6:10]:
            tag_size = (tag_size << 7) | byte
        footer_size = 10 if header[3] == 4 and header[5] & 0x10 else 0
        stream.seek(10 + tag_size + footer_size)
        return _is_mpeg_audio_frame_header(stream.read(4))
    if suffix in {".ogg", ".opus"}:
        return header.startswith(b"OggS")
    if suffix == ".wav":
        return len(header) >= 12 and header[:4] in {b"RIFF", b"RF64"} and header[8:12] == b"WAVE"
    if suffix == ".webm":
        return header.startswith(b"\x1a\x45\xdf\xa3")
    return False
