"""Descriptor-bound files, strict WAV verification, and no-overwrite publication."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
import struct
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from capicove.processing.audio.clip_models import (
    DISK_RESERVE_BYTES,
    MAX_PCM_BYTES,
    AudioClipGeometry,
    AudioClippingError,
    remaining_seconds,
)
from capicove.processing.audio.models import AudioFailureCode


_PCM_GUID = bytes.fromhex("0100000000001000800000aa00389b71")


@contextmanager
def open_directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Open every directory component without following links."""
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            if component == "..":
                raise ValueError("directory paths must be normalized")
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def same_file(first: os.stat_result, second: os.stat_result) -> bool:
    """Compare the identity of two open or descriptor-relative files."""
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


@contextmanager
def verified_source(source: Path, deadline: datetime) -> Iterator[Path]:
    """Snapshot authorized bytes through a no-follow handle before path-based intake."""
    with open_directory(source.parent) as parent:
        descriptor = os.open(
            source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
                raise AudioClippingError(
                    AudioFailureCode.INVALID_AUDIO, "source must be a nonempty regular file"
                )
            with tempfile.TemporaryDirectory(prefix="capicove-audio-source-") as directory:
                snapshot = Path(directory) / source.name
                with snapshot.open("xb") as output:
                    remaining = before.st_size
                    while remaining:
                        remaining_seconds(deadline)
                        chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                        if not chunk:
                            raise AudioClippingError(
                                AudioFailureCode.SOURCE_CHANGED, "source changed during intake"
                            )
                        output.write(chunk)
                        remaining -= len(chunk)
                after = os.fstat(descriptor)
                fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                if any(getattr(before, field) != getattr(after, field) for field in fields):
                    raise AudioClippingError(
                        AudioFailureCode.SOURCE_CHANGED, "source changed during intake"
                    )
                require_binding(source.parent, parent)
                snapshot.chmod(0o400)
                yield snapshot
        finally:
            os.close(descriptor)


def require_binding(path: Path, descriptor: int) -> None:
    """Reject a logical parent path replaced during execution."""
    with open_directory(path) as current:
        if not same_file(os.fstat(current), os.fstat(descriptor)):
            raise AudioClippingError(AudioFailureCode.SOURCE_CHANGED, "directory binding changed")


def hash_descriptor(descriptor: int, deadline: datetime) -> str:
    """Hash the open file without changing the shared descriptor position."""
    digest = hashlib.sha256()
    position = 0
    while True:
        remaining_seconds(deadline)
        chunk = os.pread(descriptor, 1024 * 1024, position)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        position += len(chunk)


def admit_directory(descriptor: int) -> None:
    """Behaviorally verify exclusive creation, hard links, no-follow, and fsync."""
    names = [f".capicove-probe-{secrets.token_hex(16)}" for _ in range(3)]
    owned: dict[str, os.stat_result] = {}
    try:
        first = os.open(
            names[0], os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=descriptor
        )
        try:
            owned[names[0]] = os.fstat(first)
            # A successful link may lose its acknowledgement; retain its expected inode.
            owned[names[1]] = owned[names[0]]
            os.link(
                names[0],
                names[1],
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
                follow_symlinks=False,
            )
            linked = os.stat(names[1], dir_fd=descriptor, follow_symlinks=False)
            if not same_file(owned[names[0]], linked) or os.fstat(first).st_nlink != 2:
                raise AudioClippingError(
                    AudioFailureCode.BACKEND_UNAVAILABLE, "hard-link semantics are unavailable"
                )
            try:
                os.link(
                    names[0],
                    names[1],
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                pass
            else:
                raise AudioClippingError(
                    AudioFailureCode.BACKEND_UNAVAILABLE,
                    "hard-link creation replaced an existing name",
                )
            os.symlink(names[0], names[2], dir_fd=descriptor)
            owned[names[2]] = os.stat(names[2], dir_fd=descriptor, follow_symlinks=False)
            try:
                unexpected = os.open(names[2], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            except OSError:
                pass
            else:
                os.close(unexpected)
                raise AudioClippingError(
                    AudioFailureCode.BACKEND_UNAVAILABLE, "no-follow semantics are unavailable"
                )
            os.fsync(first)
            os.fsync(descriptor)
        finally:
            os.close(first)
    finally:
        cleanup_failed = False
        for name, identity in owned.items():
            try:
                if same_file(os.stat(name, dir_fd=descriptor, follow_symlinks=False), identity):
                    os.unlink(name, dir_fd=descriptor)
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_failed = True
        try:
            os.fsync(descriptor)
        except OSError:
            cleanup_failed = True
        if cleanup_failed:
            raise AudioClippingError(
                AudioFailureCode.OUTPUT_FAILED, "filesystem admission cleanup failed"
            )


def verify_wav(
    descriptor: int, geometry: AudioClipGeometry, deadline: datetime
) -> dict[str, object]:
    """Parse actual RIFF bytes, including extensible PCM, before publication."""
    size = os.fstat(descriptor).st_size

    def invalid() -> AudioClippingError:
        return AudioClippingError(AudioFailureCode.OUTPUT_CORRUPTED, "invalid PCM16 WAV output")

    def read(offset: int, count: int) -> bytes:
        remaining_seconds(deadline)
        payload = os.pread(descriptor, count, offset)
        if len(payload) != count:
            raise invalid()
        return payload

    header = read(0, 12)
    if (
        header[:4] != b"RIFF"
        or header[8:] != b"WAVE"
        or int.from_bytes(header[4:8], "little") + 8 != size
    ):
        raise invalid()
    position = 12
    fmt: bytes | None = None
    data_bytes: int | None = None
    chunks = 0
    while position < size:
        chunks += 1
        if chunks > 1024:
            raise invalid()
        chunk = read(position, 8)
        length = int.from_bytes(chunk[4:], "little")
        payload_start = position + 8
        position = payload_start + length + (length % 2)
        if position > size:
            raise invalid()
        if chunk[:4] == b"fmt ":
            if fmt is not None or length not in (16, 18, 40):
                raise invalid()
            fmt = read(payload_start, length)
        elif chunk[:4] == b"data":
            if data_bytes is not None:
                raise invalid()
            data_bytes = length
    if fmt is None or data_bytes is None or not 0 < data_bytes <= MAX_PCM_BYTES:
        raise invalid()
    tag, channels, rate, byte_rate, alignment, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == 1:
        if len(fmt) not in (16, 18) or (len(fmt) == 18 and fmt[16:] != b"\0\0"):
            raise invalid()
    elif tag == 0xFFFE:
        if len(fmt) != 40 or fmt[16:20] != b"\x16\0\x10\0" or fmt[24:] != _PCM_GUID:
            raise invalid()
    else:
        raise invalid()
    if (rate, channels, bits, alignment, byte_rate) != (
        geometry.sample_rate_hz,
        geometry.channels,
        16,
        geometry.channels * 2,
        geometry.sample_rate_hz * geometry.channels * 2,
    ) or data_bytes % alignment:
        raise invalid()
    frames = data_bytes // alignment
    tolerance = int(geometry.sample_rate_hz != geometry.source_rate_hz)
    if abs(frames - geometry.expected_frames) > tolerance:
        raise invalid()
    return {
        "duration_seconds": frames / rate,
        "sample_rate_hz": rate,
        "channels": channels,
        "codec": "pcm_s16le",
        "container": "wav",
        "bit_depth": 16,
        "bit_rate_bps": byte_rate * 8,
        "audio_stream_index": 0,
        "container_stream_index": 0,
        "frame_count": frames,
    }


class ClipOutput:
    """Own one private output and acknowledge only a verified final publication."""

    def __init__(self, destination: Path, parent: int, geometry: AudioClipGeometry) -> None:
        self.destination = destination
        self.parent = parent
        self.name = f".capicove-{secrets.token_hex(16)}.partial"
        self.published = False
        self.publish_attempted = False
        self.acknowledged = False
        try:
            os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise AudioClippingError(AudioFailureCode.OUTPUT_EXISTS, "destination already exists")
        free = os.fstatvfs(parent)
        if (
            free.f_bavail * free.f_frsize
            < geometry.expected_frames * geometry.channels * 2 + DISK_RESERVE_BYTES
        ):
            raise AudioClippingError(AudioFailureCode.OUTPUT_FAILED, "insufficient output space")
        self.descriptor = os.open(
            self.name, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent
        )
        self.identity = os.fstat(self.descriptor)

    def _owned(self, name: str) -> bool:
        try:
            return same_file(
                os.stat(name, dir_fd=self.parent, follow_symlinks=False), self.identity
            )
        except FileNotFoundError:
            return False

    def publish(self) -> None:
        """Create the final link atomically; never replace or adopt another file."""
        require_binding(self.destination.parent, self.parent)
        if not self._owned(self.name) or os.fstat(self.descriptor).st_nlink != 1:
            raise AudioClippingError(
                AudioFailureCode.OUTPUT_CORRUPTED, "partial output identity changed"
            )
        os.fsync(self.descriptor)
        try:
            self.publish_attempted = True
            os.link(
                self.name,
                self.destination.name,
                src_dir_fd=self.parent,
                dst_dir_fd=self.parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise AudioClippingError(
                AudioFailureCode.OUTPUT_EXISTS, "destination already exists"
            ) from None
        self.published = self._owned(self.destination.name)
        if not self.published or os.fstat(self.descriptor).st_nlink != 2:
            raise AudioClippingError(
                AudioFailureCode.OUTPUT_CORRUPTED, "published output identity changed"
            )
        os.unlink(self.name, dir_fd=self.parent)
        os.fsync(self.parent)
        if not self._owned(self.destination.name) or os.fstat(self.descriptor).st_nlink != 1:
            raise AudioClippingError(
                AudioFailureCode.OUTPUT_CORRUPTED, "final output identity changed"
            )
        require_binding(self.destination.parent, self.parent)

    def close(self) -> None:
        """Remove only identity-owned unacknowledged files and close the descriptor."""
        try:
            names = [self.name]
            if self.publish_attempted and not self.acknowledged:
                names.append(self.destination.name)
            for name in names:
                if self._owned(name):
                    os.unlink(name, dir_fd=self.parent)
            os.fsync(self.parent)
        finally:
            os.close(self.descriptor)
