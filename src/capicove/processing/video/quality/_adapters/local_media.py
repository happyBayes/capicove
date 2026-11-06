"""Bounded no-follow intake for self-contained local media."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from capicove.processing.video.quality.video_quality_contracts import QualityError, QualityLimits
from capicove.processing.video.quality.video_quality_ports import QualityIntake, QualitySource


def check_deadline(deadline: datetime) -> None:
    """Apply the shared invocation deadline during copy, hashing and analysis."""
    if datetime.now(UTC) >= deadline:
        raise QualityError("timeout", "Video quality inspection exceeded its deadline.")


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _open_regular(path: Path) -> int:
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def _check_mov_references(stream: BinaryIO, size: int, deadline: datetime) -> None:
    """Reject external MOV data references before a media provider sees the file."""
    containers = {b"moov", b"trak", b"mdia", b"minf", b"dinf"}
    count = 0

    def boxes(start: int, end: int, depth: int) -> None:
        nonlocal count
        if depth > 12:
            raise QualityError("invalid_input", "Media container nesting exceeds its limit.")
        position = start
        while position < end:
            check_deadline(deadline)
            count += 1
            if count > 100000:
                raise QualityError("invalid_input", "Media container box count exceeds its limit.")
            stream.seek(position)
            header = stream.read(8)
            if len(header) != 8:
                raise QualityError("invalid_input", "Media container header is truncated.")
            length, kind = int.from_bytes(header[:4], "big"), header[4:]
            header_size = 8
            if length == 1:
                large = stream.read(8)
                if len(large) != 8:
                    raise QualityError("invalid_input", "Media container size is truncated.")
                length, header_size = int.from_bytes(large, "big"), 16
            elif length == 0:
                length = end - position
            if length < header_size or position + length > end:
                raise QualityError("invalid_input", "Media container box has an invalid size.")
            body = position + header_size
            if kind in {b"rmra", b"rdrf"}:
                raise QualityError("invalid_input", "External movie references are unsupported.")
            if kind in containers:
                boxes(body, position + length, depth + 1)
            elif kind == b"dref":
                stream.seek(body)
                prefix = stream.read(8)
                entries = int.from_bytes(prefix[4:], "big")
                if len(prefix) != 8 or entries > 1024:
                    raise QualityError("invalid_input", "Invalid media data reference table.")
                entry_pos = body + 8
                for _ in range(entries):
                    stream.seek(entry_pos)
                    entry = stream.read(12)
                    if len(entry) != 12:
                        raise QualityError("invalid_input", "Truncated media data reference.")
                    entry_len = int.from_bytes(entry[:4], "big")
                    if (
                        entry_len != 12
                        or entry[4:8] != b"url "
                        or int.from_bytes(entry[8:12], "big") != 1
                    ):
                        raise QualityError(
                            "invalid_input", "External media data references are unsupported."
                        )
                    entry_pos += entry_len
                if entry_pos != position + length:
                    raise QualityError("invalid_input", "Invalid media data reference boundary.")
            position += length

    boxes(0, size, 0)


def _validated_demuxer(path: Path, size: int, deadline: datetime) -> str:
    check_deadline(deadline)
    with path.open("rb") as stream:
        header = stream.read(16)
        if header[:4] == b"\x1aE\xdf\xa3":
            return "matroska"
        if header[4:8] in {b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip"}:
            _check_mov_references(stream, size, deadline)
            return "mov"
    raise QualityError(
        "invalid_input", "Only self-contained MP4/MOV and MKV/WebM files are supported."
    )


class VerifiedQualityIntake:
    """Validate an authority-owned snapshot without another unmanaged media copy.

    Linux consumers receive a file-descriptor path held through all passes. The
    workspace host independently requires Linux descriptor/executable support.
    Other POSIX compositions retain the trusted port's locator and hash guards;
    they do not gain workspace-host support through this adapter.
    """

    def __init__(self, intake: QualityIntake, limits: QualityLimits) -> None:
        self._intake, self._limits = intake, limits

    def _verify(self, descriptor: int, source: QualitySource, deadline: datetime) -> os.stat_result:
        check_deadline(deadline)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != source.size_bytes:
            raise QualityError("source_changed", "The verified source identity or size changed.")
        if before.st_size > self._limits.max_input_bytes:
            raise QualityError("resource_limit", "The verified source exceeds its byte budget.")
        digest, offset = hashlib.sha256(), 0
        while block := os.pread(descriptor, 1024 * 1024, offset):
            check_deadline(deadline)
            digest.update(block)
            offset += len(block)
            if offset > self._limits.max_input_bytes:
                raise QualityError("resource_limit", "The verified source exceeds its byte budget.")
        if (
            _identity(before) != _identity(os.fstat(descriptor))
            or digest.hexdigest() != source.sha256
        ):
            raise QualityError("source_changed", "The verified source content changed.")
        return before

    @contextmanager
    def stage(self, path: Path, deadline: datetime) -> Iterator[QualitySource]:
        with self._intake.stage(path, deadline) as source:
            if source.original_path != path:
                raise QualityError(
                    "permission_denied",
                    "The verified source does not match the authorized request.",
                )
            try:
                descriptor = os.open(
                    source.staged_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                )
            except OSError as error:
                raise QualityError(
                    "source_changed", "The verified source is unavailable."
                ) from error
            try:
                before = self._verify(descriptor, source, deadline)
                bound = Path(f"/proc/{os.getpid()}/fd/{descriptor}")
                locator = bound if bound.exists() else source.staged_path
                accepted = replace(
                    source,
                    staged_path=locator,
                    demuxer=_validated_demuxer(locator, source.size_bytes, deadline),
                )
                try:
                    yield accepted
                finally:
                    after = self._verify(descriptor, source, deadline)
                    if _identity(before) != _identity(after):
                        raise QualityError(
                            "source_changed", "The verified source changed during inspection."
                        )
            finally:
                os.close(descriptor)


class LocalQualityIntake:
    """Stage one bounded authorized file through no-follow descriptors."""

    def __init__(self, read_roots: tuple[Path, ...], limits: QualityLimits) -> None:
        self._roots = tuple(root.resolve() for root in read_roots)
        self._limits = limits

    @contextmanager
    def stage(self, path: Path, deadline: datetime) -> Iterator[QualitySource]:
        """Yield a verified private copy, cleaning it on every exit."""
        if ".." in path.parts or not any(path.is_relative_to(root) for root in self._roots):
            raise QualityError("permission_denied", "Video path is outside authorized read roots.")
        check_deadline(deadline)
        try:
            descriptor = _open_regular(path)
        except OSError as error:
            raise QualityError(
                "invalid_input", "Video source must be a readable no-follow regular file."
            ) from error
        with (
            os.fdopen(descriptor, "rb") as original,
            tempfile.TemporaryDirectory(prefix="capicove-quality-") as scratch,
        ):
            before = os.fstat(original.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > self._limits.max_input_bytes:
                raise QualityError(
                    "invalid_input", "Video source type or size exceeds its allowed scope."
                )
            staged = Path(scratch) / "source.media"
            digest = hashlib.sha256()
            total = 0
            with staged.open("xb") as output:
                while block := original.read(1024 * 1024):
                    check_deadline(deadline)
                    total += len(block)
                    if total > self._limits.max_input_bytes:
                        raise QualityError(
                            "invalid_input", "Video source exceeds the input byte limit."
                        )
                    digest.update(block)
                    output.write(block)
            if (
                _identity(before) != _identity(os.fstat(original.fileno()))
                or total != before.st_size
            ):
                raise QualityError("source_changed", "Video source changed during verified intake.")
            staged.chmod(0o400)
            demuxer = _validated_demuxer(staged, total, deadline)
            try:
                yield QualitySource(staged, path, digest.hexdigest(), total, demuxer)
            finally:
                check_deadline(deadline)
                try:
                    with staged.open("rb") as stream:
                        observed = hashlib.sha256()
                        while block := stream.read(1024 * 1024):
                            check_deadline(deadline)
                            observed.update(block)
                    if observed.hexdigest() != digest.hexdigest():
                        raise QualityError(
                            "source_changed", "The verified inspection copy changed."
                        )
                except OSError as error:
                    raise QualityError(
                        "source_changed", "The verified inspection copy is unavailable."
                    ) from error
