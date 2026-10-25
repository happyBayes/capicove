"""Streaming byte-exact video file deduplication."""

from __future__ import annotations

import hashlib
import stat
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Protocol

from capicove.processing.video.deduplication.models import (
    VideoDeduplicationFailure,
    VideoDeduplicationMode,
    VideoDeduplicationResult,
    VideoDeduplicationStats,
    VideoDuplicateGroup,
    VideoFileRecord,
)


DEFAULT_VIDEO_EXTENSIONS = frozenset(
    {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
)


class HashLike(Protocol):
    """Describe the streaming hash operations used by this module."""

    def update(self, payload: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


class VideoDeduplicationDependencyError(RuntimeError):
    """Report a missing optional hashing dependency."""


class VideoFileChangedError(OSError):
    """Report a source that changed during hashing."""


def _new_xxh3_128() -> HashLike:
    try:
        import xxhash
    except ImportError as exc:
        raise VideoDeduplicationDependencyError(
            "video deduplication requires capicove[video-dedup]"
        ) from exc
    return xxhash.xxh3_128()


def _file_identity(value: object) -> tuple[int, int, int, int]:
    return (
        value.st_dev,  # type: ignore[attr-defined]
        value.st_ino,  # type: ignore[attr-defined]
        value.st_size,  # type: ignore[attr-defined]
        value.st_mtime_ns,  # type: ignore[attr-defined]
    )


def _digest_stable_file(
    path: Path,
    factory: Callable[[], HashLike],
    chunk_size: int,
) -> str:
    before = path.stat(follow_symlinks=False)
    digest = factory()
    with path.open("rb") as stream:
        while payload := stream.read(chunk_size):
            digest.update(payload)
    after = path.stat(follow_symlinks=False)
    if _file_identity(before) != _file_identity(after):
        raise VideoFileChangedError(f"video changed while hashing: {path}")
    return digest.hexdigest()


def _xxh3_128_file(path: Path, chunk_size: int) -> str:
    return _digest_stable_file(path, _new_xxh3_128, chunk_size)


def _sha256_file(path: Path, chunk_size: int) -> str:
    return _digest_stable_file(path, hashlib.sha256, chunk_size)


def _discover_paths(
    inputs: Sequence[Path], extensions: Collection[str]
) -> tuple[tuple[Path, ...], tuple[VideoDeduplicationFailure, ...]]:
    normalized_extensions = {extension.lower() for extension in extensions}
    paths: set[Path] = set()
    failures: list[VideoDeduplicationFailure] = []
    for supplied in inputs:
        path = supplied.absolute()
        if path.is_symlink():
            failures.append(VideoDeduplicationFailure(path, "symlink", "symlinks are not followed"))
            continue
        if path.is_file():
            paths.add(path)
            continue
        if path.is_dir():
            for candidate in path.rglob("*"):
                if (
                    candidate.suffix.lower() in normalized_extensions
                    and not candidate.is_symlink()
                    and candidate.is_file()
                ):
                    paths.add(candidate.absolute())
            continue
        failures.append(VideoDeduplicationFailure(path, "missing", "input does not exist"))
    return tuple(sorted(paths, key=str)), tuple(sorted(failures, key=lambda item: str(item.path)))


def deduplicate_video_files(
    inputs: Sequence[Path],
    *,
    extensions: Collection[str] = DEFAULT_VIDEO_EXTENSIONS,
    chunk_size: int = 4 * 1024 * 1024,
) -> VideoDeduplicationResult:
    """Find byte-identical videos without modifying source files."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    paths, discovery_failures = _discover_paths(inputs, extensions)
    failures = list(discovery_failures)
    records: list[VideoFileRecord] = []
    size_buckets: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        try:
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("input is not a regular file")
        except OSError as exc:
            failures.append(VideoDeduplicationFailure(path, "unreadable", str(exc)))
            continue
        records.append(VideoFileRecord(path, metadata.st_size))
        size_buckets[metadata.st_size].append(path)

    groups: list[VideoDuplicateGroup] = []
    hash_failures: set[Path] = set()
    for size_bytes, same_size in size_buckets.items():
        if len(same_size) < 2:
            continue
        xxhash_buckets: dict[str, list[Path]] = defaultdict(list)
        for path in same_size:
            try:
                xxhash_buckets[_xxh3_128_file(path, chunk_size)].append(path)
            except (OSError, VideoDeduplicationDependencyError) as exc:
                if isinstance(exc, VideoDeduplicationDependencyError):
                    raise
                hash_failures.add(path)
                failures.append(VideoDeduplicationFailure(path, "hash_failed", str(exc)))
        for xxh3_128, same_fast_hash in xxhash_buckets.items():
            if len(same_fast_hash) < 2:
                continue
            sha_buckets: dict[str, list[Path]] = defaultdict(list)
            for path in same_fast_hash:
                try:
                    sha_buckets[_sha256_file(path, chunk_size)].append(path)
                except OSError as exc:
                    hash_failures.add(path)
                    failures.append(VideoDeduplicationFailure(path, "hash_failed", str(exc)))
            for sha256, duplicates in sha_buckets.items():
                if len(duplicates) < 2:
                    continue
                ordered = tuple(sorted(duplicates, key=str))
                groups.append(
                    VideoDuplicateGroup(
                        group_id=f"sha256:{sha256}",
                        keeper=ordered[0],
                        eliminated=ordered[1:],
                        size_bytes=size_bytes,
                        xxh3_128=xxh3_128,
                        sha256=sha256,
                    )
                )

    records = [record for record in records if record.path not in hash_failures]
    groups.sort(key=lambda group: str(group.keeper))
    eliminated = {path for group in groups for path in group.eliminated}
    unique_files = tuple(record.path for record in records if record.path not in eliminated)
    ordered_failures = tuple(sorted(failures, key=lambda item: str(item.path)))
    return VideoDeduplicationResult(
        mode=VideoDeduplicationMode.FILE,
        files=tuple(records),
        unique_files=unique_files,
        duplicate_groups=tuple(groups),
        failures=ordered_failures,
        stats=VideoDeduplicationStats(
            discovered_files=len(paths) + len(discovery_failures),
            readable_files=len(records),
            failed_files=len(ordered_failures),
            unique_files=len(unique_files),
            duplicate_files=len(eliminated),
        ),
    )
