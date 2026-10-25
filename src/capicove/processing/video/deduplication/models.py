"""Immutable values for byte-exact video deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


_SHA256 = re.compile(r"[0-9a-f]{64}")
_XXH3_128 = re.compile(r"[0-9a-f]{32}")


def _require_sorted_absolute(name: str, paths: tuple[Path, ...]) -> None:
    if any(not path.is_absolute() for path in paths):
        raise ValueError(f"{name} must contain absolute paths")
    if paths != tuple(sorted(paths, key=str)) or len(paths) != len(set(paths)):
        raise ValueError(f"{name} must contain unique lexicographically sorted paths")


class VideoDeduplicationMode(StrEnum):
    """Select the supported deduplication meaning."""

    FILE = "file"


@dataclass(frozen=True, slots=True)
class VideoFileRecord:
    """Record one readable file observed by the scanner."""

    path: Path
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("path must be absolute")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class VideoDuplicateGroup:
    """Describe one SHA-256-confirmed byte-identical group."""

    group_id: str
    keeper: Path
    eliminated: tuple[Path, ...]
    size_bytes: int
    xxh3_128: str
    sha256: str

    def __post_init__(self) -> None:
        if self.group_id != f"sha256:{self.sha256}":
            raise ValueError("group_id must identify the confirmed SHA-256")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if _XXH3_128.fullmatch(self.xxh3_128) is None:
            raise ValueError("xxh3_128 must be a lowercase XXH3-128 digest")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not self.eliminated:
            raise ValueError("eliminated must not be empty")
        _require_sorted_absolute("duplicate paths", self.all_paths)
        if self.keeper != self.all_paths[0]:
            raise ValueError("keeper must be the lexicographically first path")

    @property
    def all_paths(self) -> tuple[Path, ...]:
        """Return the keeper followed by eliminated files."""
        return (self.keeper, *self.eliminated)


@dataclass(frozen=True, slots=True)
class VideoDeduplicationFailure:
    """Report one file that could not be scanned or acted upon."""

    path: Path
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class VideoDeduplicationStats:
    """Summarize a scan without hiding failures."""

    discovered_files: int
    readable_files: int
    failed_files: int
    unique_files: int
    duplicate_files: int

    def __post_init__(self) -> None:
        if (
            min(
                self.discovered_files,
                self.readable_files,
                self.failed_files,
                self.unique_files,
                self.duplicate_files,
            )
            < 0
        ):
            raise ValueError("statistics must be non-negative")
        if self.discovered_files != self.readable_files + self.failed_files:
            raise ValueError("discovered_files must equal readable plus failed files")
        if self.readable_files != self.unique_files + self.duplicate_files:
            raise ValueError("readable_files must equal unique plus duplicate files")


@dataclass(frozen=True, slots=True)
class VideoDeduplicationResult:
    """Return deterministic exact-file deduplication decisions."""

    mode: VideoDeduplicationMode
    files: tuple[VideoFileRecord, ...]
    unique_files: tuple[Path, ...]
    duplicate_groups: tuple[VideoDuplicateGroup, ...]
    failures: tuple[VideoDeduplicationFailure, ...]
    stats: VideoDeduplicationStats

    def __post_init__(self) -> None:
        file_paths = tuple(record.path for record in self.files)
        _require_sorted_absolute("files", file_paths)
        _require_sorted_absolute("unique_files", self.unique_files)
        eliminated = {path for group in self.duplicate_groups for path in group.eliminated}
        if eliminated.intersection(self.unique_files):
            raise ValueError("unique_files cannot contain eliminated paths")
        expected_unique = tuple(path for path in file_paths if path not in eliminated)
        if self.unique_files != expected_unique:
            raise ValueError("unique_files must contain every keeper and nonduplicate")
        if self.stats.readable_files != len(self.files):
            raise ValueError("readable_files does not match files")
        if self.stats.failed_files != len(self.failures):
            raise ValueError("failed_files does not match failures")
        if self.stats.unique_files != len(self.unique_files):
            raise ValueError("unique_files statistic does not match result")
        if self.stats.duplicate_files != len(eliminated):
            raise ValueError("duplicate_files statistic does not match result")


@dataclass(frozen=True, slots=True)
class ManifestItem:
    """Describe one upload decision."""

    path: Path
    upload_mode: str
    duplicate_group_id: str | None
    keeper_path: Path | None
    size_bytes: int
    moved_to: Path | None = None


@dataclass(frozen=True, slots=True)
class DuplicateMoveRecord:
    """Audit one successful duplicate move."""

    source: Path
    destination: Path


@dataclass(frozen=True, slots=True)
class DuplicateActionResult:
    """Collect successful and failed filesystem actions."""

    moves: tuple[DuplicateMoveRecord, ...] = ()
    deleted: tuple[Path, ...] = ()
    failures: tuple[VideoDeduplicationFailure, ...] = ()
