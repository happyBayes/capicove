"""Explicit filesystem actions for confirmed duplicate video files."""

from __future__ import annotations

import stat
from pathlib import Path

from capicove.processing.video.deduplication.file import _sha256_file
from capicove.processing.video.deduplication.models import (
    DuplicateActionResult,
    DuplicateMoveRecord,
    VideoDeduplicationFailure,
    VideoDeduplicationResult,
    VideoDuplicateGroup,
)


def _confirm_source(path: Path, group: VideoDuplicateGroup) -> None:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("duplicate source is not a regular file")
    if metadata.st_size != group.size_bytes:
        raise OSError("duplicate source size changed after scanning")
    if _sha256_file(path, 4 * 1024 * 1024) != group.sha256:
        raise OSError("duplicate source content changed after scanning")


def move_duplicate_files(
    result: VideoDeduplicationResult, destination: Path
) -> DuplicateActionResult:
    """Move confirmed eliminated duplicates without overwriting files."""
    moves: list[DuplicateMoveRecord] = []
    failures: list[VideoDeduplicationFailure] = []
    destination = destination.absolute()
    for group in result.duplicate_groups:
        for source in group.eliminated:
            target = destination / group.sha256[:2] / source.name
            try:
                _confirm_source(source, group)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    failures.append(
                        VideoDeduplicationFailure(
                            source, "destination_exists", f"move destination exists: {target}"
                        )
                    )
                    continue
                source.rename(target)
                moves.append(DuplicateMoveRecord(source, target))
            except OSError as exc:
                failures.append(VideoDeduplicationFailure(source, "move_failed", str(exc)))
    return DuplicateActionResult(tuple(moves), (), tuple(failures))


def delete_duplicate_files(
    result: VideoDeduplicationResult, *, force: bool = False
) -> DuplicateActionResult:
    """Delete confirmed eliminated duplicates only with explicit force."""
    if not force:
        raise ValueError("deleting duplicates requires force=True")
    deleted: list[Path] = []
    failures: list[VideoDeduplicationFailure] = []
    for group in result.duplicate_groups:
        for source in group.eliminated:
            try:
                _confirm_source(source, group)
                source.unlink()
                deleted.append(source)
            except OSError as exc:
                failures.append(VideoDeduplicationFailure(source, "delete_failed", str(exc)))
    return DuplicateActionResult((), tuple(deleted), tuple(failures))
