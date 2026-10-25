from pathlib import Path

import pytest

from capicove.processing.video.deduplication import (
    VideoDeduplicationMode,
    VideoDeduplicationResult,
    VideoDeduplicationStats,
    VideoDuplicateGroup,
    VideoFileRecord,
)


def _group(tmp_path: Path) -> VideoDuplicateGroup:
    return VideoDuplicateGroup(
        group_id=f"sha256:{'a' * 64}",
        keeper=(tmp_path / "a.mp4").resolve(),
        eliminated=((tmp_path / "b.mp4").resolve(),),
        size_bytes=5,
        xxh3_128="b" * 32,
        sha256="a" * 64,
    )


def test_duplicate_group_exposes_keeper_before_eliminated(tmp_path: Path) -> None:
    group = _group(tmp_path)

    assert group.all_paths == (
        (tmp_path / "a.mp4").resolve(),
        (tmp_path / "b.mp4").resolve(),
    )


def test_duplicate_group_requires_lexicographic_keeper(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lexicographically"):
        VideoDuplicateGroup(
            group_id=f"sha256:{'a' * 64}",
            keeper=(tmp_path / "z.mp4").resolve(),
            eliminated=((tmp_path / "a.mp4").resolve(),),
            size_bytes=5,
            xxh3_128="b" * 32,
            sha256="a" * 64,
        )


def test_result_rejects_eliminated_path_as_unique(tmp_path: Path) -> None:
    group = _group(tmp_path)
    files = tuple(VideoFileRecord(path, 5) for path in group.all_paths)

    with pytest.raises(ValueError, match="unique_files"):
        VideoDeduplicationResult(
            mode=VideoDeduplicationMode.FILE,
            files=files,
            unique_files=group.all_paths,
            duplicate_groups=(group,),
            failures=(),
            stats=VideoDeduplicationStats(2, 2, 0, 1, 1),
        )


def test_result_accepts_consistent_sorted_membership(tmp_path: Path) -> None:
    group = _group(tmp_path)
    files = tuple(VideoFileRecord(path, 5) for path in group.all_paths)

    result = VideoDeduplicationResult(
        mode=VideoDeduplicationMode.FILE,
        files=files,
        unique_files=(group.keeper,),
        duplicate_groups=(group,),
        failures=(),
        stats=VideoDeduplicationStats(2, 2, 0, 1, 1),
    )

    assert result.unique_files == (group.keeper,)
