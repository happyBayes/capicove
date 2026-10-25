from pathlib import Path

import pytest

from capicove.processing.video.deduplication import (
    VideoDeduplicationResult,
    deduplicate_video_files,
    delete_duplicate_files,
    move_duplicate_files,
)


def _result(tmp_path: Path) -> VideoDeduplicationResult:
    keeper = tmp_path / "a.mp4"
    duplicate = tmp_path / "b.mp4"
    keeper.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    return deduplicate_video_files((keeper, duplicate))


def test_move_preserves_keeper_and_records_destination(tmp_path: Path) -> None:
    result = _result(tmp_path)

    action = move_duplicate_files(result, tmp_path / "duplicates")

    assert result.duplicate_groups[0].keeper.exists()
    assert not result.duplicate_groups[0].eliminated[0].exists()
    assert action.moves[0].destination.exists()
    assert action.failures == ()


def test_move_does_not_overwrite_existing_destination(tmp_path: Path) -> None:
    result = _result(tmp_path)
    group = result.duplicate_groups[0]
    destination = tmp_path / "duplicates" / group.sha256[:2] / "b.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"existing")

    action = move_duplicate_files(result, tmp_path / "duplicates")

    assert group.eliminated[0].exists()
    assert destination.read_bytes() == b"existing"
    assert action.failures[0].code == "destination_exists"


def test_delete_requires_force(tmp_path: Path) -> None:
    result = _result(tmp_path)

    with pytest.raises(ValueError, match="force"):
        delete_duplicate_files(result)

    assert result.duplicate_groups[0].eliminated[0].exists()


def test_forced_delete_preserves_keeper(tmp_path: Path) -> None:
    result = _result(tmp_path)

    action = delete_duplicate_files(result, force=True)

    assert result.duplicate_groups[0].keeper.exists()
    assert action.deleted == result.duplicate_groups[0].eliminated
