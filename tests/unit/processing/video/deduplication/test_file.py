from pathlib import Path

from pytest import MonkeyPatch

from capicove.processing.video.deduplication import deduplicate_video_files
from capicove.processing.video.deduplication import file as file_module


def test_identical_files_use_lexicographic_keeper(tmp_path: Path) -> None:
    later = tmp_path / "z.mp4"
    keeper = tmp_path / "a.mp4"
    later.write_bytes(b"same-video")
    keeper.write_bytes(b"same-video")

    result = deduplicate_video_files((later, keeper))

    assert result.unique_files == (keeper.resolve(),)
    assert result.duplicate_groups[0].keeper == keeper.resolve()
    assert result.duplicate_groups[0].eliminated == (later.resolve(),)
    assert result.stats.duplicate_files == 1


def test_sha256_separates_simulated_xxhash_collision(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    first = tmp_path / "a.mp4"
    second = tmp_path / "b.mp4"
    first.write_bytes(b"abc")
    second.write_bytes(b"xyz")
    monkeypatch.setattr(file_module, "_xxh3_128_file", lambda *_: "0" * 32)

    result = deduplicate_video_files((first, second))

    assert result.unique_files == (first.resolve(), second.resolve())
    assert result.duplicate_groups == ()


def test_recurses_directories_and_ignores_nonvideo_and_symlinks(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    video = nested / "clip.MP4"
    video.write_bytes(b"video")
    (nested / "notes.txt").write_text("not video", encoding="utf-8")
    (nested / "link.mp4").symlink_to(video)

    result = deduplicate_video_files((tmp_path,))

    assert result.unique_files == (video.resolve(),)
    assert result.stats.discovered_files == 1


def test_missing_explicit_input_is_reported(tmp_path: Path) -> None:
    missing = (tmp_path / "missing.mp4").resolve()

    result = deduplicate_video_files((missing,))

    assert result.unique_files == ()
    assert result.failures[0].path == missing
    assert result.failures[0].code == "missing"


def test_repeated_input_path_is_scanned_once(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")

    result = deduplicate_video_files((video, video))

    assert result.stats.discovered_files == 1
    assert result.unique_files == (video.resolve(),)
