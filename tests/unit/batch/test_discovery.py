from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
from threading import Event, Thread
from typing import BinaryIO, Literal

import pytest

from capicove.batch.discovery import (
    DirectoryDiscoverer,
    DirectoryDiscoveryError,
    SourceChangedDuringDiscoveryError,
    _observe_regular_file,
)
from capicove.batch.models import BatchModality


def _discoverer() -> DirectoryDiscoverer:
    return DirectoryDiscoverer(
        {
            ".wav": "audio/wav",
            "mp3": "audio/mpeg",
        }
    )


def test_discovery_recurses_in_relative_path_order_and_hashes_supported_files(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "calls"
    nested.mkdir()
    (tmp_path / "z.WAV").write_bytes(b"z")
    (nested / "b.txt").write_text("ignored", encoding="utf-8")
    (nested / "a.mp3").write_bytes(b"a")

    items = _discoverer().discover(tmp_path, BatchModality.AUDIO)

    assert [item.relative_path for item in items] == [
        PurePosixPath("calls/a.mp3"),
        PurePosixPath("z.WAV"),
    ]
    assert [item.media_type for item in items] == ["audio/mpeg", "audio/wav"]
    assert [item.sha256 for item in items] == [
        hashlib.sha256(b"a").hexdigest(),
        hashlib.sha256(b"z").hexdigest(),
    ]
    assert all(item.source_path.is_absolute() for item in items)
    assert [item.size_bytes for item in items] == [1, 1]


def test_discovery_item_identity_tracks_relative_path_not_mutable_content(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"one")
    first = _discoverer().discover(tmp_path, BatchModality.AUDIO)[0]

    source.write_bytes(b"two")
    second = _discoverer().discover(tmp_path, BatchModality.AUDIO)[0]
    duplicate = tmp_path / "copy.wav"
    duplicate.write_bytes(b"two")
    items = _discoverer().discover(tmp_path, BatchModality.AUDIO)
    copy = next(item for item in items if item.relative_path == PurePosixPath("copy.wav"))

    assert first.item_id == second.item_id
    assert first.sha256 != second.sha256
    assert copy.sha256 == second.sha256
    assert copy.item_id != second.item_id


def test_discovery_does_not_follow_file_or_directory_symlinks(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "private.wav").write_bytes(b"private")
    try:
        (tmp_path / "linked.wav").symlink_to(outside / "private.wav")
        (tmp_path / "linked-directory").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")
    (tmp_path / "visible.wav").write_bytes(b"visible")

    items = _discoverer().discover(tmp_path, BatchModality.AUDIO)

    assert [item.relative_path for item in items] == [PurePosixPath("visible.wav")]


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory replacement race")
def test_observe_rejects_parent_swap_before_open_without_reading_foreign_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "inputs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    source = nested / "meeting.wav"
    source.write_bytes(b"authorized recording")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "meeting.wav").write_bytes(b"foreign private recording")
    moved = root / "moved-nested"
    original_lstat = Path.lstat
    original_open = Path.open
    swapped = False
    foreign_reads: list[Path] = []

    def swap_after_parent_stat(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = original_lstat(path)
        if path == nested and not swapped:
            swapped = True
            nested.rename(moved)
            nested.symlink_to(foreign, target_is_directory=True)
        return observed

    def record_foreign_open(path: Path, mode: Literal["rb"] = "rb") -> BinaryIO:
        if path == source and nested.is_symlink():
            foreign_reads.append(path)
        return original_open(path, mode)

    monkeypatch.setattr(Path, "lstat", swap_after_parent_stat)
    monkeypatch.setattr(Path, "open", record_foreign_open)

    with pytest.raises(SourceChangedDuringDiscoveryError):
        _observe_regular_file(source)

    assert swapped
    assert foreign_reads == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO replacement race")
def test_observe_rejects_fifo_swap_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "meeting.wav"
    source.write_bytes(b"authorized recording")
    original_lstat = Path.lstat
    swapped = False
    finished = Event()
    outcomes: list[object] = []

    def swap_after_file_stat(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = original_lstat(path)
        if path == source and not swapped:
            swapped = True
            source.unlink()
            os.mkfifo(source)
        return observed

    def observe() -> None:
        try:
            outcomes.append(_observe_regular_file(source))
        except Exception as error:
            outcomes.append(error)
        finally:
            finished.set()

    monkeypatch.setattr(Path, "lstat", swap_after_file_stat)
    thread = Thread(target=observe, daemon=True)
    thread.start()
    completed_without_writer = finished.wait(timeout=1)
    if not completed_without_writer:
        release = os.open(source, os.O_RDWR | os.O_NONBLOCK)
        os.close(release)
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert swapped
    assert completed_without_writer
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], SourceChangedDuringDiscoveryError)


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_discovery_rejects_a_root_that_is_not_a_directory(
    tmp_path: Path,
    root_kind: str,
) -> None:
    root = tmp_path / "input"
    if root_kind == "file":
        root.write_bytes(b"not a directory")

    with pytest.raises(DirectoryDiscoveryError, match="directory"):
        _discoverer().discover(root, BatchModality.AUDIO)


def test_discovery_rejects_conflicting_or_empty_media_type_rules() -> None:
    with pytest.raises(ValueError, match="media type rule"):
        DirectoryDiscoverer({})
    with pytest.raises(ValueError, match="media_type"):
        DirectoryDiscoverer({".wav": " "})
    with pytest.raises(ValueError, match="conflicting"):
        DirectoryDiscoverer({".WAV": "audio/wav", "wav": "application/octet-stream"})
