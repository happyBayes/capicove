"""Tests for safe content-addressed export media paths and resolution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

import pytest

from capicove.export.media import (
    TraceMediaCopyError,
    WorkspaceExportMediaResolver,
    copy_media_reference,
)
from capicove.traces import (
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    TraceModality,
)


def _reference(payload: bytes, media_type: str) -> MediaReference:
    modality = (
        TraceModality.AUDIO
        if media_type.startswith("audio/")
        else TraceModality.VIDEO
        if media_type.startswith("video/")
        else TraceModality.IMAGE
    )
    return MediaReference(
        media_id=f"media-{media_type}",
        modality=modality,
        media_type=media_type,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


@pytest.mark.parametrize(
    ("media_type", "extension"),
    [
        ("audio/wav", ".wav"),
        ("video/mp4", ".mp4"),
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("application/octet-stream", ".bin"),
    ],
)
def test_copy_uses_fixed_content_addressed_extension_and_deduplicates(
    tmp_path: Path,
    media_type: str,
    extension: str,
) -> None:
    payload = b"verified media bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    export = tmp_path / "export"
    export.mkdir()
    reference = _reference(payload, media_type)

    first = copy_media_reference(reference, source, export)
    second = copy_media_reference(reference, source, export)

    expected = PurePosixPath(f"media/sha256/{reference.sha256[:2]}/{reference.sha256}{extension}")
    assert first.relative_path == expected
    assert first.copied
    assert not second.copied
    assert (export / expected).read_bytes() == payload


def test_copy_rejects_hash_mismatch_symlink_and_conflicting_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"actual")
    export = tmp_path / "export"
    export.mkdir()
    wrong = _reference(b"expected", "audio/wav")
    with pytest.raises(TraceMediaCopyError, match="SOURCE_MISMATCH"):
        copy_media_reference(wrong, source, export)

    symlink = tmp_path / "link.bin"
    try:
        symlink.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(TraceMediaCopyError, match="INVALID_SOURCE"):
        copy_media_reference(_reference(b"actual", "audio/wav"), symlink, export)

    valid = _reference(b"actual", "audio/wav")
    target = export / f"media/sha256/{valid.sha256[:2]}/{valid.sha256}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"wrong")
    with pytest.raises(TraceMediaCopyError, match="DESTINATION_CONFLICT"):
        copy_media_reference(valid, source, export)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory handle boundary")
def test_copy_rejects_source_under_linked_parent_even_with_matching_bytes(
    tmp_path: Path,
) -> None:
    payload = b"private matching media bytes"
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "source.bin").write_bytes(payload)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "source.bin").write_bytes(payload)
    trusted.rename(tmp_path / "trusted-held")
    trusted.symlink_to(foreign, target_is_directory=True)
    export = tmp_path / "export"
    export.mkdir()

    with pytest.raises(TraceMediaCopyError, match="INVALID_SOURCE"):
        copy_media_reference(_reference(payload, "audio/wav"), trusted / "source.bin", export)

    assert not tuple(export.rglob("*.wav"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory swap boundary")
def test_copy_rejects_same_inode_parent_swap_after_source_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"private source bytes"
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    source = trusted / "source.bin"
    source.write_bytes(payload)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    os.link(source, foreign / "source.bin")
    export = tmp_path / "export"
    export.mkdir()
    original_lstat = Path.lstat
    swapped = False

    def swap_after_lstat(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = original_lstat(path)
        if path == source and not swapped:
            swapped = True
            trusted.rename(tmp_path / "trusted-held")
            trusted.symlink_to(foreign, target_is_directory=True)
        return observed

    monkeypatch.setattr(Path, "lstat", swap_after_lstat)
    with pytest.raises(TraceMediaCopyError, match="INVALID_SOURCE"):
        copy_media_reference(_reference(payload, "audio/wav"), source, export)

    assert swapped
    assert not tuple(export.rglob("*.wav"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX export directory handle boundary")
def test_copy_never_writes_through_linked_export_parent(tmp_path: Path) -> None:
    payload = b"private media bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    export = tmp_path / "export"
    export.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (export / "media").symlink_to(foreign, target_is_directory=True)
    reference = _reference(payload, "audio/wav")

    with pytest.raises(TraceMediaCopyError, match="PATH_ESCAPE"):
        copy_media_reference(reference, source, export)

    assert not tuple(foreign.rglob("*.wav"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX export root swap boundary")
def test_copy_never_writes_through_export_root_swapped_after_lstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"private media bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    export = tmp_path / "export"
    export.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    original_lstat = Path.lstat
    swapped = False

    def swap_after_lstat(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = original_lstat(path)
        if path == export and not swapped:
            swapped = True
            export.rename(tmp_path / "export-held")
            export.symlink_to(foreign, target_is_directory=True)
        return observed

    monkeypatch.setattr(Path, "lstat", swap_after_lstat)
    with pytest.raises(TraceMediaCopyError, match="PATH_ESCAPE"):
        copy_media_reference(_reference(payload, "audio/wav"), source, export)

    assert swapped
    assert not tuple(foreign.rglob("*.wav"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX atomic link boundary")
def test_copy_rejects_disappearing_destination_after_link_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified media bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    export = tmp_path / "export"
    export.mkdir()
    reference = _reference(payload, "audio/wav")

    def collide_then_disappear(*args: object, **kwargs: object) -> None:
        raise FileExistsError("destination disappeared after a competing link")

    monkeypatch.setattr(os, "link", collide_then_disappear)
    with pytest.raises(TraceMediaCopyError, match="DESTINATION_CONFLICT"):
        copy_media_reference(reference, source, export)

    target = export / f"media/sha256/{reference.sha256[:2]}/{reference.sha256}.wav"
    assert not target.exists()


def test_workspace_resolver_allows_only_regular_paths_below_absolute_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "inputs" / "audio.wav"
    source.parent.mkdir()
    source.write_bytes(b"audio")
    resolver = WorkspaceExportMediaResolver(root)
    valid = MediaReference(
        "media-1",
        TraceModality.AUDIO,
        "audio/wav",
        hashlib.sha256(b"audio").hexdigest(),
        5,
        locator=MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, "inputs/audio.wav"),
    )
    artifact = MediaReference(
        "media-2",
        TraceModality.AUDIO,
        "audio/wav",
        hashlib.sha256(b"audio").hexdigest(),
        5,
        locator=MediaLocator(MediaLocatorKind.ARTIFACT, "artifact/one"),
    )

    assert resolver.resolve(valid) == source
    assert resolver.resolve(artifact) is None

    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    link = root / "inputs" / "outside.wav"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    escaped = MediaReference(
        "media-3",
        TraceModality.AUDIO,
        "audio/wav",
        hashlib.sha256(b"audio").hexdigest(),
        5,
        locator=MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, "inputs/outside.wav"),
    )
    assert resolver.resolve(escaped) is None
