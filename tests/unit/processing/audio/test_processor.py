from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from capicove.processing.audio import (
    AudioEvidence,
    AudioFailureCode,
    AudioImportError,
    AudioProcessor,
    AudioStreamInventory,
    AudioStreamSelection,
)
from capicove.processing.audio.ports import AudioValidationError
from capicove.processing.audio.processor import _FileSignature


_SYNTHETIC_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x64\x00\x00\x00\x00"


class _Inspector:
    def __init__(self, count: int = 1) -> None:
        self.count = count
        self.calls: list[Path] = []

    def inspect(
        self,
        source: Path,
        *,
        deadline: datetime | None = None,
    ) -> AudioStreamInventory:
        self.calls.append(source)
        return AudioStreamInventory(self.count)


class _AcceptingValidator:
    def __init__(self) -> None:
        self.indexes: list[int] = []

    def validate(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        deadline: datetime | None = None,
    ) -> None:
        self.indexes.append(audio_stream_index)


class _RejectingValidator:
    def validate(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        deadline: datetime | None = None,
    ) -> None:
        raise AudioValidationError(
            AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE,
            "audio stream is not decodable",
        )


class _IndexSensitiveValidator:
    def __init__(self, damaged_index: int) -> None:
        self.damaged_index = damaged_index
        self.indexes: list[int] = []

    def validate(
        self,
        source: Path,
        *,
        audio_stream_index: int,
        deadline: datetime | None = None,
    ) -> None:
        self.indexes.append(audio_stream_index)
        if audio_stream_index == self.damaged_index:
            raise AudioValidationError(
                AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE,
                "audio stream is not decodable",
            )


def _processor() -> AudioProcessor:
    return AudioProcessor(_Inspector(), _AcceptingValidator())


@pytest.mark.parametrize(
    ("name", "payload", "expected_media_type"),
    [
        ("clip.aac", b"\xff\xf1\x50\x80\x00\x1f\xfc", "audio/aac"),
        ("clip.flac", b"fLaC\x00\x00\x00\x00", "audio/flac"),
        ("clip.m4a", b"\x00\x00\x00\x18ftypM4A ", "audio/mp4"),
        ("clip.mp3", _SYNTHETIC_MP3, "audio/mpeg"),
        ("clip.ogg", b"OggS\x00\x02\x00\x00", "audio/ogg"),
        ("clip.opus", b"OggS\x00\x02Opus", "audio/ogg"),
        ("clip.wav", b"RIFF\x04\x00\x00\x00WAVE", "audio/wav"),
        ("clip.webm", b"\x1a\x45\xdf\xa3\x01\x00\x00\x00", "audio/webm"),
    ],
)
def test_accepts_supported_audio_signatures(
    tmp_path: Path, name: str, payload: bytes, expected_media_type: str
) -> None:
    source = tmp_path / name
    source.write_bytes(payload)

    evidence = _processor().import_file(source, tmp_path / "store")

    assert evidence.media_type == expected_media_type
    assert evidence.stored_path.read_bytes() == payload
    assert source.read_bytes() == payload


def test_rejects_undecodable_audio_before_creating_evidence(tmp_path: Path) -> None:
    source = tmp_path / "malformed.wav"
    payload = b"RIFF" + (1000).to_bytes(4, "little") + b"WAVE-not-a-real-wave-payload"
    source.write_bytes(payload)
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="audio stream is not decodable") as captured:
        AudioProcessor(_Inspector(), _RejectingValidator()).import_file(source, store)

    assert captured.value.code is AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE
    assert source.read_bytes() == payload
    assert not store.exists()


def test_import_preserves_source_and_stores_content_addressed_audio(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    payload = _SYNTHETIC_MP3
    source.write_bytes(payload)
    source_mode = source.stat().st_mode

    evidence = _processor().import_file(source, tmp_path / "store")

    expected_hash = hashlib.sha256(payload).hexdigest()
    assert evidence == AudioEvidence(
        evidence_id=f"sha256:{expected_hash}",
        sha256=expected_hash,
        size_bytes=len(payload),
        media_type="audio/mpeg",
        source_name="meeting.mp3",
        stored_path=(tmp_path / "store" / "sha256" / expected_hash[:2] / expected_hash / "blob"),
    )
    assert source.read_bytes() == payload
    assert source.stat().st_mode == source_mode
    assert evidence.stored_path.read_bytes() == payload


def test_reimport_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "note.wav"
    source.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
    store = tmp_path / "store"
    processor = _processor()

    first = processor.import_file(source, store)
    with (
        patch(
            "capicove.processing.audio.processor._create_temporary_file",
            side_effect=AssertionError("re-import must not create a temporary file"),
        ),
        patch("capicove.processing.audio.processor.os.fsync") as fsync,
    ):
        second = processor.import_file(source, store)

    assert second == first
    fsync.assert_not_called()
    assert list((store / "sha256").rglob("blob")) == [first.stored_path]
    assert list((store / ".tmp").iterdir()) == []


@pytest.mark.parametrize("name", ["recording.txt", "recording", "recording.mp4"])
def test_rejects_unsupported_input_without_stored_evidence(tmp_path: Path, name: str) -> None:
    source = tmp_path / name
    source.write_bytes(b"not-accepted")
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="unsupported audio extension") as captured:
        _processor().import_file(source, store)

    assert captured.value.code is AudioFailureCode.UNSUPPORTED_FORMAT
    assert source.read_bytes() == b"not-accepted"
    assert not store.exists()


def test_rejects_content_that_does_not_match_supported_extension(tmp_path: Path) -> None:
    source = tmp_path / "not-audio.mp3"
    payload = b"This is text, not decodable MP3 audio."
    source.write_bytes(payload)
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="content does not match"):
        _processor().import_file(source, store)

    assert source.read_bytes() == payload
    assert not store.exists()


def test_rejects_id3_tag_without_mpeg_audio_frames(tmp_path: Path) -> None:
    source = tmp_path / "metadata-only.mp3"
    payload = b"ID3\x04\x00\x00\x00\x00\x00\x00This is still not MP3 audio."
    source.write_bytes(payload)
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="content does not match"):
        _processor().import_file(source, store)

    assert source.read_bytes() == payload
    assert not store.exists()


def test_accepts_id3v24_footer_before_mpeg_audio_frame(tmp_path: Path) -> None:
    source = tmp_path / "footer.mp3"
    footer = b"3DI\x04\x00\x10\x00\x00\x00\x00"
    payload = b"ID3\x04\x00\x10\x00\x00\x00\x00" + footer + b"\xff\xfb\x90\x64"
    source.write_bytes(payload)

    evidence = _processor().import_file(source, tmp_path / "store")

    assert evidence.stored_path.read_bytes() == payload


def test_rejects_missing_and_empty_sources(tmp_path: Path) -> None:
    missing = tmp_path / "private-name.mp3"
    with pytest.raises(AudioImportError, match="not readable") as captured:
        _processor().import_file(missing, tmp_path / "store")

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO
    assert str(missing) not in str(captured.value)

    empty = tmp_path / "empty.mp3"
    empty.touch()
    with pytest.raises(AudioImportError, match="must not be empty"):
        _processor().import_file(empty, tmp_path / "store")

    assert empty.exists()
    assert not (tmp_path / "store").exists()


def test_translates_store_initialization_failure_without_exposing_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    store = tmp_path / "private-store"
    store.write_bytes(b"not-a-directory")

    with pytest.raises(
        AudioImportError,
        match=r"^audio evidence could not be stored$",
    ) as captured:
        _processor().import_file(source, store)

    assert captured.value.code is AudioFailureCode.STORE_FAILED
    assert str(store) not in str(captured.value)


def test_fails_closed_without_secure_store_primitives(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    store = tmp_path / "store"

    with (
        patch(
            "capicove.processing.audio.processor._secure_store_primitives_available",
            return_value=False,
        ),
        pytest.raises(AudioImportError, match="secure filesystem primitives") as captured,
    ):
        _processor().import_file(source, store)

    assert captured.value.code is AudioFailureCode.STORE_FAILED
    assert not store.exists()


def test_cleanup_failure_does_not_replace_storage_error(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)

    with (
        patch(
            "capicove.processing.audio.processor.os.link",
            side_effect=OSError("private storage diagnostic"),
        ),
        patch(
            "capicove.processing.audio.processor.os.unlink",
            side_effect=OSError("private cleanup diagnostic"),
        ),
        pytest.raises(
            AudioImportError,
            match=r"^audio evidence could not be stored$",
        ) as captured,
    ):
        _processor().import_file(source, tmp_path / "store")

    assert captured.value.code is AudioFailureCode.STORE_FAILED
    assert "private" not in str(captured.value)


def test_rejects_source_changed_during_import_and_cleans_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    initial = source.stat()
    signatures = [
        _FileSignature(
            device=initial.st_dev,
            inode=initial.st_ino,
            size=initial.st_size,
            modified_ns=initial.st_mtime_ns,
            changed_ns=initial.st_ctime_ns,
        ),
        _FileSignature(
            device=initial.st_dev,
            inode=initial.st_ino,
            size=initial.st_size,
            modified_ns=initial.st_mtime_ns,
            changed_ns=initial.st_ctime_ns,
        ),
        _FileSignature(
            device=initial.st_dev,
            inode=initial.st_ino,
            size=initial.st_size + 1,
            modified_ns=initial.st_mtime_ns + 1,
            changed_ns=initial.st_ctime_ns + 1,
        ),
    ]
    store = tmp_path / "store"

    with (
        patch(
            "capicove.processing.audio.processor._file_signature",
            side_effect=signatures,
        ),
        pytest.raises(AudioImportError, match="changed during import"),
    ):
        _processor().import_file(source, store)

    assert source.exists()
    assert list((store / "sha256").rglob("blob")) == []
    assert list((store / ".tmp").iterdir()) == []


def test_detects_corrupted_existing_blob_without_overwriting_it(tmp_path: Path) -> None:
    source = tmp_path / "meeting.flac"
    payload = b"fLaC-audio"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    blob = tmp_path / "store" / "sha256" / digest[:2] / digest / "blob"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"corrupted")

    with pytest.raises(AudioImportError, match="is corrupted"):
        _processor().import_file(source, tmp_path / "store")

    assert source.read_bytes() == payload
    assert blob.read_bytes() == b"corrupted"
    assert not (tmp_path / "store" / ".tmp").exists()


def test_rejects_symlinked_existing_blob_even_when_target_hash_matches(tmp_path: Path) -> None:
    source = tmp_path / "meeting.flac"
    payload = b"fLaC-audio"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    blob = tmp_path / "store" / "sha256" / digest[:2] / digest / "blob"
    blob.parent.mkdir(parents=True)
    target = tmp_path / "outside-blob"
    target.write_bytes(payload)
    blob.symlink_to(target)

    with pytest.raises(AudioImportError, match="is corrupted") as captured:
        _processor().import_file(source, tmp_path / "store")

    assert captured.value.code is AudioFailureCode.STORE_CORRUPTED
    assert blob.is_symlink()
    assert target.read_bytes() == payload


@pytest.mark.parametrize("component", ["root", ".tmp", "sha256", "prefix", "hash"])
def test_rejects_symlinked_store_directory_components(
    tmp_path: Path,
    component: str,
) -> None:
    source = tmp_path / "meeting.flac"
    payload = b"fLaC-audio"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    store = tmp_path / "store"
    outside = tmp_path / f"outside-{component}"
    outside.mkdir()

    if component == "root":
        store.symlink_to(outside, target_is_directory=True)
    elif component == ".tmp":
        store.mkdir()
        (store / ".tmp").symlink_to(outside, target_is_directory=True)
    elif component == "sha256":
        store.mkdir()
        (store / "sha256").symlink_to(outside, target_is_directory=True)
    elif component == "prefix":
        (store / "sha256").mkdir(parents=True)
        (store / "sha256" / digest[:2]).symlink_to(outside, target_is_directory=True)
    else:
        (store / "sha256" / digest[:2]).mkdir(parents=True)
        (store / "sha256" / digest[:2] / digest).symlink_to(
            outside,
            target_is_directory=True,
        )

    with pytest.raises(AudioImportError, match="is corrupted") as captured:
        _processor().import_file(source, store)

    assert captured.value.code is AudioFailureCode.STORE_CORRUPTED
    assert list(outside.iterdir()) == []


def test_atomic_publish_does_not_overwrite_competing_blob(tmp_path: Path) -> None:
    source = tmp_path / "meeting.flac"
    payload = b"fLaC-audio"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    blob = tmp_path / "store" / "sha256" / digest[:2] / digest / "blob"
    competing = b"competing-writer"

    def create_competing_blob(*args: object, **kwargs: object) -> None:
        blob.write_bytes(competing)
        raise FileExistsError

    with (
        patch("capicove.processing.audio.processor.os.link", side_effect=create_competing_blob),
        pytest.raises(AudioImportError, match="is corrupted") as captured,
    ):
        _processor().import_file(source, tmp_path / "store")

    assert captured.value.code is AudioFailureCode.STORE_CORRUPTED
    assert blob.read_bytes() == competing


def test_rejects_store_directory_replaced_during_publish(tmp_path: Path) -> None:
    source = tmp_path / "meeting.flac"
    payload = b"fLaC-audio"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    hash_directory = tmp_path / "store" / "sha256" / digest[:2] / digest
    detached = tmp_path / "detached-hash-directory"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_link = os.link

    def replace_hash_directory(
        source_name: str,
        target_name: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        real_link(
            source_name,
            target_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        hash_directory.rename(detached)
        hash_directory.symlink_to(outside, target_is_directory=True)

    with (
        patch("capicove.processing.audio.processor.os.link", side_effect=replace_hash_directory),
        pytest.raises(AudioImportError, match="is corrupted") as captured,
    ):
        _processor().import_file(source, tmp_path / "store")

    assert captured.value.code is AudioFailureCode.STORE_CORRUPTED
    assert hash_directory.is_symlink()
    assert list(outside.iterdir()) == []


def test_expired_import_deadline_prevents_evidence_writes(tmp_path: Path) -> None:
    source = tmp_path / "meeting.flac"
    source.write_bytes(b"fLaC-audio")
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="deadline") as captured:
        _processor().import_file(
            source,
            store,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT
    assert not store.exists()


@pytest.mark.parametrize("selector", [None, True, False, 1.0, "1", -1])
def test_rejects_invalid_explicit_selector_before_probe_or_storage(
    tmp_path: Path,
    selector: object,
) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    store = tmp_path / "store"
    inspector = _Inspector(2)
    validator = _AcceptingValidator()

    with pytest.raises(AudioImportError, match="non-negative integer") as captured:
        AudioProcessor(inspector, validator).import_file(
            source,
            store,
            audio_stream_index=selector,
        )

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO_STREAM_INDEX
    assert captured.value.preflight_audio_stream_count is None
    assert inspector.calls == []
    assert validator.indexes == []
    assert not store.exists()


@pytest.mark.parametrize(
    ("count", "selector", "code", "message"),
    [
        (0, 0, AudioFailureCode.NO_AUDIO_STREAM, "no audio streams"),
        (
            2,
            "omitted",
            AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED,
            "require audio_stream_index",
        ),
        (
            2,
            2,
            AudioFailureCode.AUDIO_STREAM_INDEX_OUT_OF_RANGE,
            "outside the available",
        ),
    ],
)
def test_selection_failures_are_bounded_and_have_no_side_effects(
    tmp_path: Path,
    count: int,
    selector: object,
    code: AudioFailureCode,
    message: str,
) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    store = tmp_path / "store"
    validator = _AcceptingValidator()
    processor = AudioProcessor(_Inspector(count), validator)

    with pytest.raises(AudioImportError, match=message) as captured:
        if selector == "omitted":
            processor.import_file(source, store)
        else:
            processor.import_file(source, store, audio_stream_index=selector)

    assert captured.value.code is code
    assert captured.value.preflight_audio_stream_count == count
    assert validator.indexes == []
    assert not store.exists()


def test_explicit_selection_is_passed_to_validation_and_recorded(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    validator = _AcceptingValidator()

    imported = AudioProcessor(_Inspector(2), validator).import_file_with_selection(
        source,
        tmp_path / "store",
        audio_stream_index=1,
    )

    assert validator.indexes == [1]
    assert imported.stream_selection == AudioStreamSelection(
        selected_audio_stream_index=1,
        selection_method="explicit_index",
        preflight_audio_stream_count=2,
    )
    assert imported.evidence.stored_path.read_bytes() == _SYNTHETIC_MP3


def test_single_stream_omission_records_implicit_selection(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)

    imported = _processor().import_file_with_selection(source, tmp_path / "store")

    assert imported.stream_selection == AudioStreamSelection(
        selected_audio_stream_index=0,
        selection_method="sole_stream_implicit",
        preflight_audio_stream_count=1,
    )


def test_selected_stream_failure_does_not_fall_back_or_write_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="not decodable") as captured:
        AudioProcessor(_Inspector(2), _RejectingValidator()).import_file(
            source,
            store,
            audio_stream_index=1,
        )

    assert captured.value.code is AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE
    assert captured.value.preflight_audio_stream_count == 2
    assert not store.exists()


def test_two_selected_ordinals_share_original_evidence_identity(tmp_path: Path) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    processor = AudioProcessor(_Inspector(2), _AcceptingValidator())
    store = tmp_path / "store"

    first = processor.import_file_with_selection(source, store, audio_stream_index=0)
    second = processor.import_file_with_selection(source, store, audio_stream_index=1)

    assert first.evidence == second.evidence
    assert first.stream_selection != second.stream_selection
    assert list(store.rglob("blob")) == [first.evidence.stored_path]


def test_damaged_unselected_stream_does_not_block_selected_valid_stream(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.mp3"
    source.write_bytes(_SYNTHETIC_MP3)
    store = tmp_path / "store"
    validator = _IndexSensitiveValidator(damaged_index=1)
    processor = AudioProcessor(_Inspector(2), validator)

    selected_valid = processor.import_file_with_selection(
        source,
        store,
        audio_stream_index=0,
    )

    assert selected_valid.stream_selection.selected_audio_stream_index == 0
    assert validator.indexes == [0]

    with pytest.raises(AudioImportError) as captured:
        processor.import_file_with_selection(
            source,
            store,
            audio_stream_index=1,
        )

    assert captured.value.code is AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE
    assert validator.indexes == [0, 1]
