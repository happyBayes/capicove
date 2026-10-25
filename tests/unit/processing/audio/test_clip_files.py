"""Reject corrupt WAV bytes and unsafe source/output filesystem bindings."""

from __future__ import annotations

import os
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.audio import AudioClippingError
from capicove.processing.audio.clip_models import AudioClipGeometry


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX clipping profile")


def wav_bytes() -> bytes:
    pcm = b"\x10\x01" * 160
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


@pytest.mark.parametrize(
    "damage",
    [
        "trailing",
        "truncated",
        "partial_frame",
        "byte_rate",
        "zero_frames",
        "duplicate_fmt",
        "float_pcm",
    ],
)
def test_wav_parser_rejects_corrupt_output(tmp_path: Path, damage: str) -> None:
    from capicove.processing.audio._adapters.clip_files import verify_wav

    payload = bytearray(wav_bytes())
    if damage == "trailing":
        payload += b"x"
    elif damage == "truncated":
        payload = payload[:-1]
    elif damage == "partial_frame":
        payload[40:44] = struct.pack("<I", 319)
    elif damage == "byte_rate":
        payload[28:32] = struct.pack("<I", 123)
    elif damage == "zero_frames":
        payload = payload[:44]
        payload[4:8] = struct.pack("<I", 36)
        payload[40:44] = b"\0" * 4
    elif damage == "duplicate_fmt":
        payload += payload[12:36]
        payload[4:8] = struct.pack("<I", len(payload) - 8)
    else:
        payload[20:22] = struct.pack("<H", 3)
    path = tmp_path / "bad.wav"
    path.write_bytes(payload)
    with path.open("rb") as stream, pytest.raises(AudioClippingError, match="OUTPUT_CORRUPTED"):
        verify_wav(
            stream.fileno(),
            AudioClipGeometry(0, 160, 8000, 8000, 1, 160),
            datetime.now(UTC) + timedelta(seconds=5),
        )


def test_verified_source_rejects_parent_symlinks(tmp_path: Path) -> None:
    from capicove.processing.audio._adapters.clip_files import verified_source

    directory = tmp_path / "real"
    directory.mkdir()
    (directory / "source.wav").write_bytes(wav_bytes())
    (tmp_path / "link").symlink_to(directory, target_is_directory=True)
    with (
        pytest.raises(OSError),
        verified_source(tmp_path / "link" / "source.wav", datetime.now(UTC) + timedelta(seconds=5)),
    ):
        raise AssertionError("symlinked parent accepted")


def test_verified_source_preserves_original_bytes_after_replacement(tmp_path: Path) -> None:
    from capicove.processing.audio._adapters.clip_files import verified_source

    source = tmp_path / "source.wav"
    payload = wav_bytes()
    source.write_bytes(payload)
    with verified_source(source, datetime.now(UTC) + timedelta(seconds=5)) as snapshot:
        source.write_bytes(b"replacement")
        assert snapshot.read_bytes() == payload
        assert snapshot.name == "source.wav"
    assert not snapshot.exists()


def test_output_directory_replacement_cannot_redirect_publication(tmp_path: Path) -> None:
    from capicove.processing.audio._adapters.clip_files import ClipOutput, open_directory

    directory = tmp_path / "output"
    directory.mkdir()
    destination = directory / "clip.wav"
    geometry = AudioClipGeometry(0, 160, 8000, 8000, 1, 160)
    with open_directory(directory) as parent:
        output = ClipOutput(destination, parent, geometry)
        os.write(output.descriptor, wav_bytes())
        directory.rename(tmp_path / "moved")
        directory.mkdir()
        try:
            with pytest.raises(AudioClippingError, match="SOURCE_CHANGED"):
                output.publish()
        finally:
            output.close()
    assert not destination.exists()
    assert not list((tmp_path / "moved").iterdir())


def test_ambiguous_link_failure_cleans_identity_owned_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.processing.audio._adapters.clip_files import ClipOutput, open_directory

    original = os.link

    def ambiguous(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)  # type: ignore[arg-type]
        raise OSError("acknowledgement lost")

    with open_directory(tmp_path) as parent:
        output = ClipOutput(
            tmp_path / "clip.wav", parent, AudioClipGeometry(0, 160, 8000, 8000, 1, 160)
        )
        os.write(output.descriptor, wav_bytes())
        monkeypatch.setattr(os, "link", ambiguous)
        try:
            with pytest.raises(OSError):
                output.publish()
        finally:
            output.close()
    assert not (tmp_path / "clip.wav").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_admission_cleans_probe_links_after_lost_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.processing.audio._adapters.clip_files import admit_directory, open_directory

    original = os.link

    def ambiguous(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)  # type: ignore[arg-type]
        raise OSError("acknowledgement lost")

    with open_directory(tmp_path) as parent:
        monkeypatch.setattr(os, "link", ambiguous)
        with pytest.raises(OSError):
            admit_directory(parent)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bad_duration", ["bad", "NaN", "Infinity"])
def test_malformed_probe_duration_returns_a_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_duration: str
) -> None:
    import json

    from capicove.processing.audio._adapters import audio_clip

    path = tmp_path / "source.wav"
    path.write_bytes(wav_bytes())
    payload = json.dumps(
        {"streams": [{"duration": bad_duration}], "format": {"format_name": "wav"}}
    ).encode()
    monkeypatch.setattr(audio_clip, "_run", lambda *args, **kwargs: payload)
    with path.open("rb") as source, pytest.raises(AudioClippingError, match="INVALID_AUDIO"):
        audio_clip.FfmpegAudioClipBackend().inspect(
            source.fileno(), "wav", datetime.now(UTC) + timedelta(seconds=5)
        )
