"""Exercise real sample boundaries, verified WAV output, and publication safety."""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    os.environ.get("CAPICOVE_REQUIRE_AUDIO_FFMPEG") != "1"
    and (os.name != "posix" or FFMPEG is None or FFPROBE is None),
    reason="requires POSIX and FFmpeg",
)


@pytest.fixture(scope="module", autouse=True)
def require_qualification_backend() -> None:
    if os.environ.get("CAPICOVE_REQUIRE_AUDIO_FFMPEG") == "1":
        assert os.name == "posix" and FFMPEG and FFPROBE, (
            "mandatory FFmpeg qualification backend is missing"
        )


def source_wav(path: Path, *, rate: int = 8000, channels: int = 1) -> bytes:
    samples = [((index % 1000) - 500) * 30 for index in range(rate)]
    payload = struct.pack(
        f"<{len(samples) * channels}h", *(s for s in samples for _ in range(channels))
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(payload)
    return payload


def service() -> object:
    from capicove.processing.audio import AudioClippingService, AudioProcessor
    from capicove.processing.audio._adapters.audio_clip import FfmpegAudioClipBackend
    from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
    from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector

    assert FFMPEG and FFPROBE
    return AudioClippingService(
        AudioProcessor(
            FfprobeAudioStreamInspector(FFPROBE, restrict_inputs=True),
            FfmpegAudioValidator(FFMPEG, restrict_inputs=True),
        ),
        FfmpegAudioClipBackend(FFMPEG, FFPROBE),
    )


def clip(tmp_path: Path, **options: object) -> dict[str, object]:
    from capicove.processing.audio import AudioClipRequest

    request = AudioClipRequest(
        source=tmp_path / "source.wav",
        destination=tmp_path / "clip.wav",
        start_seconds=0.01251,
        end_seconds=0.05251,
        **options,  # type: ignore[arg-type]
    )
    return service().clip_file(  # type: ignore[attr-defined,no-any-return]
        request, tmp_path / "evidence", deadline=datetime.now(UTC) + timedelta(seconds=30)
    )


def test_rounded_probe_duration_does_not_reject_valid_final_sample(tmp_path: Path) -> None:
    from capicove.processing.audio import AudioClipRequest

    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\x10\x01" * 1001)
    result = service().clip_file(  # type: ignore[attr-defined]
        AudioClipRequest(source, tmp_path / "clip.wav", 0, 0.0226984),
        tmp_path / "evidence",
        deadline=datetime.now(UTC) + timedelta(seconds=30),
    )
    assert result["output"]["frame_count"] == 1001


def test_unsupported_filesystem_fails_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    from capicove.processing.audio import AudioClippingError, AudioClipRequest
    from capicove.processing.audio._adapters import clip_files

    source_wav(tmp_path / "source.wav")

    def unsupported(descriptor: int) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(clip_files, "admit_directory", unsupported)
    with pytest.raises(AudioClippingError, match="BACKEND_UNAVAILABLE"):
        service().clip_file(  # type: ignore[attr-defined]
            AudioClipRequest(tmp_path / "source.wav", tmp_path / "clip.wav", 0, 0.1),
            tmp_path / "evidence",
            deadline=datetime.now(UTC) + timedelta(seconds=30),
        )
    assert not (tmp_path / "clip.wav").exists()


def test_clip_preserves_exact_samples_and_source_evidence(tmp_path: Path) -> None:
    payload = source_wav(tmp_path / "source.wav")
    before = (tmp_path / "source.wav").read_bytes()
    result = clip(tmp_path)
    with wave.open(str(tmp_path / "clip.wav"), "rb") as stream:
        assert stream.getnframes() == 320
        assert stream.readframes(320) == payload[202:842]
    assert (tmp_path / "source.wav").read_bytes() == before
    assert result["actual_range"] == {
        "start_frame": 101,
        "end_frame": 421,
        "start_seconds": 0.012625,
        "end_seconds": 0.052625,
    }
    evidence = result["evidence"]
    artifact = result["artifact"]
    assert isinstance(evidence, dict) and isinstance(artifact, dict)
    assert evidence["sha256"] == hashlib.sha256(before).hexdigest()
    assert artifact["sha256"] == hashlib.sha256((tmp_path / "clip.wav").read_bytes()).hexdigest()
    assert next((tmp_path / "evidence").rglob("blob")).read_bytes() == before
    assert str(tmp_path) not in str(result)


@pytest.mark.parametrize("rate,channels,frames", [(16000, 1, 640), (96000, 2, 3840)])
def test_clip_resamples_and_validates_actual_geometry(
    tmp_path: Path, rate: int, channels: int, frames: int
) -> None:
    source_wav(tmp_path / "source.wav")
    result = clip(tmp_path, sample_rate_hz=rate, channels=channels)
    output = result["output"]
    assert isinstance(output, dict)
    assert abs(int(str(output["frame_count"])) - frames) <= 1
    assert output["sample_rate_hz"] == rate and output["channels"] == channels
    assert output["bit_rate_bps"] == rate * channels * 16


@pytest.mark.parametrize("collision", ["file", "symlink"])
def test_clip_never_overwrites_existing_destination(tmp_path: Path, collision: str) -> None:
    from capicove.processing.audio import AudioClippingError

    source_wav(tmp_path / "source.wav")
    destination = tmp_path / "clip.wav"
    if collision == "file":
        destination.write_bytes(b"keep me")
    else:
        destination.symlink_to(tmp_path / "source.wav")
    with pytest.raises(AudioClippingError, match="OUTPUT_EXISTS"):
        clip(tmp_path)
    assert destination.read_bytes() == (
        b"keep me" if collision == "file" else (tmp_path / "source.wav").read_bytes()
    )
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_seconds", True),
        ("start_seconds", -1),
        ("end_seconds", float("nan")),
        ("end_seconds", float("inf")),
        ("end_seconds", 0),
        ("sample_rate_hz", True),
        ("sample_rate_hz", 7999),
        ("channels", 3),
        ("profile", "mp3"),
    ],
)
def test_invalid_requests_are_rejected_before_processing(
    tmp_path: Path, field: str, value: object
) -> None:
    from capicove.processing.audio import AudioClipRequest

    arguments: dict[str, object] = dict(
        source=tmp_path / "source.wav",
        destination=tmp_path / "clip.wav",
        start_seconds=0,
        end_seconds=1,
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        AudioClipRequest(**arguments)  # type: ignore[arg-type]
    assert not (tmp_path / "clip.wav").exists()


@pytest.mark.parametrize(
    "suffix,codec,extra",
    [
        ("aac", "aac", ["-f", "adts"]),
        ("flac", "flac", []),
        ("m4a", "aac", []),
        ("mp3", "libmp3lame", []),
        ("ogg", "libvorbis", []),
        ("opus", "libopus", []),
        ("wav", "pcm_s16le", []),
        ("webm", "libopus", []),
    ],
)
def test_every_declared_format_can_be_clipped(
    tmp_path: Path, suffix: str, codec: str, extra: list[str]
) -> None:
    from capicove.processing.audio import AudioClipRequest

    assert FFMPEG
    source = tmp_path / f"input.{suffix}"
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            codec,
            *extra,
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    request = AudioClipRequest(source, tmp_path / "clip.wav", 0.1, 0.3)
    result = service().clip_file(  # type: ignore[attr-defined]
        request, tmp_path / "evidence", deadline=datetime.now(UTC) + timedelta(seconds=30)
    )
    assert result["output"]["duration_seconds"] == pytest.approx(0.2, abs=1 / 8000)


def test_multiple_audio_streams_are_rejected_without_guessing(tmp_path: Path) -> None:
    from capicove.processing.audio import AudioClippingError, AudioClipRequest

    assert FFMPEG
    source = tmp_path / "two.m4a"
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=1",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        timeout=20,
    )
    with pytest.raises(AudioClippingError, match="AUDIO_STREAM_SELECTION_REQUIRED"):
        service().clip_file(  # type: ignore[attr-defined]
            AudioClipRequest(source, tmp_path / "clip.wav", 0, 0.1),
            tmp_path / "evidence",
            deadline=datetime.now(UTC) + timedelta(seconds=30),
        )
    assert not (tmp_path / "clip.wav").exists()


def test_range_beyond_source_duration_is_rejected(tmp_path: Path) -> None:
    from capicove.processing.audio import AudioClippingError, AudioClipRequest

    source_wav(tmp_path / "source.wav")
    with pytest.raises(AudioClippingError, match="INVALID_RANGE"):
        service().clip_file(  # type: ignore[attr-defined]
            AudioClipRequest(tmp_path / "source.wav", tmp_path / "clip.wav", 0, 2),
            tmp_path / "evidence",
            deadline=datetime.now(UTC) + timedelta(seconds=30),
        )
    assert not (tmp_path / "clip.wav").exists()


def test_expired_deadline_creates_no_output(tmp_path: Path) -> None:
    from capicove.processing.audio import AudioClippingError, AudioClipRequest

    source_wav(tmp_path / "source.wav")
    request = AudioClipRequest(tmp_path / "source.wav", tmp_path / "clip.wav", 0, 0.5)
    with pytest.raises(AudioClippingError, match="TIMEOUT"):
        service().clip_file(  # type: ignore[attr-defined]
            request, tmp_path / "evidence", deadline=datetime.now(UTC) - timedelta(seconds=1)
        )
    assert not (tmp_path / "evidence").exists()
    assert not (tmp_path / "clip.wav").exists()


def test_replaced_evidence_blob_is_never_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.processing.audio import AudioClippingError, AudioClipRequest

    source_wav(tmp_path / "source.wav")
    instance = service()
    backend = instance._backend  # type: ignore[attr-defined]
    original = backend.write

    def replace_blob(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)
        blob = next((tmp_path / "evidence").rglob("blob"), None)
        if blob is not None:
            blob.unlink()
            blob.write_bytes(b"corrupted replacement")

    monkeypatch.setattr(backend, "write", replace_blob)
    with pytest.raises(AudioClippingError, match="STORE_CORRUPTED"):
        instance.clip_file(  # type: ignore[attr-defined]
            AudioClipRequest(tmp_path / "source.wav", tmp_path / "clip.wav", 0.1, 0.3),
            tmp_path / "evidence",
            deadline=datetime.now(UTC) + timedelta(seconds=30),
        )
    assert not (tmp_path / "clip.wav").exists()
