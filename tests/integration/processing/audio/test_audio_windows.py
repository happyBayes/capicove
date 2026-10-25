"""Real FFmpeg window preparation: exact samples, stream selection and failures."""

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


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=60)


def _write_wav(path: Path, seconds: float, *, rate: int = 16_000) -> bytes:
    frames = int(seconds * rate)
    samples = [((index * 7) % 2000) - 1000 for index in range(frames)]
    payload = struct.pack(f"<{frames}h", *samples)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(payload)
    return payload


def _preparer(policy: object | None = None) -> object:
    from capicove.processing.audio import AudioProcessor
    from capicove.processing.audio._adapters.audio_windows import FfmpegAudioWindowBackend
    from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
    from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
    from capicove.processing.audio.retrieval import AudioWindowPolicy, AudioWindowPreparer

    assert FFMPEG and FFPROBE
    return AudioWindowPreparer(
        AudioProcessor(
            FfprobeAudioStreamInspector(FFPROBE, restrict_inputs=True),
            FfmpegAudioValidator(FFMPEG, restrict_inputs=True),
        ),
        FfmpegAudioWindowBackend(FFMPEG),
        policy or AudioWindowPolicy(),  # type: ignore[arg-type]
    )


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    return path


def test_thirty_one_second_source_yields_exact_pcm_windows(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    pcm = _write_wav(source, 31)
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    with _preparer().prepare(source, store, staging, deadline=_deadline()) as prepared:  # type: ignore[attr-defined]
        assert prepared.total_frames == 31 * 16_000
        assert [(w.start_frame, w.end_frame) for w in prepared.windows] == [
            (0, 480_000),
            (400_000, 496_000),
        ]
        assert prepared.evidence.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
        assert prepared.pcm_sha256 == hashlib.sha256(pcm).hexdigest()
        tail = prepared.window_pcm(1)
        assert tail == pcm[400_000 * 2 : 496_000 * 2]
        receipt = prepared.receipts[1]
        assert receipt.pcm_sha256 == hashlib.sha256(tail).hexdigest()
        assert receipt.audio_stream_index == 0
        assert receipt.decoder.startswith("ffmpeg ")
        staged = list(staging.iterdir())
        assert len(staged) == 1 and staged[0].stat().st_mode & 0o777 == 0o600
    assert list(staging.iterdir()) == []


def test_resampled_source_preserves_target_geometry(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    _write_wav(source, 2, rate=8_000)
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    with _preparer().prepare(source, store, staging, deadline=_deadline()) as prepared:  # type: ignore[attr-defined]
        assert prepared.total_frames == 32_000
        assert len(prepared.window_pcm(0)) == 64_000


def test_multiple_streams_require_explicit_selection(tmp_path: Path) -> None:
    from capicove.processing.audio.models import AudioFailureCode
    from capicove.processing.audio.retrieval import AudioWindowError

    left, right = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(left, 1)
    _write_wav(right, 2)
    source = tmp_path / "two.webm"
    assert FFMPEG
    subprocess.run(
        [
            FFMPEG,
            "-v",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "libopus",
            str(source),
        ],
        check=True,
    )
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    with pytest.raises(AudioWindowError) as error:
        _preparer().prepare(source, store, staging, deadline=_deadline())  # type: ignore[attr-defined]
    assert error.value.code is AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED
    with _preparer().prepare(  # type: ignore[attr-defined]
        source, store, staging, stream_index=1, deadline=_deadline()
    ) as prepared:
        assert prepared.selection.selected_audio_stream_index == 1
        assert abs(prepared.total_frames - 32_000) <= 400


def test_pcm_tampering_after_preparation_fails_closed(tmp_path: Path) -> None:
    from capicove.processing.audio.retrieval import AudioWindowError

    source = tmp_path / "tone.wav"
    _write_wav(source, 1)
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    with _preparer().prepare(source, store, staging, deadline=_deadline()) as prepared:  # type: ignore[attr-defined]
        staged = next(staging.iterdir())
        with staged.open("r+b") as stream:
            stream.write(b"\x7f\x7f")
        with pytest.raises(AudioWindowError):
            prepared.window_pcm(0)


def test_overlong_source_is_rejected_before_windowing(tmp_path: Path) -> None:
    from capicove.processing.audio.retrieval import AudioWindowError, AudioWindowPolicy

    source = tmp_path / "tone.wav"
    _write_wav(source, 3)
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    preparer = _preparer(AudioWindowPolicy(max_duration_seconds=2))
    with pytest.raises(AudioWindowError):
        preparer.prepare(source, store, staging, deadline=_deadline())  # type: ignore[attr-defined]
    assert list(staging.iterdir()) == []


def test_undecodable_source_reports_failure_and_cleans_staging(tmp_path: Path) -> None:
    from capicove.processing.audio.retrieval import AudioWindowError

    source = tmp_path / "broken.wav"
    source.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 64)
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    with pytest.raises(AudioWindowError):
        _preparer().prepare(source, store, staging, deadline=_deadline())  # type: ignore[attr-defined]
    assert list(staging.iterdir()) == []


def test_preserved_evidence_decodes_identically_and_detects_tampering(tmp_path: Path) -> None:
    from capicove.processing.audio.retrieval import AudioWindowError

    source = tmp_path / "tone.wav"
    _write_wav(source, 31)
    store, staging = _private(tmp_path / "store"), _private(tmp_path / "staging")
    preparer = _preparer()
    with preparer.prepare(source, store, staging, deadline=_deadline()) as first:  # type: ignore[attr-defined]
        evidence, selection, receipts = first.evidence, first.selection, first.receipts
    source.write_bytes(b"changed after import")
    with preparer.prepare_preserved(  # type: ignore[attr-defined]
        evidence, selection, ".wav", staging, deadline=_deadline()
    ) as again:
        assert again.receipts == receipts
    evidence.stored_path.chmod(0o600)
    with evidence.stored_path.open("r+b") as stream:
        stream.seek(100)
        stream.write(b"\x01\x02")
    with pytest.raises(AudioWindowError):
        preparer.prepare_preserved(  # type: ignore[attr-defined]
            evidence, selection, ".wav", staging, deadline=_deadline()
        )
    assert list(staging.iterdir()) == []
