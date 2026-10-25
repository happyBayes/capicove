from __future__ import annotations

import shutil
import subprocess
import wave
from array import array
from math import sqrt
from pathlib import Path

import pytest

from capicove.processing.audio import (
    AudioFailureCode,
    AudioImportError,
    AudioProcessor,
)
from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
from capicove.processing.audio._adapters.whisper_cpp import WhisperCppTranscriber


pytestmark = pytest.mark.integration


@pytest.fixture
def ffmpeg_executable() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is required for this integration test")
    return executable


@pytest.fixture
def ffprobe_executable() -> str:
    executable = shutil.which("ffprobe")
    if executable is None:
        pytest.skip("ffprobe is required for this integration test")
    return executable


@pytest.mark.parametrize(
    ("suffix", "codec", "container_arguments"),
    [
        (".aac", "aac", ("-f", "adts")),
        (".flac", "flac", ()),
        (".m4a", "aac", ()),
        (".mp3", "libmp3lame", ()),
        (".ogg", "vorbis", ("-ac", "2", "-strict", "-2")),
        (".opus", "libopus", ()),
        (".wav", "pcm_s16le", ()),
        (".webm", "libopus", ("-f", "webm")),
    ],
)
def test_ffmpeg_validator_accepts_every_declared_format_and_processor_preserves_it(
    tmp_path: Path,
    ffmpeg_executable: str,
    ffprobe_executable: str,
    suffix: str,
    codec: str,
    container_arguments: tuple[str, ...],
) -> None:
    source = tmp_path / f"generated{suffix}"
    subprocess.run(
        [
            ffmpeg_executable,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.1",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            codec,
            *container_arguments,
            str(source),
        ],
        check=True,
        capture_output=True,
        start_new_session=True,
    )
    payload = source.read_bytes()

    evidence = AudioProcessor(
        FfprobeAudioStreamInspector(ffprobe_executable),
        FfmpegAudioValidator(ffmpeg_executable),
    ).import_file(source, tmp_path / "store")

    assert source.read_bytes() == payload
    assert evidence.stored_path.read_bytes() == payload


def test_ffmpeg_validator_rejects_malformed_container_before_evidence_storage(
    tmp_path: Path,
    ffmpeg_executable: str,
    ffprobe_executable: str,
) -> None:
    source = tmp_path / "malformed.wav"
    payload = b"RIFF" + (1000).to_bytes(4, "little") + b"WAVE-not-a-real-wave-payload"
    source.write_bytes(payload)
    store = tmp_path / "store"

    with pytest.raises(AudioImportError, match="inventory is invalid") as captured:
        AudioProcessor(
            FfprobeAudioStreamInspector(ffprobe_executable),
            FfmpegAudioValidator(ffmpeg_executable),
        ).import_file(source, store)

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO
    assert source.read_bytes() == payload
    assert not store.exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("suffix", "codec", "container_arguments"),
    [
        (".m4a", "aac", ()),
        (".webm", "libopus", ("-f", "webm")),
    ],
)
def test_multistream_container_requires_selector_and_preserves_one_evidence_identity(
    tmp_path: Path,
    ffmpeg_executable: str,
    ffprobe_executable: str,
    suffix: str,
    codec: str,
    container_arguments: tuple[str, ...],
) -> None:
    source = tmp_path / f"two-streams{suffix}"
    subprocess.run(
        [
            ffmpeg_executable,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=16000:d=0.15",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=0.15",
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-c:a",
            codec,
            "-disposition:a:0",
            "0",
            "-disposition:a:1",
            "default",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=zho",
            "-metadata:s:a:1",
            "title=commentary",
            *container_arguments,
            str(source),
        ],
        check=True,
        capture_output=True,
        start_new_session=True,
    )
    processor = AudioProcessor(
        FfprobeAudioStreamInspector(ffprobe_executable),
        FfmpegAudioValidator(ffmpeg_executable),
    )
    store = tmp_path / "store"

    with pytest.raises(AudioImportError) as ambiguous:
        processor.import_file(source, store)

    assert ambiguous.value.code is AudioFailureCode.AUDIO_STREAM_SELECTION_REQUIRED
    assert ambiguous.value.preflight_audio_stream_count == 2
    assert not store.exists()

    silent = processor.import_file_with_selection(
        source,
        store,
        audio_stream_index=0,
    )
    audible = processor.import_file_with_selection(
        source,
        store,
        audio_stream_index=1,
    )

    assert silent.evidence == audible.evidence
    assert silent.stream_selection.selected_audio_stream_index == 0
    assert audible.stream_selection.selected_audio_stream_index == 1
    assert audible.stream_selection.preflight_audio_stream_count == 2
    assert list(store.rglob("blob")) == [silent.evidence.stored_path]

    normalizer = WhisperCppTranscriber(
        tmp_path / "unused-model",
        vad_model_path=tmp_path / "unused-vad-model",
        ffmpeg_executable=ffmpeg_executable,
    )
    silent_wav = tmp_path / "silent.wav"
    audible_wav = tmp_path / "audible.wav"
    normalizer._normalize(source, silent_wav, 0, 30.0)
    normalizer._normalize(source, audible_wav, 1, 30.0)
    silent_rms = _wav_rms(silent_wav)
    audible_rms = _wav_rms(audible_wav)
    assert audible_rms > max(100.0, silent_rms * 10)


@pytest.mark.parametrize("channel_layout", ["mono", "stereo", "5.1"])
def test_single_multichannel_stream_remains_implicit_stream_zero(
    tmp_path: Path,
    ffmpeg_executable: str,
    ffprobe_executable: str,
    channel_layout: str,
) -> None:
    source = tmp_path / f"{channel_layout}.wav"
    subprocess.run(
        [
            ffmpeg_executable,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=channel_layout={channel_layout}:sample_rate=16000:d=0.1",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        capture_output=True,
        start_new_session=True,
    )

    imported = AudioProcessor(
        FfprobeAudioStreamInspector(ffprobe_executable),
        FfmpegAudioValidator(ffmpeg_executable),
    ).import_file_with_selection(source, tmp_path / "store")

    explicit = AudioProcessor(
        FfprobeAudioStreamInspector(ffprobe_executable),
        FfmpegAudioValidator(ffmpeg_executable),
    ).import_file_with_selection(
        source,
        tmp_path / "store",
        audio_stream_index=0,
    )

    assert imported.stream_selection.preflight_audio_stream_count == 1
    assert imported.stream_selection.selected_audio_stream_index == 0
    assert imported.stream_selection.selection_method == "sole_stream_implicit"
    assert explicit.evidence == imported.evidence
    assert explicit.stream_selection.selection_method == "explicit_index"


def test_video_only_webm_reports_no_audio_before_evidence_storage(
    tmp_path: Path,
    ffmpeg_executable: str,
    ffprobe_executable: str,
) -> None:
    source = tmp_path / "video-only.webm"
    subprocess.run(
        [
            ffmpeg_executable,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-an",
            "-c:v",
            "libvpx-vp9",
            str(source),
        ],
        check=True,
        capture_output=True,
        start_new_session=True,
    )
    store = tmp_path / "store"

    with pytest.raises(AudioImportError) as captured:
        AudioProcessor(
            FfprobeAudioStreamInspector(ffprobe_executable),
            FfmpegAudioValidator(ffmpeg_executable),
        ).import_file(source, store)

    assert captured.value.code is AudioFailureCode.NO_AUDIO_STREAM
    assert captured.value.preflight_audio_stream_count == 0
    assert not store.exists()


def _wav_rms(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        samples = array("h", stream.readframes(stream.getnframes()))
    if not samples:
        return 0.0
    return sqrt(sum(sample * sample for sample in samples) / len(samples))
