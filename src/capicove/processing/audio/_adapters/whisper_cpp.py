"""Offline whisper.cpp adapter for timestamped local ASR."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess  # nosec B404
import tempfile
import wave
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from capicove.processing.audio._adapters.ffmpeg import restricted_input_arguments
from capicove.processing.audio.models import (
    AudioBackendLineage,
    AudioFailureCode,
    AudioTranscript,
    AudioTranscriptionSegment,
    AudioWarningCode,
)
from capicove.processing.audio.transcription import AudioTranscriptionError


class WhisperCppTranscriber:
    """Run an explicit local whisper.cpp model on CPU without network access."""

    def __init__(
        self,
        model_path: Path,
        *,
        vad_model_path: Path,
        executable: str = "whisper-cli",
        vad_executable: str = "whisper-vad-speech-segments",
        ffmpeg_executable: str = "ffmpeg",
        backend_version: str = "unknown",
        timeout_seconds: float = 300.0,
        segment_language_detection: bool = True,
        restrict_inputs: bool = False,
        max_duration_seconds: int | None = None,
        expected_model_sha256: str | None = None,
        expected_vad_sha256: str | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not executable or not vad_executable or not ffmpeg_executable:
            raise ValueError("executable names must not be blank")
        if max_duration_seconds is not None and (
            type(max_duration_seconds) is not int or not 1 <= max_duration_seconds <= 7200
        ):
            raise ValueError("invalid decoded duration limit")
        self._max_duration_seconds = max_duration_seconds
        self._expected_model_sha256 = expected_model_sha256
        self._expected_vad_sha256 = expected_vad_sha256
        self._model_path = model_path.expanduser()
        self._vad_model_path = vad_model_path.expanduser()
        self._executable = executable
        self._vad_executable = vad_executable
        self._ffmpeg_executable = ffmpeg_executable
        self._backend_version = backend_version
        self._timeout_seconds = timeout_seconds
        self._segment_language_detection = segment_language_detection
        self._restrict_inputs = restrict_inputs

    def transcribe(
        self,
        source: Path,
        *,
        audio_stream_index: int = 0,
        language: str,
        deadline: datetime,
    ) -> AudioTranscript:
        """Normalize local evidence, run whisper.cpp, and validate its JSON."""
        _remaining_timeout(deadline, self._timeout_seconds)
        try:
            model = self._model_path.resolve(strict=True)
        except OSError as error:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_UNAVAILABLE,
                "local ASR model is unavailable",
            ) from error
        if not model.is_file():
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_UNAVAILABLE,
                "local ASR model is unavailable",
            )
        try:
            vad_model = self._vad_model_path.resolve(strict=True)
        except OSError as error:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_UNAVAILABLE,
                "local VAD model is unavailable",
            ) from error
        if not vad_model.is_file():
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_UNAVAILABLE,
                "local VAD model is unavailable",
            )
        try:
            model_hash = _hash_file(model, deadline=deadline)
            vad_model_hash = _hash_file(vad_model, deadline=deadline)
        except OSError as error:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_UNAVAILABLE,
                "local ASR or VAD model is unavailable",
            ) from error
        if (
            self._expected_model_sha256 is not None and model_hash != self._expected_model_sha256
        ) or (
            self._expected_vad_sha256 is not None and vad_model_hash != self._expected_vad_sha256
        ):
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_UNAVAILABLE, "audio model digest mismatch"
            )
        lineage = AudioBackendLineage(
            backend="whisper.cpp",
            backend_version=self._backend_version,
            model=model.name,
            model_sha256=model_hash,
            device="cpu",
            precision="model-defined",
            vad_model=vad_model.name,
            vad_model_sha256=vad_model_hash,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="capicove-audio-asr-") as temporary:
                work = Path(temporary)
                normalized = work / "input.wav"
                self._normalize(
                    source,
                    normalized,
                    audio_stream_index,
                    _remaining_timeout(deadline, self._timeout_seconds),
                )
                _require_deadline(deadline)
                duration_seconds = _wav_duration_seconds(normalized, deadline)
                if (
                    self._max_duration_seconds is not None
                    and duration_seconds > self._max_duration_seconds
                ):
                    raise AudioTranscriptionError(
                        AudioFailureCode.INVALID_AUDIO, "decoded audio exceeds duration limit"
                    )
                if language == "auto" and self._segment_language_detection:
                    vad_segments = self._detect_vad_segments(
                        normalized,
                        vad_model,
                        _remaining_timeout(deadline, self._timeout_seconds),
                        deadline,
                    )
                    return self._transcribe_vad_segments(
                        normalized,
                        work,
                        model,
                        vad_model,
                        vad_segments,
                        lineage,
                        duration_seconds,
                        deadline,
                    )
                payload, encoding_repaired = self._run_whisper(
                    normalized,
                    work / "transcript",
                    model,
                    vad_model,
                    language,
                    _remaining_timeout(deadline, self._timeout_seconds),
                    deadline,
                    use_vad=True,
                )
                _require_unchanged_model(model, model_hash, deadline)
                _require_unchanged_model(vad_model, vad_model_hash, deadline)
        except FileNotFoundError as error:
            raise AudioTranscriptionError(
                AudioFailureCode.DECODER_UNAVAILABLE,
                "local audio transcription executable is unavailable",
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AudioTranscriptionError(
                AudioFailureCode.TIMEOUT,
                "audio transcription timed out",
            ) from error
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_LOAD_FAILED,
                "local ASR returned an invalid result",
            ) from error
        try:
            return _parse_transcript(
                payload,
                lineage,
                duration_seconds=duration_seconds,
                encoding_repaired=encoding_repaired,
                input_duration_seconds=duration_seconds,
                deadline=deadline,
            )
        except AudioTranscriptionError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_LOAD_FAILED,
                "local ASR returned an invalid result",
            ) from error

    def _normalize(
        self,
        source: Path,
        output: Path,
        audio_stream_index: int,
        timeout: float,
    ) -> None:
        completed = subprocess.run(  # nosec B603
            [
                self._ffmpeg_executable,
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                *(restricted_input_arguments(source) if self._restrict_inputs else []),
                "-i",
                str(source),
                "-map",
                f"0:a:{audio_stream_index}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                *(
                    ["-t", str(self._max_duration_seconds + 1)]
                    if self._max_duration_seconds is not None
                    else []
                ),
                str(output),
            ],
            check=False,
            capture_output=True,
            timeout=timeout,
            start_new_session=True,
            shell=False,
        )
        if completed.returncode != 0 or not output.is_file():
            raise AudioTranscriptionError(
                AudioFailureCode.INVALID_AUDIO,
                "audio could not be normalized for transcription",
            )

    def _run_whisper(
        self,
        source: Path,
        output_prefix: Path,
        model: Path,
        vad_model: Path,
        language: str,
        timeout: float,
        deadline: datetime,
        *,
        use_vad: bool,
    ) -> tuple[object, bool]:
        backend_language = language.split("-", maxsplit=1)[0]
        vad_arguments = ["--vad", "--vad-model", str(vad_model)] if use_vad else []
        completed = subprocess.run(  # nosec B603
            [
                self._executable,
                "--model",
                str(model),
                "--file",
                str(source),
                "--language",
                backend_language,
                *vad_arguments,
                "--output-json",
                "--output-file",
                str(output_prefix),
                "--no-prints",
                "--no-gpu",
            ],
            check=False,
            capture_output=True,
            timeout=timeout,
            start_new_session=True,
            shell=False,
        )
        if completed.returncode != 0:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_LOAD_FAILED,
                "local ASR could not complete transcription",
            )
        _require_deadline(deadline)
        try:
            raw_json = output_prefix.with_suffix(".json").read_bytes()
        except FileNotFoundError as error:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_LOAD_FAILED,
                "local ASR did not produce the expected result",
            ) from error
        _require_deadline(deadline)
        encoding_repaired = False
        try:
            decoded = raw_json.decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw_json.decode("utf-8", errors="replace")
            encoding_repaired = True
        payload = json.loads(decoded)
        _require_deadline(deadline)
        return payload, encoding_repaired

    def _detect_vad_segments(
        self,
        source: Path,
        vad_model: Path,
        timeout: float,
        deadline: datetime,
    ) -> tuple[tuple[float, float], ...]:
        completed = subprocess.run(  # nosec B603
            [
                self._vad_executable,
                "--file",
                str(source),
                "--vad-model",
                str(vad_model),
                "--no-prints",
            ],
            check=False,
            capture_output=True,
            timeout=timeout,
            start_new_session=True,
            shell=False,
        )
        if completed.returncode != 0:
            raise AudioTranscriptionError(
                AudioFailureCode.MODEL_LOAD_FAILED,
                "local VAD could not detect speech segments",
            )
        _require_deadline(deadline)
        return _parse_vad_segments(completed.stdout, deadline=deadline)

    def _transcribe_vad_segments(
        self,
        source: Path,
        work: Path,
        model: Path,
        vad_model: Path,
        vad_segments: tuple[tuple[float, float], ...],
        lineage: AudioBackendLineage,
        input_duration_seconds: float,
        deadline: datetime,
    ) -> AudioTranscript:
        segments: list[AudioTranscriptionSegment] = []
        warnings: list[AudioWarningCode] = []
        language_durations: dict[str, float] = {}
        for group_index, (group_start, detected_group_end) in enumerate(vad_segments):
            group_end = min(detected_group_end, input_duration_seconds)
            if detected_group_end > input_duration_seconds:
                _append_warning(warnings, AudioWarningCode.TIMESTAMP_CLAMPED)
            if group_start >= group_end:
                _append_warning(warnings, AudioWarningCode.TIMESTAMP_CLAMPED)
                continue
            segment_audio = work / f"speech-{group_index:03d}.wav"
            _require_deadline(deadline)
            _slice_wav(source, segment_audio, group_start, group_end, deadline)
            payload, encoding_repaired = self._run_whisper(
                segment_audio,
                work / f"transcript-{group_index:03d}",
                model,
                vad_model,
                "auto",
                _remaining_timeout(deadline, self._timeout_seconds),
                deadline,
                use_vad=False,
            )
            partial = _parse_transcript(
                payload,
                lineage,
                duration_seconds=group_end - group_start,
                encoding_repaired=encoding_repaired,
                deadline=deadline,
            )
            if partial.segments:
                recognized_duration = sum(
                    segment.end_seconds - segment.start_seconds for segment in partial.segments
                )
                language_durations[partial.language] = (
                    language_durations.get(partial.language, 0.0) + recognized_duration
                )
            for warning in partial.warnings:
                if warning is not AudioWarningCode.NO_SPEECH_DETECTED:
                    _append_warning(warnings, warning)
            for partial_segment in partial.segments:
                text = partial_segment.text
                if segments and _needs_word_separator(segments[-1].text, text):
                    text = f" {text}"
                segments.append(
                    AudioTranscriptionSegment(
                        index=len(segments),
                        start_seconds=round(
                            group_start + partial_segment.start_seconds,
                            3,
                        ),
                        end_seconds=round(
                            group_start + partial_segment.end_seconds,
                            3,
                        ),
                        text=text,
                    )
                )
        _require_unchanged_model(model, lineage.model_sha256, deadline)
        if lineage.vad_model_sha256 is None:
            raise ValueError("VAD lineage is required")
        _require_unchanged_model(vad_model, lineage.vad_model_sha256, deadline)
        transcript_text = "".join(segment.text for segment in segments)
        if not segments:
            _append_warning(warnings, AudioWarningCode.NO_SPEECH_DETECTED)
        elif _has_excessive_repetition(transcript_text, deadline):
            _append_warning(warnings, AudioWarningCode.POTENTIAL_HALLUCINATION)
        substantial_languages = [
            detected_language
            for detected_language, duration in language_durations.items()
            if duration >= 1.0
        ]
        if not language_durations:
            language = "und"
        elif len(substantial_languages) >= 2:
            language = "multilingual"
        else:
            language = max(language_durations, key=language_durations.__getitem__)
        return AudioTranscript(
            text=transcript_text,
            language=language,
            segments=tuple(segments),
            warnings=tuple(warnings),
            lineage=lineage,
            input_duration_seconds=input_duration_seconds,
        )


_VAD_SEGMENT_PATTERN = re.compile(
    rb"^Speech segment \d+: start = ([0-9]+(?:\.[0-9]+)?), "
    rb"end = ([0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)

_VAD_SEGMENT_COUNT_PATTERN = re.compile(
    rb"^Detected ([0-9]+) speech segments:$",
    re.MULTILINE,
)


def _parse_vad_segments(
    output: bytes,
    *,
    deadline: datetime | None = None,
) -> tuple[tuple[float, float], ...]:
    summary = _VAD_SEGMENT_COUNT_PATTERN.search(output)
    if summary is None:
        raise ValueError("VAD output did not contain a valid segment summary")
    declared_count = int(summary.group(1))
    segments: list[tuple[float, float]] = []
    previous_end = 0.0
    for match in _VAD_SEGMENT_PATTERN.finditer(output):
        if deadline is not None:
            _require_deadline(deadline)
        start_seconds = float(match.group(1)) / 100
        end_seconds = float(match.group(2)) / 100
        if start_seconds < previous_end or end_seconds <= start_seconds:
            raise ValueError("VAD segments must be ordered and non-overlapping")
        segments.append((start_seconds, end_seconds))
        previous_end = end_seconds
    if len(segments) != declared_count:
        raise ValueError("VAD segment count did not match the declared summary")
    return tuple(segments)


def _slice_wav(
    source: Path,
    output: Path,
    start_seconds: float,
    end_seconds: float,
    deadline: datetime,
) -> None:
    _require_deadline(deadline)
    with wave.open(str(source), "rb") as input_stream:
        frame_rate = input_stream.getframerate()
        frame_count = input_stream.getnframes()
        start_frame = min(round(start_seconds * frame_rate), frame_count)
        end_frame = min(round(end_seconds * frame_rate), frame_count)
        if end_frame <= start_frame:
            raise ValueError("VAD segment is outside normalized audio duration")
        input_stream.setpos(start_frame)
        _require_deadline(deadline)
        frames = input_stream.readframes(end_frame - start_frame)
        _require_deadline(deadline)
        with wave.open(str(output), "wb") as output_stream:
            output_stream.setparams(input_stream.getparams())
            output_stream.writeframes(frames)
    _require_deadline(deadline)


def _needs_word_separator(previous: str, current: str) -> bool:
    if previous[-1:].isspace() or current[:1].isspace():
        return False
    previous_character = previous.rstrip()[-1]
    current_character = current.lstrip()[0]
    return (previous_character.isascii() and previous_character.isalnum()) or (
        current_character.isascii() and current_character.isalnum()
    )


def _parse_transcript(
    payload: object,
    lineage: AudioBackendLineage,
    *,
    duration_seconds: float,
    encoding_repaired: bool,
    input_duration_seconds: float | None = None,
    deadline: datetime | None = None,
) -> AudioTranscript:
    if deadline is not None:
        _require_deadline(deadline)
    root = _mapping(payload, "root")
    result = _mapping(root["result"], "result")
    language = _string(result["language"], "language")
    raw_segments = _sequence(root["transcription"], "transcription")
    segments: list[AudioTranscriptionSegment] = []
    warnings: list[AudioWarningCode] = []
    if encoding_repaired:
        warnings.append(AudioWarningCode.TRANSCRIPT_ENCODING_REPAIRED)
    for raw_segment in raw_segments:
        if deadline is not None:
            _require_deadline(deadline)
        segment = _mapping(raw_segment, "transcription segment")
        offsets = _mapping(segment["offsets"], "offsets")
        start_ms = _integer(offsets["from"], "offsets.from")
        end_ms = _integer(offsets["to"], "offsets.to")
        raw_text_value = segment["text"]
        if not isinstance(raw_text_value, str):
            raise ValueError("text must be a string")
        raw_text = raw_text_value
        stripped = raw_text.strip()
        if not stripped or _is_no_speech_marker(stripped):
            continue
        start_seconds = start_ms / 1000
        end_seconds = end_ms / 1000
        if end_seconds > duration_seconds:
            if end_seconds - duration_seconds > 0.05:
                _append_warning(warnings, AudioWarningCode.TIMESTAMP_CLAMPED)
            end_seconds = duration_seconds
        if start_seconds >= duration_seconds or end_seconds <= start_seconds:
            _append_warning(warnings, AudioWarningCode.TIMESTAMP_CLAMPED)
            continue
        text = f" {stripped}" if segments and raw_text[:1].isspace() else stripped
        if segments and segments[-1].text.endswith(" ") and text.startswith(" "):
            text = text.lstrip()
        segments.append(
            AudioTranscriptionSegment(
                len(segments),
                start_seconds,
                end_seconds,
                text,
            )
        )
    transcript_text = "".join(segment.text for segment in segments)
    if not segments:
        _append_warning(warnings, AudioWarningCode.NO_SPEECH_DETECTED)
    elif _has_excessive_repetition(transcript_text, deadline):
        _append_warning(warnings, AudioWarningCode.POTENTIAL_HALLUCINATION)
    if deadline is not None:
        _require_deadline(deadline)
    try:
        return AudioTranscript(
            text=transcript_text,
            language=language,
            segments=tuple(segments),
            warnings=tuple(warnings),
            lineage=lineage,
            input_duration_seconds=input_duration_seconds,
        )
    except ValueError as error:
        raise AudioTranscriptionError(
            AudioFailureCode.MODEL_LOAD_FAILED,
            "local ASR returned invalid transcript timing",
        ) from error


def _remaining_timeout(deadline: datetime, configured_timeout: float) -> float:
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise AudioTranscriptionError(
            AudioFailureCode.TIMEOUT,
            "audio transcription deadline has expired",
        )
    return min(configured_timeout, remaining)


def _require_deadline(deadline: datetime) -> None:
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    if deadline <= datetime.now(UTC):
        raise AudioTranscriptionError(
            AudioFailureCode.TIMEOUT,
            "audio transcription deadline has expired",
        )


def _hash_file(path: Path, *, deadline: datetime | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if deadline is not None:
                _require_deadline(deadline)
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_unchanged_model(
    path: Path,
    expected_sha256: str,
    deadline: datetime,
) -> None:
    try:
        actual_sha256 = _hash_file(path, deadline=deadline)
    except OSError as error:
        raise AudioTranscriptionError(
            AudioFailureCode.MODEL_LOAD_FAILED,
            "local ASR or VAD model changed during transcription",
        ) from error
    if actual_sha256 != expected_sha256:
        raise AudioTranscriptionError(
            AudioFailureCode.MODEL_LOAD_FAILED,
            "local ASR or VAD model changed during transcription",
        )


def _wav_duration_seconds(path: Path, deadline: datetime) -> float:
    _require_deadline(deadline)
    try:
        with wave.open(str(path), "rb") as stream:
            frame_rate = stream.getframerate()
            if frame_rate <= 0:
                raise ValueError("WAV frame rate must be positive")
            duration = stream.getnframes() / frame_rate
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError("normalized WAV metadata is invalid") from error
    _require_deadline(deadline)
    return duration


def _is_no_speech_marker(text: str) -> bool:
    normalized = text.upper().replace(" ", "_")
    return normalized in {"[BLANK_AUDIO]", "[NO_SPEECH]", "[SILENCE]"}


def _has_excessive_repetition(text: str, deadline: datetime | None = None) -> bool:
    if deadline is not None:
        _require_deadline(deadline)
    normalized = re.sub(r"[^\w\u3400-\u9fff]", "", text.casefold())
    if len(normalized) < 20:
        return False
    bigrams = [normalized[index : index + 2] for index in range(len(normalized) - 1)]
    most_frequent = max(Counter(bigrams).values())
    if deadline is not None:
        _require_deadline(deadline)
    return most_frequent / len(bigrams) >= 0.35


def _append_warning(
    warnings: list[AudioWarningCode],
    warning: AudioWarningCode,
) -> None:
    if warning not in warnings:
        warnings.append(warning)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be an array")
    return cast(Sequence[object], value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value
