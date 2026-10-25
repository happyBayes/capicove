"""Immutable public values for audio evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_positive_int(field_name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{field_name} must be positive")


def _require_sha256(field_name: str, value: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


class AudioFailureCode(StrEnum):
    """Classify stable audio intake failures without exposing private input."""

    INVALID_AUDIO = "INVALID_AUDIO"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    INVALID_AUDIO_STREAM_INDEX = "INVALID_AUDIO_STREAM_INDEX"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    AUDIO_STREAM_SELECTION_REQUIRED = "AUDIO_STREAM_SELECTION_REQUIRED"
    AUDIO_STREAM_INDEX_OUT_OF_RANGE = "AUDIO_STREAM_INDEX_OUT_OF_RANGE"
    SELECTED_AUDIO_STREAM_UNDECODABLE = "SELECTED_AUDIO_STREAM_UNDECODABLE"
    DECODER_UNAVAILABLE = "DECODER_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    SOURCE_CHANGED = "SOURCE_CHANGED"
    STORE_CORRUPTED = "STORE_CORRUPTED"
    STORE_FAILED = "STORE_FAILED"
    INVALID_RANGE = "INVALID_RANGE"
    UNSUPPORTED_AUDIO_PROFILE = "UNSUPPORTED_AUDIO_PROFILE"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    OUTPUT_EXISTS = "OUTPUT_EXISTS"
    OUTPUT_FAILED = "OUTPUT_FAILED"
    OUTPUT_CORRUPTED = "OUTPUT_CORRUPTED"


class AudioWarningCode(StrEnum):
    """Describe non-fatal audio transcription conditions."""

    NO_SPEECH_DETECTED = "NO_SPEECH_DETECTED"
    POTENTIAL_HALLUCINATION = "POTENTIAL_HALLUCINATION"
    TIMESTAMP_CLAMPED = "TIMESTAMP_CLAMPED"
    TRANSCRIPT_ENCODING_REPAIRED = "TRANSCRIPT_ENCODING_REPAIRED"
    UNSELECTED_AUDIO_STREAMS_IGNORED = "UNSELECTED_AUDIO_STREAMS_IGNORED"


@dataclass(frozen=True, slots=True)
class AudioEvidence:
    """Describe one preserved, content-addressed original audio file."""

    evidence_id: str
    sha256: str
    size_bytes: int
    media_type: str
    source_name: str
    stored_path: Path

    def __post_init__(self) -> None:
        _require_non_blank("evidence_id", self.evidence_id)
        _require_sha256("sha256", self.sha256)
        _require_positive_int("size_bytes", self.size_bytes)
        _require_non_blank("media_type", self.media_type)
        _require_non_blank("source_name", self.source_name)
        if self.evidence_id != f"sha256:{self.sha256}":
            raise ValueError("evidence_id must be derived from sha256")
        if not self.stored_path.is_absolute():
            raise ValueError("stored_path must be absolute")


@dataclass(frozen=True, slots=True)
class AudioStreamInventory:
    """Record the bounded number of audio streams observed before intake."""

    audio_stream_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.audio_stream_count, int)
            or isinstance(self.audio_stream_count, bool)
            or not 0 <= self.audio_stream_count <= 64
        ):
            raise ValueError("audio_stream_count must be an integer from zero through 64")


@dataclass(frozen=True, slots=True)
class AudioStreamSelection:
    """Bind one explicit audio ordinal to its preflight inventory."""

    selected_audio_stream_index: int
    selection_method: str
    preflight_audio_stream_count: int
    schema_version: int = 1
    index_semantics: str = "zero_based_audio_stream_ordinal"
    validation_scope: str = "selected_audio_stream_only"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if self.index_semantics != "zero_based_audio_stream_ordinal":
            raise ValueError("index_semantics is invalid")
        if self.validation_scope != "selected_audio_stream_only":
            raise ValueError("validation_scope is invalid")
        if self.selection_method not in {"explicit_index", "sole_stream_implicit"}:
            raise ValueError("selection_method is invalid")
        inventory = AudioStreamInventory(self.preflight_audio_stream_count)
        if inventory.audio_stream_count == 0:
            raise ValueError("a stream selection requires at least one audio stream")
        if (
            not isinstance(self.selected_audio_stream_index, int)
            or isinstance(self.selected_audio_stream_index, bool)
            or not 0 <= self.selected_audio_stream_index < inventory.audio_stream_count
        ):
            raise ValueError("selected_audio_stream_index is outside the inventory")
        if self.selection_method == "sole_stream_implicit" and (
            inventory.audio_stream_count != 1 or self.selected_audio_stream_index != 0
        ):
            raise ValueError("sole_stream_implicit requires exactly stream zero")


@dataclass(frozen=True, slots=True)
class AudioImportResult:
    """Bind preserved original evidence to the selected audio stream."""

    evidence: AudioEvidence
    stream_selection: AudioStreamSelection


@dataclass(frozen=True, slots=True)
class AudioBackendLineage:
    """Identify the exact local backend and model used for one transcript."""

    backend: str
    backend_version: str
    model: str
    model_sha256: str
    device: str
    precision: str
    vad_model: str | None = None
    vad_model_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("backend", "backend_version", "model", "device", "precision"):
            _require_non_blank(field_name, getattr(self, field_name))
        _require_sha256("model_sha256", self.model_sha256)
        if (self.vad_model is None) != (self.vad_model_sha256 is None):
            raise ValueError("vad_model and vad_model_sha256 must be provided together")
        if self.vad_model is not None and self.vad_model_sha256 is not None:
            _require_non_blank("vad_model", self.vad_model)
            _require_sha256("vad_model_sha256", self.vad_model_sha256)


@dataclass(frozen=True, slots=True)
class AudioTranscriptionSegment:
    """Represent one ordered, timestamped ASR segment."""

    index: int
    start_seconds: float
    end_seconds: float
    text: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("index must not be negative")
        if not isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("start_seconds must be finite and non-negative")
        if not isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be finite and greater than start_seconds")
        _require_non_blank("text", self.text)


@dataclass(frozen=True, slots=True)
class AudioTranscript:
    """Return complete text, ordered segments, warnings, and model lineage."""

    text: str
    language: str
    segments: tuple[AudioTranscriptionSegment, ...]
    warnings: tuple[AudioWarningCode, ...]
    lineage: AudioBackendLineage
    input_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_non_blank("language", self.language)
        _require_sequential_segments(self.segments)
        if self.text != "".join(segment.text for segment in self.segments):
            raise ValueError("text must equal concatenated segment text")
        if not self.segments and self.text:
            raise ValueError("text must be empty when segments are empty")
        if len(set(self.warnings)) != len(self.warnings):
            raise ValueError("warnings must not contain duplicates")
        if self.input_duration_seconds is not None and (
            not isfinite(self.input_duration_seconds) or self.input_duration_seconds <= 0
        ):
            raise ValueError("input_duration_seconds must be finite and positive")


@dataclass(frozen=True, slots=True)
class AudioTranscriptionResult:
    """Bind a transcript to its immutable original evidence."""

    evidence: AudioEvidence
    transcript: AudioTranscript
    stream_selection: AudioStreamSelection


def _require_sequential_segments(segments: Sequence[AudioTranscriptionSegment]) -> None:
    previous_end = 0.0
    for expected_index, segment in enumerate(segments):
        if segment.index != expected_index:
            raise ValueError("segment indexes must be consecutive from zero")
        if expected_index and segment.start_seconds < previous_end:
            raise ValueError("transcript segments must not overlap")
        previous_end = segment.end_seconds
