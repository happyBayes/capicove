"""Immutable values for offline video representations."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from capicove.contracts.retrieval import (
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    SearchableUnit,
    TemporalRange,
)
from capicove.processing.video.models import VideoSource


def nonblank(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reference must be non-empty")


def sha256(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA-256")


def timestamp(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("timestamp must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """A bounded decoded preview candidate and quality measurement."""

    timestamp_ms: int
    quality_score: float
    preview_candidate_ref: str
    black_frame: bool = False
    corrupt: bool = False

    def __post_init__(self) -> None:
        timestamp(self.timestamp_ms)
        if not isfinite(self.quality_score) or not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be finite in [0, 1]")
        nonblank(self.preview_candidate_ref)


@dataclass(frozen=True, slots=True)
class SceneMetric:
    """One normalized adjacent-frame scene difference."""

    timestamp_ms: int
    score: float

    def __post_init__(self) -> None:
        timestamp(self.timestamp_ms)
        if not isfinite(self.score) or self.score < 0:
            raise ValueError("scene score must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class VideoDecodeObservation:
    """Reusable output of a single video decode pass."""

    duration_ms: int
    frames: tuple[FrameObservation, ...]
    scene_metrics: tuple[SceneMetric, ...]

    def __post_init__(self) -> None:
        timestamp(self.duration_ms)
        if not self.duration_ms:
            raise ValueError("duration must be positive")
        times = [f.timestamp_ms for f in self.frames]
        if times != sorted(set(times)) or any(t >= self.duration_ms for t in times):
            raise ValueError("frames must be unique, ordered and within duration")


@dataclass(frozen=True, slots=True)
class RetrievalKeyframe:
    """Semantic representative with verified preview identity."""

    unit_id: str
    timestamp_ms: int
    scene_range: TemporalRange
    preview_resource_ref: str
    preview_sha256: str
    quality_score: float
    processing_fingerprint: ProcessingFingerprint

    def __post_init__(self) -> None:
        nonblank(self.unit_id)
        nonblank(self.preview_resource_ref)
        sha256(self.preview_sha256)
        timestamp(self.timestamp_ms)
        if not self.scene_range.start_ms <= self.timestamp_ms < self.scene_range.end_ms:
            raise ValueError("keyframe must be inside scene")
        if not isfinite(self.quality_score) or not 0 < self.quality_score <= 1:
            raise ValueError("quality_score must be finite in (0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalOcrText:
    """Recognized preview text with temporal evidence."""

    text: str
    time_range: TemporalRange
    evidence_ref: str
    representation_ref: str

    def __post_init__(self) -> None:
        for value in (self.text, self.evidence_ref, self.representation_ref):
            nonblank(value)


@dataclass(frozen=True, slots=True)
class CandidateVideoSegment:
    """A metadata interval, never a rendered clip."""

    unit_id: str
    time_range: TemporalRange
    preview_resource_ref: str
    keyframe_unit_ids: tuple[str, ...]
    asr_unit_ids: tuple[str, ...]
    ocr_unit_ids: tuple[str, ...]
    processing_fingerprint: ProcessingFingerprint

    def __post_init__(self) -> None:
        nonblank(self.unit_id)
        nonblank(self.preview_resource_ref)
        if not self.keyframe_unit_ids:
            raise ValueError("keyframe_unit_ids must be non-empty")
        for name in ("keyframe_unit_ids", "asr_unit_ids", "ocr_unit_ids"):
            values = tuple(getattr(self, name))
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
            for value in values:
                nonblank(value)
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True)
class VideoRetrievalPreparationRequest:
    """Caller identities and an absolute offline deadline."""

    source: VideoSource
    artifact_ref: str
    evidence_ref: str
    asr_representation_ref: str
    processing_run_ref: str
    profile_ref: str
    processing_fingerprint: ProcessingFingerprint
    asr_processing_fingerprint: ProcessingFingerprint
    ocr_processing_fingerprint: ProcessingFingerprint
    deadline: datetime

    def __post_init__(self) -> None:
        for value in (
            self.artifact_ref,
            self.evidence_ref,
            self.asr_representation_ref,
            self.processing_run_ref,
            self.profile_ref,
        ):
            nonblank(value)
        if self.deadline.tzinfo is None:
            raise ValueError("deadline must be timezone aware")


@dataclass(frozen=True, slots=True)
class VideoRetrievalPreparationResult:
    """Frozen output consumed by offline index publication."""

    source_hash: str
    keyframes: tuple[RetrievalKeyframe, ...]
    segments: tuple[CandidateVideoSegment, ...]
    searchable_units: tuple[SearchableUnit, ...]
    keyframe_vectors: tuple[tuple[float, ...], ...]
    segment_vectors: tuple[tuple[float, ...], ...]
    embedding_space: EmbeddingSpaceRef
    unit_manifest_sha256: str
    representation_refs: tuple[str, ...]
    warning_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        sha256(self.source_hash)
        sha256(self.unit_manifest_sha256)
        ids = [u.unit_id for u in self.searchable_units]
        if len(ids) != len(set(ids)):
            raise ValueError("unit IDs must be unique")
        for units, vectors in (
            (self.keyframes, self.keyframe_vectors),
            (self.segments, self.segment_vectors),
        ):
            if len(units) != len(vectors):
                raise ValueError("vector count mismatch")
            for vector in vectors:
                if len(vector) != self.embedding_space.dimension or any(
                    not isfinite(v) for v in vector
                ):
                    raise ValueError("vector dimension or finite-value mismatch")
