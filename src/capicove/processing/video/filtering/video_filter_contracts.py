"""Immutable contracts for unified local video filtering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

from capicove.processing.video.models import VideoSource


class Aggregation(StrEnum):
    """Combine boolean observations."""

    ALL = "all"
    ANY = "any"


class Reduction(StrEnum):
    """Reduce ordered frame scores to one video score."""

    AVG = "avg"
    MAX = "max"
    MIN = "min"


class SamplingMethod(StrEnum):
    """Select a bounded frame-sampling strategy."""

    UNIFORM = "uniform"
    ALL_KEYFRAMES = "all_keyframes"


class CriterionType(StrEnum):
    """Identify one supported video filter criterion."""

    METADATA = "metadata"
    MOTION = "motion"
    OCR_AREA = "ocr_area"
    AESTHETICS = "aesthetics"
    NSFW = "nsfw"
    WATERMARK = "watermark"
    TAGS = "tags"
    FRAME_TEXT_SIMILARITY = "frame_text_similarity"


class VideoFilterStatus(StrEnum):
    """Describe the capability-level outcome."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VideoFilterFailureCode(StrEnum):
    """Classify stable video filter failures."""

    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    SOURCE_CHANGED = "source_changed"
    UNSUPPORTED_MEDIA = "unsupported_media"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    MODEL_INCOMPATIBLE = "model_incompatible"
    DECODE_FAILED = "decode_failed"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


class VideoFilterWarningCode(StrEnum):
    """Classify non-fatal video filter diagnostics."""

    SAMPLE_LIMIT_REACHED = "sample_limit_reached"
    METADATA_ORIENTATION_APPLIED = "metadata_orientation_applied"
    MODEL_SCORE_NOT_CALIBRATED = "model_score_not_calibrated"
    DUPLICATE_SOURCE_REUSED = "duplicate_source_reused"


def _require_nonblank(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _require_finite(name: str, value: float, *, non_negative: bool = False) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if non_negative and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_range(
    minimum: float | int | None,
    maximum: float | int | None,
    *,
    name: str,
    non_negative: bool = False,
) -> None:
    for field_name, value in ((f"min_{name}", minimum), (f"max_{name}", maximum)):
        if value is not None:
            _require_finite(field_name, float(value), non_negative=non_negative)
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(f"{name} range minimum must not exceed maximum")


def _normalize_strings(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        _require_nonblank(name, value)
        lowered = value.strip().casefold()
        if lowered not in normalized:
            normalized.append(lowered)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class FrameSamplingConfig:
    """Configure deterministic bounded frame sampling."""

    method: SamplingMethod = SamplingMethod.UNIFORM
    frame_count: int = 3
    max_frames: int = 64

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.max_frames <= 0 or self.max_frames > 256:
            raise ValueError("max_frames must be between 1 and 256")


@dataclass(frozen=True, slots=True)
class LocalModelReference:
    """Identify an already-local model or locally cached model revision."""

    reference: str
    revision: str | None = None
    cache_directory: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank("model reference", self.reference)
        if "://" in self.reference:
            raise ValueError("model reference must identify a local path or local cache entry")
        if self.revision is not None:
            _require_nonblank("model revision", self.revision)
        if self.cache_directory is not None:
            _require_nonblank("model cache_directory", self.cache_directory)


@dataclass(frozen=True, slots=True)
class MetadataCriterion:
    """Filter normalized video metadata."""

    criterion_id: str
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    min_display_width: int | None = None
    max_display_width: int | None = None
    min_display_height: int | None = None
    max_display_height: int | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.METADATA)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        ranges = (
            (self.min_duration_seconds, self.max_duration_seconds, "duration_seconds"),
            (self.min_width, self.max_width, "width"),
            (self.min_height, self.max_height, "height"),
            (self.min_display_width, self.max_display_width, "display_width"),
            (self.min_display_height, self.max_display_height, "display_height"),
            (self.min_aspect_ratio, self.max_aspect_ratio, "aspect_ratio"),
        )
        if all(minimum is None and maximum is None for minimum, maximum, _ in ranges):
            raise ValueError("metadata criterion requires at least one configured range")
        for minimum, maximum, name in ranges:
            _validate_range(minimum, maximum, name=name, non_negative=True)


@dataclass(frozen=True, slots=True)
class MotionCriterion:
    """Filter mean dense optical-flow magnitude."""

    criterion_id: str
    sampling: FrameSamplingConfig = FrameSamplingConfig()
    min_score: float | None = 0.0
    max_score: float | None = None
    sampling_fps: float = 2.0
    resize: tuple[int, int] | None = None
    relative: bool = False
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.MOTION)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        _validate_range(self.min_score, self.max_score, name="score", non_negative=True)
        _require_finite("sampling_fps", self.sampling_fps)
        if self.sampling_fps <= 0:
            raise ValueError("sampling_fps must be positive")
        if self.resize is not None and (len(self.resize) != 2 or min(self.resize) <= 0):
            raise ValueError("resize must contain positive width and height")


@dataclass(frozen=True, slots=True)
class OcrAreaCriterion:
    """Filter mean detected text-area ratio."""

    criterion_id: str
    sampling: FrameSamplingConfig = FrameSamplingConfig()
    min_area_ratio: float | None = 0.0
    max_area_ratio: float | None = 1.0
    languages: tuple[str, ...] = ("ch_sim", "en")
    reduction: Reduction = Reduction.AVG
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.OCR_AREA)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        _validate_range(self.min_area_ratio, self.max_area_ratio, name="area_ratio")
        if self.min_area_ratio is not None and self.min_area_ratio < 0:
            raise ValueError("area ratio must be within [0, 1]")
        if self.max_area_ratio is not None and self.max_area_ratio > 1:
            raise ValueError("area ratio must be within [0, 1]")
        object.__setattr__(self, "languages", _normalize_strings("language", self.languages))
        if not self.languages:
            raise ValueError("languages must not be empty")


@dataclass(frozen=True, slots=True)
class AestheticsCriterion:
    """Filter local model aesthetics scores."""

    criterion_id: str
    model: LocalModelReference
    sampling: FrameSamplingConfig
    min_score: float | None = 0.4
    max_score: float | None = 1.0
    reduction: Reduction = Reduction.AVG
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.AESTHETICS)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        _validate_range(self.min_score, self.max_score, name="score")


@dataclass(frozen=True, slots=True)
class NsfwCriterion:
    """Filter local model NSFW probabilities."""

    criterion_id: str
    model: LocalModelReference
    sampling: FrameSamplingConfig
    target_label: str | None = None
    min_score: float | None = 0.0
    max_score: float | None = 0.5
    reduction: Reduction = Reduction.MAX
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.NSFW)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        if self.target_label is not None:
            _require_nonblank("target_label", self.target_label)
        _validate_range(self.min_score, self.max_score, name="score")


@dataclass(frozen=True, slots=True)
class WatermarkCriterion:
    """Filter local model watermark probabilities."""

    criterion_id: str
    model: LocalModelReference
    sampling: FrameSamplingConfig
    target_label: str | None = None
    max_probability: float = 0.8
    reduction: Reduction = Reduction.MAX
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.WATERMARK)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        if self.target_label is not None:
            _require_nonblank("target_label", self.target_label)
        _require_finite("max_probability", self.max_probability)
        if not 0 <= self.max_probability <= 1:
            raise ValueError("max_probability must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class TagCriterion:
    """Filter visual tags emitted by a local VideoTagger."""

    criterion_id: str
    model: LocalModelReference
    sampling: FrameSamplingConfig
    tags: tuple[str, ...]
    vocabulary_path: str | None = None
    contain: Aggregation = Aggregation.ANY
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(init=False, default=CriterionType.TAGS)

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        object.__setattr__(self, "tags", _normalize_strings("tag", self.tags))
        if not self.tags:
            raise ValueError("tags must not be empty")
        if self.vocabulary_path is not None:
            _require_nonblank("vocabulary_path", self.vocabulary_path)


@dataclass(frozen=True, slots=True)
class FrameTextSimilarityCriterion:
    """Filter local CLIP-compatible frame-text similarity."""

    criterion_id: str
    model: LocalModelReference
    sampling: FrameSamplingConfig
    min_score: float | None = 0.1
    max_score: float | None = 1.0
    reduction: Reduction = Reduction.AVG
    videos_aggregation: Aggregation = Aggregation.ALL
    criterion_type: CriterionType = field(
        init=False,
        default=CriterionType.FRAME_TEXT_SIMILARITY,
    )

    def __post_init__(self) -> None:
        _require_nonblank("criterion_id", self.criterion_id)
        _validate_range(self.min_score, self.max_score, name="score")


VideoFilterCriterion: TypeAlias = (
    MetadataCriterion
    | MotionCriterion
    | OcrAreaCriterion
    | AestheticsCriterion
    | NsfwCriterion
    | WatermarkCriterion
    | TagCriterion
    | FrameTextSimilarityCriterion
)


@dataclass(frozen=True, slots=True)
class VideoFilterItem:
    """Associate one immutable video source with optional descriptive text."""

    item_id: str
    source: VideoSource
    text: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank("item_id", self.item_id)
        if self.text is not None:
            _require_nonblank("text", self.text)


@dataclass(frozen=True, slots=True)
class VideoFilterRequest:
    """Request one transport-neutral multi-criterion video evaluation."""

    request_id: str
    trace_id: str
    attempt_id: str
    operation_id: str
    execution_policy_ref: str
    items: tuple[VideoFilterItem, ...]
    criteria: tuple[VideoFilterCriterion, ...]
    deadline: datetime
    aggregation: Aggregation = Aggregation.ALL

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "trace_id",
            "attempt_id",
            "operation_id",
            "execution_policy_ref",
        ):
            _require_nonblank(name, getattr(self, name))
        if not self.items:
            raise ValueError("items must contain at least one item")
        if not self.criteria:
            raise ValueError("criteria must contain at least one criterion")
        item_ids = tuple(item.item_id for item in self.items)
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item_id values must be unique")
        criterion_ids = tuple(criterion.criterion_id for criterion in self.criteria)
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("criterion_id values must be unique")
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        if any(
            criterion.criterion_type is CriterionType.FRAME_TEXT_SIMILARITY
            for criterion in self.criteria
        ) and any(item.text is None for item in self.items):
            raise ValueError("text is required for frame-text similarity")


@dataclass(frozen=True, slots=True)
class VideoFilterMetadata:
    """Describe normalized media metadata used by filter criteria."""

    duration_seconds: float
    encoded_width: int
    encoded_height: int
    display_width: int
    display_height: int
    rotation_degrees: int = 0
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    backend: str = "unknown"
    backend_version: str | None = None

    def __post_init__(self) -> None:
        _require_finite("duration_seconds", self.duration_seconds, non_negative=True)
        for name in ("encoded_width", "encoded_height", "display_width", "display_height"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        _require_nonblank("backend", self.backend)

    @property
    def aspect_ratio(self) -> float:
        """Return display width divided by display height."""
        return self.display_width / self.display_height


@dataclass(frozen=True, slots=True)
class FrameMetric:
    """Record one bounded frame-level numeric observation."""

    frame_index: int
    timestamp_seconds: float
    value: float
    raw_value: float | None = None


@dataclass(frozen=True, slots=True)
class VideoCriterionResult:
    """Record one criterion's completed result for one requested item."""

    item_id: str
    criterion_id: str
    passed: bool
    score: float | None = None
    frame_metrics: tuple[FrameMetric, ...] = ()
    tags: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """Record complete video aggregation for one criterion."""

    criterion_id: str
    criterion_type: CriterionType
    video_results: tuple[VideoCriterionResult, ...]
    aggregation: Aggregation
    passed: bool


@dataclass(frozen=True, slots=True)
class VideoFilterItemResult:
    """Record accepted-source and metadata diagnostics for one item."""

    item_id: str
    source_id: str
    source_sha256: str
    evidence_ref: str
    metadata: VideoFilterMetadata | None = None


@dataclass(frozen=True, slots=True)
class VideoFilterWarning:
    """Describe one non-fatal filter diagnostic."""

    code: VideoFilterWarningCode
    message: str
    item_id: str | None = None
    criterion_id: str | None = None


@dataclass(frozen=True, slots=True)
class VideoFilterFailure:
    """Describe one stable capability failure."""

    code: VideoFilterFailureCode
    message: str
    item_id: str | None = None
    criterion_id: str | None = None


@dataclass(frozen=True, slots=True)
class VideoFilterResult:
    """Return one evidence-aware unified video filter outcome."""

    status: VideoFilterStatus
    keep: bool | None
    item_results: tuple[VideoFilterItemResult, ...]
    criterion_results: tuple[CriterionResult, ...]
    warnings: tuple[VideoFilterWarning, ...]
    failures: tuple[VideoFilterFailure, ...]
    aggregation: Aggregation
    request_id: str
    trace_id: str
    attempt_id: str
    operation_id: str

    def __post_init__(self) -> None:
        if self.status is VideoFilterStatus.SUCCEEDED:
            if self.keep is None or self.failures:
                raise ValueError("successful result requires keep and no failures")
        elif self.keep is not None or not self.failures:
            raise ValueError("failed result requires keep=None and at least one failure")
