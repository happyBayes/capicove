"""Immutable public values for video splitting."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SceneOption: TypeAlias = str | int | float | bool


def _require_nonblank(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def _require_sha256(name: str, value: str) -> None:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class VideoSplitStrategy(StrEnum):
    """Select the algorithm used to plan video ranges."""

    DURATION = "duration"
    KEYFRAME = "keyframe"
    SCENE = "scene"


class VideoOutputFormat(StrEnum):
    """Select how generated segments are represented."""

    PATH = "path"
    BYTES = "bytes"


class VideoBackend(StrEnum):
    """Select a concrete media backend or automatic resolution."""

    AUTO = "auto"
    FFMPEG = "ffmpeg"
    AV = "av"


class VideoWriteMode(StrEnum):
    """Select accurate transcoding or keyframe-aligned stream copying."""

    ACCURATE = "accurate"
    COPY = "copy"


class SceneDetectorName(StrEnum):
    """Name a supported scene detection algorithm."""

    CONTENT = "content"
    ADAPTIVE = "adaptive"
    THRESHOLD = "threshold"


class VideoSplitStatus(StrEnum):
    """Describe the capability-level outcome."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VideoSplitFailureCode(StrEnum):
    """Classify processing failures before Tool mapping."""

    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


class VideoSplitWarningCode(StrEnum):
    """Classify visible best-effort warnings."""

    SEGMENT_FAILED = "segment_failed"
    NO_KEYFRAMES = "no_keyframes"
    INVALID_TIMESTAMPS = "invalid_timestamps"
    SCENE_DETECTION_FAILED = "scene_detection_failed"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    OUTPUT_CLEANUP_FAILED = "output_cleanup_failed"
    FALLBACK_TO_ORIGINAL = "fallback_to_original"


@dataclass(frozen=True, slots=True)
class VideoSource:
    """Describe exactly one caller-provided video representation."""

    path: Path | None = None
    data: bytes | None = None
    artifact_ref: str | None = None
    artifact_sha256: str | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        representations = (
            self.path is not None,
            self.data is not None,
            self.artifact_ref is not None,
        )
        if sum(representations) != 1:
            raise ValueError("VideoSource requires exactly one representation")
        if self.artifact_ref is not None:
            _require_nonblank("artifact_ref", self.artifact_ref)
            if self.artifact_sha256 is None:
                raise ValueError("artifact_sha256 is required for an artifact source")
            _require_sha256("artifact_sha256", self.artifact_sha256)
        elif self.artifact_sha256 is not None:
            raise ValueError("artifact_sha256 is valid only with artifact_ref")
        if self.source_id is not None:
            _require_nonblank("source_id", self.source_id)


@dataclass(frozen=True, slots=True)
class AcceptedVideoSource:
    """Carry an authorized and content-verified media source."""

    source_id: str
    sha256: str
    mime_type: str
    path: Path | None = None
    data: bytes | None = None
    artifact_ref: str | None = None

    def __post_init__(self) -> None:
        _require_nonblank("source_id", self.source_id)
        _require_sha256("sha256", self.sha256)
        _require_nonblank("mime_type", self.mime_type)
        if (self.path is None) == (self.data is None):
            raise ValueError("AcceptedVideoSource requires exactly one materialized representation")


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Describe a non-empty half-open media time range in seconds."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("start_seconds must be finite and non-negative")
        if not math.isfinite(self.end_seconds) or self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be finite and greater than start_seconds")


@dataclass(frozen=True, slots=True)
class DurationSplitConfig:
    """Configure fixed-duration range planning."""

    split_duration: float
    overlap_duration: float = 0.0
    min_last_split_duration: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.split_duration) or self.split_duration <= 0:
            raise ValueError("split_duration must be finite and positive")
        if not math.isfinite(self.overlap_duration) or self.overlap_duration < 0:
            raise ValueError("overlap_duration must be finite and non-negative")
        if self.overlap_duration >= self.split_duration:
            raise ValueError("overlap_duration must be less than split_duration")
        if not math.isfinite(self.min_last_split_duration) or self.min_last_split_duration < 0:
            raise ValueError("min_last_split_duration must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class KeyframeSplitConfig:
    """Configure keyframe range planning."""

    min_segment_duration: float = 10.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_segment_duration) or self.min_segment_duration < 0:
            raise ValueError("min_segment_duration must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SceneSplitConfig:
    """Configure scene range detection."""

    detector: SceneDetectorName = SceneDetectorName.CONTENT
    options: Mapping[str, SceneOption] = field(default_factory=dict)

    def __post_init__(self) -> None:
        copied: dict[str, SceneOption] = {}
        for key, value in self.options.items():
            _require_nonblank("scene option", key)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"scene option {key!r} must be finite")
            copied[key] = value
        object.__setattr__(self, "options", MappingProxyType(copied))


SplitConfig: TypeAlias = DurationSplitConfig | KeyframeSplitConfig | SceneSplitConfig


@dataclass(frozen=True, slots=True)
class VideoSplitRequest:
    """Request one transport-neutral video split operation."""

    request_id: str
    trace_id: str
    attempt_id: str
    operation_id: str
    execution_policy_ref: str
    source: VideoSource
    strategy: VideoSplitStrategy
    config: SplitConfig
    output_format: VideoOutputFormat
    backend: VideoBackend
    deadline: datetime
    output_directory: Path | None = None
    write_mode: VideoWriteMode = VideoWriteMode.ACCURATE

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "trace_id",
            "attempt_id",
            "operation_id",
            "execution_policy_ref",
        ):
            _require_nonblank(name, getattr(self, name))
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("deadline must be timezone-aware")
        expected_type = {
            VideoSplitStrategy.DURATION: DurationSplitConfig,
            VideoSplitStrategy.KEYFRAME: KeyframeSplitConfig,
            VideoSplitStrategy.SCENE: SceneSplitConfig,
        }[self.strategy]
        if not isinstance(self.config, expected_type):
            raise ValueError("strategy must match its configuration type")
        if self.output_format is VideoOutputFormat.PATH and self.output_directory is None:
            raise ValueError("output_directory is required for path output")
        if self.output_format is VideoOutputFormat.BYTES and self.output_directory is not None:
            raise ValueError("output_directory is not valid for byte output")


@dataclass(frozen=True, slots=True)
class VideoAsset:
    """Describe preserved original evidence."""

    source_id: str
    sha256: str
    mime_type: str
    path: Path | None = None
    data: bytes | None = None
    artifact_ref: str | None = None


@dataclass(frozen=True, slots=True)
class VideoSegment:
    """Describe one successfully generated video segment and its lineage."""

    index: int
    time_range: TimeRange
    output_id: str
    sha256: str
    mime_type: str
    source_id: str
    strategy: VideoSplitStrategy
    operation_id: str
    configuration_hash: str
    backend: str
    backend_version: str | None
    path: Path | None = None
    data: bytes | None = None


@dataclass(frozen=True, slots=True)
class VideoSplitWarning:
    """Report a visible degraded or failed operation detail."""

    code: VideoSplitWarningCode
    message: str
    segment_index: int | None = None
    time_range: TimeRange | None = None


@dataclass(frozen=True, slots=True)
class VideoSplitResult:
    """Return original evidence, successful segments, and canonical diagnostics."""

    status: VideoSplitStatus
    original: VideoAsset
    segments: tuple[VideoSegment, ...]
    warnings: tuple[VideoSplitWarning, ...]
    degraded: bool
    strategy: VideoSplitStrategy
    normalized_config: Mapping[str, object]
    configuration_hash: str
    backend: str
    backend_version: str | None
    request_id: str
    trace_id: str
    attempt_id: str
    operation_id: str
    failure_code: VideoSplitFailureCode | None = None
    failure_message: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "normalized_config", MappingProxyType(dict(self.normalized_config))
        )
        if self.status is VideoSplitStatus.SUCCEEDED and self.failure_code is not None:
            raise ValueError("a successful split cannot contain a failure")
        if self.status is VideoSplitStatus.FAILED and self.failure_code is None:
            raise ValueError("a failed split requires a failure code")
