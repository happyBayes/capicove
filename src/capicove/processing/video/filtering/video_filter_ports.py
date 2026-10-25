"""Ports and typed capability errors for unified video filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    VideoCriterionResult,
    VideoFilterCriterion,
    VideoFilterFailureCode,
    VideoFilterItem,
    VideoFilterMetadata,
    VideoFilterWarning,
)
from capicove.processing.video.models import AcceptedVideoSource


class VideoFilterCapabilityError(RuntimeError):
    """Base class for a stable video-filter capability failure."""

    failure_code = VideoFilterFailureCode.INTERNAL


class InvalidVideoFilterInputError(VideoFilterCapabilityError):
    """Report invalid evaluator input."""

    failure_code = VideoFilterFailureCode.INVALID_INPUT


class PermissionDeniedFilterError(VideoFilterCapabilityError):
    """Report a policy-denied source or resource."""

    failure_code = VideoFilterFailureCode.PERMISSION_DENIED


class SourceChangedFilterError(VideoFilterCapabilityError):
    """Report a source hash mismatch."""

    failure_code = VideoFilterFailureCode.SOURCE_CHANGED


class UnsupportedMediaFilterError(VideoFilterCapabilityError):
    """Report unsupported or invalid media."""

    failure_code = VideoFilterFailureCode.UNSUPPORTED_MEDIA


class BackendUnavailableFilterError(VideoFilterCapabilityError):
    """Report a missing local executable or optional dependency."""

    failure_code = VideoFilterFailureCode.BACKEND_UNAVAILABLE


class ModelIncompatibleFilterError(VideoFilterCapabilityError):
    """Report an incompatible local model contract."""

    failure_code = VideoFilterFailureCode.MODEL_INCOMPATIBLE


class DecodeFailedFilterError(VideoFilterCapabilityError):
    """Report a media decode failure."""

    failure_code = VideoFilterFailureCode.DECODE_FAILED


class VideoFilterTimeoutError(VideoFilterCapabilityError):
    """Report an expired invocation deadline."""

    failure_code = VideoFilterFailureCode.TIMEOUT


@dataclass(frozen=True, slots=True)
class SampledFrame:
    """Carry one private adapter frame payload with canonical coordinates."""

    index: int
    timestamp_seconds: float
    width: int
    height: int
    payload: object
    key_frame: bool = False


@dataclass(frozen=True, slots=True)
class FrameSampleBatch:
    """Return bounded ordered frames plus sampling warnings."""

    frames: tuple[SampledFrame, ...]
    warnings: tuple[VideoFilterWarning, ...] = ()


class VideoMetadataProbe(Protocol):
    """Probe normalized metadata without exposing backend-specific types."""

    @property
    def identity(self) -> tuple[str, str | None]: ...

    def probe(
        self,
        source: AcceptedVideoSource,
        deadline: datetime,
    ) -> VideoFilterMetadata: ...


class VideoFrameSampler(Protocol):
    """Sample bounded chronological frames from accepted evidence."""

    @property
    def identity(self) -> tuple[str, str | None]: ...

    def sample(
        self,
        source: AcceptedVideoSource,
        config: FrameSamplingConfig,
        deadline: datetime,
    ) -> FrameSampleBatch: ...


class VideoCriterionEvaluator(Protocol):
    """Evaluate one configured criterion for one accepted item."""

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult: ...


@dataclass(frozen=True, slots=True)
class FrameTags:
    """Return normalized tags for one frame."""

    tags: tuple[str, ...]
    confidences: tuple[float, ...] = ()


class VideoTagger(Protocol):
    """Generate visual tags without exposing provider model types."""

    @property
    def identity(self) -> tuple[str, str | None]: ...

    def tag(
        self, frames: tuple[SampledFrame, ...], deadline: datetime
    ) -> tuple[FrameTags, ...]: ...
