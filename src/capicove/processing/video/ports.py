"""Ports consumed by the video splitting capability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from capicove.processing.video.models import (
    AcceptedVideoSource,
    SceneSplitConfig,
    TimeRange,
    VideoBackend,
    VideoWriteMode,
)


class VideoProcessingError(RuntimeError):
    """Base class for typed video processing failures."""


class BackendUnavailableError(VideoProcessingError):
    """Report that a selected media backend is unavailable."""


class VideoInspectionError(VideoProcessingError):
    """Report invalid or unreadable video metadata."""


class ClipWriteError(VideoProcessingError):
    """Report a failure to write one planned clip."""


class SceneDetectionError(VideoProcessingError):
    """Report a scene detection failure."""


class OutputStoreError(VideoProcessingError):
    """Report output allocation, verification, or cleanup failure."""


class PathPolicyError(VideoProcessingError):
    """Report a path outside the trusted execution policy."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Return normalized backend-independent video metadata."""

    duration_seconds: float
    keyframe_timestamps: tuple[float, ...]
    backend: str
    backend_version: str | None = None


@dataclass(frozen=True, slots=True)
class PendingOutput:
    """Describe a store-owned incomplete output."""

    identity: str
    path: Path


@dataclass(frozen=True, slots=True)
class WrittenClip:
    """Describe bytes written by a clip backend."""

    path: Path
    sha256: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class StoredOutput:
    """Describe a verified finalized output."""

    identity: str
    path: Path
    sha256: str
    mime_type: str


class ArtifactResolver(Protocol):
    """Materialize an immutable Artifact for local processing."""

    def resolve(
        self,
        ref: str,
        expected_sha256: str,
        deadline: datetime,
    ) -> AcceptedVideoSource: ...


class PathAuthorizer(Protocol):
    """Authorize resolved filesystem paths under a trusted policy."""

    def authorize_read(self, policy_ref: str, path: Path) -> None: ...

    def authorize_write(self, policy_ref: str, path: Path) -> None: ...


class VideoInspector(Protocol):
    """Inspect source metadata without exposing backend-specific values."""

    def inspect(self, source: AcceptedVideoSource, deadline: datetime) -> VideoMetadata: ...


class ClipWriter(Protocol):
    """Write one requested range into a store-owned destination."""

    def write(
        self,
        source: AcceptedVideoSource,
        time_range: TimeRange,
        destination: PendingOutput,
        deadline: datetime,
    ) -> WrittenClip: ...


class SceneDetector(Protocol):
    """Detect scene ranges without writing output clips."""

    @property
    def identity(self) -> tuple[str, str | None]: ...

    def detect(
        self,
        source: AcceptedVideoSource,
        config: SceneSplitConfig,
        deadline: datetime,
    ) -> tuple[TimeRange, ...]: ...


class OutputStore(Protocol):
    """Allocate, verify, expose, and clean managed outputs."""

    def allocate(self, identity: str, suffix: str) -> PendingOutput: ...

    def finalize(self, pending: PendingOutput, written: WrittenClip) -> StoredOutput: ...

    def load_bytes(self, stored: StoredOutput) -> bytes: ...

    def discard(self, pending: PendingOutput) -> None: ...


@dataclass(frozen=True, slots=True)
class VideoBackendBundle:
    """Bind one concrete inspector and writer implementation."""

    name: str
    version: str | None
    inspector: VideoInspector
    writer: ClipWriter


class VideoBackendResolver(Protocol):
    """Resolve a requested backend without import-time probing."""

    def resolve(
        self,
        requested: VideoBackend,
        write_mode: VideoWriteMode,
    ) -> VideoBackendBundle: ...
