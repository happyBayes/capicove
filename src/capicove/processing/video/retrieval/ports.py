"""Processing-owned ports; no provider SDK types cross this boundary."""

from datetime import datetime
from typing import Protocol, runtime_checkable

from capicove.contracts.retrieval import (
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    SearchableUnit,
    TemporalRange,
)
from capicove.processing.audio.models import AudioTranscript
from capicove.processing.video.models import AcceptedVideoSource, VideoSource
from capicove.processing.video.retrieval.models import (
    CandidateVideoSegment,
    FrameObservation,
    RetrievalKeyframe,
    SceneMetric,
    TemporalOcrText,
    VideoDecodeObservation,
)


@runtime_checkable
class KeyframeExtractor(Protocol):
    """Produce persisted semantic representatives from an existing decode pass."""

    def extract(
        self,
        scenes: tuple[TemporalRange, ...],
        frames: tuple[FrameObservation, ...],
        *,
        deadline: datetime,
    ) -> tuple[RetrievalKeyframe, ...]: ...


@runtime_checkable
class TemporalWindowExtractor(Protocol):
    """Build immutable metadata intervals without materializing video clips."""

    def build(
        self,
        *,
        source_hash: str,
        duration_ms: int,
        scenes: tuple[TemporalRange, ...],
        keyframes: tuple[RetrievalKeyframe, ...],
        asr_units: tuple[SearchableUnit, ...],
        ocr_units: tuple[SearchableUnit, ...],
        processing_fingerprint: ProcessingFingerprint,
    ) -> tuple[CandidateVideoSegment, ...]: ...


@runtime_checkable
class SourceAcceptor(Protocol):
    """Authorize a source through the existing video source boundary."""

    def accept(self, source: VideoSource, *, deadline: datetime) -> AcceptedVideoSource: ...


@runtime_checkable
class OfflineVideoDecodePass(Protocol):
    """Decode bounded observations exactly once per compatible source."""

    @property
    def fingerprint(self) -> ProcessingFingerprint: ...
    def decode(
        self, source: AcceptedVideoSource, *, deadline: datetime
    ) -> VideoDecodeObservation: ...


@runtime_checkable
class SceneBoundaryExtractor(Protocol):
    """Derive scenes from already decoded metrics."""

    @property
    def fingerprint(self) -> ProcessingFingerprint: ...
    def extract(
        self, metrics: tuple[SceneMetric, ...], *, duration_ms: int, deadline: datetime
    ) -> tuple[TemporalRange, ...]: ...


@runtime_checkable
class ASRProcessor(Protocol):
    """Reuse the public transcription service."""

    @property
    def fingerprint(self) -> ProcessingFingerprint: ...

    def transcribe(
        self, source: AcceptedVideoSource, *, deadline: datetime
    ) -> AudioTranscript | None: ...


@runtime_checkable
class OCRProcessor(Protocol):
    """Read only selected persisted previews."""

    @property
    def fingerprint(self) -> ProcessingFingerprint: ...

    def recognize(
        self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
    ) -> tuple[TemporalOcrText, ...]: ...


@runtime_checkable
class MediaRepresentationEncoder(Protocol):
    """Encode local previews into one declared joint space."""

    @property
    def embedding_space(self) -> EmbeddingSpaceRef: ...
    def encode_keyframes(
        self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]: ...


@runtime_checkable
class PreviewStore(Protocol):
    """Content-addressed bounded representation storage."""

    def put(self, payload: bytes) -> str: ...
    def read(self, reference: str) -> bytes: ...


@runtime_checkable
class ProcessingManifestStore(Protocol):
    """Publish canonical manifests without replacing previous evidence."""

    def put(self, payload: bytes) -> str: ...
    def read(self, reference: str) -> bytes: ...


@runtime_checkable
class DerivedCacheCatalog(Protocol):
    """Resolve immutable layer manifests by complete fingerprint."""

    def lookup(self, fingerprint: ProcessingFingerprint) -> str | None: ...
    def publish(self, fingerprint: ProcessingFingerprint, manifest_ref: str) -> None: ...
