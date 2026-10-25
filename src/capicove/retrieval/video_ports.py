"""Provider-neutral ports for video visual indexes and generation sets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from capicove.contracts import EmbeddingSpaceRef, SearchableUnit, SearchFilters
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    PinnedVisualIndexGeneration,
    VideoGenerationSetEvent,
    VideoGenerationSetRecord,
    VisualCandidate,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    VisualIndexValidationReport,
)


@runtime_checkable
class VisualIndexWriter(Protocol):
    """Write entries to one building immutable visual generation."""

    provider_id: str

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[VisualIndexEntry, ...],
    ) -> None: ...


@runtime_checkable
class VisualCandidateRetriever(Protocol):
    """Search one explicitly pinned visual generation."""

    provider_id: str

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedVisualIndexGeneration,
        top_k: int,
    ) -> tuple[VisualCandidate, ...]: ...


@runtime_checkable
class VisualIndexGenerationLifecycle(Protocol):
    """Create, validate, and activate immutable visual generations."""

    provider_id: str

    def create(self, manifest: VisualIndexGenerationManifest) -> str: ...

    def validate(self, manifest: VisualIndexGenerationManifest) -> VisualIndexValidationReport: ...

    def activate(
        self,
        manifest: VisualIndexGenerationManifest,
        validation: VisualIndexValidationReport,
    ) -> PinnedVisualIndexGeneration: ...


@runtime_checkable
class VisualRetrieverBackend(Protocol):
    """Group one provider's independent visual retrieval roles."""

    provider_id: str
    index_writer: VisualIndexWriter
    candidate_retriever: VisualCandidateRetriever
    generation_lifecycle: VisualIndexGenerationLifecycle


@runtime_checkable
class VideoUnitCatalogWriter(Protocol):
    """Publish one immutable searchable-unit catalog."""

    def publish(
        self,
        *,
        units: tuple[SearchableUnit, ...],
        unit_manifest_sha256: str,
        keyframe_to_segments: Mapping[str, tuple[str, ...]],
        text_to_segments: Mapping[str, tuple[tuple[str, int], ...]],
    ) -> str: ...


@runtime_checkable
class VideoUnitCatalogReader(Protocol):
    """Read bounded units and precomputed temporal relations."""

    @property
    def unit_manifest_sha256(self) -> str: ...

    def get(self, unit_id: str) -> SearchableUnit | None: ...

    def get_many(self, unit_ids: tuple[str, ...]) -> tuple[SearchableUnit, ...]: ...

    def segments_for_keyframe(self, unit_id: str) -> tuple[str, ...]: ...

    def segments_for_keyframes(self, unit_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]: ...

    def segments_for_text(self, unit_id: str) -> tuple[tuple[str, int], ...]: ...

    def segments_for_texts(
        self, unit_ids: tuple[str, ...]
    ) -> dict[str, tuple[tuple[str, int], ...]]: ...


@runtime_checkable
class VideoGenerationSetCatalog(Protocol):
    """Persist complete generation sets, aliases, events, and leases."""

    def register(self, record: VideoGenerationSetRecord) -> VideoGenerationSetRecord: ...

    def get(self, set_id: str) -> VideoGenerationSetRecord | None: ...

    def record_activation(self, pin: PinnedVideoGenerationSet) -> VideoGenerationSetRecord: ...

    def active(
        self, *, workspace_ref: str, profile_ref: str
    ) -> VideoGenerationSetRecord | None: ...

    def history(self, set_id: str) -> tuple[VideoGenerationSetEvent, ...]: ...

    def acquire_active_lease(
        self, *, workspace_ref: str, profile_ref: str
    ) -> tuple[str, PinnedVideoGenerationSet]: ...

    def release_lease(self, lease_id: str) -> None: ...

    def lease_count(self, set_id: str) -> int: ...


@runtime_checkable
class VideoGenerationLease(Protocol):
    """Yield one pinned complete generation set for a bounded query lifetime."""

    def __enter__(self) -> PinnedVideoGenerationSet: ...

    def __exit__(self, *args: object) -> None: ...


__all__ = [
    "VideoGenerationLease",
    "VideoGenerationSetCatalog",
    "VideoUnitCatalogReader",
    "VideoUnitCatalogWriter",
    "VisualCandidateRetriever",
    "VisualIndexGenerationLifecycle",
    "VisualIndexWriter",
    "VisualRetrieverBackend",
]
