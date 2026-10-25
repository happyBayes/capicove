"""Ports consumed by production reverse-image retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, TypeVar, runtime_checkable

from PIL import Image

from capicove.contracts.retrieval import (
    EmbeddingSpaceRef,
    ExecutionProfile,
    ImageQuery,
    SearchableUnit,
    SearchFilters,
)
from capicove.retrieval.cursor import SearchCursorBinding, SearchCursorPosition
from capicove.retrieval.image_generations import GenerationSetManifest
from capicove.retrieval.search_models import (
    Candidate,
    CandidateLane,
    DeferredCleanupReservation,
    FusedCandidate,
    InvocationAuthorizationContext,
    LaneScheduleSnapshot,
    PerceptualHashQuery,
    QueryRepresentation,
    QueryVector,
    ResolvedImageQuery,
    RetrievalLane,
    ScheduledLaneJob,
)


_T = TypeVar("_T")


@runtime_checkable
class QueryEncoder(Protocol):
    """Encode one query in an exact, verified embedding space."""

    embedding_space: EmbeddingSpaceRef
    bundle_manifest_sha256: str

    def encode(
        self,
        image: Image.Image,
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> QueryVector: ...


@runtime_checkable
class QueryEncoderResolver(Protocol):
    """Resolve an exact lane, space, bundle, and execution-profile binding."""

    def resolve(
        self,
        lane: RetrievalLane,
        space: EmbeddingSpaceRef,
        target_bundle_manifest_sha256: str,
        execution_profile: ExecutionProfile,
    ) -> QueryEncoder: ...


@runtime_checkable
class PerceptualQueryEncoder(Protocol):
    """Encode one content-bound query pHash under an absolute deadline."""

    def encode(
        self,
        image: Image.Image,
        source_sha256: str,
        max_hamming_distance: int,
        deadline: datetime,
    ) -> PerceptualHashQuery: ...


@runtime_checkable
class ImageQueryResolver(Protocol):
    """Authorize and decode one query source exactly once."""

    def resolve(
        self,
        query: ImageQuery,
        *,
        authorization: InvocationAuthorizationContext,
        data_policy_ref: str,
        deadline: datetime,
    ) -> ResolvedImageQuery: ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Fuse bounded lane rankings without provider score coupling."""

    policy_id: str

    def fuse(
        self,
        lanes: tuple[CandidateLane, ...],
        *,
        limit: int,
    ) -> tuple[FusedCandidate, ...]: ...


@runtime_checkable
class OrdinalSelection(Protocol):
    """Expose a catalog-bound all-or-sparse ordinal selection."""

    @property
    def filter_sha256(self) -> str: ...

    @property
    def ordinals(self) -> tuple[int, ...] | None: ...


@runtime_checkable
class UnitCatalog(Protocol):
    """Read immutable searchable-unit metadata and exact-content matches."""

    @property
    def unit_count(self) -> int: ...

    def lookup_content_candidates(
        self,
        source_sha256: str,
        filters: SearchFilters,
        *,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]: ...

    def select_ordinals(
        self,
        filters: SearchFilters,
        *,
        deadline: datetime,
    ) -> OrdinalSelection: ...

    def get_many(
        self,
        unit_ids: tuple[str, ...],
        *,
        deadline: datetime,
    ) -> tuple[SearchableUnit, ...]: ...


@runtime_checkable
class IndexReader(Protocol):
    """Search one immutable, already generation-bound lane index."""

    def search(
        self,
        query: QueryRepresentation,
        selection: tuple[int, ...] | None,
        *,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]: ...


@runtime_checkable
class SearchGenerationLease(Protocol):
    """Expose one fixed complete generation set for one bounded search."""

    @property
    def manifest(self) -> GenerationSetManifest: ...

    @property
    def unit_catalog(self) -> UnitCatalog: ...

    @property
    def readers(self) -> Mapping[RetrievalLane, IndexReader]: ...

    @property
    def qualification_tier(self) -> str: ...

    def close(self) -> None: ...

    def __enter__(self) -> SearchGenerationLease: ...

    def __exit__(self, *_exc: object) -> None: ...


@runtime_checkable
class GenerationSetProvider(Protocol):
    """Acquire exactly one immutable active generation-set lease."""

    def acquire(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
        cleanup_reservation: DeferredCleanupReservation | None = None,
    ) -> SearchGenerationLease: ...


@runtime_checkable
class LaneScheduler(Protocol):
    """Run independent lane jobs within a runtime-owned resource budget."""

    @property
    def execution_policy(self) -> LaneScheduleSnapshot: ...

    def run(
        self,
        jobs: tuple[ScheduledLaneJob[_T], ...],
        *,
        deadline: datetime,
    ) -> tuple[_T, ...]: ...


@runtime_checkable
class CursorCodec(Protocol):
    """Encode and verify a cursor bound to an immutable search snapshot."""

    def encode(self, position: SearchCursorPosition) -> str: ...

    def decode(
        self,
        cursor: str,
        *,
        expected: SearchCursorBinding,
        now: datetime,
    ) -> SearchCursorPosition: ...


__all__ = [
    "CursorCodec",
    "FusionStrategy",
    "GenerationSetProvider",
    "ImageQueryResolver",
    "IndexReader",
    "LaneScheduler",
    "OrdinalSelection",
    "PerceptualQueryEncoder",
    "QueryEncoder",
    "QueryEncoderResolver",
    "SearchGenerationLease",
    "UnitCatalog",
]
