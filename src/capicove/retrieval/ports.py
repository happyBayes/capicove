"""Ports consumed by multimodal retrieval operations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from capicove.contracts import EmbeddingSpaceRef, SearchFilters, TextQuery
from capicove.retrieval.generations import (
    PinnedTextIndexGeneration,
    TextIndexGenerationEvent,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexValidationReport,
)
from capicove.retrieval.models import (
    ProviderMatch,
    TextCandidate,
    TextEmbedding,
    TextEvidence,
    TextIndexEntry,
)


@runtime_checkable
class TextBatchEncoder(Protocol):
    """Encode each text at the same tuple index, including duplicate inputs.

    Roles are exactly query/document. Implementations must return dense vectors
    in the requested complete space, reject overlong text without truncation,
    and preserve input count/order. Vectors carry no per-input correlation IDs;
    ordering is a provider obligation, qualified by adapter conformance tests.
    """

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]: ...


class MultimodalEmbeddingProvider(Protocol):
    """Provide aligned image/text vectors and model-native matching scores."""

    model_id: str
    model_revision: str
    device: str

    def rank(
        self,
        image: Image.Image,
        texts: tuple[str, ...],
    ) -> tuple[ProviderMatch, ...]: ...

    def embed_images(
        self,
        images: tuple[Image.Image, ...],
    ) -> tuple[tuple[float, ...], ...]: ...

    def embed_texts(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]: ...


@runtime_checkable
class TextIndexWriter(Protocol):
    """Append canonical text entries to one building index generation."""

    provider_id: str

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[TextIndexEntry, ...],
    ) -> None: ...


@runtime_checkable
class LexicalCandidateRetriever(Protocol):
    """Retrieve lexical text candidates from one pinned index generation."""

    provider_id: str

    def retrieve(
        self,
        *,
        query: TextQuery,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]: ...


@runtime_checkable
class DenseCandidateRetriever(Protocol):
    """Retrieve dense text candidates from one pinned index generation."""

    provider_id: str

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]: ...


@runtime_checkable
class TextQueryEmbeddingProvider(Protocol):
    """Encode a canonical text query in one explicit embedding space."""

    provider_id: str

    def embed_query(
        self,
        *,
        query: TextQuery,
        embedding_space: EmbeddingSpaceRef,
    ) -> tuple[float, ...]: ...


@runtime_checkable
class TextEvidenceReader(Protocol):
    """Read complete text through a permission-checked evidence boundary."""

    def read(self, *, evidence_ref: str) -> TextEvidence: ...


@runtime_checkable
class TextIndexGenerationLifecycle(Protocol):
    """Manage one provider's derived immutable text index generations."""

    provider_id: str

    def mapping_sha256(
        self,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...],
    ) -> str: ...

    def create(self, manifest: TextIndexGenerationManifest) -> str: ...

    def validate(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexValidationReport: ...

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration: ...

    def resolve_active(self) -> PinnedTextIndexGeneration: ...


@runtime_checkable
class TextIndexGenerationCatalog(Protocol):
    """Persist authoritative generation state and append-only history."""

    def register(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexGenerationRecord: ...

    def get(self, generation_id: str) -> TextIndexGenerationRecord | None: ...

    def record_physical_index(
        self,
        generation_id: str,
        physical_index_ref: str,
    ) -> TextIndexGenerationRecord: ...

    def record_validation(
        self,
        report: TextIndexValidationReport,
    ) -> TextIndexGenerationRecord: ...

    def record_activation(
        self,
        pinned: PinnedTextIndexGeneration,
    ) -> TextIndexGenerationRecord: ...

    def active(
        self,
        *,
        workspace_ref: str,
        retriever_id: str,
    ) -> TextIndexGenerationRecord | None: ...

    def history(self, generation_id: str) -> tuple[TextIndexGenerationEvent, ...]: ...


@runtime_checkable
class TextRetrieverBackend(Protocol):
    """Group one provider's independent text indexing and candidate lanes."""

    provider_id: str
    index_writer: TextIndexWriter
    lexical_retriever: LexicalCandidateRetriever
    dense_retriever: DenseCandidateRetriever
    generation_lifecycle: TextIndexGenerationLifecycle
