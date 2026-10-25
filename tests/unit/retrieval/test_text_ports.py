"""Structural tests for provider-neutral text retrieval ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
    VectorNormalization,
)
from capicove.retrieval import (
    DenseCandidateRetriever,
    LexicalCandidateRetriever,
    PinnedTextIndexGeneration,
    TextCandidate,
    TextEvidence,
    TextEvidenceReader,
    TextIndexEntry,
    TextIndexGenerationCatalog,
    TextIndexGenerationEvent,
    TextIndexGenerationEventType,
    TextIndexGenerationLifecycle,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationState,
    TextIndexValidationReport,
    TextIndexWriter,
    TextQueryEmbeddingProvider,
)


SOURCE_HASH = "a" * 64


def text_unit() -> SearchableUnit:
    """Return one canonical ASR text unit."""
    return SearchableUnit(
        unit_id="text-1",
        unit_type=TargetType.TEXT_CHUNK,
        source_hash=SOURCE_HASH,
        artifact_ref="artifact-1",
        evidence_ref="evidence-1",
        representation_refs=("representation-1",),
        processing_run_ref="run-1",
        processing_version="asr-chunk-v1",
        processing_fingerprint=ProcessingFingerprint.from_manifest(
            {"source_hash": SOURCE_HASH, "processor_id": "capicove.asr"}
        ),
        text_locator=TextChunkLocator(
            TextOrigin.ASR,
            CharacterRange(100, 125),
            time_range=TemporalRange(1_000, 2_000),
        ),
    )


def embedding_space() -> EmbeddingSpaceRef:
    """Return one text-to-text embedding space."""
    return EmbeddingSpaceRef(
        space_id="text-dense-v1",
        provider_id="embedding-provider",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def candidate() -> TextCandidate:
    """Return one provider-neutral text candidate."""
    return TextCandidate(
        unit_id="text-1",
        evidence_ref="evidence-1",
        text_locator=text_unit().text_locator,  # type: ignore[arg-type]
        snippet="A matching ASR passage.",
        raw_score=4.2,
        score_semantics="bm25",
        rank=1,
        provider_id="retriever-provider",
        index_generation_id="generation-42",
    )


@dataclass
class RecordingIndexWriter:
    """Record index writes through the public port shape."""

    provider_id: str = "retriever-provider"
    calls: list[tuple[str, tuple[TextIndexEntry, ...]]] = field(default_factory=list)

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[TextIndexEntry, ...],
    ) -> None:
        self.calls.append((index_generation_id, entries))


@dataclass
class RecordingLexicalRetriever:
    """Record lexical candidate requests through the public port shape."""

    result: tuple[TextCandidate, ...]
    provider_id: str = "retriever-provider"
    call: tuple[TextQuery, SearchFilters, PinnedTextIndexGeneration, int] | None = None

    def retrieve(
        self,
        *,
        query: TextQuery,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        self.call = (query, filters, generation, top_k)
        return self.result


@dataclass
class RecordingDenseRetriever:
    """Record dense candidate requests through the public port shape."""

    result: tuple[TextCandidate, ...]
    provider_id: str = "retriever-provider"
    call: (
        tuple[
            tuple[float, ...],
            EmbeddingSpaceRef,
            SearchFilters,
            PinnedTextIndexGeneration,
            int,
        ]
        | None
    ) = None

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        self.call = (
            query_vector,
            embedding_space,
            filters,
            generation,
            top_k,
        )
        return self.result


@dataclass
class RecordingQueryEmbeddingProvider:
    """Record canonical query encoding through the public port shape."""

    provider_id: str = "embedding-provider"
    call: tuple[TextQuery, EmbeddingSpaceRef] | None = None

    def embed_query(
        self,
        *,
        query: TextQuery,
        embedding_space: EmbeddingSpaceRef,
    ) -> tuple[float, ...]:
        self.call = (query, embedding_space)
        return (0.1, 0.2, 0.3)


@dataclass
class RecordingEvidenceReader:
    """Record permissioned evidence reads through the public port shape."""

    evidence: TextEvidence
    call: str | None = None

    def read(self, *, evidence_ref: str) -> TextEvidence:
        self.call = evidence_ref
        return self.evidence


@dataclass
class RecordingGenerationLifecycle:
    """Record provider-neutral generation lifecycle operations."""

    provider_id: str = "retriever-provider"
    calls: list[str] = field(default_factory=list)

    def mapping_sha256(
        self,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...],
    ) -> str:
        self.calls.append(f"mapping:{len(embedding_spaces)}")
        return "f" * 64

    def create(self, manifest: TextIndexGenerationManifest) -> str:
        self.calls.append(f"create:{manifest.generation_id}")
        return "provider-physical-index"

    def validate(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexValidationReport:
        self.calls.append(f"validate:{manifest.generation_id}")
        return TextIndexValidationReport(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref="provider-physical-index",
            mapping_sha256=manifest.mapping_sha256,
            expected_unit_count=manifest.expected_unit_count,
            observed_unit_count=manifest.expected_unit_count,
            issue_codes=(),
        )

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration:
        del validation
        self.calls.append(f"activate:{manifest.generation_id}")
        return PinnedTextIndexGeneration(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref="provider-physical-index",
        )

    def resolve_active(self) -> PinnedTextIndexGeneration:
        self.calls.append("resolve")
        return PinnedTextIndexGeneration(
            generation_id="generation-42",
            manifest_sha256="e" * 64,
            physical_index_ref="provider-physical-index",
        )


@dataclass
class RecordingGenerationCatalog:
    """Expose the complete provider-neutral generation catalog shape."""

    records: dict[str, TextIndexGenerationRecord] = field(default_factory=dict)

    def register(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexGenerationRecord:
        record = TextIndexGenerationRecord(
            manifest=manifest,
            state=TextIndexGenerationState.BUILDING,
            revision=0,
        )
        self.records[manifest.generation_id] = record
        return record

    def get(self, generation_id: str) -> TextIndexGenerationRecord | None:
        return self.records.get(generation_id)

    def record_physical_index(
        self,
        generation_id: str,
        physical_index_ref: str,
    ) -> TextIndexGenerationRecord:
        record = self.records[generation_id]
        updated = TextIndexGenerationRecord(
            manifest=record.manifest,
            state=record.state,
            physical_index_ref=physical_index_ref,
            revision=record.revision + 1,
        )
        self.records[generation_id] = updated
        return updated

    def record_validation(
        self,
        report: TextIndexValidationReport,
    ) -> TextIndexGenerationRecord:
        record = self.records[report.generation_id]
        updated = TextIndexGenerationRecord(
            manifest=record.manifest,
            state=(
                TextIndexGenerationState.READY if report.valid else TextIndexGenerationState.CORRUPT
            ),
            physical_index_ref=report.physical_index_ref,
            validation=report,
            revision=record.revision + 1,
        )
        self.records[report.generation_id] = updated
        return updated

    def record_activation(
        self,
        pinned: PinnedTextIndexGeneration,
    ) -> TextIndexGenerationRecord:
        record = self.records[pinned.generation_id]
        updated = TextIndexGenerationRecord(
            manifest=record.manifest,
            state=TextIndexGenerationState.ACTIVE,
            physical_index_ref=record.physical_index_ref,
            validation=record.validation,
            revision=record.revision + 1,
        )
        self.records[pinned.generation_id] = updated
        return updated

    def active(
        self,
        *,
        workspace_ref: str,
        retriever_id: str,
    ) -> TextIndexGenerationRecord | None:
        return next(
            (
                record
                for record in self.records.values()
                if record.state is TextIndexGenerationState.ACTIVE
                and record.manifest.workspace_ref == workspace_ref
                and record.manifest.retriever_id == retriever_id
            ),
            None,
        )

    def history(self, generation_id: str) -> tuple[TextIndexGenerationEvent, ...]:
        record = self.records[generation_id]
        return (
            TextIndexGenerationEvent(
                sequence=1,
                event_type=TextIndexGenerationEventType.REGISTERED,
                record=record,
                recorded_at=datetime(2026, 8, 26, tzinfo=UTC),
            ),
        )


def test_text_index_writer_adds_entries_to_one_named_generation() -> None:
    writer = RecordingIndexWriter()
    entry = TextIndexEntry(unit=text_unit(), text="Complete ASR transcript text.")

    assert isinstance(writer, TextIndexWriter)
    writer.add(index_generation_id="generation-42", entries=(entry,))

    assert writer.calls == [("generation-42", (entry,))]


def test_candidate_ports_preserve_closed_filters_and_pinned_generation() -> None:
    expected = (candidate(),)
    query = TextQuery("matching passage", language="en")
    filters = SearchFilters(text_origins=frozenset({TextOrigin.ASR}))
    space = embedding_space()
    lexical = RecordingLexicalRetriever(expected)
    dense = RecordingDenseRetriever(expected)
    generation = PinnedTextIndexGeneration(
        generation_id="generation-42",
        manifest_sha256="f" * 64,
        physical_index_ref="capicove-text-generation-42",
    )

    assert isinstance(lexical, LexicalCandidateRetriever)
    assert isinstance(dense, DenseCandidateRetriever)
    assert (
        lexical.retrieve(
            query=query,
            filters=filters,
            generation=generation,
            top_k=20,
        )
        == expected
    )
    assert (
        dense.retrieve(
            query_vector=(0.1, 0.2, 0.3),
            embedding_space=space,
            filters=filters,
            generation=generation,
            top_k=20,
        )
        == expected
    )
    assert lexical.call == (query, filters, generation, 20)
    assert dense.call == (
        (0.1, 0.2, 0.3),
        space,
        filters,
        generation,
        20,
    )


def test_query_embedding_and_evidence_ports_use_canonical_values() -> None:
    query = TextQuery("matching passage")
    space = embedding_space()
    embedding_provider = RecordingQueryEmbeddingProvider()
    evidence = TextEvidence(text_unit(), "Complete ASR transcript text.")
    evidence_reader = RecordingEvidenceReader(evidence)

    assert isinstance(embedding_provider, TextQueryEmbeddingProvider)
    assert isinstance(evidence_reader, TextEvidenceReader)
    assert embedding_provider.embed_query(query=query, embedding_space=space) == (
        0.1,
        0.2,
        0.3,
    )
    assert evidence_reader.read(evidence_ref="evidence-1") is evidence
    assert embedding_provider.call == (query, space)
    assert evidence_reader.call == "evidence-1"


def test_generation_lifecycle_is_a_provider_neutral_runtime_port() -> None:
    lifecycle = RecordingGenerationLifecycle()

    assert isinstance(lifecycle, TextIndexGenerationLifecycle)
    pinned = lifecycle.resolve_active()

    assert pinned.generation_id == "generation-42"
    assert lifecycle.calls == ["resolve"]


def test_generation_catalog_is_a_provider_neutral_runtime_port() -> None:
    catalog = RecordingGenerationCatalog()

    assert isinstance(catalog, TextIndexGenerationCatalog)
