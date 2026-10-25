"""Tests for provider-neutral text search orchestration."""

from dataclasses import dataclass, replace

import pytest

from capicove.contracts import (
    ArtifactSource,
    CharacterRange,
    DetailLevel,
    ImageQuery,
    ProcessingFingerprint,
    SearchFilters,
    SearchRequest,
    SearchStatus,
    TargetType,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
)
from capicove.retrieval import (
    PinnedTextIndexGeneration,
    StaticTextRetrievalProfileRegistry,
    TextCandidate,
    TextIndexGenerationEvent,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationState,
    TextIndexValidationReport,
    TextRetrievalProfile,
    TextSearchError,
    TextSearchService,
)
from tests.unit.retrieval.test_text_models import text_embedding_space


PIN = PinnedTextIndexGeneration("generation-1", "f" * 64, "physical-index-1")


def manifest() -> TextIndexGenerationManifest:
    """Return one active text-index manifest."""
    return TextIndexGenerationManifest(
        generation_id=PIN.generation_id,
        workspace_ref="workspace-1",
        corpus_refs=("corpus-1",),
        searchable_unit_manifest_sha256="a" * 64,
        content_sha256="b" * 64,
        mapping_sha256="c" * 64,
        embedding_spaces=(text_embedding_space(),),
        retriever_id="capicove.retriever.opensearch",
        index_format_version="opensearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint.from_manifest({"build": 1}),
        expected_unit_count=2,
    )


def active_record() -> TextIndexGenerationRecord:
    """Return an authoritative active generation matching the provider pin."""
    value = manifest()
    object.__setattr__(PIN, "manifest_sha256", value.manifest_sha256)
    report = TextIndexValidationReport(
        generation_id=value.generation_id,
        manifest_sha256=value.manifest_sha256,
        physical_index_ref=PIN.physical_index_ref,
        mapping_sha256=value.mapping_sha256,
        expected_unit_count=2,
        observed_unit_count=2,
        issue_codes=(),
    )
    return TextIndexGenerationRecord(
        manifest=value,
        state=TextIndexGenerationState.ACTIVE,
        revision=3,
        physical_index_ref=PIN.physical_index_ref,
        validation=report,
    )


def candidate(unit_id: str, rank: int) -> TextCandidate:
    """Return one candidate from the pinned generation."""
    return TextCandidate(
        unit_id=unit_id,
        evidence_ref=f"evidence-{unit_id}",
        text_locator=TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 8)),
        snippet=f"passage {unit_id}",
        raw_score=2.0,
        score_semantics="provider-score",
        rank=rank,
        provider_id="capicove.retriever.opensearch",
        index_generation_id=PIN.generation_id,
    )


class LaneFailure(RuntimeError):
    """Represent one sanitized provider test failure."""


@dataclass
class RecordingLane:
    result: tuple[TextCandidate, ...] = ()
    failure: Exception | None = None
    provider_id: str = "capicove.retriever.opensearch"
    pins: list[PinnedTextIndexGeneration] | None = None

    def _record(self, generation: PinnedTextIndexGeneration) -> tuple[TextCandidate, ...]:
        if self.pins is None:
            self.pins = []
        self.pins.append(generation)
        if self.failure is not None:
            raise self.failure
        return self.result

    def retrieve(self, **values: object) -> tuple[TextCandidate, ...]:
        return self._record(values["generation"])  # type: ignore[arg-type]


@dataclass
class Lifecycle:
    calls: int = 0
    provider_id: str = "capicove.retriever.opensearch"

    def resolve_active(self) -> PinnedTextIndexGeneration:
        self.calls += 1
        return PIN


@dataclass
class Catalog:
    record: TextIndexGenerationRecord

    def get(self, generation_id: str) -> TextIndexGenerationRecord | None:
        return self.record if generation_id == self.record.manifest.generation_id else None

    def active(self, **values: str) -> TextIndexGenerationRecord | None:
        return self.record

    def history(self, generation_id: str) -> tuple[TextIndexGenerationEvent, ...]:
        return ()


@dataclass
class Embedder:
    failure: Exception | None = None
    provider_id: str = "embedder"

    def embed_query(self, **values: object) -> tuple[float, ...]:
        if self.failure is not None:
            raise self.failure
        return (0.1, 0.2, 0.3)


@dataclass
class Backend:
    lexical_retriever: RecordingLane
    dense_retriever: RecordingLane
    generation_lifecycle: Lifecycle
    provider_id: str = "capicove.retriever.opensearch"
    index_writer: object = object()


def service(
    *,
    lexical: RecordingLane,
    dense: RecordingLane,
    dense_required: bool = False,
    embedder: Embedder | None = None,
) -> tuple[TextSearchService, Lifecycle]:
    """Return a configured service and its recording lifecycle."""
    lifecycle = Lifecycle()
    profile = TextRetrievalProfile(
        profile_id="local-default",
        lexical_weight=2.0,
        dense_weight=1.0,
        lexical_required=True,
        dense_required=dense_required,
        lexical_candidate_count=20,
        dense_candidate_count=20,
        embedding_space=text_embedding_space(),
    )
    return (
        TextSearchService(
            backend=Backend(lexical, dense, lifecycle),  # type: ignore[arg-type]
            catalog=Catalog(active_record()),  # type: ignore[arg-type]
            profiles=StaticTextRetrievalProfileRegistry((profile,)),
            query_embedder=embedder or Embedder(),
            query_id_factory=lambda: "query-1",
        ),
        lifecycle,
    )


def request() -> SearchRequest:
    """Return one canonical compact text search request."""
    return SearchRequest(
        query=TextQuery("matching passage"),
        targets=frozenset({TargetType.TEXT_CHUNK}),
        top_k=2,
        filters=SearchFilters(corpus_refs=frozenset({"corpus-1"})),
    )


def test_search_pins_once_and_fuses_both_lanes_into_canonical_hits() -> None:
    lexical = RecordingLane((candidate("a", 1), candidate("b", 2)))
    dense = RecordingLane((candidate("b", 1), candidate("a", 2)))
    operation, lifecycle = service(lexical=lexical, dense=dense)

    result = operation.search(request())

    assert result.status is SearchStatus.SUCCEEDED
    assert result.query_id == "query-1"
    assert [hit.id for hit in result.hits] == ["a", "b"]
    assert lifecycle.calls == 1
    assert lexical.pins == [PIN]
    assert dense.pins == [PIN]


def test_optional_dense_failure_returns_stable_degraded_lexical_result() -> None:
    operation, _ = service(
        lexical=RecordingLane((candidate("a", 1),)),
        dense=RecordingLane(failure=LaneFailure("SECRET_PROVIDER_DETAIL")),
    )

    result = operation.search(request())

    assert result.status is SearchStatus.SUCCEEDED
    assert result.degraded is True
    assert result.warning_codes == ("text_dense_lane_unavailable",)
    assert [hit.id for hit in result.hits] == ["a"]


@pytest.mark.parametrize("lane", ["lexical", "dense"])
@pytest.mark.parametrize("field", ["index_generation_id", "provider_id"])
@pytest.mark.parametrize("shared_unit", [False, True])
def test_search_rejects_candidates_outside_the_pinned_backend(
    lane: str, field: str, shared_unit: bool
) -> None:
    original = candidate("a" if shared_unit else "b", 1)
    invalid = (
        replace(original, index_generation_id="other")
        if field == "index_generation_id"
        else replace(original, provider_id="other")
    )
    valid = candidate("a", 1)
    operation, _ = service(
        lexical=RecordingLane((invalid if lane == "lexical" else valid,)),
        dense=RecordingLane((invalid if lane == "dense" else valid,)),
    )

    with pytest.raises(TextSearchError, match=r"^index_corrupt$"):
        operation.search(request())


def test_required_dense_failure_raises_only_a_stable_error() -> None:
    operation, _ = service(
        lexical=RecordingLane((candidate("a", 1),)),
        dense=RecordingLane(failure=LaneFailure("SECRET_PROVIDER_DETAIL")),
        dense_required=True,
    )

    with pytest.raises(TextSearchError, match="provider_unavailable") as raised:
        operation.search(request())

    assert "SECRET_PROVIDER_DETAIL" not in str(raised.value)


def test_search_rejects_non_compact_request_before_provider_access() -> None:
    operation, lifecycle = service(lexical=RecordingLane(), dense=RecordingLane())
    invalid = SearchRequest(
        query=TextQuery("query"),
        targets=frozenset({TargetType.TEXT_CHUNK}),
        detail_level=DetailLevel.FULL,
    )

    with pytest.raises(TextSearchError, match="invalid_query"):
        operation.search(invalid)

    assert lifecycle.calls == 0


def test_search_rejects_non_text_request_before_provider_access() -> None:
    operation, lifecycle = service(lexical=RecordingLane(), dense=RecordingLane())
    invalid = SearchRequest(
        query=ImageQuery(ArtifactSource("artifact-image")),
        targets=frozenset({TargetType.TEXT_CHUNK}),
    )

    with pytest.raises(TextSearchError, match="invalid_query"):
        operation.search(invalid)

    assert lifecycle.calls == 0
