"""Tests for the lazy Elasticsearch text retriever adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from capicove_retriever_elasticsearch.adapter import (
    ElasticsearchOperationError,
    ElasticsearchTextRetriever,
)
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration

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
    PinnedTextIndexGeneration,
    TextEmbedding,
    TextIndexEntry,
    TextRetrieverBackend,
)


SOURCE_HASH = "a" * 64
EMPTY_RESPONSE: dict[str, object] = {"hits": {"hits": []}}


@pytest.mark.parametrize("dense", [False, True])
def test_malformed_provider_response_remains_an_integrity_error(dense: bool) -> None:
    client = RecordingClient(search_responses=[{"hits": {"hits": "malformed"}}])
    backend = ElasticsearchTextRetriever(configuration(), client_factory=lambda config: client)
    with pytest.raises(ElasticsearchOperationError) as caught:
        if dense:
            backend.dense_retriever.retrieve(
                query_vector=(1.0, 0.0, 0.0),
                embedding_space=embedding_space(),
                filters=SearchFilters(),
                generation=pinned_generation(),
                top_k=10,
            )
        else:
            backend.lexical_retriever.retrieve(
                query=TextQuery("authored"),
                filters=SearchFilters(),
                generation=pinned_generation(),
                top_k=10,
            )
    assert isinstance(caught.value, ValueError)
    assert "malformed" not in str(caught.value)


def configuration(*, multiplier: int = 10) -> ElasticsearchConfiguration:
    """Return one secure local test configuration."""
    return ElasticsearchConfiguration(
        hosts=("https://localhost:9200",),
        index_prefix="capicove-text",
        knn_candidate_multiplier=multiplier,
    )


def embedding_space() -> EmbeddingSpaceRef:
    """Return one compatible dense text space."""
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


def pinned_generation() -> PinnedTextIndexGeneration:
    """Return one logical generation pinned to a concrete provider index."""
    return PinnedTextIndexGeneration(
        generation_id="generation-42",
        manifest_sha256="f" * 64,
        physical_index_ref="capicove-text-g-79091da8c3ddbe5b32099e4c",
    )


def text_entry(*, text: str = "A complete matching passage.") -> TextIndexEntry:
    """Return one vector-bearing document text entry."""
    space = embedding_space()
    unit = SearchableUnit(
        unit_id="text-1",
        unit_type=TargetType.TEXT_CHUNK,
        source_hash=SOURCE_HASH,
        artifact_ref="artifact-1",
        evidence_ref="evidence-1",
        representation_refs=("representation-1",),
        processing_run_ref="run-1",
        processing_version="text-v1",
        processing_fingerprint=ProcessingFingerprint.from_manifest(
            {"source_hash": SOURCE_HASH, "processor_id": "capicove.text"}
        ),
        embedding_spaces=(space,),
        text_locator=TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, len(text))),
    )
    return TextIndexEntry(
        unit=unit,
        text=text,
        embeddings=(TextEmbedding(space, (0.1, 0.2, 0.3)),),
        corpus_refs=("corpus-1",),
        media_type="text/plain",
        language="en",
    )


@dataclass
class UnexpectedIndicesClient:
    """Reject index administration operations outside lifecycle tests."""

    def get_settings(self, **kwargs: object) -> object:
        raise AssertionError("unexpected index settings operation")

    def create(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.create call: {sorted(kwargs)}")

    def refresh(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.refresh call: {sorted(kwargs)}")

    def get_mapping(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.get_mapping call: {sorted(kwargs)}")

    def update_aliases(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.update_aliases call: {sorted(kwargs)}")

    def get_alias(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.get_alias call: {sorted(kwargs)}")

    def exists_alias(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.exists_alias call: {sorted(kwargs)}")

    def add_block(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected indices.add_block call: {sorted(kwargs)}")


@dataclass
class RecordingClient:
    """Record complete client calls and return controlled provider responses."""

    bulk_response: object = field(default_factory=lambda: {"errors": False})
    search_responses: list[object] = field(default_factory=list)
    bulk_calls: list[dict[str, object]] = field(default_factory=list)
    search_calls: list[dict[str, object]] = field(default_factory=list)
    indices: UnexpectedIndicesClient = field(default_factory=UnexpectedIndicesClient)

    def bulk(self, **kwargs: object) -> object:
        self.bulk_calls.append(dict(kwargs))
        return self.bulk_response

    def search(self, **kwargs: object) -> object:
        self.search_calls.append(dict(kwargs))
        if not self.search_responses:
            raise AssertionError("unexpected search call")
        return self.search_responses.pop(0)

    def count(self, **kwargs: object) -> object:
        raise AssertionError(f"unexpected count call: {sorted(kwargs)}")


class RaisingClient(RecordingClient):
    """Raise provider errors that contain content the adapter must suppress."""

    def bulk(self, **kwargs: object) -> object:
        del kwargs
        raise RuntimeError("VERY_SECRET_CHUNK")

    def search(self, **kwargs: object) -> object:
        del kwargs
        raise RuntimeError("VERY_SECRET_QUERY")


def runtime_for(
    client: RecordingClient,
    *,
    multiplier: int = 10,
) -> ElasticsearchTextRetriever:
    """Return one runtime with an injected non-network client."""
    return ElasticsearchTextRetriever(
        configuration(multiplier=multiplier),
        client_factory=lambda _: client,
    )


def test_runtime_constructs_no_client_and_shares_one_client_across_lanes() -> None:
    client = RecordingClient(search_responses=[EMPTY_RESPONSE, EMPTY_RESPONSE])
    factory_calls: list[ElasticsearchConfiguration] = []

    def factory(value: ElasticsearchConfiguration) -> RecordingClient:
        factory_calls.append(value)
        return client

    runtime = ElasticsearchTextRetriever(configuration(), client_factory=factory)

    assert isinstance(runtime, TextRetrieverBackend)
    assert factory_calls == []
    runtime.lexical_retriever.retrieve(
        query=TextQuery("matching passage"),
        filters=SearchFilters(),
        generation=pinned_generation(),
        top_k=5,
    )
    runtime.dense_retriever.retrieve(
        query_vector=(0.1, 0.2, 0.3),
        embedding_space=embedding_space(),
        filters=SearchFilters(),
        generation=pinned_generation(),
        top_k=5,
    )

    assert factory_calls == [configuration()]


def test_bulk_writer_uses_create_actions_in_one_pinned_physical_generation() -> None:
    client = RecordingClient()
    runtime = runtime_for(client)
    entry = text_entry()

    runtime.index_writer.add(index_generation_id="generation-42", entries=(entry,))

    call = client.bulk_calls[0]
    assert call["index"] == "capicove-text-g-79091da8c3ddbe5b32099e4c"
    assert call["refresh"] is False
    operations = cast(list[dict[str, Any]], call["operations"])
    assert operations[0] == {"create": {"_id": "text-1"}}
    assert operations[1]["unit_id"] == "text-1"
    assert operations[1]["text"] == "A complete matching passage."
    assert operations[1]["preview"] == "A complete matching passage."


def test_bulk_writer_rejects_duplicate_units_before_opening_client() -> None:
    client = RecordingClient()
    runtime = runtime_for(client)
    entry = text_entry()

    with pytest.raises(ValueError, match="duplicate unit"):
        runtime.index_writer.add(
            index_generation_id="generation-42",
            entries=(entry, entry),
        )

    assert client.bulk_calls == []


@pytest.mark.parametrize(
    "client",
    [RecordingClient(bulk_response={"errors": True}), RaisingClient()],
)
def test_bulk_writer_maps_failures_without_exposing_content(
    client: RecordingClient,
) -> None:
    runtime = runtime_for(client)

    with pytest.raises(ElasticsearchOperationError, match="bulk operation failed") as raised:
        runtime.index_writer.add(
            index_generation_id="generation-42",
            entries=(text_entry(text="VERY_SECRET_CHUNK"),),
        )

    assert "VERY_SECRET_CHUNK" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_lexical_retrieval_translates_only_closed_filters_and_decodes_candidates() -> None:
    client = RecordingClient(
        search_responses=[
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "text-asr",
                            "_score": 8.5,
                            "_source": {
                                "unit_id": "text-asr",
                                "evidence_ref": "evidence-asr",
                                "origin": "asr",
                                "start_char": 10,
                                "end_char": 30,
                                "start_ms": 1_100,
                                "end_ms": 1_900,
                                "preview": "A matching transcript passage.",
                            },
                            "highlight": {"text": ["A <em>matching</em> transcript."]},
                        }
                    ]
                }
            }
        ]
    )
    runtime = runtime_for(client)
    filters = SearchFilters(
        corpus_refs=frozenset({"corpus-2", "corpus-1"}),
        artifact_refs=frozenset({"artifact-1"}),
        media_types=frozenset({"audio/wav"}),
        text_origins=frozenset({TextOrigin.ASR}),
        time_range=TemporalRange(1_000, 2_000),
    )

    candidates = runtime.lexical_retriever.retrieve(
        query=TextQuery("VERY_SECRET_QUERY", language="en"),
        filters=filters,
        generation=pinned_generation(),
        top_k=5,
    )

    assert candidates[0].snippet == "A matching transcript."
    assert candidates[0].raw_score == 8.5
    assert candidates[0].score_semantics == "elasticsearch_bm25_score"
    assert client.search_calls == [
        {
            "index": "capicove-text-g-79091da8c3ddbe5b32099e4c",
            "size": 5,
            "query": {
                "bool": {
                    "must": [{"match": {"text": {"query": "VERY_SECRET_QUERY"}}}],
                    "filter": [
                        {"terms": {"corpus_refs": ["corpus-1", "corpus-2"]}},
                        {"terms": {"artifact_ref": ["artifact-1"]}},
                        {"terms": {"media_type": ["audio/wav"]}},
                        {"terms": {"origin": ["asr"]}},
                        {"terms": {"origin": ["asr", "ocr"]}},
                        {"range": {"start_ms": {"lt": 2_000}}},
                        {"range": {"end_ms": {"gt": 1_000}}},
                    ],
                }
            },
            "source": [
                "unit_id",
                "evidence_ref",
                "origin",
                "start_char",
                "end_char",
                "page_number",
                "start_ms",
                "end_ms",
                "preview",
            ],
            "highlight": {
                "fields": {"text": {"fragment_size": 2_048, "number_of_fragments": 1}},
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
            },
        }
    ]


def test_dense_retrieval_uses_top_level_knn_with_bounded_pre_filters() -> None:
    client = RecordingClient(search_responses=[EMPTY_RESPONSE])
    runtime = runtime_for(client, multiplier=12)

    candidates = runtime.dense_retriever.retrieve(
        query_vector=(0.1, 0.2, 0.3),
        embedding_space=embedding_space(),
        filters=SearchFilters(text_origins=frozenset({TextOrigin.DOCUMENT})),
        generation=pinned_generation(),
        top_k=3,
    )

    assert candidates == ()
    assert client.search_calls == [
        {
            "index": "capicove-text-g-79091da8c3ddbe5b32099e4c",
            "size": 3,
            "knn": {
                "field": "vector_77bad12a9fc0e553b1a845c3",
                "query_vector": [0.1, 0.2, 0.3],
                "k": 3,
                "num_candidates": 36,
                "filter": [{"terms": {"origin": ["document"]}}],
            },
            "source": [
                "unit_id",
                "evidence_ref",
                "origin",
                "start_char",
                "end_char",
                "page_number",
                "start_ms",
                "end_ms",
                "preview",
            ],
        }
    ]


@pytest.mark.parametrize("top_k", [0, 301, True])
def test_candidate_lanes_reject_invalid_limits_before_opening_client(top_k: int) -> None:
    client = RecordingClient()
    runtime = runtime_for(client)

    with pytest.raises(ValueError, match="top_k"):
        runtime.lexical_retriever.retrieve(
            query=TextQuery("matching passage"),
            filters=SearchFilters(),
            generation=pinned_generation(),
            top_k=top_k,
        )

    assert client.search_calls == []


@pytest.mark.parametrize(
    "vector",
    [(0.1, 0.2), (0.1, math.nan, 0.3), (0.1, True, 0.3)],
)
def test_dense_lane_rejects_invalid_vectors_before_opening_client(
    vector: tuple[float, ...],
) -> None:
    client = RecordingClient()
    runtime = runtime_for(client)

    with pytest.raises(ValueError, match="query_vector"):
        runtime.dense_retriever.retrieve(
            query_vector=vector,
            embedding_space=embedding_space(),
            filters=SearchFilters(),
            generation=pinned_generation(),
            top_k=5,
        )

    assert client.search_calls == []


@pytest.mark.parametrize("lane", ["lexical", "dense"])
def test_search_failures_do_not_expose_queries_or_provider_errors(lane: str) -> None:
    runtime = runtime_for(RaisingClient())

    with pytest.raises(
        ElasticsearchOperationError,
        match="search operation failed",
    ) as raised:
        if lane == "lexical":
            runtime.lexical_retriever.retrieve(
                query=TextQuery("VERY_SECRET_QUERY"),
                filters=SearchFilters(),
                generation=pinned_generation(),
                top_k=5,
            )
        else:
            runtime.dense_retriever.retrieve(
                query_vector=(0.1, 0.2, 0.3),
                embedding_space=embedding_space(),
                filters=SearchFilters(),
                generation=pinned_generation(),
                top_k=5,
            )

    assert "VERY_SECRET_QUERY" not in str(raised.value)
    assert raised.value.__cause__ is None
