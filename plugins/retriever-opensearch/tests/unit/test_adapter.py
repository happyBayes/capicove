"""Tests for independent OpenSearch indexing and candidate lanes."""

from types import SimpleNamespace
from typing import Any

import pytest
from capicove_retriever_opensearch.adapter import OpenSearchOperationError, OpenSearchTextRetriever
from capicove_retriever_opensearch.config import OpenSearchConfiguration

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    SearchFilters,
    TargetType,
    TextOrigin,
    TextQuery,
    VectorNormalization,
)
from capicove.retrieval import PinnedTextIndexGeneration


def text_embedding_space() -> EmbeddingSpaceRef:
    """Return one real-compatible text embedding space."""
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


class RecordingIndices:
    """Record index administration calls."""

    def __init__(self) -> None:
        self.alias_calls: list[dict[str, Any]] = []

    def update_aliases(self, **values: Any) -> dict[str, bool]:
        self.alias_calls.append(values)
        return {"acknowledged": True}

    def get_alias(self, **values: Any) -> dict[str, object]:
        self.alias_calls.append(values)
        return {}


class RecordingClient:
    """Record OpenSearch calls and return complete empty responses."""

    def __init__(self) -> None:
        self.indices = RecordingIndices()
        self.search_calls: list[dict[str, Any]] = []

    def search(self, **values: Any) -> dict[str, object]:
        self.search_calls.append(values)
        return {"hits": {"hits": []}}

    def bulk(self, **values: Any) -> dict[str, object]:
        return {"errors": False}

    def count(self, **values: Any) -> dict[str, object]:
        return {"count": 0}


def runtime(client: RecordingClient) -> OpenSearchTextRetriever:
    """Return a runtime backed by one recording client."""
    return OpenSearchTextRetriever(
        OpenSearchConfiguration(
            hosts=("http://127.0.0.1:9200",),
            verify_certs=False,
        ),
        client_factory=lambda _: client,
    )


def pin() -> PinnedTextIndexGeneration:
    """Return one concrete OpenSearch generation."""
    return PinnedTextIndexGeneration("generation-1", "f" * 64, "physical-index-1")


def test_activation_never_removes_aliases_from_unrelated_or_system_indexes() -> None:
    client = RecordingClient()
    backend = runtime(client)
    manifest = SimpleNamespace(generation_id="generation-1", manifest_sha256="f" * 64)
    validation = SimpleNamespace(
        valid=True, generation_id="generation-1", physical_index_ref="capicove-text-g-123"
    )
    backend.generation_lifecycle.activate(manifest, validation)
    remove = client.indices.alias_calls[0]["body"]["actions"][0]["remove"]
    assert remove["index"] == "capicove-text-g-*"


def test_resolution_never_scans_unrelated_or_system_indexes() -> None:
    client = RecordingClient()
    with pytest.raises(OpenSearchOperationError):
        runtime(client).generation_lifecycle.resolve_active()
    assert client.indices.alias_calls == [
        {
            "index": "capicove-text-g-*",
            "name": "capicove-text-active",
        }
    ]


def test_lexical_lane_queries_only_the_pinned_physical_index() -> None:
    client = RecordingClient()

    result = runtime(client).lexical_retriever.retrieve(
        query=TextQuery("matching passage"),
        filters=SearchFilters(text_origins=frozenset({TextOrigin.DOCUMENT})),
        generation=pin(),
        top_k=5,
    )

    assert result == ()
    assert client.search_calls == [
        {
            "index": "physical-index-1",
            "body": {
                "size": 5,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": {"query": "matching passage"}}}],
                        "filter": [{"terms": {"origin": ["document"]}}],
                    }
                },
                "highlight": {
                    "fields": {"text": {"fragment_size": 2048, "number_of_fragments": 1}},
                    "pre_tags": ["<em>"],
                    "post_tags": ["</em>"],
                },
                "_source": True,
            },
        }
    ]


def test_dense_lane_uses_opensearch_knn_inside_closed_bool_filter() -> None:
    client = RecordingClient()
    space = text_embedding_space()

    result = runtime(client).dense_retriever.retrieve(
        query_vector=(0.1, 0.2, 0.3),
        embedding_space=space,
        filters=SearchFilters(text_origins=frozenset({TextOrigin.DOCUMENT})),
        generation=pin(),
        top_k=3,
    )

    assert result == ()
    body = client.search_calls[0]["body"]
    assert isinstance(body, dict)
    query = body["query"]
    assert query == {
        "bool": {
            "must": [{"knn": {"embedding_text-dense-v1": {"vector": [0.1, 0.2, 0.3], "k": 3}}}],
            "filter": [{"terms": {"origin": ["document"]}}],
        }
    }
