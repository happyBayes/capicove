"""Lazy independent OpenSearch implementations of text retrieval ports."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from capicove.contracts import EmbeddingSpaceRef, SearchFilters, TextOrigin, TextQuery
from capicove.retrieval import (
    DenseCandidateRetriever,
    LexicalCandidateRetriever,
    PinnedTextIndexGeneration,
    TextCandidate,
    TextIndexEntry,
    TextIndexGenerationLifecycle,
    TextIndexGenerationManifest,
    TextIndexValidationReport,
    TextIndexWriter,
)
from capicove_retriever_opensearch.codec import (
    build_managed_text_index_mapping,
    decode_text_candidates,
    encode_text_index_entry,
    physical_index_name,
    vector_field_name,
)
from capicove_retriever_opensearch.config import OpenSearchConfiguration


_PROVIDER_ID = "capicove.retriever.opensearch"


class OpenSearchOperationError(RuntimeError):
    """Report one sanitized OpenSearch operation failure."""


class _Client(Protocol):
    indices: Any

    def search(self, **values: Any) -> object: ...
    def bulk(self, **values: Any) -> object: ...
    def count(self, **values: Any) -> object: ...


_ClientFactory = Callable[[OpenSearchConfiguration], _Client]


class _Owner:
    def __init__(self, configuration: OpenSearchConfiguration, factory: _ClientFactory) -> None:
        self.configuration = configuration
        self._factory = factory
        self._client: _Client | None = None
        self._lock = threading.Lock()

    def client(self) -> _Client:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._factory(self.configuration)
        return self._client


class OpenSearchTextRetriever:
    """Group OpenSearch indexing, lifecycle, and independent candidate lanes."""

    provider_id = _PROVIDER_ID

    def __init__(
        self,
        configuration: OpenSearchConfiguration,
        *,
        client_factory: _ClientFactory | None = None,
    ) -> None:
        if not isinstance(configuration, OpenSearchConfiguration):
            raise TypeError("configuration must be an OpenSearchConfiguration")
        owner = _Owner(configuration, client_factory or _default_client_factory)
        self.index_writer: TextIndexWriter = _Writer(owner)
        self.lexical_retriever: LexicalCandidateRetriever = _Lexical(owner)
        self.dense_retriever: DenseCandidateRetriever = _Dense(owner)
        self.generation_lifecycle: TextIndexGenerationLifecycle = _Lifecycle(owner)


class _Writer:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _Owner) -> None:
        self._owner = owner

    def add(self, *, index_generation_id: str, entries: tuple[TextIndexEntry, ...]) -> None:
        materialized = tuple(entries)
        if len({entry.unit.unit_id for entry in materialized}) != len(materialized):
            raise ValueError("entries must not contain duplicate unit IDs")
        if not materialized:
            return
        body: list[object] = []
        for entry in materialized:
            body.extend(({"create": {"_id": entry.unit.unit_id}}, encode_text_index_entry(entry)))
        try:
            response = self._owner.client().bulk(
                index=physical_index_name(
                    self._owner.configuration.index_prefix, index_generation_id
                ),
                body=body,
                refresh=False,
            )
            if not isinstance(response, Mapping) or response.get("errors") is not False:
                raise ValueError
        except Exception:
            raise OpenSearchOperationError("OpenSearch bulk operation failed") from None


class _Lexical:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _Owner) -> None:
        self._owner = owner

    def retrieve(
        self,
        *,
        query: TextQuery,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        _validate_retrieval(query, filters, generation, top_k)
        bool_query: dict[str, object] = {"must": [{"match": {"text": {"query": query.text}}}]}
        clauses = _filters(filters)
        if clauses:
            bool_query["filter"] = clauses
        body = {
            "size": top_k,
            "query": {"bool": bool_query},
            "highlight": {
                "fields": {"text": {"fragment_size": 2_048, "number_of_fragments": 1}},
                "pre_tags": ["<em>"],
                "post_tags": ["</em>"],
            },
            "_source": True,
        }
        try:
            response = self._owner.client().search(index=generation.physical_index_ref, body=body)
            return decode_text_candidates(
                response,
                provider_id=self.provider_id,
                generation_id=generation.generation_id,
                score_semantics="opensearch_bm25_score",
            )
        except Exception:
            raise OpenSearchOperationError("OpenSearch search operation failed") from None


class _Dense:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _Owner) -> None:
        self._owner = owner

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        _validate_retrieval(TextQuery("dense"), filters, generation, top_k)
        vector = tuple(query_vector)
        if len(vector) != embedding_space.dimension or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in vector
        ):
            raise ValueError("query_vector must match the embedding space")
        query: dict[str, object] = {
            "bool": {
                "must": [
                    {
                        "knn": {
                            vector_field_name(embedding_space.space_id): {
                                "vector": [float(item) for item in vector],
                                "k": top_k,
                            }
                        }
                    }
                ]
            }
        }
        clauses = _filters(filters)
        if clauses:
            cast(dict[str, object], query["bool"])["filter"] = clauses
        try:
            response = self._owner.client().search(
                index=generation.physical_index_ref,
                body={"size": top_k, "query": query, "_source": True},
            )
            return decode_text_candidates(
                response,
                provider_id=self.provider_id,
                generation_id=generation.generation_id,
                score_semantics="opensearch_knn_score",
            )
        except Exception:
            raise OpenSearchOperationError("OpenSearch search operation failed") from None


class _Lifecycle:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _Owner) -> None:
        self._owner = owner

    def mapping_sha256(self, embedding_spaces: tuple[EmbeddingSpaceRef, ...]) -> str:
        import hashlib
        import json

        payload = [(space.space_id, space.dimension) for space in embedding_spaces]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def create(self, manifest: TextIndexGenerationManifest) -> str:
        physical = physical_index_name(
            self._owner.configuration.index_prefix, manifest.generation_id
        )
        try:
            response = self._owner.client().indices.create(
                index=physical,
                body=build_managed_text_index_mapping(manifest),
            )
            _acknowledged(response)
        except Exception:
            raise OpenSearchOperationError("OpenSearch index creation failed") from None
        return physical

    def validate(self, manifest: TextIndexGenerationManifest) -> TextIndexValidationReport:
        physical = physical_index_name(
            self._owner.configuration.index_prefix, manifest.generation_id
        )
        try:
            client = self._owner.client()
            client.indices.put_settings(index=physical, body={"index.blocks.write": True})
            client.indices.refresh(index=physical)
            mapping = client.indices.get_mapping(index=physical)
            count = client.count(index=physical)
            observed_value = cast(Mapping[str, object], count)["count"]
            if isinstance(observed_value, bool) or not isinstance(observed_value, int):
                raise ValueError
            observed = observed_value
            issues: list[str] = []
            actual = cast(Mapping[str, object], mapping).get(physical)
            if not isinstance(actual, Mapping):
                issues.append("mapping_identity_mismatch")
            else:
                mappings = actual.get("mappings")
                meta = mappings.get("_meta") if isinstance(mappings, Mapping) else None
                if (
                    not isinstance(meta, Mapping)
                    or meta.get("capicove_manifest_sha256") != manifest.manifest_sha256
                ):
                    issues.append("mapping_identity_mismatch")
            if observed != manifest.expected_unit_count:
                issues.append("document_count_mismatch")
            return TextIndexValidationReport(
                manifest.generation_id,
                manifest.manifest_sha256,
                physical,
                manifest.mapping_sha256,
                manifest.expected_unit_count,
                observed,
                tuple(issues),
            )
        except OpenSearchOperationError:
            raise
        except Exception:
            raise OpenSearchOperationError("OpenSearch index validation failed") from None

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration:
        if not validation.valid or validation.generation_id != manifest.generation_id:
            raise ValueError("activation requires valid generation evidence")
        alias = self._owner.configuration.active_alias
        generation_pattern = self._owner.configuration.index_prefix + "-g-*"
        actions: list[dict[str, object]] = [
            {"remove": {"index": generation_pattern, "alias": alias}}
        ]
        actions.append({"add": {"index": validation.physical_index_ref, "alias": alias}})
        try:
            response = self._owner.client().indices.update_aliases(body={"actions": actions})
            _acknowledged(response)
        except Exception:
            raise OpenSearchOperationError("OpenSearch index activation failed") from None
        return PinnedTextIndexGeneration(
            manifest.generation_id, manifest.manifest_sha256, validation.physical_index_ref
        )

    def resolve_active(self) -> PinnedTextIndexGeneration:
        alias = self._owner.configuration.active_alias
        try:
            response = self._owner.client().indices.get_alias(
                index=self._owner.configuration.index_prefix + "-g-*", name=alias
            )
            if not isinstance(response, Mapping) or len(response) != 1:
                raise ValueError
            physical, body = next(iter(response.items()))
            if not isinstance(physical, str) or not isinstance(body, Mapping):
                raise ValueError
            mapping_response = self._owner.client().indices.get_mapping(index=physical)
            mapping = cast(Mapping[str, object], mapping_response)[physical]
            mappings = cast(Mapping[str, object], mapping)["mappings"]
            meta = cast(Mapping[str, object], mappings)["_meta"]
            return PinnedTextIndexGeneration(
                generation_id=cast(str, cast(Mapping[str, object], meta)["capicove_generation_id"]),
                manifest_sha256=cast(
                    str, cast(Mapping[str, object], meta)["capicove_manifest_sha256"]
                ),
                physical_index_ref=physical,
            )
        except Exception:
            raise OpenSearchOperationError(
                "OpenSearch active generation resolution failed"
            ) from None


def _filters(filters: SearchFilters) -> list[dict[str, object]]:
    clauses: list[dict[str, object]] = []
    for values, field in (
        (filters.corpus_refs, "corpus_refs"),
        (filters.artifact_refs, "artifact_ref"),
        (filters.media_types, "media_type"),
    ):
        if values:
            clauses.append({"terms": {field: sorted(values)}})
    if filters.text_origins:
        clauses.append({"terms": {"origin": sorted(item.value for item in filters.text_origins)}})
    if filters.time_range is not None:
        clauses.extend(
            (
                {"term": {"origin": TextOrigin.ASR.value}},
                {"range": {"start_ms": {"lt": filters.time_range.end_ms}}},
                {"range": {"end_ms": {"gt": filters.time_range.start_ms}}},
            )
        )
    return clauses


def _validate_retrieval(
    query: TextQuery,
    filters: SearchFilters,
    generation: PinnedTextIndexGeneration,
    top_k: int,
) -> None:
    if not isinstance(query, TextQuery) or not isinstance(filters, SearchFilters):
        raise TypeError("query and filters must be canonical values")
    if not isinstance(generation, PinnedTextIndexGeneration):
        raise TypeError("generation must be pinned")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be from 1 to 100")


def _acknowledged(response: object) -> None:
    if not isinstance(response, Mapping) or response.get("acknowledged") is not True:
        raise ValueError


def _default_client_factory(configuration: OpenSearchConfiguration) -> _Client:
    from opensearchpy import OpenSearch

    auth = None
    if configuration.username is not None and configuration.password is not None:
        auth = (configuration.username, configuration.password)
    return cast(
        _Client,
        OpenSearch(
            hosts=list(configuration.hosts),
            http_auth=auth,
            verify_certs=configuration.verify_certs,
            ca_certs=configuration.ca_certs,
            timeout=configuration.request_timeout,
            max_retries=configuration.max_retries,
            retry_on_timeout=True,
        ),
    )


__all__ = ["OpenSearchOperationError", "OpenSearchTextRetriever"]
