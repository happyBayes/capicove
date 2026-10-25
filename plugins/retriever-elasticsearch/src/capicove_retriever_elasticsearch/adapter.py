"""Lazy Elasticsearch implementations of Capicove text retrieval ports."""

from __future__ import annotations

import importlib
import math
import threading
from collections.abc import Callable, Mapping
from typing import Protocol, cast

from capicove.contracts import (
    EmbeddingSpaceRef,
    QueryModality,
    SearchFilters,
    TargetType,
    TextOrigin,
    TextQuery,
)
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
from capicove_retriever_elasticsearch.codec import (
    build_managed_text_index_mapping,
    decode_managed_text_index_identity,
    decode_text_candidates,
    encode_text_index_entry,
    inspect_managed_text_index_mapping,
    physical_index_name,
    text_index_mapping_sha256,
    vector_field_name,
)
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration


_PROVIDER_ID = "capicove.retriever.elasticsearch"
_INDEX_FORMAT_VERSION = "elasticsearch-text-v1"
_SOURCE_FIELDS = (
    "unit_id",
    "evidence_ref",
    "origin",
    "start_char",
    "end_char",
    "page_number",
    "start_ms",
    "end_ms",
    "preview",
)
_HIGHLIGHT = {
    "fields": {"text": {"fragment_size": 2_048, "number_of_fragments": 1}},
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
}


class ElasticsearchOperationError(RuntimeError):
    """Report one sanitized Elasticsearch operation failure."""


class _ElasticsearchDataError(ElasticsearchOperationError, ValueError):
    """Preserve legacy operation-error catches and fail-closed data semantics."""


class _ElasticsearchIndicesClient(Protocol):
    def get_settings(self, **kwargs: object) -> object:
        """Read physical write-protection settings during idempotent validation."""

    def add_block(self, **kwargs: object) -> object:
        """Seal one physical index against further writes."""

    def create(self, **kwargs: object) -> object:
        """Create one physical index."""

    def refresh(self, **kwargs: object) -> object:
        """Refresh one physical index."""

    def get_mapping(self, **kwargs: object) -> object:
        """Read one physical index mapping."""

    def update_aliases(self, **kwargs: object) -> object:
        """Apply one atomic aliases operation."""

    def get_alias(self, **kwargs: object) -> object:
        """Resolve one configured index alias."""

    def exists_alias(self, **kwargs: object) -> object:
        """Return whether one configured index alias exists."""


class _ElasticsearchClient(Protocol):
    @property
    def indices(self) -> _ElasticsearchIndicesClient:
        """Return the index administration namespace."""

    def bulk(self, **kwargs: object) -> object:
        """Execute one bulk operation."""

    def search(self, **kwargs: object) -> object:
        """Execute one search operation."""

    def count(self, **kwargs: object) -> object:
        """Count visible documents in one physical index."""


_ClientFactory = Callable[[ElasticsearchConfiguration], _ElasticsearchClient]


class _LazyClientOwner:
    """Construct and share one Elasticsearch client on first operation."""

    def __init__(
        self,
        configuration: ElasticsearchConfiguration,
        client_factory: _ClientFactory,
    ) -> None:
        self.configuration = configuration
        self._client_factory = client_factory
        self._client: _ElasticsearchClient | None = None
        self._lock = threading.Lock()

    def client(self) -> _ElasticsearchClient:
        """Return the shared client, constructing it at most once."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._client_factory(self.configuration)
        return self._client


class ElasticsearchTextRetriever:
    """Group Elasticsearch text indexing and independent candidate lanes."""

    provider_id = _PROVIDER_ID

    def __init__(
        self,
        configuration: ElasticsearchConfiguration,
        *,
        client_factory: _ClientFactory | None = None,
    ) -> None:
        if not isinstance(configuration, ElasticsearchConfiguration):
            raise TypeError("configuration must be an ElasticsearchConfiguration")
        owner = _LazyClientOwner(configuration, client_factory or _default_client_factory)
        self.index_writer: TextIndexWriter = _ElasticsearchTextIndexWriter(owner)
        self.lexical_retriever: LexicalCandidateRetriever = _ElasticsearchLexicalRetriever(owner)
        self.dense_retriever: DenseCandidateRetriever = _ElasticsearchDenseRetriever(owner)
        self.generation_lifecycle: TextIndexGenerationLifecycle = (
            _ElasticsearchTextIndexGenerationLifecycle(owner)
        )


class _ElasticsearchTextIndexGenerationLifecycle:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _LazyClientOwner) -> None:
        self._owner = owner

    def mapping_sha256(
        self,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...],
    ) -> str:
        return text_index_mapping_sha256(embedding_spaces)

    def create(self, manifest: TextIndexGenerationManifest) -> str:
        physical_ref = self._physical_ref(manifest)
        mapping = build_managed_text_index_mapping(manifest)
        try:
            response = self._owner.client().indices.create(
                index=physical_ref,
                mappings=mapping,
            )
            _require_acknowledged(response)
        except Exception:
            raise ElasticsearchOperationError("Elasticsearch index creation failed") from None
        return physical_ref

    def validate(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexValidationReport:
        physical_ref = self._physical_ref(manifest)
        try:
            client = self._owner.client()
            block_response = client.indices.add_block(
                index=physical_ref,
                block="write",
            )
            block = _provider_mapping(block_response)
            if (
                block.get("acknowledged") is True
                and block.get("indices") == []
                and block.get("shards_acknowledged") is False
            ):
                settings = _provider_mapping(
                    client.indices.get_settings(
                        index=physical_ref, name="index.blocks.write", flat_settings=True
                    )
                )
                target = _provider_mapping(settings.get(physical_ref))
                value = _provider_mapping(target.get("settings")).get("index.blocks.write")
                if value is not True and value != "true":
                    raise ValueError("physical write block is absent")
            else:
                _require_write_block(block_response, physical_ref)
            refresh_response = client.indices.refresh(index=physical_ref)
            _require_no_failed_shards(refresh_response)
            mapping_response = client.indices.get_mapping(index=physical_ref)
            mapping = _extract_physical_mapping(mapping_response, physical_ref)
            count_response = client.count(index=physical_ref)
            observed_unit_count = _decode_count(count_response)
            issues = list(inspect_managed_text_index_mapping(mapping, manifest))
            if observed_unit_count != manifest.expected_unit_count:
                issues.append("document_count_mismatch")
            return TextIndexValidationReport(
                generation_id=manifest.generation_id,
                manifest_sha256=manifest.manifest_sha256,
                physical_index_ref=physical_ref,
                mapping_sha256=manifest.mapping_sha256,
                expected_unit_count=manifest.expected_unit_count,
                observed_unit_count=observed_unit_count,
                issue_codes=tuple(issues),
            )
        except Exception:
            raise ElasticsearchOperationError("Elasticsearch index validation failed") from None

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration:
        physical_ref = self._physical_ref(manifest)
        _require_ready_validation(manifest, validation, physical_ref)
        alias = self._owner.configuration.active_alias
        try:
            indices = self._owner.client().indices
            actions: list[dict[str, object]] = []
            exists = indices.exists_alias(name=alias)
            if exists:
                alias_response = indices.get_alias(name=alias)
                current_ref = _extract_single_alias_target(alias_response, alias)
                actions.append(
                    {
                        "remove": {
                            "index": current_ref,
                            "alias": alias,
                            "must_exist": True,
                        }
                    }
                )
            actions.append(
                {
                    "add": {
                        "index": physical_ref,
                        "alias": alias,
                        "is_write_index": False,
                    }
                }
            )
            response = indices.update_aliases(actions=actions)
            _require_acknowledged(response)
            if isinstance(response, Mapping) and response.get("errors") is True:
                raise ValueError
        except Exception:
            raise ElasticsearchOperationError("Elasticsearch index activation failed") from None
        return PinnedTextIndexGeneration(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref=physical_ref,
        )

    def resolve_active(self) -> PinnedTextIndexGeneration:
        alias = self._owner.configuration.active_alias
        try:
            indices = self._owner.client().indices
            alias_response = indices.get_alias(name=alias)
            physical_ref = _extract_single_alias_target(alias_response, alias)
            mapping_response = indices.get_mapping(index=physical_ref)
            mapping = _extract_physical_mapping(mapping_response, physical_ref)
            generation_id, manifest_sha256 = decode_managed_text_index_identity(mapping)
            if physical_ref != physical_index_name(
                self._owner.configuration.index_prefix,
                generation_id,
            ):
                raise ValueError
            return PinnedTextIndexGeneration(
                generation_id=generation_id,
                manifest_sha256=manifest_sha256,
                physical_index_ref=physical_ref,
            )
        except Exception:
            raise ElasticsearchOperationError(
                "Elasticsearch active generation resolution failed"
            ) from None

    def _physical_ref(self, manifest: TextIndexGenerationManifest) -> str:
        if not isinstance(manifest, TextIndexGenerationManifest):
            raise TypeError("manifest must be a TextIndexGenerationManifest")
        if manifest.retriever_id != self.provider_id:
            raise ValueError("manifest retriever_id does not match this provider")
        if manifest.index_format_version != _INDEX_FORMAT_VERSION:
            raise ValueError("manifest index_format_version is unsupported")
        build_managed_text_index_mapping(manifest)
        return physical_index_name(
            self._owner.configuration.index_prefix,
            manifest.generation_id,
        )


class _ElasticsearchTextIndexWriter:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _LazyClientOwner) -> None:
        self._owner = owner

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[TextIndexEntry, ...],
    ) -> None:
        index = physical_index_name(
            self._owner.configuration.index_prefix,
            index_generation_id,
        )
        materialized = tuple(entries)
        if any(not isinstance(entry, TextIndexEntry) for entry in materialized):
            raise TypeError("entries must contain TextIndexEntry values")
        unit_ids = tuple(entry.unit.unit_id for entry in materialized)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("entries must not contain duplicate unit IDs")
        if not materialized:
            return
        operations: list[object] = []
        for entry in materialized:
            operations.append({"create": {"_id": entry.unit.unit_id}})
            operations.append(encode_text_index_entry(entry))
        try:
            response = self._owner.client().bulk(
                index=index,
                operations=operations,
                refresh=False,
            )
            if _provider_mapping(response).get("errors") is not False:
                raise ValueError
        except Exception:
            raise ElasticsearchOperationError("Elasticsearch bulk operation failed") from None


class _ElasticsearchLexicalRetriever:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _LazyClientOwner) -> None:
        self._owner = owner

    def retrieve(
        self,
        *,
        query: TextQuery,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        if not isinstance(query, TextQuery):
            raise TypeError("query must be a TextQuery")
        _validate_filters(filters)
        _validate_top_k(top_k)
        if not isinstance(generation, PinnedTextIndexGeneration):
            raise TypeError("generation must be a PinnedTextIndexGeneration")
        index = generation.physical_index_ref
        bool_query: dict[str, object] = {
            "must": [{"match": {"text": {"query": query.text}}}],
        }
        filter_clauses = _translate_filters(filters)
        if filter_clauses:
            bool_query["filter"] = filter_clauses
        try:
            response = self._owner.client().search(
                index=index,
                size=top_k,
                query={"bool": bool_query},
                source=list(_SOURCE_FIELDS),
                highlight=_HIGHLIGHT,
            )
        except Exception:
            raise ElasticsearchOperationError("Elasticsearch search operation failed") from None
        try:
            return decode_text_candidates(
                response,
                provider_id=self.provider_id,
                generation_id=generation.generation_id,
                score_semantics="elasticsearch_bm25_score",
            )
        except (ValueError, TypeError, KeyError):
            raise _ElasticsearchDataError("Elasticsearch result integrity failed") from None


class _ElasticsearchDenseRetriever:
    provider_id = _PROVIDER_ID

    def __init__(self, owner: _LazyClientOwner) -> None:
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
        vector = _validate_query_vector(query_vector, embedding_space)
        _validate_filters(filters)
        _validate_top_k(top_k)
        if not isinstance(generation, PinnedTextIndexGeneration):
            raise TypeError("generation must be a PinnedTextIndexGeneration")
        index = generation.physical_index_ref
        knn: dict[str, object] = {
            "field": vector_field_name(embedding_space.space_id),
            "query_vector": list(vector),
            "k": top_k,
            "num_candidates": min(
                top_k * self._owner.configuration.knn_candidate_multiplier,
                10_000,
            ),
        }
        filter_clauses = _translate_filters(filters)
        if filter_clauses:
            knn["filter"] = filter_clauses
        try:
            response = self._owner.client().search(
                index=index,
                size=top_k,
                knn=knn,
                source=list(_SOURCE_FIELDS),
            )
        except Exception:
            raise ElasticsearchOperationError("Elasticsearch search operation failed") from None
        try:
            return decode_text_candidates(
                response,
                provider_id=self.provider_id,
                generation_id=generation.generation_id,
                score_semantics="elasticsearch_knn_score",
            )
        except (ValueError, TypeError, KeyError):
            raise _ElasticsearchDataError("Elasticsearch result integrity failed") from None


def _require_ready_validation(
    manifest: TextIndexGenerationManifest,
    validation: TextIndexValidationReport,
    physical_ref: str,
) -> None:
    if not isinstance(validation, TextIndexValidationReport):
        raise TypeError("validation must be a TextIndexValidationReport")
    if not validation.valid:
        raise ValueError("generation must have a ready validation report")
    if validation.generation_id != manifest.generation_id:
        raise ValueError("validation generation does not match manifest")
    if validation.manifest_sha256 != manifest.manifest_sha256:
        raise ValueError("validation manifest digest does not match manifest")
    if validation.mapping_sha256 != manifest.mapping_sha256:
        raise ValueError("validation mapping digest does not match manifest")
    if validation.physical_index_ref != physical_ref:
        raise ValueError("validation physical index does not match manifest")
    if validation.expected_unit_count != manifest.expected_unit_count:
        raise ValueError("validation unit count does not match manifest")


def _extract_physical_mapping(response: object, physical_ref: str) -> Mapping[object, object]:
    root = _provider_mapping(response)
    if set(root) != {physical_ref}:
        raise ValueError
    index = _provider_mapping(root[physical_ref])
    return _provider_mapping(index["mappings"])


def _extract_single_alias_target(response: object, alias: str) -> str:
    root = _provider_mapping(response)
    if len(root) != 1:
        raise ValueError
    physical_ref = next(iter(root))
    if not isinstance(physical_ref, str) or not physical_ref.strip():
        raise ValueError
    target = _provider_mapping(root[physical_ref])
    aliases = _provider_mapping(target["aliases"])
    if set(aliases) != {alias}:
        raise ValueError
    return physical_ref


def _decode_count(response: object) -> int:
    root = _provider_mapping(response)
    _require_no_failed_shards(root)
    count = root.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError
    return count


def _require_acknowledged(response: object) -> None:
    root = _provider_mapping(response)
    if root.get("acknowledged") is not True:
        raise ValueError


def _require_write_block(response: object, physical_ref: str) -> None:
    root = _provider_mapping(response)
    if root.get("acknowledged") is not True or root.get("shards_acknowledged") is not True:
        raise ValueError
    indices = root.get("indices")
    if not isinstance(indices, list) or len(indices) != 1:
        raise ValueError
    result = _provider_mapping(indices[0])
    if result.get("name") != physical_ref or result.get("blocked") is not True:
        raise ValueError


def _require_no_failed_shards(response: object) -> None:
    root = _provider_mapping(response)
    shards = _provider_mapping(root.get("_shards"))
    failed = shards.get("failed")
    if isinstance(failed, bool) or not isinstance(failed, int) or failed != 0:
        raise ValueError


def _provider_mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        value = getattr(value, "body", None)
    if not isinstance(value, Mapping):
        raise TypeError
    return value


def _translate_filters(filters: SearchFilters) -> list[dict[str, object]]:
    clauses: list[dict[str, object]] = []
    if filters.corpus_refs:
        clauses.append({"terms": {"corpus_refs": sorted(filters.corpus_refs)}})
    if filters.artifact_refs:
        clauses.append({"terms": {"artifact_ref": sorted(filters.artifact_refs)}})
    if filters.media_types:
        clauses.append({"terms": {"media_type": sorted(filters.media_types)}})
    if filters.text_origins:
        clauses.append(
            {"terms": {"origin": sorted(origin.value for origin in filters.text_origins)}}
        )
    if filters.time_range is not None:
        clauses.extend(
            (
                {"terms": {"origin": [TextOrigin.ASR.value, TextOrigin.OCR.value]}},
                {"range": {"start_ms": {"lt": filters.time_range.end_ms}}},
                {"range": {"end_ms": {"gt": filters.time_range.start_ms}}},
            )
        )
    return clauses


def _validate_filters(filters: SearchFilters) -> None:
    if not isinstance(filters, SearchFilters) or filters.schema_version != 2:
        raise ValueError("filters must use SearchFilters schema version 2")


def _validate_top_k(top_k: int) -> None:
    # Candidate depth is internal; the public search request remains capped at 100.
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 300:
        raise ValueError("top_k must be an integer from 1 to 300")


def _validate_query_vector(
    query_vector: tuple[float, ...],
    embedding_space: EmbeddingSpaceRef,
) -> tuple[float, ...]:
    if not isinstance(embedding_space, EmbeddingSpaceRef) or not embedding_space.accepts(
        QueryModality.TEXT,
        TargetType.TEXT_CHUNK,
    ):
        raise ValueError("embedding_space must accept text queries and text chunks")
    vector = tuple(query_vector)
    if len(vector) != embedding_space.dimension:
        raise ValueError("query_vector dimension must match embedding_space")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("query_vector must contain only finite numbers")
    return tuple(float(value) for value in vector)


def _default_client_factory(
    configuration: ElasticsearchConfiguration,
) -> _ElasticsearchClient:
    """Import and construct the optional SDK only on the first operation."""
    module = importlib.import_module("elasticsearch")
    constructor = module.Elasticsearch
    keyword_arguments: dict[str, object] = {
        "hosts": list(configuration.hosts),
        "verify_certs": configuration.verify_certs,
        "request_timeout": configuration.request_timeout,
    }
    if configuration.api_key is not None:
        keyword_arguments["api_key"] = configuration.api_key
    if configuration.ca_certs is not None:
        keyword_arguments["ca_certs"] = configuration.ca_certs
    return cast(_ElasticsearchClient, constructor(**keyword_arguments))


__all__ = [
    "ElasticsearchOperationError",
    "ElasticsearchTextRetriever",
]
