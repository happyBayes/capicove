"""Provider-neutral orchestration for canonical text search."""

from __future__ import annotations

from collections.abc import Callable

from capicove.contracts import (
    CompactHit,
    DetailLevel,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TextQuery,
)
from capicove.retrieval.generations import TextIndexGenerationCatalogError, TextIndexGenerationState
from capicove.retrieval.models import TextCandidate
from capicove.retrieval.ports import (
    TextIndexGenerationCatalog,
    TextQueryEmbeddingProvider,
    TextRetrieverBackend,
)
from capicove.retrieval.text_fusion import weighted_reciprocal_rank_fusion
from capicove.retrieval.text_profiles import StaticTextRetrievalProfileRegistry


class TextSearchError(RuntimeError):
    """Report one stable provider-neutral text search failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class TextSearchService:
    """Pin one generation, execute configured lanes, and project canonical hits."""

    def __init__(
        self,
        *,
        backend: TextRetrieverBackend,
        catalog: TextIndexGenerationCatalog,
        profiles: StaticTextRetrievalProfileRegistry,
        query_embedder: TextQueryEmbeddingProvider,
        query_id_factory: Callable[[], str],
    ) -> None:
        self._backend = backend
        self._catalog = catalog
        self._profiles = profiles
        self._query_embedder = query_embedder
        self._query_id_factory = query_id_factory

    def search(
        self, request: SearchRequest, *, strict_provider_errors: bool = False
    ) -> SearchResult:
        """Execute one compact canonical text search."""
        query = self._validate_request(request)
        try:
            profile = self._profiles.resolve(request.retrieval_profile_ref)
        except LookupError:
            raise TextSearchError("invalid_query") from None
        try:
            generation = self._backend.generation_lifecycle.resolve_active()
        except Exception as exc:
            if strict_provider_errors and not isinstance(
                exc, (OSError, TextSearchError, TextIndexGenerationCatalogError)
            ):
                raise
            raise TextSearchError("index_generation_unavailable") from None
        record = self._catalog.get(generation.generation_id)
        if (
            record is None
            or record.state is not TextIndexGenerationState.ACTIVE
            or record.manifest.manifest_sha256 != generation.manifest_sha256
            or record.physical_index_ref != generation.physical_index_ref
        ):
            raise TextSearchError("index_corrupt")
        if profile.embedding_space is not None and profile.embedding_space not in (
            record.manifest.embedding_spaces
        ):
            raise TextSearchError("embedding_space_incompatible")

        lexical: tuple[TextCandidate, ...] = ()
        dense: tuple[TextCandidate, ...] = ()
        warnings: list[SearchWarningCode] = []
        successes = 0
        if profile.lexical_weight > 0:
            try:
                lexical = self._backend.lexical_retriever.retrieve(
                    query=query,
                    filters=request.filters,
                    generation=generation,
                    top_k=profile.lexical_candidate_count,
                )
                successes += 1
            except Exception as exc:
                if strict_provider_errors and not isinstance(
                    exc, (OSError, TextSearchError, TextIndexGenerationCatalogError)
                ):
                    raise
                if profile.lexical_required:
                    raise TextSearchError("provider_unavailable") from None
                warnings.append(SearchWarningCode.TEXT_LEXICAL_LANE_UNAVAILABLE)
        if profile.dense_weight > 0:
            try:
                if profile.embedding_space is None:
                    raise ValueError
                vector = self._query_embedder.embed_query(
                    query=query,
                    embedding_space=profile.embedding_space,
                )
            except Exception as exc:
                if strict_provider_errors and not isinstance(
                    exc, (OSError, TextSearchError, TextIndexGenerationCatalogError)
                ):
                    raise
                if profile.dense_required:
                    raise TextSearchError("provider_unavailable") from None
                warnings.append(SearchWarningCode.TEXT_DENSE_EMBEDDING_UNAVAILABLE)
            else:
                try:
                    dense = self._backend.dense_retriever.retrieve(
                        query_vector=vector,
                        embedding_space=profile.embedding_space,
                        filters=request.filters,
                        generation=generation,
                        top_k=profile.dense_candidate_count,
                    )
                    successes += 1
                except Exception as exc:
                    if strict_provider_errors and not isinstance(
                        exc, (OSError, TextSearchError, TextIndexGenerationCatalogError)
                    ):
                        raise
                    if profile.dense_required:
                        raise TextSearchError("provider_unavailable") from None
                    warnings.append(SearchWarningCode.TEXT_DENSE_LANE_UNAVAILABLE)
        if successes == 0:
            raise TextSearchError("provider_unavailable")
        try:
            for candidates in (lexical, dense):
                for candidate in candidates:
                    if (
                        not isinstance(candidate, TextCandidate)
                        or candidate.index_generation_id != generation.generation_id
                        or candidate.provider_id != self._backend.provider_id
                    ):
                        raise ValueError("candidate does not belong to the pinned backend")
            fused = weighted_reciprocal_rank_fusion(
                lexical=lexical,
                dense=dense,
                profile=profile,
                top_k=request.top_k,
            )
        except (TypeError, ValueError):
            raise TextSearchError("index_corrupt") from None
        hits = tuple(
            CompactHit(
                id=item.candidate.unit_id,
                target_type=TargetType.TEXT_CHUNK,
                brief_score=item.score,
                evidence_id=item.candidate.evidence_ref,
                text_locator=item.candidate.text_locator,
                snippet=item.candidate.snippet,
            )
            for item in fused
        )
        return SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id=self._query_id_factory(),
            degraded=bool(warnings),
            warning_codes=tuple(warnings),
            hits=hits,
        )

    @staticmethod
    def _validate_request(request: SearchRequest) -> TextQuery:
        if not isinstance(request, SearchRequest):
            raise TextSearchError("invalid_query")
        if (
            not isinstance(request.query, TextQuery)
            or request.targets != frozenset({TargetType.TEXT_CHUNK})
            or request.detail_level is not DetailLevel.COMPACT
            or request.cursor is not None
        ):
            raise TextSearchError("invalid_query")
        return request.query


__all__ = ["TextSearchError", "TextSearchService"]
