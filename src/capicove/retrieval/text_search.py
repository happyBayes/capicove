"""Provider-neutral orchestration for canonical text search."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

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
from capicove.retrieval.text_fusion import FusedTextCandidate, weighted_reciprocal_rank_fusion
from capicove.retrieval.text_profiles import StaticTextRetrievalProfileRegistry


if TYPE_CHECKING:
    from capicove.retrieval.generations import PinnedTextIndexGeneration
    from capicove.retrieval.pipeline_profiles import RerankPolicy
    from capicove.retrieval.reranking import TextReranker

# Resolve the full text of fused candidates for reranking; defaults to snippets.
RerankTextSource = Callable[
    ["PinnedTextIndexGeneration", tuple[TextCandidate, ...]], tuple[str, ...]
]


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
        rerank_policy: RerankPolicy | None = None,
        reranker: TextReranker | None = None,
        rerank_texts: RerankTextSource | None = None,
    ) -> None:
        """Bind ports; an enabled ``rerank_policy`` reranks the fused head.

        Without a policy (or with reranking disabled) the service behaves exactly
        as before. ``rerank_texts`` supplies full candidate text; snippets are
        used otherwise.
        """
        self._rerank_policy = rerank_policy
        self._reranker = reranker
        self._rerank_texts = rerank_texts
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
            reranking = self._rerank_policy is not None and self._rerank_policy.enabled
            depth = request.top_k
            if reranking:
                assert self._rerank_policy is not None
                depth = min(100, max(request.top_k, self._rerank_policy.candidate_depth))
            fused = weighted_reciprocal_rank_fusion(
                lexical=lexical,
                dense=dense,
                profile=profile,
                top_k=depth,
            )
        except (TypeError, ValueError):
            raise TextSearchError("index_corrupt") from None
        if reranking:
            fused, rerank_warning = self._rerank(query, fused, generation)
            fused = fused[: request.top_k]
            if rerank_warning:
                # The canonical warning set has no rerank code; a dedicated one
                # is proposed in RFC-0034 rather than changing Tool schemas here.
                warnings.append(SearchWarningCode.RETRIEVAL_LANE_UNAVAILABLE)
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

    def _rerank(
        self, query: TextQuery, fused: tuple[FusedTextCandidate, ...], generation: object
    ) -> tuple[tuple[FusedTextCandidate, ...], bool]:
        """Reorder the fused head; return whether the fused order was kept degraded."""
        from capicove.retrieval.rerank_stage import RerankUnavailableError, rerank_fused

        assert self._rerank_policy is not None
        candidates = tuple(item.candidate for item in fused)
        try:
            if self._rerank_texts is None:
                texts = tuple(candidate.snippet for candidate in candidates)
            else:
                texts = self._rerank_texts(generation, candidates)  # type: ignore[arg-type]
                if len(texts) != len(candidates):
                    raise ValueError("rerank text count mismatch")
        except Exception:
            if self._rerank_policy.on_unavailable == "fail":
                raise TextSearchError("provider_unavailable") from None
            return fused, True
        try:
            outcome = rerank_fused(query.text, texts, self._rerank_policy, self._reranker)
        except RerankUnavailableError:
            raise TextSearchError("provider_unavailable") from None
        return tuple(fused[index] for index in outcome.order), outcome.status == "degraded"

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
