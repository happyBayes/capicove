"""Use the existing versioned text-search service as the optional memory dense lane.

The selected host profile must be dense-only. Index entries use memory target IDs
as unit IDs and the scoped corpus below; normal memory filters revalidate all hits.
No model, service, index, or network connection is created by this adapter.
"""

from capicove.contracts import SearchFilters, SearchRequest, TargetType, TextQuery
from capicove.contracts.knowledge import digest
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge.errors import OptionalMemoryLaneUnavailable
from capicove.retrieval.text_profiles import TextRetrievalProfile
from capicove.retrieval.text_search import TextSearchError, TextSearchService


def memory_corpus(workspace: str, scope: str) -> str:
    return "memory:" + digest([workspace, scope])


class MemoryTextSearchLane:
    def __init__(self, service: TextSearchService, *, profile: TextRetrievalProfile) -> None:
        if profile.lexical_weight != 0 or profile.dense_weight <= 0:
            raise ValueError("memory dense lane requires a dense-only text profile")
        self.service, self.profile = service, profile

    def __call__(self, query: MemoryQuery) -> tuple[str, ...]:
        if not query.text.strip():
            return ()
        try:
            result = self.service.search(
                SearchRequest(
                    TextQuery(query.text),
                    frozenset({TargetType.TEXT_CHUNK}),
                    top_k=100,
                    retrieval_profile_ref=self.profile.profile_id,
                    filters=SearchFilters(
                        corpus_refs=frozenset({memory_corpus(query.workspace_ref, query.scope_ref)})
                    ),
                ),
                strict_provider_errors=True,
            )
        except TextSearchError as exc:
            if exc.code not in {
                "provider_unavailable",
                "index_generation_unavailable",
                "index_corrupt",
                "embedding_space_incompatible",
            }:
                raise
            raise OptionalMemoryLaneUnavailable("embedding_unavailable") from exc
        if result.degraded:
            raise OptionalMemoryLaneUnavailable("embedding_unavailable")
        return tuple(hit.id for hit in result.hits)
