"""Transport-neutral multimodal retrieval capabilities."""

from capicove.retrieval.evaluation import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from capicove.retrieval.fusion import WeightedReciprocalRankFusion
from capicove.retrieval.generation_service import TextIndexGenerationService
from capicove.retrieval.generations import (
    PinnedTextIndexGeneration,
    TextIndexGenerationCatalogError,
    TextIndexGenerationConflictError,
    TextIndexGenerationEvent,
    TextIndexGenerationEventType,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationState,
    TextIndexGenerationTransitionError,
    TextIndexValidationReport,
)
from capicove.retrieval.models import (
    EmbeddingResult,
    ProviderMatch,
    RankedMatch,
    RankingResult,
    TextCandidate,
    TextEmbedding,
    TextEvidence,
    TextIndexEntry,
)
from capicove.retrieval.ports import (
    DenseCandidateRetriever,
    LexicalCandidateRetriever,
    MultimodalEmbeddingProvider,
    TextBatchEncoder,
    TextEvidenceReader,
    TextIndexGenerationCatalog,
    TextIndexGenerationLifecycle,
    TextIndexWriter,
    TextQueryEmbeddingProvider,
    TextRetrieverBackend,
)
from capicove.retrieval.profiles import (
    RetrievalProfile,
    RetrievalProfileRegistry,
    default_retrieval_profiles,
)
from capicove.retrieval.reranking import RerankItem, RerankResult, TextReranker
from capicove.retrieval.search import RetrievalService
from capicove.retrieval.search_models import (
    Candidate,
    CandidateLane,
    FusedCandidate,
    InvocationAuthorizationContext,
    PerceptualHashQuery,
    QueryVector,
    ResolvedImageQuery,
    RetrievalLane,
    SearchFailure,
)
from capicove.retrieval.search_ports import (
    FusionStrategy,
    ImageQueryResolver,
    PerceptualQueryEncoder,
    QueryEncoder,
    QueryEncoderResolver,
)
from capicove.retrieval.service import Siglip2RetrievalService
from capicove.retrieval.text_embedding import TextEmbeddingService, validate_vector
from capicove.retrieval.text_fusion import (
    FusedTextCandidate,
    weighted_reciprocal_rank_fusion,
)
from capicove.retrieval.text_profiles import (
    StaticTextRetrievalProfileRegistry,
    TextRetrievalProfile,
)
from capicove.retrieval.text_search import TextSearchError, TextSearchService
from capicove.retrieval.video_encoding import VideoEmbeddingAdapter
from capicove.retrieval.video_profiles import (
    StaticVideoRetrievalProfileRegistry,
    VideoRetrievalProfile,
)
from capicove.retrieval.video_search import VideoSearchError, VideoSearchService


__all__ = [
    "Candidate",
    "CandidateLane",
    "DenseCandidateRetriever",
    "EmbeddingResult",
    "FusedCandidate",
    "FusedTextCandidate",
    "FusionStrategy",
    "ImageQueryResolver",
    "InvocationAuthorizationContext",
    "LexicalCandidateRetriever",
    "MultimodalEmbeddingProvider",
    "PerceptualHashQuery",
    "PerceptualQueryEncoder",
    "PinnedTextIndexGeneration",
    "ProviderMatch",
    "QueryEncoder",
    "QueryEncoderResolver",
    "QueryVector",
    "RankedMatch",
    "RankingResult",
    "RerankItem",
    "RerankResult",
    "ResolvedImageQuery",
    "RetrievalLane",
    "RetrievalProfile",
    "RetrievalProfileRegistry",
    "RetrievalService",
    "SearchFailure",
    "Siglip2RetrievalService",
    "StaticTextRetrievalProfileRegistry",
    "StaticVideoRetrievalProfileRegistry",
    "TextBatchEncoder",
    "TextCandidate",
    "TextEmbedding",
    "TextEmbeddingService",
    "TextEvidence",
    "TextEvidenceReader",
    "TextIndexEntry",
    "TextIndexGenerationCatalog",
    "TextIndexGenerationCatalogError",
    "TextIndexGenerationConflictError",
    "TextIndexGenerationEvent",
    "TextIndexGenerationEventType",
    "TextIndexGenerationLifecycle",
    "TextIndexGenerationManifest",
    "TextIndexGenerationRecord",
    "TextIndexGenerationService",
    "TextIndexGenerationState",
    "TextIndexGenerationTransitionError",
    "TextIndexValidationReport",
    "TextIndexWriter",
    "TextQueryEmbeddingProvider",
    "TextReranker",
    "TextRetrievalProfile",
    "TextRetrieverBackend",
    "TextSearchError",
    "TextSearchService",
    "VideoEmbeddingAdapter",
    "VideoRetrievalProfile",
    "VideoSearchError",
    "VideoSearchService",
    "WeightedReciprocalRankFusion",
    "default_retrieval_profiles",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "validate_vector",
    "weighted_reciprocal_rank_fusion",
]
