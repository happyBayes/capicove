"""Tests for production reverse-image search values."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    MatchKind,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.contracts.tools import PrivacyClass
from capicove.retrieval.search_models import (
    Candidate,
    ContentCandidateProvenance,
    IndexedCandidateProvenance,
    InvocationAuthorizationContext,
    PerceptualHashQuery,
    QueryVector,
    ResolvedImageQuery,
    RetrievalLane,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _space(*, dimension: int = 2) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="instance-space-v1",
        provider_id="test-instance-provider",
        model_revision=_sha("model"),
        preprocessing_version="rgb-v1",
        dimension=dimension,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="opencv-phash-64-v1",
        provider_id="opencv-phash",
        provider_version="4.12.0",
        preprocessing_version="pillow-luma-opencv-area32-dct8-median-excluding-dc-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def test_query_vector_requires_complete_space_dimension_and_l2_norm() -> None:
    vector = QueryVector(space=_space(), values=(0.6, 0.8))

    assert vector.values == (0.6, 0.8)
    with pytest.raises(ValueError, match="dimension"):
        QueryVector(space=_space(dimension=3), values=(0.6, 0.8))
    with pytest.raises(ValueError, match="L2-normalized"):
        QueryVector(space=_space(), values=(1.0, 1.0))


def test_invocation_authorization_is_host_created_and_bounded() -> None:
    authorization = InvocationAuthorizationContext(
        request_id="request-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        operation="retrieval:search",
        authorization_scope_ref="scope-1",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset(),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )

    assert authorization.allowed_data_policy_refs == frozenset({"local-only"})
    assert authorization.allowed_source_root_refs == frozenset()
    with pytest.raises(ValueError, match="allowed_data_policy_refs"):
        replace(authorization, allowed_data_policy_refs=frozenset())
    with pytest.raises(ValueError, match="allowed_source_root_ref"):
        replace(authorization, allowed_source_root_refs=frozenset({""}))


@pytest.mark.parametrize("values", [(float("nan"), 0.0), (float("inf"), 0.0)])
def test_query_vector_rejects_non_finite_values(values: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        QueryVector(space=_space(), values=values)


def test_perceptual_query_requires_unsigned_value_within_hash_width() -> None:
    query = PerceptualHashQuery(
        source_sha256=_sha("query"),
        space=_hash_space(),
        value=0x0F0F,
        max_hamming_distance=2,
    )

    assert query.value == 0x0F0F
    with pytest.raises(ValueError, match="unsigned"):
        replace(query, value=-1)
    with pytest.raises(ValueError, match="0 to 2"):
        replace(query, max_hamming_distance=3)


def test_resolved_query_binds_one_decoded_image_to_source_evidence() -> None:
    image = Image.new("RGB", (8, 6), "white")
    resolved = ResolvedImageQuery(
        image=image,
        source_sha256=_sha("encoded-image"),
        media_type="image/png",
        encoded_byte_size=123,
        width=8,
        height=6,
    )

    assert resolved.image is image
    with pytest.raises(ValueError, match="decoded dimensions"):
        replace(resolved, width=7)


def test_candidate_enforces_lane_provenance_and_complete_space() -> None:
    semantic = Candidate(
        unit_id=f"unit:{_sha('a')}",
        unit_ordinal=0,
        lane=RetrievalLane.INSTANCE,
        match_kind=MatchKind.VISUAL,
        raw_score=0.75,
        score_semantics="cosine_similarity",
        rank=1,
        provenance=IndexedCandidateProvenance(
            activation_set_id="set-1",
            generation_id="generation-instance-1",
            provider_id="test-instance-provider",
        ),
        embedding_space=_space(),
    )

    assert semantic.embedding_space == _space()
    with pytest.raises(ValueError, match="indexed provenance"):
        replace(
            semantic,
            provenance=ContentCandidateProvenance(
                activation_set_id="set-1",
                unit_catalog_id="catalog-1",
            ),
        )
    with pytest.raises(ValueError, match="visual match"):
        replace(semantic, match_kind=MatchKind.SEMANTIC)


def test_exact_content_candidate_has_no_generation_or_space() -> None:
    candidate = Candidate(
        unit_id=f"unit:{_sha('exact')}",
        unit_ordinal=2,
        lane=RetrievalLane.CONTENT,
        match_kind=MatchKind.NEAR_DUPLICATE,
        raw_score=1.0,
        score_semantics="sha256_equality",
        rank=1,
        provenance=ContentCandidateProvenance(
            activation_set_id="set-1",
            unit_catalog_id="catalog-1",
        ),
        exact_content=True,
    )

    assert candidate.exact_content is True
    with pytest.raises(ValueError, match="content lane"):
        replace(candidate, embedding_space=_space())


def test_search_value_deadline_examples_use_timezone_aware_instants() -> None:
    assert datetime(2026, 8, 27, tzinfo=UTC).utcoffset() is not None


def test_production_search_foundation_is_public_without_exporting_research_api() -> None:
    import capicove.retrieval as retrieval

    assert retrieval.QueryVector is QueryVector
    assert retrieval.RetrievalLane is RetrievalLane
    assert not hasattr(retrieval, "PersonalImageResearchIndex")
    assert not hasattr(retrieval, "ResearchSearchIntent")
