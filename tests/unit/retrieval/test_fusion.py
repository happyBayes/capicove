"""Tests for deterministic rank-only reverse-image fusion."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    MatchKind,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.fusion import WeightedReciprocalRankFusion
from capicove.retrieval.search_models import (
    Candidate,
    CandidateLane,
    CandidateProvenance,
    ContentCandidateProvenance,
    IndexedCandidateProvenance,
    RetrievalLane,
)


def _unit(label: str) -> str:
    return f"unit:{hashlib.sha256(label.encode()).hexdigest()}"


def _space(lane: RetrievalLane) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-space-v1",
        provider_id=f"provider-{lane.value}",
        model_revision="a" * 64,
        preprocessing_version="rgb-v1",
        dimension=2,
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
        preprocessing_version="rgb-luma-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _candidate(
    label: str,
    *,
    lane: RetrievalLane,
    rank: int,
    raw_score: float,
    exact: bool = False,
    ordinal: int = 0,
) -> Candidate:
    provenance: CandidateProvenance
    if lane is RetrievalLane.CONTENT:
        provenance = ContentCandidateProvenance("set-1", "catalog-1")
        match_kind = MatchKind.NEAR_DUPLICATE
    else:
        provenance = IndexedCandidateProvenance(
            "set-1",
            f"generation-{lane.value}",
            f"provider-{lane.value}",
        )
        match_kind = MatchKind.VISUAL if lane is RetrievalLane.INSTANCE else MatchKind.SEMANTIC
        if lane is RetrievalLane.PERCEPTUAL:
            match_kind = MatchKind.NEAR_DUPLICATE
    embedding_space = (
        _space(lane)
        if lane
        in {
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }
        else None
    )
    hash_space = _hash_space() if lane is RetrievalLane.PERCEPTUAL else None
    return Candidate(
        unit_id=_unit(label),
        unit_ordinal=ordinal,
        lane=lane,
        match_kind=match_kind,
        raw_score=raw_score,
        score_semantics=f"{lane.value}-native",
        rank=rank,
        provenance=provenance,
        embedding_space=embedding_space,
        hash_space=hash_space,
        exact_content=exact,
    )


def test_weighted_rrf_never_compares_incompatible_raw_scores() -> None:
    fused = WeightedReciprocalRankFusion().fuse(
        (
            CandidateLane(
                RetrievalLane.PERCEPTUAL,
                weight=2.0,
                candidates=(
                    _candidate(
                        "near",
                        lane=RetrievalLane.PERCEPTUAL,
                        rank=1,
                        raw_score=2.0,
                    ),
                ),
            ),
            CandidateLane(
                RetrievalLane.INSTANCE,
                weight=1.0,
                candidates=(
                    _candidate(
                        "semantic",
                        lane=RetrievalLane.INSTANCE,
                        rank=1,
                        raw_score=0.99,
                        ordinal=1,
                    ),
                ),
            ),
        ),
        limit=2,
    )

    assert [item.unit_id for item in fused] == [_unit("near"), _unit("semantic")]
    assert [item.brief_score for item in fused] == [1.0, 0.5]


def test_exact_content_precedes_higher_rrf_candidate() -> None:
    exact = _candidate(
        "exact",
        lane=RetrievalLane.CONTENT,
        rank=1,
        raw_score=1.0,
        exact=True,
        ordinal=1,
    )
    stronger = _candidate(
        "stronger",
        lane=RetrievalLane.PERCEPTUAL,
        rank=1,
        raw_score=0.0,
    )

    fused = WeightedReciprocalRankFusion().fuse(
        (
            CandidateLane(RetrievalLane.CONTENT, 1.0, (exact,)),
            CandidateLane(RetrievalLane.PERCEPTUAL, 100.0, (stronger,)),
        ),
        limit=2,
    )

    assert [item.unit_id for item in fused] == [_unit("exact"), _unit("stronger")]


def test_candidate_present_in_multiple_lanes_unions_evidence() -> None:
    instance = _candidate(
        "shared",
        lane=RetrievalLane.INSTANCE,
        rank=2,
        raw_score=-99.0,
        ordinal=7,
    )
    concept = replace(
        _candidate(
            "shared",
            lane=RetrievalLane.CONCEPT,
            rank=1,
            raw_score=999.0,
            ordinal=7,
        )
    )

    fused = WeightedReciprocalRankFusion().fuse(
        (
            CandidateLane(RetrievalLane.INSTANCE, 1.0, (instance,)),
            CandidateLane(RetrievalLane.CONCEPT, 1.0, (concept,)),
        ),
        limit=1,
    )

    assert fused[0].match_kinds == (MatchKind.VISUAL, MatchKind.SEMANTIC)
    assert [observation.lane for observation in fused[0].lane_observations] == [
        RetrievalLane.INSTANCE,
        RetrievalLane.CONCEPT,
    ]


def test_stable_unit_id_breaks_equal_rank_ties() -> None:
    left = _candidate("a", lane=RetrievalLane.INSTANCE, rank=1, raw_score=-10.0)
    right = _candidate("b", lane=RetrievalLane.INSTANCE, rank=1, raw_score=10.0, ordinal=1)

    fused = WeightedReciprocalRankFusion().fuse(
        (CandidateLane(RetrievalLane.INSTANCE, 1.0, (right, left)),),
        limit=2,
    )

    assert [item.unit_id for item in fused] == sorted((_unit("a"), _unit("b")))


def test_lane_rejects_duplicate_units_and_fusion_rejects_duplicate_lanes() -> None:
    candidate = _candidate("a", lane=RetrievalLane.INSTANCE, rank=1, raw_score=0.0)
    with pytest.raises(ValueError, match="duplicate unit"):
        CandidateLane(RetrievalLane.INSTANCE, 1.0, (candidate, candidate))

    lane = CandidateLane(RetrievalLane.INSTANCE, 1.0, (candidate,))
    with pytest.raises(ValueError, match="duplicate lane"):
        WeightedReciprocalRankFusion().fuse((lane, lane), limit=1)


@pytest.mark.parametrize("limit", [0, 10_001, True])
def test_fusion_requires_bounded_integer_limit(limit: object) -> None:
    with pytest.raises(ValueError, match="1 to 10,000"):
        WeightedReciprocalRankFusion().fuse((), limit=limit)  # type: ignore[arg-type]
