"""Tests for deterministic provider-neutral weighted RRF."""

import pytest

from capicove.contracts import CharacterRange, TextChunkLocator, TextOrigin
from capicove.retrieval import (
    TextCandidate,
    TextRetrievalProfile,
    weighted_reciprocal_rank_fusion,
)


def profile() -> TextRetrievalProfile:
    """Return one lexical-only-compatible hybrid weighting policy for fusion."""
    from tests.unit.retrieval.test_text_models import text_embedding_space

    return TextRetrievalProfile(
        profile_id="hybrid",
        lexical_weight=2.0,
        dense_weight=1.0,
        lexical_required=True,
        dense_required=False,
        lexical_candidate_count=20,
        dense_candidate_count=20,
        embedding_space=text_embedding_space(),
    )


def candidate(unit_id: str, *, rank: int, score: float = 1.0) -> TextCandidate:
    """Return one agreed provider candidate."""
    return TextCandidate(
        unit_id=unit_id,
        evidence_ref=f"evidence-{unit_id}",
        text_locator=TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 5)),
        snippet=f"text {unit_id}",
        raw_score=score,
        score_semantics="provider-score",
        rank=rank,
        provider_id="capicove.retriever.opensearch",
        index_generation_id="generation-1",
    )


def test_weighted_rrf_normalizes_by_successful_lane_maximum() -> None:
    fused = weighted_reciprocal_rank_fusion(
        lexical=(candidate("a", rank=1), candidate("b", rank=2)),
        dense=(candidate("b", rank=1), candidate("a", rank=2)),
        profile=profile(),
        top_k=2,
    )

    expected_a = ((2 / 61) + (1 / 62)) / (3 / 61)
    expected_b = ((2 / 62) + (1 / 61)) / (3 / 61)
    assert [item.candidate.unit_id for item in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(expected_a)
    assert fused[1].score == pytest.approx(expected_b)


def test_weighted_rrf_uses_unit_id_for_exact_score_ties_and_applies_top_k() -> None:
    fused = weighted_reciprocal_rank_fusion(
        lexical=(candidate("b", rank=1), candidate("a", rank=1)),
        dense=(),
        profile=profile(),
        top_k=1,
    )

    assert [item.candidate.unit_id for item in fused] == ["a"]
    assert fused[0].score == pytest.approx(1.0)


def test_weighted_rrf_rejects_duplicate_units_inside_one_lane() -> None:
    with pytest.raises(ValueError, match="duplicate unit"):
        weighted_reciprocal_rank_fusion(
            lexical=(candidate("a", rank=1), candidate("a", rank=2)),
            dense=(),
            profile=profile(),
            top_k=10,
        )


def test_weighted_rrf_rejects_cross_lane_identity_disagreement() -> None:
    dense = candidate("a", rank=1)
    object.__setattr__(dense, "evidence_ref", "different-evidence")

    with pytest.raises(ValueError, match="identity disagreement"):
        weighted_reciprocal_rank_fusion(
            lexical=(candidate("a", rank=1),),
            dense=(dense,),
            profile=profile(),
            top_k=10,
        )


@pytest.mark.parametrize("top_k", [0, 101, True])
def test_weighted_rrf_rejects_invalid_public_limits(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        weighted_reciprocal_rank_fusion(
            lexical=(candidate("a", rank=1),),
            dense=(),
            profile=profile(),
            top_k=top_k,
        )


def test_shared_fusion_preserves_invalid_candidate_error() -> None:
    with pytest.raises(TypeError, match="TextCandidate"):
        weighted_reciprocal_rank_fusion(
            lexical=(object(),),  # type: ignore[arg-type]
            dense=(),
            profile=profile(),
            top_k=10,
        )
