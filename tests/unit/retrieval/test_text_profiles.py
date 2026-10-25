"""Tests for closed provider-neutral text retrieval profiles."""

from dataclasses import FrozenInstanceError

import pytest

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval import (
    StaticTextRetrievalProfileRegistry,
    TextRetrievalProfile,
)


def embedding_space() -> EmbeddingSpaceRef:
    """Return one compatible dense text space."""
    return EmbeddingSpaceRef(
        space_id="text-v1",
        provider_id="embedder",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def test_profile_freezes_closed_hybrid_policy() -> None:
    profile = TextRetrievalProfile(
        profile_id="local-hybrid",
        lexical_weight=2.0,
        dense_weight=1.0,
        lexical_required=True,
        dense_required=False,
        lexical_candidate_count=50,
        dense_candidate_count=40,
        embedding_space=embedding_space(),
    )

    assert profile.fusion_algorithm == "rrf-v1"
    assert profile.rrf_k == 60
    with pytest.raises(FrozenInstanceError):
        profile.lexical_weight = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"profile_id": " "},
        {"fusion_algorithm": "raw-score-v1"},
        {"rrf_k": 0},
        {"lexical_weight": -1.0},
        {"dense_weight": float("nan")},
        {"lexical_weight": 0.0, "dense_weight": 0.0, "embedding_space": None},
        {"lexical_weight": 0.0, "lexical_required": True},
        {"dense_weight": 0.0, "dense_required": True, "embedding_space": None},
        {"dense_weight": 1.0, "embedding_space": None},
        {"dense_weight": 0.0, "embedding_space": embedding_space()},
        {"lexical_candidate_count": 0},
        {"dense_candidate_count": 1001},
    ],
)
def test_profile_rejects_ambiguous_or_unbounded_policy(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "profile_id": "local-hybrid",
        "lexical_weight": 1.0,
        "dense_weight": 1.0,
        "lexical_required": True,
        "dense_required": False,
        "lexical_candidate_count": 50,
        "dense_candidate_count": 50,
        "embedding_space": embedding_space(),
    }
    values.update(changes)

    with pytest.raises((TypeError, ValueError)):
        TextRetrievalProfile(**values)  # type: ignore[arg-type]


def test_registry_resolves_profiles_without_exposing_mutable_state() -> None:
    lexical = TextRetrievalProfile(
        profile_id="lexical",
        lexical_weight=1.0,
        dense_weight=0.0,
        lexical_required=True,
        dense_required=False,
        lexical_candidate_count=20,
        dense_candidate_count=20,
    )
    registry = StaticTextRetrievalProfileRegistry((lexical,))

    assert registry.resolve("lexical") is lexical
    with pytest.raises(LookupError, match="unknown text retrieval profile"):
        registry.resolve("missing")
    with pytest.raises(ValueError, match="duplicate"):
        StaticTextRetrievalProfileRegistry((lexical, lexical))
