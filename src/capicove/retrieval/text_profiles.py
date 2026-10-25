"""Closed provider-neutral policy for text retrieval orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from capicove.contracts import EmbeddingSpaceRef, QueryModality, TargetType


@dataclass(frozen=True, slots=True)
class TextRetrievalProfile:
    """Define one reviewed lexical and dense retrieval policy."""

    profile_id: str
    lexical_weight: float
    dense_weight: float
    lexical_required: bool
    dense_required: bool
    lexical_candidate_count: int
    dense_candidate_count: int
    embedding_space: EmbeddingSpaceRef | None = None
    fusion_algorithm: str = "rrf-v1"
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must not be blank")
        if self.fusion_algorithm != "rrf-v1":
            raise ValueError("fusion_algorithm must be rrf-v1")
        _require_positive_int("rrf_k", self.rrf_k)
        for name in ("lexical_weight", "dense_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite non-negative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            object.__setattr__(self, name, float(value))
        for name in ("lexical_required", "dense_required"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in ("lexical_candidate_count", "dense_candidate_count"):
            value = getattr(self, name)
            _require_positive_int(name, value)
            if value > 1_000:
                raise ValueError(f"{name} must not exceed 1000")
        if self.lexical_weight == 0 and self.dense_weight == 0:
            raise ValueError("at least one retrieval lane must have positive weight")
        if self.lexical_required and self.lexical_weight == 0:
            raise ValueError("a required lexical lane must have positive weight")
        if self.dense_required and self.dense_weight == 0:
            raise ValueError("a required dense lane must have positive weight")
        if self.dense_weight > 0:
            if not isinstance(self.embedding_space, EmbeddingSpaceRef):
                raise ValueError("embedding_space is required for dense retrieval")
            if not self.embedding_space.accepts(QueryModality.TEXT, TargetType.TEXT_CHUNK):
                raise ValueError("embedding_space must accept text queries and text chunks")
        elif self.embedding_space is not None:
            raise ValueError("embedding_space is forbidden when dense retrieval is disabled")


class StaticTextRetrievalProfileRegistry:
    """Resolve profiles from one immutable startup collection."""

    def __init__(self, profiles: tuple[TextRetrievalProfile, ...]) -> None:
        materialized = tuple(profiles)
        if any(not isinstance(profile, TextRetrievalProfile) for profile in materialized):
            raise TypeError("profiles must contain TextRetrievalProfile values")
        identifiers = tuple(profile.profile_id for profile in materialized)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("profiles must not contain duplicate profile IDs")
        self._profiles = {profile.profile_id: profile for profile in materialized}

    def resolve(self, profile_id: str) -> TextRetrievalProfile:
        """Resolve one configured profile by exact identity."""
        try:
            return self._profiles[profile_id]
        except KeyError:
            raise LookupError(f"unknown text retrieval profile: {profile_id}") from None


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = ["StaticTextRetrievalProfileRegistry", "TextRetrievalProfile"]
