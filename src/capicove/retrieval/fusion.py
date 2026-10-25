"""Deterministic rank-only fusion for reverse-image retrieval lanes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from capicove.contracts.retrieval import MatchKind
from capicove.retrieval.search_models import (
    Candidate,
    CandidateLane,
    FusedCandidate,
    IndexedCandidateProvenance,
    LaneObservation,
)


MAX_FUSION_CANDIDATES = 10_000


@dataclass(slots=True)
class _Accumulator:
    unit_id: str
    unit_ordinal: int
    activation_set_id: str
    exact_content: bool
    fused_score: float
    best_lane_rank: int
    match_kinds: list[MatchKind]
    lane_observations: list[LaneObservation]
    generation_ids: list[str]
    provider_ids: list[str]


class WeightedReciprocalRankFusion:
    """Fuse stable lane ranks without comparing model-native raw scores."""

    def __init__(
        self,
        *,
        policy_id: str = "weighted-rrf-v1",
        rrf_k: int = 60,
    ) -> None:
        if not isinstance(policy_id, str) or not policy_id.strip():
            raise ValueError("policy_id must be a non-blank string")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        self.policy_id = policy_id
        self.rrf_k = rrf_k

    def fuse(
        self,
        lanes: tuple[CandidateLane, ...],
        *,
        limit: int,
    ) -> tuple[FusedCandidate, ...]:
        """Return a bounded stable ranking with exact-content precedence."""
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_FUSION_CANDIDATES
        ):
            raise ValueError("limit must be an integer from 1 to 10,000")
        materialized = tuple(lanes)
        if any(not isinstance(lane, CandidateLane) for lane in materialized):
            raise TypeError("lanes must contain CandidateLane values")
        if len({lane.lane for lane in materialized}) != len(materialized):
            raise ValueError("fusion input must not contain a duplicate lane")

        accumulated: dict[str, _Accumulator] = {}
        for lane in materialized:
            for candidate in lane.candidates:
                contribution = lane.weight / (self.rrf_k + candidate.rank)
                current = accumulated.get(candidate.unit_id)
                activation_set_id = candidate.provenance.activation_set_id
                if current is None:
                    current = _Accumulator(
                        unit_id=candidate.unit_id,
                        unit_ordinal=candidate.unit_ordinal,
                        activation_set_id=activation_set_id,
                        exact_content=candidate.exact_content,
                        fused_score=0.0,
                        best_lane_rank=candidate.rank,
                        match_kinds=[],
                        lane_observations=[],
                        generation_ids=[],
                        provider_ids=[],
                    )
                    accumulated[candidate.unit_id] = current
                self._require_consistent_identity(current, candidate)
                current.exact_content = current.exact_content or candidate.exact_content
                current.fused_score += contribution
                current.best_lane_rank = min(current.best_lane_rank, candidate.rank)
                if candidate.match_kind not in current.match_kinds:
                    current.match_kinds.append(candidate.match_kind)
                current.lane_observations.append(
                    LaneObservation(
                        lane=candidate.lane,
                        raw_score=candidate.raw_score,
                        score_semantics=candidate.score_semantics,
                        rank=candidate.rank,
                        contribution=contribution,
                    )
                )
                if isinstance(candidate.provenance, IndexedCandidateProvenance):
                    _append_unique(current.generation_ids, candidate.provenance.generation_id)
                    _append_unique(current.provider_ids, candidate.provenance.provider_id)

        ranked = sorted(
            accumulated.values(),
            key=lambda item: (
                not item.exact_content,
                -item.fused_score,
                item.best_lane_rank,
                item.unit_id,
            ),
        )[:limit]
        projected = tuple(self._project(item) for item in ranked)
        return tuple(
            replace(item, brief_score=1.0 / rank) for rank, item in enumerate(projected, start=1)
        )

    @staticmethod
    def _require_consistent_identity(current: _Accumulator, candidate: Candidate) -> None:
        if current.unit_ordinal != candidate.unit_ordinal:
            raise ValueError("one unit_id cannot refer to multiple ordinals")
        if current.activation_set_id != candidate.provenance.activation_set_id:
            raise ValueError("one fused result cannot mix activation sets")

    @staticmethod
    def _project(item: _Accumulator) -> FusedCandidate:
        if not math.isfinite(item.fused_score) or item.fused_score <= 0:
            raise ValueError("fused score must be positive and finite")
        return FusedCandidate(
            unit_id=item.unit_id,
            unit_ordinal=item.unit_ordinal,
            exact_content=item.exact_content,
            fused_score=item.fused_score,
            best_lane_rank=item.best_lane_rank,
            match_kinds=tuple(item.match_kinds),
            lane_observations=tuple(item.lane_observations),
            activation_set_id=item.activation_set_id,
            generation_ids=tuple(item.generation_ids),
            provider_ids=tuple(item.provider_ids),
            brief_score=0.0,
        )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


__all__ = ["MAX_FUSION_CANDIDATES", "WeightedReciprocalRankFusion"]
