"""Rank-based video fusion independent of provider score magnitudes."""

from __future__ import annotations

import math
from dataclasses import dataclass

from capicove.contracts import TargetType


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    """Identify one candidate within one pinned lane."""

    unit_id: str
    target_type: TargetType
    lane_id: str
    raw_score: float
    rank: int
    provider_id: str
    generation_id: str
    score_semantics: str

    def __post_init__(self) -> None:
        for value in (
            self.unit_id,
            self.lane_id,
            self.provider_id,
            self.generation_id,
            self.score_semantics,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("candidate identity missing")
        if not isinstance(self.target_type, TargetType):
            raise ValueError("invalid target")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("invalid rank")
        if isinstance(self.raw_score, bool) or not math.isfinite(self.raw_score):
            raise ValueError("nonfinite score")


@dataclass(frozen=True, slots=True)
class CandidateLane:
    """One complete candidate batch and its calibrated rank weight."""

    lane_id: str
    weight: float
    candidates: tuple[VideoCandidate, ...]

    def __post_init__(self) -> None:
        if (
            not self.lane_id
            or isinstance(self.weight, bool)
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("invalid lane weight or id")
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True, slots=True)
class FusedVideoCandidate:
    """A rank score plus immutable evidence for projection and temporal mapping."""

    unit_id: str
    fused_score: float
    contributions: tuple[VideoCandidate, ...]
    lane_scores: tuple[tuple[str, float], ...]


def weighted_video_rrf(
    lanes: tuple[CandidateLane, ...], *, rrf_k: int = 60
) -> tuple[FusedVideoCandidate, ...]:
    """Validate complete batches before rank-only fusion."""
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("invalid rrf_k")
    if len({lane.lane_id for lane in lanes}) != len(lanes):
        raise ValueError("duplicate lane")
    scores: dict[str, float] = {}
    contributions: dict[str, list[VideoCandidate]] = {}
    lane_scores: dict[str, list[tuple[str, float]]] = {}
    for lane in lanes:
        batch = lane.candidates
        if len({c.unit_id for c in batch}) != len(batch):
            raise ValueError("duplicate candidate")
        if any(c.lane_id != lane.lane_id or c.rank != rank for rank, c in enumerate(batch, 1)):
            raise ValueError("inconsistent lane or noncontiguous rank")
        if (
            len({(c.generation_id, c.provider_id, c.score_semantics, c.target_type) for c in batch})
            > 1
        ):
            raise ValueError("inconsistent candidate batch")
        for candidate in batch:
            scores[candidate.unit_id] = scores.get(candidate.unit_id, 0) + lane.weight / (
                rrf_k + candidate.rank
            )
            contributions.setdefault(candidate.unit_id, []).append(candidate)
            lane_scores.setdefault(candidate.unit_id, []).append(
                (lane.lane_id, lane.weight / (rrf_k + candidate.rank))
            )
    result = (
        FusedVideoCandidate(key, score, tuple(contributions[key]), tuple(lane_scores[key]))
        for key, score in scores.items()
    )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                -item.fused_score,
                min(c.rank for c in item.contributions),
                item.unit_id,
            ),
        )
    )
