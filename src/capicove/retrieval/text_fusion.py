"""Pure deterministic fusion for ranked text candidate lanes."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.contracts.rank_fusion import normalized_rrf
from capicove.retrieval.models import TextCandidate
from capicove.retrieval.text_profiles import TextRetrievalProfile


@dataclass(frozen=True, slots=True)
class FusedTextCandidate:
    """Pair one canonical text candidate with its normalized RRF score."""

    candidate: TextCandidate
    score: float


def weighted_reciprocal_rank_fusion(
    *,
    lexical: tuple[TextCandidate, ...],
    dense: tuple[TextCandidate, ...],
    profile: TextRetrievalProfile,
    top_k: int,
) -> tuple[FusedTextCandidate, ...]:
    """Fuse successful lexical and dense lanes without comparing raw scores."""
    if not isinstance(profile, TextRetrievalProfile):
        raise TypeError("profile must be a TextRetrievalProfile")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    lanes = (
        ("lexical", tuple(lexical), profile.lexical_weight),
        ("dense", tuple(dense), profile.dense_weight),
    )
    representatives: dict[str, TextCandidate] = {}
    ranked_lanes: list[tuple[float, tuple[tuple[str, int], ...]]] = []
    for lane_name, candidates, weight in lanes:
        if weight == 0 or not candidates:
            continue
        seen: set[str] = set()
        for item in candidates:
            if not isinstance(item, TextCandidate):
                raise TypeError(f"{lane_name} candidates must contain TextCandidate values")
            if item.unit_id in seen:
                raise ValueError(f"{lane_name} lane contains a duplicate unit")
            seen.add(item.unit_id)
            existing = representatives.get(item.unit_id)
            if existing is not None and _candidate_identity(existing) != _candidate_identity(item):
                raise ValueError("candidate identity disagreement across retrieval lanes")
            representatives.setdefault(item.unit_id, item)
        ranked_lanes.append((weight, tuple((item.unit_id, item.rank) for item in candidates)))
    scores = normalized_rrf(ranked_lanes, k=profile.rrf_k)
    fused = tuple(
        FusedTextCandidate(
            representative,
            scores[unit_id],
        )
        for unit_id, representative in representatives.items()
    )
    return tuple(sorted(fused, key=lambda item: (-item.score, item.candidate.unit_id))[:top_k])


def _candidate_identity(candidate: TextCandidate) -> tuple[object, ...]:
    return (
        candidate.evidence_ref,
        candidate.text_locator,
        candidate.snippet,
        candidate.provider_id,
        candidate.index_generation_id,
    )


__all__ = ["FusedTextCandidate", "weighted_reciprocal_rank_fusion"]
