"""Ranking checks use adversarial scores rather than implementation mirrors."""

import pytest

from capicove.contracts import TargetType
from capicove.retrieval.video_fusion import CandidateLane, VideoCandidate, weighted_video_rrf


def candidate(unit: str, lane: str, rank: int, raw: float) -> VideoCandidate:
    return VideoCandidate(unit, TargetType.VIDEO_SEGMENT, lane, raw, rank, "p", "g", "similarity")


def test_ranks_win_over_incomparable_raw_scores() -> None:
    lanes = (
        CandidateLane("a", 1.0, (candidate("x", "a", 1, 1e9), candidate("y", "a", 2, -1e9))),
        CandidateLane(
            "b", 2.0, (candidate("y", "b", 1, 0.00001), candidate("x", "b", 2, 0.000001))
        ),
    )
    assert weighted_video_rrf(lanes)[0].unit_id == "y"


def test_duplicate_rank_and_nonfinite_fail() -> None:
    with pytest.raises(ValueError):
        weighted_video_rrf(
            (CandidateLane("a", 1, (candidate("x", "a", 1, 1), candidate("x", "a", 2, 1))),)
        )
    with pytest.raises(ValueError):
        weighted_video_rrf((CandidateLane("a", 1, (candidate("x", "a", 2, 1),)),))
    with pytest.raises(ValueError):
        candidate("x", "a", 1, float("nan"))


def test_fusion_preserves_each_lane_weight_for_temporal_assembly() -> None:
    (result,) = weighted_video_rrf(
        (
            CandidateLane("a", 1.0, (candidate("x", "a", 1, 10),)),
            CandidateLane("b", 3.0, (candidate("x", "b", 1, 20),)),
        )
    )
    assert dict(result.lane_scores) == pytest.approx({"a": 1 / 61, "b": 3 / 61})
