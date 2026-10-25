"""Weighted RRF of the unified profile against a handwritten rank calculation."""

from __future__ import annotations

import math

from capicove.runtime.unified_retrieval import RRF_K, _fuse


def test_fusion_matches_handwritten_reciprocal_ranks_and_stable_ties() -> None:
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    lexical = [("c", 7.0), ("d", 3.0)]
    fused = _fuse({"shared_dense": dense, "lexical": lexical})
    expected = {
        "a": 1 / (RRF_K + 1),
        "b": 1 / (RRF_K + 2),
        "c": 1 / (RRF_K + 3) + 1 / (RRF_K + 1),
        "d": 1 / (RRF_K + 2),
    }
    order = sorted(expected, key=lambda unit: (-expected[unit], unit))
    assert [unit for unit, _ in fused] == order == ["c", "a", "b", "d"]
    observations = dict(fused)
    assert observations["c"] == {"shared_dense": (3, 0.1), "lexical": (1, 7.0)}
    # b and d tie exactly; unit ID decides, never raw scores from different lanes.
    assert math.isclose(expected["b"], expected["d"])


def test_empty_lanes_fuse_to_nothing() -> None:
    assert _fuse({"shared_dense": [], "lexical": []}) == []
