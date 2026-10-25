"""Tests for deliberate public evolution exports."""

from __future__ import annotations

import capicove.evolution as evolution


def test_public_evolution_api_exports_trace_clustering() -> None:
    expected = {"TraceClusteringResult", "cluster_trace_vectors"}

    assert expected.issubset(set(evolution.__all__))
    assert all(hasattr(evolution, name) for name in expected)
