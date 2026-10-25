"""Unit tests for deterministic trace-vector clustering."""

from __future__ import annotations

import numpy as np
import pytest

from capicove.evolution.clustering import cluster_trace_vectors


def test_single_trace_vector_returns_one_cluster() -> None:
    result = cluster_trace_vectors([[2.0, 4.0]])

    assert result.selected_k == 1
    assert result.labels == (0,)
    assert result.centers == ((2.0, 4.0),)
    assert result.inertia == 0.0
    assert result.elbow_curve == ((1, 0.0),)


def test_duplicate_trace_vectors_return_one_cluster() -> None:
    result = cluster_trace_vectors([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])

    assert result.selected_k == 1
    assert result.labels == (0, 0, 0)
    assert result.centers == ((1.0, 3.0),)
    assert result.inertia == 0.0
    assert result.elbow_curve == ((1, 0.0),)


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [1.0, 2.0],
        np.empty((2, 0)),
        [["not-a-number"]],
        [[1.0, float("nan")]],
        [[1.0, float("inf")]],
    ],
)
def test_trace_vector_clustering_rejects_invalid_vectors(vectors: object) -> None:
    with pytest.raises(ValueError):
        cluster_trace_vectors(vectors)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_k", True),
        ("max_k", 0),
        ("max_k", 3),
        ("n_init", False),
        ("n_init", 0),
        ("max_iterations", False),
        ("max_iterations", 0),
    ],
)
def test_trace_vector_clustering_rejects_invalid_integer_parameters(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        cluster_trace_vectors([[0.0], [1.0]], **{name: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_trace_vector_clustering_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        cluster_trace_vectors([[0.0], [1.0]], tolerance=tolerance)


def test_kmeans_evaluates_each_candidate_with_non_increasing_inertia() -> None:
    vectors = [
        [-10.0, -10.0],
        [-10.2, -9.8],
        [0.0, 0.0],
        [0.2, -0.2],
        [10.0, 10.0],
        [9.8, 10.2],
    ]

    result = cluster_trace_vectors(vectors, max_k=3, random_state=7)

    assert tuple(k for k, _ in result.elbow_curve) == (1, 2, 3)
    inertias = tuple(inertia for _, inertia in result.elbow_curve)
    assert inertias[0] >= inertias[1] >= inertias[2]
    assert inertias[2] == pytest.approx(0.12)


def test_kmeans_is_deterministic_for_a_fixed_seed() -> None:
    vectors = [[-2.0], [-1.9], [2.0], [2.1], [8.0], [8.1]]

    first = cluster_trace_vectors(vectors, max_k=3, random_state=19)
    second = cluster_trace_vectors(vectors, max_k=3, random_state=19)

    assert first == second


def test_kmeans_limits_candidates_to_distinct_vectors() -> None:
    result = cluster_trace_vectors(
        [[0.0], [0.0], [5.0], [5.0], [5.0]],
        max_k=5,
    )

    assert tuple(k for k, _ in result.elbow_curve) == (1, 2)


def test_elbow_method_selects_three_separated_trace_clusters() -> None:
    vectors = [
        *[[-10.0 + offset, -10.0 - offset] for offset in (0.0, 0.1, -0.1, 0.2, -0.2)],
        *[[0.0 + offset, 0.0 - offset] for offset in (0.0, 0.1, -0.1, 0.2, -0.2)],
        *[[10.0 + offset, 10.0 - offset] for offset in (0.0, 0.1, -0.1, 0.2, -0.2)],
    ]

    result = cluster_trace_vectors(vectors, random_state=23)

    assert result.selected_k == 3
    assert len(result.labels) == len(vectors)
    assert len(result.centers) == 3
    assert result.centers == tuple(sorted(result.centers))
    assert set(result.labels) == {0, 1, 2}
    assert result.inertia == dict(result.elbow_curve)[3]


def test_elbow_method_falls_back_to_one_without_an_interior_candidate() -> None:
    result = cluster_trace_vectors([[0.0], [0.1], [10.0], [10.1]], max_k=2)

    assert result.selected_k == 1
    assert tuple(k for k, _ in result.elbow_curve) == (1, 2)
