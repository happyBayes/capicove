"""Deterministic K-Means analysis for caller-provided trace vectors."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, SupportsIndex, cast

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TraceClusteringResult:
    """Record a deterministic trace-vector clustering result."""

    selected_k: int
    labels: tuple[int, ...]
    centers: tuple[tuple[float, ...], ...]
    inertia: float
    elbow_curve: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class _KMeansFit:
    """Hold one internal K-Means fit before public tuple conversion."""

    labels: npt.NDArray[np.int64]
    centers: npt.NDArray[np.float64]
    inertia: float


def cluster_trace_vectors(
    vectors: npt.ArrayLike,
    *,
    max_k: int | np.integer[Any] | None = None,
    random_state: int | np.integer[Any] = 0,
    n_init: int | np.integer[Any] = 10,
    max_iterations: int | np.integer[Any] = 300,
    tolerance: float = 1e-4,
) -> TraceClusteringResult:
    """Cluster numeric trace vectors, selecting K with an elbow method."""
    matrix = _validate_vectors(vectors)
    sample_count = matrix.shape[0]
    requested_max_k = _normalize_max_k(max_k, sample_count)
    normalized_random_state = _normalize_non_negative_integer("random_state", random_state)
    normalized_n_init = _normalize_positive_integer("n_init", n_init)
    normalized_max_iterations = _normalize_positive_integer("max_iterations", max_iterations)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    distinct_count = np.unique(matrix, axis=0).shape[0]
    effective_max_k = min(requested_max_k, distinct_count)
    seed_sequences = np.random.SeedSequence(normalized_random_state).spawn(
        effective_max_k * normalized_n_init
    )
    fits: list[_KMeansFit] = []
    seed_index = 0
    for cluster_count in range(1, effective_max_k + 1):
        best_fit: _KMeansFit | None = None
        for _ in range(normalized_n_init):
            fit = _fit_kmeans(
                matrix,
                cluster_count=cluster_count,
                rng=np.random.default_rng(seed_sequences[seed_index]),
                max_iterations=normalized_max_iterations,
                tolerance=tolerance,
            )
            seed_index += 1
            if best_fit is None or fit.inertia < best_fit.inertia:
                best_fit = fit
        if best_fit is None:  # pragma: no cover - n_init validation makes this unreachable
            raise RuntimeError("K-Means produced no fit")
        fits.append(best_fit)

    selected_k = _select_elbow_k(tuple(fit.inertia for fit in fits))
    selected_fit = fits[selected_k - 1]
    return TraceClusteringResult(
        selected_k=selected_k,
        labels=tuple(int(label) for label in selected_fit.labels),
        centers=tuple(tuple(float(value) for value in center) for center in selected_fit.centers),
        inertia=selected_fit.inertia,
        elbow_curve=tuple((k, fit.inertia) for k, fit in enumerate(fits, start=1)),
    )


def _select_elbow_k(inertias: tuple[float, ...]) -> int:
    """Select the smallest strongest elbow from an ordered inertia curve."""
    if len(inertias) < 3:
        return 1
    first_inertia = inertias[0]
    last_inertia = inertias[-1]
    inertia_range = first_inertia - last_inertia
    scale = max(abs(first_inertia), abs(last_inertia), np.finfo(np.float64).tiny)
    if inertia_range <= np.finfo(np.float64).eps * scale:
        return 1

    denominator = len(inertias) - 1
    distances = tuple(
        (1.0 - index / denominator) - (inertias[index] - last_inertia) / inertia_range
        for index in range(1, len(inertias) - 1)
    )
    greatest_distance = max(distances)
    if greatest_distance <= np.finfo(np.float64).eps * 10:
        return 1
    return distances.index(greatest_distance) + 2


def _fit_kmeans(
    matrix: npt.NDArray[np.float64],
    *,
    cluster_count: int,
    rng: np.random.Generator,
    max_iterations: int,
    tolerance: float,
) -> _KMeansFit:
    """Fit one deterministic Lloyd K-Means restart."""
    centers = _initialize_kmeans_plus_plus(matrix, cluster_count, rng)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(max_iterations):
        labels = _assign_with_empty_recovery(matrix, centers)
        updated_centers = np.vstack(
            [np.mean(matrix[labels == index], axis=0) for index in range(cluster_count)]
        )
        displacement = np.sqrt(np.sum((updated_centers - centers) ** 2, axis=1))
        centers = updated_centers
        if float(np.max(displacement)) <= tolerance:
            break

    labels = _assign_with_empty_recovery(matrix, centers)
    centers = np.vstack(
        [np.mean(matrix[labels == index], axis=0) for index in range(cluster_count)]
    )
    squared_distances = np.sum((matrix - centers[labels]) ** 2, axis=1)
    return _canonicalize_clusters(labels, centers, float(np.sum(squared_distances)))


def _initialize_kmeans_plus_plus(
    matrix: npt.NDArray[np.float64],
    cluster_count: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Choose K-Means++ centers from input rows."""
    sample_count, feature_count = matrix.shape
    centers = np.empty((cluster_count, feature_count), dtype=np.float64)
    first_index = int(rng.integers(sample_count))
    centers[0] = matrix[first_index]
    nearest_squared = np.sum((matrix - centers[0]) ** 2, axis=1)
    for center_index in range(1, cluster_count):
        total = float(np.sum(nearest_squared))
        if total == 0.0:
            chosen_index = int(np.flatnonzero(nearest_squared == 0.0)[0])
        else:
            chosen_index = int(rng.choice(sample_count, p=nearest_squared / total))
        centers[center_index] = matrix[chosen_index]
        candidate_squared = np.sum((matrix - centers[center_index]) ** 2, axis=1)
        nearest_squared = np.minimum(nearest_squared, candidate_squared)
    return centers


def _assign_with_empty_recovery(
    matrix: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]:
    """Assign nearest centers and deterministically populate empty clusters."""
    squared_distances = np.sum((matrix[:, np.newaxis, :] - centers) ** 2, axis=2)
    labels = cast(
        npt.NDArray[np.int64],
        np.argmin(squared_distances, axis=1).astype(np.int64),
    )
    counts = np.bincount(labels, minlength=centers.shape[0])
    assigned_squared = squared_distances[np.arange(matrix.shape[0]), labels]
    candidate_order = np.argsort(-assigned_squared, kind="stable")
    for empty_cluster in np.flatnonzero(counts == 0):
        for sample_index in candidate_order:
            donor_cluster = labels[sample_index]
            if counts[donor_cluster] > 1:
                counts[donor_cluster] -= 1
                labels[sample_index] = empty_cluster
                counts[empty_cluster] = 1
                break
    return labels


def _canonicalize_clusters(
    labels: npt.NDArray[np.int64],
    centers: npt.NDArray[np.float64],
    inertia: float,
) -> _KMeansFit:
    """Sort centers lexicographically and remap labels to stable identifiers."""
    keys = tuple(centers[:, index] for index in range(centers.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    old_to_new = np.empty(centers.shape[0], dtype=np.int64)
    old_to_new[order] = np.arange(centers.shape[0], dtype=np.int64)
    return _KMeansFit(labels=old_to_new[labels], centers=centers[order], inertia=inertia)


def _validate_vectors(vectors: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a validated two-dimensional float64 matrix."""
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("vectors must contain numeric values") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("vectors must contain only finite values")
    return matrix


def _normalize_max_k(max_k: object, sample_count: int) -> int:
    """Return the requested candidate upper bound."""
    if max_k is None:
        return max(1, min(10, sample_count - 1))
    normalized = _normalize_positive_integer("max_k", max_k)
    if normalized > sample_count:
        raise ValueError("max_k must not exceed the sample count")
    return normalized


def _normalize_positive_integer(name: str, value: object) -> int:
    """Return a positive Python integer for an integer-like input."""
    normalized = _normalize_integer(name, value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _normalize_non_negative_integer(name: str, value: object) -> int:
    """Return a non-negative Python integer for an integer-like input."""
    normalized = _normalize_integer(name, value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _normalize_integer(name: str, value: object) -> int:
    """Normalize a Python or NumPy integer while rejecting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(cast(SupportsIndex, value))
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
