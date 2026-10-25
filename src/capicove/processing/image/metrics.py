"""Deterministic overlap metrics for image regions."""

from __future__ import annotations

import math
import operator
from collections import Counter
from collections.abc import Hashable, Iterable
from typing import Any, SupportsIndex, cast

import numpy as np
import numpy.typing as npt

from capicove.processing.image.models import BoundingBox


def information_gain(
    parent: Iterable[Hashable],
    children: Iterable[Iterable[Hashable]],
) -> float:
    """Calculate the Shannon information gain of a categorical split in bits.

    The child collections must form a complete multiset partition of the parent
    labels. Empty child collections are allowed and contribute zero entropy.

    Args:
        parent: Categorical labels before the split.
        children: Categorical labels in each child after the split.

    Returns:
        Parent entropy minus the size-weighted child entropy, in bits.

    Raises:
        ValueError: If the parent is empty, no child partitions are supplied, or
            the children do not form a complete partition of the parent labels.
    """
    parent_counts = Counter(parent)
    parent_size = parent_counts.total()
    if parent_size == 0:
        raise ValueError("parent labels must not be empty")

    combined_child_counts: Counter[Hashable] = Counter()
    weighted_child_entropies: list[float] = []
    child_count = 0
    for child in children:
        child_count += 1
        child_counts = Counter(child)
        child_size = child_counts.total()
        combined_child_counts.update(child_counts)
        if child_size > 0:
            weighted_child_entropies.append(
                child_size / parent_size * _shannon_entropy(child_counts, child_size)
            )

    if child_count == 0:
        raise ValueError("children must contain at least one partition")
    if combined_child_counts != parent_counts:
        raise ValueError("children must form a complete partition of parent labels")

    parent_entropy = _shannon_entropy(parent_counts, parent_size)
    conditional_entropy = math.fsum(weighted_child_entropies)
    return max(0.0, parent_entropy - conditional_entropy)


def _shannon_entropy(counts: Counter[Hashable], sample_size: int) -> float:
    """Return base-two Shannon entropy from categorical label counts."""
    if sample_size == 0:
        return 0.0
    return -math.fsum(
        (count / sample_size) * math.log2(count / sample_size) for count in counts.values()
    )


def coefficient_of_variation(
    values: npt.ArrayLike,
    *,
    ddof: int | np.integer[Any] = 0,
) -> float:
    """Calculate standard deviation divided by the absolute arithmetic mean.

    Args:
        values: One-dimensional numeric observations.
        ddof: Delta degrees of freedom used by the standard deviation. Use zero
            for a population and one for a sample.

    Returns:
        The dimensionless coefficient. Multiply by 100 for a percentage.

    Raises:
        ValueError: If observations are empty, non-finite, not one-dimensional,
            have a numerically zero mean, or do not outnumber ``ddof``.
    """
    observations = np.asarray(values, dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("values must be a non-empty one-dimensional collection")
    if not np.all(np.isfinite(observations)):
        raise ValueError("values must be finite")
    normalized_ddof = _normalize_ddof(ddof, observations.size)

    mean = math.fsum(float(value) for value in observations) / observations.size
    scale = float(np.max(np.abs(observations)))
    zero_tolerance = np.finfo(np.float64).eps * observations.size * scale
    if abs(mean) <= zero_tolerance:
        raise ValueError(
            "coefficient of variation is undefined when the mean is "
            "numerically indistinguishable from zero"
        )
    return float(np.std(observations, ddof=normalized_ddof)) / abs(mean)


def _normalize_ddof(ddof: object, sample_size: int) -> int:
    """Return a Python integer for a valid integer-like ``ddof`` value."""
    if isinstance(ddof, bool):
        raise ValueError("ddof must be a non-negative integer smaller than the sample size")
    try:
        normalized_ddof = operator.index(cast(SupportsIndex, ddof))
    except TypeError as error:
        raise ValueError(
            "ddof must be a non-negative integer smaller than the sample size"
        ) from error
    if normalized_ddof < 0 or normalized_ddof >= sample_size:
        raise ValueError("ddof must be a non-negative integer smaller than the sample size")
    return normalized_ddof


def bounding_box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate IoU for two axis-aligned pixel-space bounding boxes.

    The boxes use Capicove's ``(x, y, width, height)`` representation. Their
    confidence values do not affect the geometric overlap.
    """
    intersection_width = max(
        0.0,
        min(first.x + first.width, second.x + second.width) - max(first.x, second.x),
    )
    intersection_height = max(
        0.0,
        min(first.y + first.height, second.y + second.height) - max(first.y, second.y),
    )
    intersection_area = intersection_width * intersection_height
    union_area = first.width * first.height + second.width * second.height - intersection_area
    return intersection_area / union_area


def binary_mask_iou(
    first: npt.NDArray[Any],
    second: npt.NDArray[Any],
) -> float:
    """Calculate IoU for two equally shaped masks, treating nonzero as foreground.

    Two masks with no foreground pixels have an IoU of 1.0 because they agree
    perfectly. A shape mismatch raises ``ValueError``.
    """
    if first.shape != second.shape:
        raise ValueError("binary masks must have the same shape")

    first_foreground = first != 0
    second_foreground = second != 0
    intersection_area = int(np.count_nonzero(first_foreground & second_foreground))
    union_area = int(np.count_nonzero(first_foreground | second_foreground))
    if union_area == 0:
        return 1.0
    return intersection_area / union_area
