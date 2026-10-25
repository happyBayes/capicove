"""Unit tests for image overlap metrics."""

from __future__ import annotations

import numpy as np
import pytest

from capicove.processing.image.metrics import (
    binary_mask_iou,
    bounding_box_iou,
    coefficient_of_variation,
    information_gain,
)
from capicove.processing.image.models import BoundingBox


def test_bounding_box_iou_returns_intersection_over_union() -> None:
    first = BoundingBox(x=0.0, y=0.0, width=4.0, height=4.0)
    second = BoundingBox(x=2.0, y=2.0, width=4.0, height=4.0)

    assert bounding_box_iou(first, second) == pytest.approx(4.0 / 28.0)


def test_bounding_box_iou_returns_zero_for_disjoint_boxes() -> None:
    first = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
    second = BoundingBox(x=2.0, y=2.0, width=1.0, height=1.0)

    assert bounding_box_iou(first, second) == 0.0


def test_binary_mask_iou_treats_nonzero_values_as_foreground() -> None:
    first = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    second = np.array([[0, 2], [0, 2]], dtype=np.uint8)

    assert binary_mask_iou(first, second) == pytest.approx(1.0 / 3.0)


def test_binary_mask_iou_returns_one_when_both_masks_are_empty() -> None:
    first = np.zeros((2, 3), dtype=np.uint8)
    second = np.zeros((2, 3), dtype=bool)

    assert binary_mask_iou(first, second) == 1.0


def test_binary_mask_iou_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        binary_mask_iou(np.zeros((2, 2)), np.zeros((2, 3)))


def test_coefficient_of_variation_uses_population_standard_deviation() -> None:
    assert coefficient_of_variation([1.0, 2.0, 3.0]) == pytest.approx(np.sqrt(1.0 / 6.0))


def test_coefficient_of_variation_supports_sample_standard_deviation() -> None:
    assert coefficient_of_variation([1.0, 2.0, 3.0], ddof=1) == pytest.approx(0.5)


def test_coefficient_of_variation_accepts_numpy_integer_ddof() -> None:
    assert coefficient_of_variation([1.0, 2.0, 3.0], ddof=np.int64(1)) == pytest.approx(0.5)


def test_coefficient_of_variation_accepts_small_nonzero_data_scale() -> None:
    assert coefficient_of_variation([1e-10, 2e-10, 3e-10]) == pytest.approx(np.sqrt(1.0 / 6.0))


def test_coefficient_of_variation_rejects_mean_lost_to_float_cancellation() -> None:
    with pytest.raises(ValueError, match="indistinguishable from zero"):
        coefficient_of_variation([0.1, 0.2, -0.3])


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [-1.0, 1.0]])
def test_coefficient_of_variation_rejects_undefined_inputs(values: list[float]) -> None:
    with pytest.raises(ValueError):
        coefficient_of_variation(values)


def test_coefficient_of_variation_rejects_invalid_ddof() -> None:
    with pytest.raises(ValueError, match="ddof"):
        coefficient_of_variation([1.0], ddof=1)


def test_information_gain_returns_one_for_a_perfect_binary_split() -> None:
    assert information_gain(
        ["yes", "yes", "no", "no"],
        [["yes", "yes"], ["no", "no"]],
    ) == pytest.approx(1.0)


def test_information_gain_weights_child_entropy_by_child_size() -> None:
    assert information_gain(
        ["yes", "yes", "no", "no"],
        [["yes", "yes", "no"], ["no"]],
    ) == pytest.approx(0.31127812445913283)


def test_information_gain_returns_zero_when_split_preserves_distributions() -> None:
    assert information_gain(
        ["yes", "yes", "no", "no"],
        [["yes", "no"], ["yes", "no"]],
    ) == pytest.approx(0.0)


def test_information_gain_clamps_rounding_error_to_zero() -> None:
    child = ["minority", "majority", "majority"]
    children = [child] * 13

    assert (
        information_gain(
            [label for labels in children for label in labels],
            children,
        )
        == 0.0
    )


def test_information_gain_rejects_children_that_do_not_partition_parent() -> None:
    with pytest.raises(ValueError, match="partition"):
        information_gain(["yes", "no"], [["yes"], ["yes"]])
