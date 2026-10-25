import math

import pytest

from capicove.processing.video.filtering.video_filter_contracts import (
    Aggregation,
    Reduction,
)
from capicove.processing.video.filtering.video_filter_rules import (
    aggregate,
    in_range,
    reduce_values,
)


@pytest.mark.parametrize(
    ("value", "minimum", "maximum", "expected"),
    [
        (1.0, 1.0, 2.0, True),
        (2.0, 1.0, 2.0, True),
        (0.5, 1.0, None, False),
        (3.0, None, 2.0, False),
    ],
)
def test_in_range_uses_closed_optional_bounds(
    value: float,
    minimum: float | None,
    maximum: float | None,
    expected: bool,
) -> None:
    assert in_range(value, minimum, maximum) is expected


def test_in_range_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        in_range(math.nan, 0.0, 1.0)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (Reduction.AVG, 2.0),
        (Reduction.MAX, 3.0),
        (Reduction.MIN, 1.0),
    ],
)
def test_reduce_values_applies_selected_mode(mode: Reduction, expected: float) -> None:
    assert reduce_values((1.0, 2.0, 3.0), mode) == expected


def test_reduce_values_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        reduce_values((), Reduction.AVG)


@pytest.mark.parametrize(
    ("values", "mode", "expected"),
    [
        ((True, True), Aggregation.ALL, True),
        ((True, False), Aggregation.ALL, False),
        ((False, True), Aggregation.ANY, True),
        ((False, False), Aggregation.ANY, False),
    ],
)
def test_aggregate_applies_selected_boolean_mode(
    values: tuple[bool, ...],
    mode: Aggregation,
    expected: bool,
) -> None:
    assert aggregate(values, mode) is expected


def test_aggregate_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate((), Aggregation.ALL)
