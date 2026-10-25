import math

import pytest

from capicove.processing.video import DurationSplitConfig
from capicove.processing.video.strategies import PlanningFailure, plan_duration_ranges


@pytest.mark.parametrize(
    ("duration", "config", "expected"),
    [
        (4.0, DurationSplitConfig(5.0), ()),
        (5.0, DurationSplitConfig(5.0), ()),
        (12.0, DurationSplitConfig(5.0), ((0.0, 5.0), (5.0, 10.0), (10.0, 12.0))),
        (
            12.0,
            DurationSplitConfig(5.0, overlap_duration=1.0),
            ((0.0, 5.0), (4.0, 9.0), (8.0, 12.0)),
        ),
        (
            12.0,
            DurationSplitConfig(5.0, min_last_split_duration=3.0),
            ((0.0, 5.0), (5.0, 10.0)),
        ),
        (10.0000000001, DurationSplitConfig(5.0), ((0.0, 5.0), (5.0, 10.0))),
    ],
)
def test_plan_duration_ranges(
    duration: float,
    config: DurationSplitConfig,
    expected: tuple[tuple[float, float], ...],
) -> None:
    ranges = plan_duration_ranges(duration, config)

    assert tuple((item.start_seconds, item.end_seconds) for item in ranges) == expected
    assert all(item.end_seconds > item.start_seconds for item in ranges)
    assert all(item.end_seconds <= duration for item in ranges)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan, math.inf])
def test_plan_duration_ranges_rejects_invalid_source_duration(duration: float) -> None:
    with pytest.raises(PlanningFailure, match="duration"):
        plan_duration_ranges(duration, DurationSplitConfig(5.0))
