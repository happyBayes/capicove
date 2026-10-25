import math

import pytest

from capicove.processing.video import KeyframeSplitConfig, VideoSplitWarningCode
from capicove.processing.video.strategies import PlanningFailure, plan_keyframe_ranges


@pytest.mark.parametrize(
    ("timestamps", "ranges", "warning", "degraded"),
    [
        ((), (), VideoSplitWarningCode.NO_KEYFRAMES, True),
        ((0.0,), ((0.0, 12.0),), None, False),
        ((4.0,), ((0.0, 4.0), (4.0, 12.0)), None, False),
        ((0.0, 4.0, 9.0), ((0.0, 4.0), (4.0, 9.0), (9.0, 12.0)), None, False),
        (
            (0.0, 4.0, 4.0, 9.0),
            ((0.0, 4.0), (4.0, 9.0), (9.0, 12.0)),
            None,
            False,
        ),
    ],
)
def test_plan_keyframe_ranges(
    timestamps: tuple[float, ...],
    ranges: tuple[tuple[float, float], ...],
    warning: VideoSplitWarningCode | None,
    degraded: bool,
) -> None:
    plan = plan_keyframe_ranges(12.0, timestamps)

    assert tuple((item.start_seconds, item.end_seconds) for item in plan.ranges) == ranges
    assert (plan.warning.code if plan.warning else None) is warning
    assert plan.degraded is degraded


@pytest.mark.parametrize(
    "timestamps",
    [
        (0.0, 9.0, 4.0),
        (-1.0, 4.0),
        (math.nan,),
        (math.inf,),
        (13.0,),
    ],
)
def test_plan_keyframe_ranges_rejects_corrupt_timestamps(
    timestamps: tuple[float, ...],
) -> None:
    with pytest.raises(PlanningFailure) as captured:
        plan_keyframe_ranges(12.0, timestamps)

    assert captured.value.warning_code is VideoSplitWarningCode.INVALID_TIMESTAMPS


def test_keyframe_ranges_group_short_gops_and_merge_a_short_tail() -> None:
    plan = plan_keyframe_ranges(
        12.0,
        tuple(float(second) for second in range(12)),
        KeyframeSplitConfig(min_segment_duration=5.0),
    )

    assert tuple((item.start_seconds, item.end_seconds) for item in plan.ranges) == (
        (0.0, 5.0),
        (5.0, 12.0),
    )
