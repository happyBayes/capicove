import pytest

from capicove.processing.video import SceneDetectorName, SceneSplitConfig, TimeRange
from capicove.processing.video.strategies import (
    PlanningFailure,
    normalize_scene_ranges,
    validate_scene_config,
)


@pytest.mark.parametrize(
    ("detector", "options"),
    [
        (SceneDetectorName.CONTENT, {"threshold": 27.0, "min_scene_len": 15, "luma_only": True}),
        (
            SceneDetectorName.ADAPTIVE,
            {"adaptive_threshold": 3.0, "min_scene_len": 15, "window_width": 2},
        ),
        (
            SceneDetectorName.THRESHOLD,
            {"threshold": 12.0, "min_scene_len": 15, "fade_bias": 0.0},
        ),
    ],
)
def test_validate_scene_config_accepts_documented_options(
    detector: SceneDetectorName,
    options: dict[str, float | int | bool],
) -> None:
    validate_scene_config(SceneSplitConfig(detector=detector, options=options))


@pytest.mark.parametrize("detector", list(SceneDetectorName))
def test_validate_scene_config_rejects_unknown_options(detector: SceneDetectorName) -> None:
    with pytest.raises(PlanningFailure, match="unknown_option"):
        validate_scene_config(SceneSplitConfig(detector=detector, options={"unknown_option": 1}))


def test_validate_scene_config_rejects_unknown_detector() -> None:
    with pytest.raises((ValueError, PlanningFailure), match="detector"):
        validate_scene_config(
            SceneSplitConfig(detector="unknown")  # type: ignore[arg-type]
        )


def test_zero_or_one_scene_produces_no_split() -> None:
    assert normalize_scene_ranges(12.0, ()) == ()
    assert normalize_scene_ranges(12.0, (TimeRange(0.0, 12.0),)) == ()


def test_multiple_scenes_remain_chronological() -> None:
    ranges = (TimeRange(0.0, 4.0), TimeRange(4.0, 9.0), TimeRange(9.0, 12.0))
    assert normalize_scene_ranges(12.0, ranges) == ranges


@pytest.mark.parametrize(
    "ranges",
    [
        (TimeRange(1.0, 4.0), TimeRange(4.0, 12.0)),
        (TimeRange(0.0, 5.0), TimeRange(4.0, 12.0)),
        (TimeRange(0.0, 4.0), TimeRange(5.0, 12.0)),
        (TimeRange(0.0, 4.0), TimeRange(4.0, 13.0)),
    ],
)
def test_normalize_scene_ranges_rejects_invalid_coverage(
    ranges: tuple[TimeRange, ...],
) -> None:
    with pytest.raises(PlanningFailure):
        normalize_scene_ranges(12.0, ranges)


def test_scene_range_normalization_clamps_sub_millisecond_backend_drift() -> None:
    normalized = normalize_scene_ranges(
        12.0,
        (TimeRange(0.0, 4.0), TimeRange(4.0, 12.0000004)),
    )

    assert normalized == (TimeRange(0.0, 4.0), TimeRange(4.0, 12.0))
