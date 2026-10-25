from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import capicove.processing.video as video
from capicove.processing.video import (
    DurationSplitConfig,
    KeyframeSplitConfig,
    SceneDetectorName,
    SceneSplitConfig,
    TimeRange,
    VideoBackend,
    VideoOutputFormat,
    VideoSource,
    VideoSplitRequest,
    VideoSplitStrategy,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def test_write_mode_defaults_to_frame_accurate_output() -> None:
    assert video.VideoWriteMode.ACCURATE.value == "accurate"
    assert video.VideoWriteMode.COPY.value == "copy"

    request = VideoSplitRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        source=VideoSource(data=b"video"),
        strategy=VideoSplitStrategy.DURATION,
        config=DurationSplitConfig(5.0),
        output_format=VideoOutputFormat.BYTES,
        backend=VideoBackend.AUTO,
        deadline=_deadline(),
    )

    assert request.write_mode is video.VideoWriteMode.ACCURATE


def test_video_source_requires_exactly_one_representation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VideoSource()
    with pytest.raises(ValueError, match="exactly one"):
        VideoSource(path=Path("input.mp4"), data=b"video")


def test_artifact_source_requires_verified_sha256() -> None:
    with pytest.raises(ValueError, match="artifact_sha256"):
        VideoSource(artifact_ref="artifact:video:1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split_duration": 0.0}, "split_duration"),
        ({"split_duration": 5.0, "overlap_duration": -1.0}, "overlap_duration"),
        ({"split_duration": 5.0, "overlap_duration": 5.0}, "overlap_duration"),
        ({"split_duration": 5.0, "min_last_split_duration": -1.0}, "min_last"),
    ],
)
def test_duration_config_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DurationSplitConfig(**kwargs)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_keyframe_config_rejects_invalid_minimum_duration(value: float) -> None:
    with pytest.raises(ValueError, match="min_segment_duration"):
        KeyframeSplitConfig(min_segment_duration=value)


def test_scene_options_are_copied_and_immutable() -> None:
    options: dict[str, float] = {"threshold": 27.0}
    config = SceneSplitConfig(detector=SceneDetectorName.CONTENT, options=options)
    options["threshold"] = 10.0

    assert config.options["threshold"] == 27.0
    with pytest.raises(TypeError):
        config.options["threshold"] = 12.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1.0, 1.0), (1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))],
)
def test_time_range_rejects_invalid_bounds(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        TimeRange(start, end)


def test_request_rejects_strategy_config_mismatch() -> None:
    with pytest.raises(ValueError, match="strategy"):
        VideoSplitRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            source=VideoSource(data=b"video"),
            strategy=VideoSplitStrategy.DURATION,
            config=KeyframeSplitConfig(),
            output_format=VideoOutputFormat.BYTES,
            backend=VideoBackend.AUTO,
            deadline=_deadline(),
        )


def test_path_output_requires_an_output_directory() -> None:
    with pytest.raises(ValueError, match="output_directory"):
        VideoSplitRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            source=VideoSource(data=b"video"),
            strategy=VideoSplitStrategy.DURATION,
            config=DurationSplitConfig(5.0),
            output_format=VideoOutputFormat.PATH,
            backend=VideoBackend.AUTO,
            deadline=_deadline(),
        )


def test_byte_output_rejects_an_output_directory() -> None:
    with pytest.raises(ValueError, match="output_directory"):
        VideoSplitRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref="policy-1",
            source=VideoSource(data=b"video"),
            strategy=VideoSplitStrategy.DURATION,
            config=DurationSplitConfig(5.0),
            output_format=VideoOutputFormat.BYTES,
            output_directory=Path("outputs"),
            backend=VideoBackend.AUTO,
            deadline=_deadline(),
        )


def test_request_requires_nonblank_execution_policy() -> None:
    with pytest.raises(ValueError, match="execution_policy_ref"):
        VideoSplitRequest(
            request_id="request-1",
            trace_id="trace-1",
            attempt_id="attempt-1",
            operation_id="operation-1",
            execution_policy_ref=" ",
            source=VideoSource(data=b"video"),
            strategy=VideoSplitStrategy.DURATION,
            config=DurationSplitConfig(5.0),
            output_format=VideoOutputFormat.BYTES,
            backend=VideoBackend.AUTO,
            deadline=_deadline(),
        )
