from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.video import (
    AcceptedVideoSource,
    DurationSplitConfig,
    KeyframeSplitConfig,
    SceneDetectorName,
    SceneSplitConfig,
    TimeRange,
    VideoBackend,
    VideoOutputFormat,
    VideoSource,
    VideoSplitFailureCode,
    VideoSplitRequest,
    VideoSplitStatus,
    VideoSplitStrategy,
    VideoSplitWarningCode,
    VideoWriteMode,
)
from capicove.processing.video.ports import (
    BackendUnavailableError,
    SceneDetectionError,
    VideoBackendBundle,
    VideoMetadata,
)
from capicove.processing.video.splitter import VideoSplitter
from tests.unit.processing.video.fakes import (
    FakeBackendResolver,
    FakeDetector,
    FakeInspector,
    FakePathAuthorizer,
    FakeStore,
    FakeWriter,
)


def make_request(
    *,
    strategy: VideoSplitStrategy = VideoSplitStrategy.DURATION,
    config: DurationSplitConfig | KeyframeSplitConfig | SceneSplitConfig | None = None,
    deadline: datetime | None = None,
) -> VideoSplitRequest:
    selected_config = config or DurationSplitConfig(5.0)
    return VideoSplitRequest(
        request_id="request-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        execution_policy_ref="policy-1",
        source=VideoSource(data=b"source-video"),
        strategy=strategy,
        config=selected_config,
        output_format=VideoOutputFormat.BYTES,
        backend=VideoBackend.AUTO,
        deadline=deadline or datetime.now(UTC) + timedelta(minutes=1),
    )


def make_splitter(
    tmp_path: Path,
    *,
    duration: float = 12.0,
    keyframes: tuple[float, ...] = (0.0, 4.0, 9.0),
    failed_indices: frozenset[int] = frozenset(),
    fail_allocate_indices: frozenset[int] = frozenset(),
    fail_finalize_indices: frozenset[int] = frozenset(),
    fail_discard: bool = False,
    detector: FakeDetector | None = None,
    unavailable: bool = False,
    authorizer: FakePathAuthorizer | None = None,
    inspector_error: Exception | None = None,
) -> tuple[VideoSplitter, FakeInspector, FakeWriter, FakeStore]:
    inspector = FakeInspector(
        VideoMetadata(duration, keyframes, "fake", "1.0"),
        inspector_error,
    )
    writer = FakeWriter(failed_indices)
    store = FakeStore(
        tmp_path,
        fail_allocate_indices=fail_allocate_indices,
        fail_finalize_indices=fail_finalize_indices,
        fail_discard=fail_discard,
    )
    splitter = VideoSplitter(
        backend_resolver=FakeBackendResolver(inspector, writer, unavailable),
        output_store=store,
        path_authorizer=authorizer or FakePathAuthorizer(),
        scene_detector=detector,
    )
    return splitter, inspector, writer, store


def test_duration_split_returns_original_and_chronological_segments(tmp_path: Path) -> None:
    splitter, inspector, writer, _ = make_splitter(tmp_path)

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.SUCCEEDED
    assert result.original.data == b"source-video"
    assert tuple(segment.index for segment in result.segments) == (0, 1, 2)
    assert tuple(segment.time_range for segment in result.segments) == (
        TimeRange(0.0, 5.0),
        TimeRange(5.0, 10.0),
        TimeRange(10.0, 12.0),
    )
    assert all(segment.source_id == result.original.source_id for segment in result.segments)
    assert all(segment.operation_id == "operation-1" for segment in result.segments)
    assert result.warnings == ()
    assert result.degraded is False
    assert inspector.calls == 1
    assert len(writer.calls) == 3


def test_splitter_passes_the_requested_write_mode_to_backend_resolution(
    tmp_path: Path,
) -> None:
    inspector = FakeInspector(VideoMetadata(6.0, (0.0,), "fake", "1.0"))
    writer = FakeWriter()

    class RecordingResolver:
        def __init__(self) -> None:
            self.write_mode = VideoWriteMode.ACCURATE

        def resolve(
            self,
            requested: VideoBackend,
            write_mode: VideoWriteMode = VideoWriteMode.ACCURATE,
        ) -> VideoBackendBundle:
            del requested
            self.write_mode = write_mode
            return VideoBackendBundle("fake", "1.0", inspector, writer)

    resolver = RecordingResolver()
    splitter = VideoSplitter(
        backend_resolver=resolver,
        output_store=FakeStore(tmp_path),
        path_authorizer=FakePathAuthorizer(),
    )
    request = replace(make_request(), write_mode=VideoWriteMode.COPY)

    result = splitter.split(request)

    assert resolver.write_mode is VideoWriteMode.COPY
    assert result.normalized_config["write_mode"] == "copy"


@pytest.mark.parametrize(
    ("strategy", "config"),
    [
        (VideoSplitStrategy.DURATION, KeyframeSplitConfig()),
        (VideoSplitStrategy.KEYFRAME, SceneSplitConfig(SceneDetectorName.CONTENT)),
        (VideoSplitStrategy.SCENE, DurationSplitConfig(5.0)),
    ],
)
def test_planning_rejects_strategy_configuration_mismatch_at_runtime(
    tmp_path: Path,
    strategy: VideoSplitStrategy,
    config: DurationSplitConfig | KeyframeSplitConfig | SceneSplitConfig,
) -> None:
    splitter, inspector, writer, _ = make_splitter(tmp_path)
    request = make_request()
    object.__setattr__(request, "strategy", strategy)
    object.__setattr__(request, "config", config)
    backend = VideoBackendBundle("fake", "1.0", inspector, writer)

    with pytest.raises(ValueError, match="strategy must match its configuration type"):
        splitter._plan(
            request,
            AcceptedVideoSource(
                source_id="source-1",
                sha256="a" * 64,
                mime_type="video/mp4",
                data=b"video",
            ),
            backend,
        )


def test_source_needing_no_split_does_not_write(tmp_path: Path) -> None:
    splitter, _, writer, store = make_splitter(tmp_path, duration=5.0)

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.SUCCEEDED
    assert result.segments == ()
    assert result.degraded is False
    assert writer.calls == []
    assert store.allocated == []


def test_partial_failure_retains_successes_and_original(tmp_path: Path) -> None:
    splitter, _, _, store = make_splitter(tmp_path, failed_indices=frozenset({1}))

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.SUCCEEDED
    assert tuple(segment.index for segment in result.segments) == (0, 2)
    assert result.original.data == b"source-video"
    assert result.degraded is True
    assert tuple(warning.code for warning in result.warnings) == (
        VideoSplitWarningCode.SEGMENT_FAILED,
    )
    assert result.warnings[0].segment_index == 1
    assert store.discarded == [store.allocated[1]]


def test_output_allocation_failure_is_a_range_local_warning(tmp_path: Path) -> None:
    splitter, _, _, _ = make_splitter(
        tmp_path,
        fail_allocate_indices=frozenset({1}),
    )

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.SUCCEEDED
    assert tuple(segment.index for segment in result.segments) == (0, 2)
    assert result.degraded is True
    assert result.warnings[0].code is VideoSplitWarningCode.SEGMENT_FAILED
    assert result.warnings[0].segment_index == 1


def test_all_segment_failures_return_failed_original(tmp_path: Path) -> None:
    splitter, _, _, _ = make_splitter(tmp_path, failed_indices=frozenset({0, 1, 2}))

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.FAILED
    assert result.original.data == b"source-video"
    assert result.segments == ()
    assert result.failure_code is VideoSplitFailureCode.INTERNAL
    assert [warning.code for warning in result.warnings].count(
        VideoSplitWarningCode.SEGMENT_FAILED
    ) == 3
    assert result.warnings[-1].code is VideoSplitWarningCode.FALLBACK_TO_ORIGINAL


def test_cleanup_failure_is_visible(tmp_path: Path) -> None:
    splitter, _, _, _ = make_splitter(
        tmp_path,
        failed_indices=frozenset({0}),
        fail_discard=True,
    )

    result = splitter.split(make_request())

    assert VideoSplitWarningCode.OUTPUT_CLEANUP_FAILED in {
        warning.code for warning in result.warnings
    }


def test_no_keyframes_is_successful_degraded_fallback(tmp_path: Path) -> None:
    splitter, _, writer, _ = make_splitter(tmp_path, keyframes=())

    result = splitter.split(
        make_request(
            strategy=VideoSplitStrategy.KEYFRAME,
            config=KeyframeSplitConfig(),
        )
    )

    assert result.status is VideoSplitStatus.SUCCEEDED
    assert result.segments == ()
    assert result.degraded is True
    assert tuple(warning.code for warning in result.warnings) == (
        VideoSplitWarningCode.NO_KEYFRAMES,
        VideoSplitWarningCode.FALLBACK_TO_ORIGINAL,
    )
    assert writer.calls == []


def test_scene_detection_failure_returns_failed_original(tmp_path: Path) -> None:
    detector = FakeDetector((), SceneDetectionError("detector failed"))
    splitter, _, _, _ = make_splitter(tmp_path, detector=detector)

    result = splitter.split(
        make_request(
            strategy=VideoSplitStrategy.SCENE,
            config=SceneSplitConfig(SceneDetectorName.CONTENT),
        )
    )

    assert result.status is VideoSplitStatus.FAILED
    assert result.failure_code is VideoSplitFailureCode.INTERNAL
    assert tuple(warning.code for warning in result.warnings) == (
        VideoSplitWarningCode.SCENE_DETECTION_FAILED,
        VideoSplitWarningCode.FALLBACK_TO_ORIGINAL,
    )


def test_expired_deadline_stops_before_backend_resolution(tmp_path: Path) -> None:
    splitter, inspector, writer, _ = make_splitter(tmp_path)

    result = splitter.split(make_request(deadline=datetime.now(UTC) - timedelta(seconds=1)))

    assert result.status is VideoSplitStatus.FAILED
    assert result.failure_code is VideoSplitFailureCode.TIMEOUT
    assert inspector.calls == 0
    assert writer.calls == []


def test_backend_unavailable_retains_accepted_original(tmp_path: Path) -> None:
    splitter, _, _, _ = make_splitter(tmp_path, unavailable=True)

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.FAILED
    assert result.failure_code is VideoSplitFailureCode.BACKEND_UNAVAILABLE
    assert result.original.data == b"source-video"
    assert result.warnings[0].code is VideoSplitWarningCode.BACKEND_UNAVAILABLE


def test_inspection_backend_unavailable_keeps_canonical_classification(tmp_path: Path) -> None:
    splitter, _, _, _ = make_splitter(
        tmp_path,
        inspector_error=BackendUnavailableError("backend unavailable"),
    )

    result = splitter.split(make_request())

    assert result.status is VideoSplitStatus.FAILED
    assert result.failure_code is VideoSplitFailureCode.BACKEND_UNAVAILABLE
    assert result.warnings[0].code is VideoSplitWarningCode.BACKEND_UNAVAILABLE


def test_scene_backend_unavailable_keeps_canonical_classification(tmp_path: Path) -> None:
    detector = FakeDetector((), BackendUnavailableError("scene backend unavailable"))
    splitter, _, _, _ = make_splitter(tmp_path, detector=detector)

    result = splitter.split(
        make_request(
            strategy=VideoSplitStrategy.SCENE,
            config=SceneSplitConfig(SceneDetectorName.CONTENT),
        )
    )

    assert result.status is VideoSplitStatus.FAILED
    assert result.failure_code is VideoSplitFailureCode.BACKEND_UNAVAILABLE
    assert result.warnings[0].code is VideoSplitWarningCode.BACKEND_UNAVAILABLE


@pytest.mark.parametrize(
    "ranges",
    [(), (TimeRange(0.0, 12.0),)],
)
def test_zero_or_one_scene_is_an_intentional_no_split(
    tmp_path: Path,
    ranges: tuple[TimeRange, ...],
) -> None:
    detector = FakeDetector(ranges)
    splitter, _, writer, _ = make_splitter(tmp_path, detector=detector)

    result = splitter.split(
        make_request(
            strategy=VideoSplitStrategy.SCENE,
            config=SceneSplitConfig(SceneDetectorName.CONTENT),
        )
    )

    assert result.status is VideoSplitStatus.SUCCEEDED
    assert result.segments == ()
    assert result.degraded is False
    assert writer.calls == []
