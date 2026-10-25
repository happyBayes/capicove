import importlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from capicove.processing.video import AcceptedVideoSource
from capicove.processing.video.filtering._adapters.pyav_frame_sampler import (
    PyavVideoFrameSampler,
)
from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    SamplingMethod,
    VideoFilterWarningCode,
)
from capicove.processing.video.filtering.video_filter_ports import (
    DecodeFailedFilterError,
    SampledFrame,
    VideoFilterTimeoutError,
)


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


def _source() -> AcceptedVideoSource:
    return AcceptedVideoSource(
        source_id="source-1",
        sha256="a" * 64,
        mime_type="video/mp4",
        data=b"video",
    )


def _frames() -> tuple[SampledFrame, ...]:
    return tuple(
        SampledFrame(
            index=index,
            timestamp_seconds=float(index),
            width=10,
            height=10,
            payload=f"frame-{index}",
            key_frame=index in {0, 2, 4},
        )
        for index in range(5)
    )


def test_uniform_sampling_keeps_endpoints_and_midpoint() -> None:
    sampler = PyavVideoFrameSampler(decoder=lambda source, deadline: _frames())

    batch = sampler.sample(
        _source(),
        FrameSamplingConfig(SamplingMethod.UNIFORM, frame_count=3),
        _deadline(),
    )

    assert [frame.index for frame in batch.frames] == [0, 2, 4]


def test_one_uniform_frame_uses_middle_frame() -> None:
    sampler = PyavVideoFrameSampler(decoder=lambda source, deadline: _frames())

    batch = sampler.sample(
        _source(),
        FrameSamplingConfig(SamplingMethod.UNIFORM, frame_count=1),
        _deadline(),
    )

    assert [frame.index for frame in batch.frames] == [2]


def test_keyframe_sampling_is_bounded_and_warns() -> None:
    sampler = PyavVideoFrameSampler(decoder=lambda source, deadline: _frames())

    batch = sampler.sample(
        _source(),
        FrameSamplingConfig(SamplingMethod.ALL_KEYFRAMES, max_frames=2),
        _deadline(),
    )

    assert [frame.index for frame in batch.frames] == [0, 4]
    assert batch.warnings[0].code is VideoFilterWarningCode.SAMPLE_LIMIT_REACHED


def test_sampling_rejects_video_without_decodable_frames() -> None:
    sampler = PyavVideoFrameSampler(decoder=lambda source, deadline: ())

    with pytest.raises(DecodeFailedFilterError, match="decodable"):
        sampler.sample(_source(), FrameSamplingConfig(), _deadline())


def test_sampling_rejects_expired_deadline_before_decode() -> None:
    called = False

    def decoder(
        source: AcceptedVideoSource,
        deadline: datetime,
    ) -> tuple[SampledFrame, ...]:
        nonlocal called
        del source, deadline
        called = True
        return _frames()

    sampler = PyavVideoFrameSampler(decoder=decoder)

    with pytest.raises(VideoFilterTimeoutError):
        sampler.sample(
            _source(),
            FrameSamplingConfig(),
            datetime.now(UTC) - timedelta(seconds=1),
        )

    assert called is False


def test_default_decoder_converts_only_selected_uniform_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converted: list[int] = []

    class FakeFrame:
        def __init__(self, index: int) -> None:
            self.index = index
            self.time = float(index)
            self.pts = index
            self.width = 10
            self.height = 10
            self.key_frame = index % 10 == 0

        def to_ndarray(self, *, format: str) -> str:
            assert format == "bgr24"
            converted.append(self.index)
            return f"frame-{self.index}"

    class FakeContainer:
        def __init__(self) -> None:
            self.stream = SimpleNamespace(frames=100, time_base=1)
            self.streams = SimpleNamespace(video=[self.stream])

        def decode(self, stream: object) -> Iterator[FakeFrame]:
            assert stream is self.stream
            return (FakeFrame(index) for index in range(100))

        def close(self) -> None:
            return None

    fake_av = SimpleNamespace(open=lambda path, mode: FakeContainer())
    real_import_module = importlib.import_module

    def import_module(name: str, package: str | None = None) -> object:
        if name == "av":
            return fake_av
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    batch = PyavVideoFrameSampler().sample(
        _source(),
        FrameSamplingConfig(SamplingMethod.UNIFORM, frame_count=3),
        _deadline(),
    )

    assert [frame.index for frame in batch.frames] == [0, 50, 99]
    assert converted == [0, 50, 99]
