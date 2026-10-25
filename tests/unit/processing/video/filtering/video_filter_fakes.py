from __future__ import annotations

from datetime import datetime

from capicove.processing.video import AcceptedVideoSource
from capicove.processing.video.filtering.video_filter_contracts import (
    FrameSamplingConfig,
    VideoCriterionResult,
    VideoFilterCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    FrameSampleBatch,
    SampledFrame,
    VideoFilterCapabilityError,
)


class FakeMetadataProbe:
    identity = ("fake-probe", "1.0")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def probe(
        self,
        source: AcceptedVideoSource,
        deadline: datetime,
    ) -> VideoFilterMetadata:
        del deadline
        self.calls.append(source.sha256)
        return VideoFilterMetadata(
            duration_seconds=10.0,
            encoded_width=640,
            encoded_height=360,
            display_width=640,
            display_height=360,
            backend="fake-probe",
            backend_version="1.0",
        )


class FakeFrameSampler:
    identity = ("fake-sampler", "1.0")

    def __init__(self) -> None:
        self.calls: list[tuple[str, FrameSamplingConfig]] = []

    def sample(
        self,
        source: AcceptedVideoSource,
        config: FrameSamplingConfig,
        deadline: datetime,
    ) -> FrameSampleBatch:
        del deadline
        self.calls.append((source.sha256, config))
        return FrameSampleBatch(
            frames=(
                SampledFrame(0, 0.0, 640, 360, payload=b"frame-0"),
                SampledFrame(1, 1.0, 640, 360, payload=b"frame-1"),
            )
        )


class FakeEvaluator:
    def __init__(
        self,
        decisions: dict[tuple[str, str], bool],
        *,
        failure: tuple[str, str, VideoFilterCapabilityError] | None = None,
    ) -> None:
        self.decisions = decisions
        self.failure = failure
        self.calls: list[tuple[str, str, int]] = []

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        del source, metadata, deadline
        self.calls.append((item.item_id, criterion.criterion_id, len(frames)))
        if self.failure is not None and self.failure[:2] == (
            item.item_id,
            criterion.criterion_id,
        ):
            raise self.failure[2]
        passed = self.decisions[(item.item_id, criterion.criterion_id)]
        return VideoCriterionResult(
            item_id=item.item_id,
            criterion_id=criterion.criterion_id,
            passed=passed,
            score=1.0 if passed else 0.0,
        )
