"""OpenCV Farneback evaluator for video motion filtering."""

from __future__ import annotations

import importlib
import math
from datetime import UTC, datetime
from typing import ClassVar

from capicove.processing.video.filtering.video_filter_contracts import (
    FrameMetric,
    MotionCriterion,
    VideoCriterionResult,
    VideoFilterCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    DecodeFailedFilterError,
    InvalidVideoFilterInputError,
    SampledFrame,
    VideoFilterTimeoutError,
)
from capicove.processing.video.filtering.video_filter_rules import in_range
from capicove.processing.video.models import AcceptedVideoSource


class OpenCvFarnebackMotionEvaluator:
    """Compute mean dense optical-flow magnitude across sampled frame pairs."""

    _parameters: ClassVar[dict[str, float | int]] = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    }

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        """Evaluate one motion criterion for one accepted source."""
        del source, metadata
        if not isinstance(criterion, MotionCriterion):
            raise InvalidVideoFilterInputError("Farneback requires a motion criterion.")
        if len(frames) < 2:
            raise DecodeFailedFilterError("Motion scoring requires at least two frames.")
        try:
            cv2 = importlib.import_module("cv2")
            np = importlib.import_module("numpy")
        except ModuleNotFoundError as error:
            raise BackendUnavailableFilterError(
                "OpenCV and NumPy are required; install capicove[video-filter]."
            ) from error

        previous = None
        metrics: list[FrameMetric] = []
        for frame in frames:
            if datetime.now(UTC) >= deadline:
                raise VideoFilterTimeoutError("The motion scoring deadline expired.")
            image = np.asarray(frame.payload)
            if image.ndim != 3 or image.shape[2] != 3:
                raise DecodeFailedFilterError("Motion frames must use three-channel BGR pixels.")
            if criterion.resize is not None:
                image = cv2.resize(image, criterion.resize, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, **self._parameters)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                raw_score = float(np.mean(magnitude))
                score = raw_score
                if criterion.relative:
                    height, width = gray.shape[:2]
                    score /= math.hypot(height, width)
                metrics.append(
                    FrameMetric(
                        frame_index=frame.index,
                        timestamp_seconds=frame.timestamp_seconds,
                        value=score,
                        raw_value=raw_score,
                    )
                )
            previous = gray

        score = sum(metric.value for metric in metrics) / len(metrics)
        return VideoCriterionResult(
            item_id=item.item_id,
            criterion_id=criterion.criterion_id,
            passed=in_range(score, criterion.min_score, criterion.max_score),
            score=score,
            frame_metrics=tuple(metrics),
            details={
                "sampling_fps": criterion.sampling_fps,
                "relative": criterion.relative,
                "resize": criterion.resize,
                "backend": "opencv-farneback",
            },
        )
