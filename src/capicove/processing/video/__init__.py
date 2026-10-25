"""Public video processing contracts and operations."""

from capicove.processing.video.deduplication import (
    VideoDeduplicationResult,
    deduplicate_video_files,
)
from capicove.processing.video.filtering.video_filter_pipeline import VideoFilterPipeline
from capicove.processing.video.models import (
    AcceptedVideoSource,
    DurationSplitConfig,
    KeyframeSplitConfig,
    SceneDetectorName,
    SceneSplitConfig,
    TimeRange,
    VideoAsset,
    VideoBackend,
    VideoOutputFormat,
    VideoSegment,
    VideoSource,
    VideoSplitFailureCode,
    VideoSplitRequest,
    VideoSplitResult,
    VideoSplitStatus,
    VideoSplitStrategy,
    VideoSplitWarning,
    VideoSplitWarningCode,
    VideoWriteMode,
)
from capicove.processing.video.splitter import VideoSplitter


__all__ = [
    "AcceptedVideoSource",
    "DurationSplitConfig",
    "KeyframeSplitConfig",
    "SceneDetectorName",
    "SceneSplitConfig",
    "TimeRange",
    "VideoAsset",
    "VideoBackend",
    "VideoDeduplicationResult",
    "VideoFilterPipeline",
    "VideoOutputFormat",
    "VideoSegment",
    "VideoSource",
    "VideoSplitFailureCode",
    "VideoSplitRequest",
    "VideoSplitResult",
    "VideoSplitStatus",
    "VideoSplitStrategy",
    "VideoSplitWarning",
    "VideoSplitWarningCode",
    "VideoSplitter",
    "VideoWriteMode",
    "deduplicate_video_files",
]
