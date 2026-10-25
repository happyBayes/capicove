"""Byte-exact video deduplication contracts and operations."""

from capicove.processing.video.deduplication.actions import (
    delete_duplicate_files,
    move_duplicate_files,
)
from capicove.processing.video.deduplication.file import (
    DEFAULT_VIDEO_EXTENSIONS,
    VideoDeduplicationDependencyError,
    deduplicate_video_files,
)
from capicove.processing.video.deduplication.manifest import (
    build_manifest_items,
    write_deduplication_manifest,
)
from capicove.processing.video.deduplication.models import (
    DuplicateActionResult,
    DuplicateMoveRecord,
    ManifestItem,
    VideoDeduplicationFailure,
    VideoDeduplicationMode,
    VideoDeduplicationResult,
    VideoDeduplicationStats,
    VideoDuplicateGroup,
    VideoFileRecord,
)


__all__ = [
    "DEFAULT_VIDEO_EXTENSIONS",
    "DuplicateActionResult",
    "DuplicateMoveRecord",
    "ManifestItem",
    "VideoDeduplicationDependencyError",
    "VideoDeduplicationFailure",
    "VideoDeduplicationMode",
    "VideoDeduplicationResult",
    "VideoDeduplicationStats",
    "VideoDuplicateGroup",
    "VideoFileRecord",
    "build_manifest_items",
    "deduplicate_video_files",
    "delete_duplicate_files",
    "move_duplicate_files",
    "write_deduplication_manifest",
]
