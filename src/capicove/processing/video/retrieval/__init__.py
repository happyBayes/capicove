"""Offline video retrieval preparation public values and operation."""

from capicove.processing.video.retrieval.local import (
    create_local_video_preparation_service,
    read_local_preparation_resource,
)
from capicove.processing.video.retrieval.models import (
    VideoRetrievalPreparationRequest,
    VideoRetrievalPreparationResult,
)
from capicove.processing.video.retrieval.pipeline import VideoRetrievalPreparationService
from capicove.processing.video.retrieval.shared_ocr import SharedImageOCRProcessor


__all__ = [
    "SharedImageOCRProcessor",
    "VideoRetrievalPreparationRequest",
    "VideoRetrievalPreparationResult",
    "VideoRetrievalPreparationService",
    "create_local_video_preparation_service",
    "read_local_preparation_resource",
]
