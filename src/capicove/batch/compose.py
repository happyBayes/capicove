"""Closed batch operation names, media types, and image operation composition.

Shared by the standalone `capicove batch run` command and the workspace operator
host (RFC-0016 amendment 1), so both select the same allowlisted Tools.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING

from capicove.batch.models import BatchModality


if TYPE_CHECKING:
    from pathlib import Path

    from capicove.batch.tool_operations import BatchToolInvoker, ImageToolBatchOperation


IMAGE_OPERATIONS = (
    "image_aspect_ratio_filter",
    "image_shape_filter",
    "image_size_filter",
    "image_subplot_filter",
)
VIDEO_OPERATIONS = ("video_filter", "video_split")
OPERATIONS = (*IMAGE_OPERATIONS, *VIDEO_OPERATIONS)
IMAGE_MEDIA_TYPES: Mapping[str, str] = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}
VIDEO_MEDIA_TYPES: Mapping[str, str] = {
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".webm": "video/webm",
}


def media_types(modality: BatchModality) -> Mapping[str, str]:
    """Return the discovery extensions for one supported batch modality."""
    if modality is BatchModality.IMAGE:
        return IMAGE_MEDIA_TYPES
    if modality is BatchModality.VIDEO:
        return VIDEO_MEDIA_TYPES
    raise ValueError("batch CLI currently supports image and video operations")


def build_image_batch_operation(
    name: str,
    parameters: Mapping[str, object],
    *,
    input_root: Path,
    provider_fingerprint: str,
    invoker: BatchToolInvoker | None = None,
    workspace_id: str = "batch-local",
    actor_id: str = "batch-worker",
    environment_id: str = "batch-local",
) -> ImageToolBatchOperation:
    """Compose one allowlisted image filter bound to a single read root.

    Args:
        name: One of ``IMAGE_OPERATIONS``.
        parameters: Provider settings applied to the filter's defaults.
        input_root: The only directory the Tool handler may read.
        provider_fingerprint: Host-owned provider identity for cache keys.
        invoker: Optional tracing invoker.
        workspace_id: Invocation workspace identity.
        actor_id: Invocation actor identity.
        environment_id: Invocation environment identity.

    Returns:
        The batch operation.

    Raises:
        ValueError: The name or parameters are not accepted.
    """
    from capicove.batch.tool_operations import ImageToolBatchOperation
    from capicove.processing.image import (
        ImageAspectRatioFilter,
        ImageShapeFilter,
        ImageSizeFilter,
        ImageSubplotFilter,
    )
    from capicove.tools.image import build_image_tool_handler

    defaults: dict[str, object] = {
        "image_aspect_ratio_filter": ImageAspectRatioFilter(),
        "image_shape_filter": ImageShapeFilter(),
        "image_size_filter": ImageSizeFilter(),
        "image_subplot_filter": ImageSubplotFilter(),
    }
    if name not in defaults:
        raise ValueError("unsupported image batch operation")
    try:
        capability = dataclasses.replace(defaults[name], **parameters)  # type: ignore[type-var]
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid parameters for {name}") from error
    handler = build_image_tool_handler(name, capability, read_roots=(input_root,))
    return ImageToolBatchOperation(
        name,
        handler,
        operation_parameters=parameters,
        provider_fingerprint=provider_fingerprint,
        invoker=invoker,
        workspace_id=workspace_id,
        actor_id=actor_id,
        environment_id=environment_id,
    )


__all__ = [
    "IMAGE_MEDIA_TYPES",
    "IMAGE_OPERATIONS",
    "OPERATIONS",
    "VIDEO_MEDIA_TYPES",
    "VIDEO_OPERATIONS",
    "build_image_batch_operation",
    "media_types",
]
