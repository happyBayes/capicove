"""Accept and hash path, byte, and Artifact video sources."""

from __future__ import annotations

import hashlib
import hmac
import mimetypes
from datetime import datetime
from pathlib import Path

from capicove.processing.video.models import AcceptedVideoSource, VideoSource
from capicove.processing.video.ports import ArtifactResolver, PathAuthorizer


class VideoSourceError(ValueError):
    """Report an invalid or unverifiable video source."""


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def accept_source(
    source: VideoSource,
    deadline: datetime,
    *,
    artifact_resolver: ArtifactResolver | None = None,
    path_authorizer: PathAuthorizer | None = None,
    policy_ref: str | None = None,
) -> AcceptedVideoSource:
    """Authorize, materialize, and hash a video source without modifying it."""
    if source.data is not None:
        digest = hashlib.sha256(source.data).hexdigest()
        return AcceptedVideoSource(
            data=source.data,
            source_id=source.source_id or f"sha256:{digest}",
            sha256=digest,
            mime_type="video/mp4",
        )

    if source.path is not None:
        if path_authorizer is None or policy_ref is None:
            raise VideoSourceError("a path authorizer and policy ref are required")
        resolved = source.path.resolve()
        path_authorizer.authorize_read(policy_ref, resolved)
        if not resolved.is_file():
            raise VideoSourceError(f"video source does not exist: {resolved}")
        digest = _hash_path(resolved)
        mime_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return AcceptedVideoSource(
            path=resolved,
            source_id=source.source_id or f"sha256:{digest}",
            sha256=digest,
            mime_type=mime_type,
        )

    if artifact_resolver is None or source.artifact_ref is None or source.artifact_sha256 is None:
        raise VideoSourceError("an artifact resolver is required for an Artifact source")
    accepted = artifact_resolver.resolve(source.artifact_ref, source.artifact_sha256, deadline)
    if not hmac.compare_digest(accepted.sha256, source.artifact_sha256):
        raise VideoSourceError("resolved Artifact hash does not match the expected hash")
    return accepted
