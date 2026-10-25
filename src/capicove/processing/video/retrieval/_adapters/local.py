"""Local verified content and checkpoint storage plus metric scene extraction."""

import hashlib
import itertools
import os
import tempfile
from datetime import datetime
from pathlib import Path

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange
from capicove.processing.video.models import AcceptedVideoSource, VideoSource
from capicove.processing.video.ports import ArtifactResolver, PathAuthorizer
from capicove.processing.video.retrieval.models import SceneMetric, sha256
from capicove.processing.video.retrieval.pipeline import check_deadline
from capicove.processing.video.source import accept_source


def _publish(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("immutable content conflict") from None
    finally:
        Path(temporary).unlink(missing_ok=True)


class LocalContentStore:
    """Content-addressed local objects checked on every read."""

    def __init__(self, root: Path, *, max_object_bytes: int = 64 * 1024 * 1024) -> None:
        self.root = root.resolve()
        self.max_object_bytes = max_object_bytes

    def put(self, payload: bytes) -> str:
        """Publish one immutable bounded object."""
        if len(payload) > self.max_object_bytes:
            raise ValueError("object capacity exceeded")
        digest = hashlib.sha256(payload).hexdigest()
        _publish(self.root / digest, payload)
        return "resource:sha256:" + digest

    def read(self, reference: str) -> bytes:
        """Resolve only this store's validated content references."""
        if not reference.startswith(("resource:sha256:", "resource:preview:")):
            raise ValueError("invalid content reference")
        digest = reference.split(":", 2)[2]
        sha256(digest)
        path = self.root / digest
        if path.is_symlink() or path.stat().st_size > self.max_object_bytes:
            raise ValueError("invalid content object")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("content hash mismatch")
        return data


class LocalDerivedCacheCatalog:
    """Append-only fingerprint-to-manifest mappings, safe across restarts."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def lookup(self, fingerprint: ProcessingFingerprint) -> str | None:
        """Find a prior manifest without unpickling runtime objects."""
        path = self.root / fingerprint.sha256
        return path.read_text() if path.is_file() else None

    def publish(self, fingerprint: ProcessingFingerprint, manifest_ref: str) -> None:
        """Reject replacing an immutable checkpoint mapping."""
        _publish(self.root / fingerprint.sha256, manifest_ref.encode())


class MetricSceneExtractor:
    """Threshold already-decoded differences without reopening media."""

    def __init__(self, threshold: float = 0.35, minimum_scene_ms: int = 2000) -> None:
        if not 0 < threshold <= 1 or minimum_scene_ms <= 0:
            raise ValueError("invalid scene configuration")
        self.threshold = threshold
        self.minimum_scene_ms = minimum_scene_ms

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Include all scene policy inputs."""
        return ProcessingFingerprint.from_manifest(
            {
                "algorithm": "mean-rgb-diff-v1",
                "threshold": self.threshold,
                "minimum_scene_ms": self.minimum_scene_ms,
            }
        )

    def extract(
        self, metrics: tuple[SceneMetric, ...], *, duration_ms: int, deadline: datetime
    ) -> tuple[TemporalRange, ...]:
        """Build non-overlapping source-contained scenes."""
        edges = [0]
        for metric in metrics:
            check_deadline(deadline)
            if (
                metric.score >= self.threshold
                and metric.timestamp_ms - edges[-1] >= self.minimum_scene_ms
                and duration_ms - metric.timestamp_ms >= self.minimum_scene_ms
            ):
                edges.append(metric.timestamp_ms)
        edges.append(duration_ms)
        return tuple(TemporalRange(a, b) for a, b in itertools.pairwise(edges))


class LocalVideoSourceAcceptor:
    """Use existing video authorization and evidence acceptance."""

    def __init__(
        self,
        *,
        path_authorizer: PathAuthorizer | None = None,
        policy_ref: str | None = None,
        artifact_resolver: ArtifactResolver | None = None,
    ) -> None:
        self._authorizer = path_authorizer
        self._policy = policy_ref
        self._resolver = artifact_resolver

    def accept(self, source: VideoSource, *, deadline: datetime) -> AcceptedVideoSource:
        """Accept explicitly configured local source representations."""
        check_deadline(deadline)
        return accept_source(
            source,
            deadline,
            path_authorizer=self._authorizer,
            policy_ref=self._policy,
            artifact_resolver=self._resolver,
        )
