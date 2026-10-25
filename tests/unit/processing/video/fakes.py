from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from capicove.processing.video import (
    AcceptedVideoSource,
    SceneSplitConfig,
    TimeRange,
    VideoBackend,
    VideoWriteMode,
)
from capicove.processing.video.ports import (
    BackendUnavailableError,
    ClipWriteError,
    OutputStoreError,
    PathPolicyError,
    PendingOutput,
    StoredOutput,
    VideoBackendBundle,
    VideoMetadata,
    WrittenClip,
)


class FakeInspector:
    def __init__(self, metadata: VideoMetadata, error: Exception | None = None) -> None:
        self.metadata = metadata
        self.error = error
        self.calls = 0

    def inspect(self, source: AcceptedVideoSource, deadline: datetime) -> VideoMetadata:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.metadata


class FakeWriter:
    def __init__(self, failed_indices: frozenset[int] = frozenset()) -> None:
        self.failed_indices = failed_indices
        self.calls: list[TimeRange] = []

    def write(
        self,
        source: AcceptedVideoSource,
        time_range: TimeRange,
        destination: PendingOutput,
        deadline: datetime,
    ) -> WrittenClip:
        index = len(self.calls)
        self.calls.append(time_range)
        if index in self.failed_indices:
            raise ClipWriteError(f"planned segment {index} failed")
        payload = f"segment:{time_range.start_seconds:.6f}:{time_range.end_seconds:.6f}".encode()
        destination.path.write_bytes(payload)
        return WrittenClip(
            path=destination.path,
            sha256=hashlib.sha256(payload).hexdigest(),
            mime_type="video/mp4",
        )


class FakeStore:
    def __init__(
        self,
        root: Path,
        *,
        fail_allocate_indices: frozenset[int] = frozenset(),
        fail_finalize_indices: frozenset[int] = frozenset(),
        fail_discard: bool = False,
    ) -> None:
        self.root = root
        self.fail_allocate_indices = fail_allocate_indices
        self.fail_finalize_indices = fail_finalize_indices
        self.fail_discard = fail_discard
        self.allocated: list[PendingOutput] = []
        self.finalized: list[StoredOutput] = []
        self.discarded: list[PendingOutput] = []

    def allocate(self, identity: str, suffix: str) -> PendingOutput:
        if len(self.allocated) in self.fail_allocate_indices:
            self.allocated.append(PendingOutput(identity, self.root / "allocation-failed"))
            raise OutputStoreError("allocation failed")
        pending = PendingOutput(identity, self.root / f"{len(self.allocated)}{suffix}.partial")
        self.allocated.append(pending)
        return pending

    def finalize(self, pending: PendingOutput, written: WrittenClip) -> StoredOutput:
        index = len(self.finalized)
        if index in self.fail_finalize_indices:
            raise OutputStoreError("finalization failed")
        stored = StoredOutput(
            identity=pending.identity,
            path=written.path,
            sha256=written.sha256,
            mime_type=written.mime_type,
        )
        self.finalized.append(stored)
        return stored

    def load_bytes(self, stored: StoredOutput) -> bytes:
        return stored.path.read_bytes()

    def discard(self, pending: PendingOutput) -> None:
        self.discarded.append(pending)
        if self.fail_discard:
            raise OutputStoreError("discard failed")
        pending.path.unlink(missing_ok=True)


class FakeBackendResolver:
    def __init__(
        self, inspector: FakeInspector, writer: FakeWriter, unavailable: bool = False
    ) -> None:
        self.inspector = inspector
        self.writer = writer
        self.unavailable = unavailable
        self.requests: list[VideoBackend] = []

    def resolve(
        self,
        requested: VideoBackend,
        write_mode: VideoWriteMode,
    ) -> VideoBackendBundle:
        del write_mode
        self.requests.append(requested)
        if self.unavailable:
            raise BackendUnavailableError("backend unavailable")
        return VideoBackendBundle("fake", "1.0", self.inspector, self.writer)


class FakeDetector:
    identity = ("fake-scenes", "1.0")

    def __init__(self, ranges: tuple[TimeRange, ...], error: Exception | None = None) -> None:
        self.ranges = ranges
        self.error = error
        self.calls = 0

    def detect(
        self,
        source: AcceptedVideoSource,
        config: SceneSplitConfig,
        deadline: datetime,
    ) -> tuple[TimeRange, ...]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.ranges


class FakePathAuthorizer:
    def __init__(self, deny: bool = False) -> None:
        self.deny = deny
        self.reads: list[Path] = []
        self.writes: list[Path] = []

    def authorize_read(self, policy_ref: str, path: Path) -> None:
        self.reads.append(path)
        if self.deny:
            raise PathPolicyError("read denied")

    def authorize_write(self, policy_ref: str, path: Path) -> None:
        self.writes.append(path)
        if self.deny:
            raise PathPolicyError("write denied")
