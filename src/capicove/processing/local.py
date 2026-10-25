"""Public local workspace operation for governed batch execution."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from capicove.contracts.data import DatasetSchema, DatasetSnapshotRef, LineageEdge
from capicove.processing._adapters.filesystem import (
    DatasetPublicationError,
    LocalDatasetPublisher,
    PublicationRequest,
)
from capicove.processing._adapters.sqlite import SqliteProcessingStore
from capicove.processing.models import ProcessingRun, RunState


class ProcessingRecoveryError(RuntimeError):
    """Report a local publication that cannot be reconciled safely."""


class LocalProcessingWorkspace:
    """Coordinate one local catalog and its crash-recoverable publications."""

    def __init__(self, output_root: Path) -> None:
        if not isinstance(output_root, Path) or not output_root.is_absolute():
            raise ValueError("output_root must be an absolute Path")
        self._output_root = output_root.resolve()
        metadata_root = self._output_root / ".capicove"
        if os.path.lexists(metadata_root) and metadata_root.is_symlink():
            raise ProcessingRecoveryError("metadata root must not be a symbolic link")
        self._store = SqliteProcessingStore(self._output_root / ".capicove" / "processing.sqlite3")
        self._publisher = LocalDatasetPublisher(self._output_root)
        with self._workspace_lock(blocking=False) as acquired:
            if acquired:
                self._reconcile_pending()

    @contextmanager
    def execution_lock(self) -> Iterator[None]:
        """Serialize execution and recovery for this output root."""
        with self._workspace_lock(blocking=True) as acquired:
            if not acquired:  # pragma: no cover - blocking acquisition invariant
                raise RuntimeError("processing workspace lock was not acquired")
            self._reconcile_pending()
            yield

    def get(self, snapshot_id: str, /) -> DatasetSnapshotRef | None:
        """Return one immutable snapshot by ID."""
        return self._store.get(snapshot_id)

    def latest(self, dataset_id: str, /) -> DatasetSnapshotRef | None:
        """Return the latest committed dataset version."""
        return self._store.latest(dataset_id)

    def validate_snapshot(
        self, snapshot_id: str, /, *, validator: Callable[[Path], bool]
    ) -> DatasetSnapshotRef | None:
        """Verify an exact committed snapshot, then run a bounded local reader.

        Returns None only for an unknown catalog identity. Unsafe paths, manifest
        or part drift, and a rejected validator raise ProcessingRecoveryError.
        The caller holds execution_lock when coordinating reads with publication.
        This verifies storage integrity, not source authorization or semantics.
        """
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            return None
        try:
            self._publisher.validate(snapshot, validator=validator)
        except (DatasetPublicationError, OSError, ValueError) as error:
            raise ProcessingRecoveryError("committed snapshot validation failed") from error
        return snapshot

    def create(self, run: ProcessingRun, /) -> None:
        """Create one submitted run idempotently."""
        self._store.create(run)

    def get_run(self, run_id: str, /) -> ProcessingRun | None:
        """Return one durable processing run."""
        return self._store.get_run(run_id)

    def transition(
        self,
        run_id: str,
        expected: RunState,
        target: RunState,
        /,
        *,
        native_run_id: str | None = None,
        output_snapshot_id: str | None = None,
        detail: str | None = None,
    ) -> ProcessingRun:
        """Compare and set one legal run transition."""
        return self._store.transition(
            run_id,
            expected,
            target,
            native_run_id=native_run_id,
            output_snapshot_id=output_snapshot_id,
            detail=detail,
        )

    def stage(self, run_id: str, /) -> Path:
        """Create one private staging directory."""
        return self._publisher.stage(run_id)

    def discard(self, staging_path: Path, /) -> None:
        """Discard one unpublished staging directory."""
        self._publisher.discard(staging_path)

    def publish(
        self,
        *,
        run_id: str,
        dataset_id: str,
        version: int,
        staging_path: Path,
        schema: DatasetSchema,
        processing_fingerprint: str,
        parent_snapshot_ids: tuple[str, ...],
        validator: Callable[[Path], bool],
    ) -> DatasetSnapshotRef:
        """Atomically rename one verified dataset with a recovery marker."""
        return self._publisher.publish(
            PublicationRequest(
                run_id=run_id,
                dataset_id=dataset_id,
                version=version,
                staging_path=staging_path,
                schema=schema,
                processing_fingerprint=processing_fingerprint,
                parent_snapshot_ids=parent_snapshot_ids,
            ),
            validator=validator,
        )

    def commit_success(
        self,
        run_id: str,
        snapshot: DatasetSnapshotRef,
        lineage: tuple[LineageEdge, ...],
        /,
        *,
        native_run_id: str,
    ) -> ProcessingRun:
        """Commit snapshot, lineage, and successful run state atomically."""
        return self._store.commit_success(
            run_id,
            snapshot,
            lineage,
            native_run_id=native_run_id,
        )

    def finalize(self, run_id: str, snapshot: DatasetSnapshotRef, /) -> bool:
        """Best-effort clear a successful publication's recovery marker."""
        try:
            self._publisher.finalize(run_id, snapshot)
        except (DatasetPublicationError, OSError):
            return False
        return True

    def retract(self, snapshot: DatasetSnapshotRef, /) -> bool:
        """Best-effort remove one exact uncommitted publication."""
        try:
            self._publisher.retract(snapshot)
        except (DatasetPublicationError, OSError):
            return False
        return True

    def _reconcile_pending(self) -> None:
        for pending in self._publisher.pending_publications():
            run = self._store.get_run(pending.run_id)
            snapshot = (
                self._store.get(pending.snapshot_id) if pending.snapshot_id is not None else None
            )
            if (
                run is not None
                and run.state is RunState.SUCCEEDED
                and pending.snapshot_id is not None
                and run.output_snapshot_id == pending.snapshot_id
                and snapshot is not None
            ):
                self._publisher.finalize_pending(pending, snapshot)
                continue
            if snapshot is not None or (run is not None and run.state is RunState.SUCCEEDED):
                raise ProcessingRecoveryError("pending publication metadata is inconsistent")
            try:
                self._publisher.retract_pending(pending)
            except (DatasetPublicationError, OSError) as error:
                raise ProcessingRecoveryError("pending publication cleanup failed") from error
            if run is not None and run.state in {RunState.SUBMITTED, RunState.RUNNING}:
                self._store.transition(
                    run.run_id,
                    run.state,
                    RunState.FAILED,
                    native_run_id=run.native_run_id,
                    detail="RECOVERED_INCOMPLETE_PUBLICATION",
                )

    @contextmanager
    def _workspace_lock(self, *, blocking: bool) -> Iterator[bool]:
        import fcntl

        lock_path = self._output_root / ".capicove" / "processing.lock"
        if lock_path.parent.is_symlink():
            raise ProcessingRecoveryError("processing lock parent must not be a symbolic link")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ProcessingRecoveryError("processing lock path is unsafe") from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ProcessingRecoveryError("processing lock must be a regular file")
        acquired = False
        try:
            operation = fcntl.LOCK_EX
            if not blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, operation)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
