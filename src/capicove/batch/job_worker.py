"""Explicit local worker for durable, allowlisted batch jobs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite
from threading import Event, Thread

from capicove.batch._adapters.sqlite import SqliteBatchStore
from capicove.batch.executor import _MAX_WORKERS, BatchExecutor
from capicove.batch.job_checkpoint_store import JobCheckpointStores
from capicove.batch.job_registry import (
    _IDENTIFIER,
    JobClaimLostError,
    JobRecord,
    JobTransitionError,
)
from capicove.batch.job_service import BatchJobService
from capicove.batch.ports import BatchOperation
from capicove.batch.retry import RetryPolicy


class JobWorker:
    """Run one queued local job without depending on an MCP client lifetime."""

    def __init__(
        self,
        service: BatchJobService,
        store: SqliteBatchStore,
        operation: BatchOperation,
        *,
        worker_id: str,
        max_workers: int,
        retry_policy: RetryPolicy,
        lease_seconds: float = 30.0,
    ) -> None:
        if operation.fingerprint != service.operation_fingerprint:
            raise ValueError("operation fingerprint does not match the configured job service")
        if not isinstance(worker_id, str) or _IDENTIFIER.fullmatch(worker_id) is None:
            raise ValueError("worker_id must be a bounded identifier")
        if type(max_workers) is not int or not 1 <= max_workers <= _MAX_WORKERS:
            raise ValueError(f"max_workers must be between 1 and {_MAX_WORKERS}")
        if (
            type(lease_seconds) not in (int, float)
            or not isfinite(lease_seconds)
            or not 0.3 <= lease_seconds <= 3600
        ):
            raise ValueError("lease_seconds must be between 0.3 and 3600")
        self._service = service
        self._cache_store = store
        self._checkpoint_stores = JobCheckpointStores(store)
        self._operation = operation
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._max_workers = max_workers
        self._retry_policy = retry_policy

    def run_next(self) -> JobRecord | None:
        """Run the oldest queued job, or return when the queue is empty."""
        queued = self._service.next_queued()
        if queued is None:
            return None
        try:
            return self.run_one(queued.job_id)
        except JobClaimLostError:
            return None

    def recover_next(self, *, now: datetime) -> JobRecord | None:
        """Recover one expired Job using its own durable checkpoints."""
        expired = self._service.next_expired(now=now)
        if expired is None:
            return None
        try:
            checkpoints = self._checkpoint_stores.for_job(
                expired.job_id,
                require_existing=expired.checkpoint_prepared,
            )
        except FileNotFoundError as error:
            raise JobTransitionError("checkpoint database is unavailable") from error
        try:
            if expired.state == "cancelling":
                return self._service.recover_cancelled(
                    job_id=expired.job_id,
                    now=now,
                    checkpoint_store=checkpoints,
                )
            if self._service.has_pending_terminal(expired.job_id):
                return self._service.finalize_terminal(
                    job_id=expired.job_id,
                    now=now,
                    checkpoint_store=checkpoints,
                )
            return self._service.recover_expired(
                job_id=expired.job_id,
                now=now,
                checkpoint_store=checkpoints,
            )
        except JobTransitionError:
            latest = self._service.status(expired.job_id)
            if latest.revision != expired.revision or latest.state != expired.state:
                return None
            raise

    def run_until_stopped(self, stop: Event, *, poll_seconds: float = 1.0) -> None:
        """Process queued jobs until the operator signals the local worker to stop."""
        if (
            type(poll_seconds) not in (int, float)
            or not isfinite(poll_seconds)
            or not 0.01 <= poll_seconds <= 60
        ):
            raise ValueError("poll_seconds must be between 0.01 and 60")
        while not stop.is_set():
            if self.recover_next(now=datetime.now(UTC)) is not None:
                continue
            if self.run_next() is None:
                stop.wait(poll_seconds)

    def run_one(self, job_id: str) -> JobRecord:
        """Claim, execute, and publish one durable job under a renewed lease."""
        if self._operation.fingerprint != self._service.operation_fingerprint:
            raise ValueError("operation fingerprint does not match the configured job service")
        queued = self._service.status(job_id)
        if queued.state != "queued":
            raise JobClaimLostError("job is no longer queued")
        checkpoints = self._checkpoint_stores.for_job(job_id)
        now = datetime.now(UTC)
        prepared, plan = self._service.claim_and_prepare(
            job_id=job_id,
            worker_id=self._worker_id,
            now=now,
            lease_until=now + timedelta(seconds=self._lease_seconds),
            checkpoint_store=checkpoints,
        )
        if not prepared.checkpoint_prepared:
            raise JobTransitionError("worker checkpoints were not prepared")
        stop = Event()
        heartbeat_failures: list[Exception] = []

        def renew() -> None:
            while not stop.wait(self._lease_seconds / 3):
                observed = datetime.now(UTC)
                try:
                    for attempt in range(2):
                        current = self._service.status(job_id)
                        if current.worker_id != self._worker_id or current.state not in {
                            "running",
                            "cancelling",
                        }:
                            raise JobTransitionError("worker lease changed during heartbeat")
                        try:
                            self._service.renew_lease(
                                job_id=job_id,
                                worker_id=self._worker_id,
                                expected_revision=current.revision,
                                now=observed,
                                lease_until=observed + timedelta(seconds=self._lease_seconds),
                            )
                        except JobTransitionError:
                            if attempt:
                                raise
                        else:
                            break
                except Exception as error:
                    heartbeat_failures.append(error)
                    stop.set()
                    return

        heartbeat = Thread(target=renew, name="capicove-job-heartbeat", daemon=True)
        heartbeat.start()

        def should_cancel() -> bool:
            if stop.is_set():
                raise JobTransitionError("worker lease renewal failed")
            return self._service.status(job_id).state == "cancelling"

        try:
            BatchExecutor(
                checkpoints,
                self._cache_store,
                max_workers=self._max_workers,
                retry_policy=self._retry_policy,
            ).execute(
                plan,
                self._operation,
                should_cancel=should_cancel,
            )
        finally:
            stop.set()
            heartbeat.join()
        if heartbeat_failures:
            raise JobTransitionError("worker lease renewal failed") from heartbeat_failures[0]
        current = self._service.status(job_id)
        if current.state not in {"running", "cancelling"} or current.worker_id != self._worker_id:
            raise JobTransitionError("worker lease changed before terminal publication")
        self._service.stage_terminal(
            job_id=job_id,
            worker_id=self._worker_id,
            expected_revision=current.revision,
            now=datetime.now(UTC),
            checkpoint_store=checkpoints,
        )
        return self._service.finalize_terminal(
            job_id=job_id,
            worker_id=self._worker_id,
            now=datetime.now(UTC),
            checkpoint_store=checkpoints,
        )
