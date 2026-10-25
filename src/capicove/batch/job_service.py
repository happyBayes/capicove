"""Operator-scoped planning and durable queueing for local batch jobs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from capicove.batch.checkpoint import CheckpointStore
from capicove.batch.discovery import (
    DirectoryDiscoverer,
    DirectoryDiscoveryError,
    _observe_regular_file,
)
from capicove.batch.job_plan_receipt import encode_plan_receipt
from capicove.batch.job_registry import JobRecord, JobTransitionError, SqliteJobRegistry
from capicove.batch.job_terminal import TerminalReceipt, build_terminal_receipt
from capicove.batch.models import BatchModality, BatchPlan, BatchState, OperationFingerprint
from capicove.batch.planner import BatchPlanner


class BatchJobService:
    """Bind batch operations to roots and providers selected at startup."""

    def __init__(
        self,
        *,
        database_path: Path,
        workspace_id: str,
        input_root: Path,
        input_root_alias: str,
        input_root_sha256: str,
        output_root_alias: str,
        output_root_sha256: str,
        modality: BatchModality,
        media_types: Mapping[str, str],
        operation: OperationFingerprint,
    ) -> None:
        if not input_root.is_absolute() or input_root.resolve(strict=True) != input_root:
            raise ValueError("batch input root must be absolute and free of symlinks")
        if modality not in operation.modalities:
            raise ValueError("batch operation does not support the selected modality")
        self._database_path = database_path
        self._workspace_id = workspace_id
        self._input_root = input_root
        self._input_root_alias = input_root_alias
        self._input_root_sha256 = input_root_sha256
        self._output_root_alias = output_root_alias
        self._output_root_sha256 = output_root_sha256
        self._modality = modality
        self._discoverer = DirectoryDiscoverer(media_types)
        self._operation = operation

    def plan(self) -> BatchPlan:
        """Compute a bounded current plan without creating scheduling state."""
        items = self._discoverer.discover(self._input_root, self._modality)
        return BatchPlanner().plan(items, self._operation)

    @property
    def operation_fingerprint(self) -> OperationFingerprint:
        """Expose the startup-authorized operation identity to the local worker."""
        return self._operation

    def enqueue(self, *, plan_id: str, idempotency_key: str) -> JobRecord:
        """Recompute a requested plan before atomically persisting its job."""
        current = self.plan()
        if current.plan_id != plan_id:
            raise ValueError("plan_changed")
        return SqliteJobRegistry(self._database_path).enqueue_plan(
            workspace_id=self._workspace_id,
            plan=current,
            input_root=self._input_root,
            input_root_alias=self._input_root_alias,
            input_root_sha256=self._input_root_sha256,
            output_root_alias=self._output_root_alias,
            output_root_sha256=self._output_root_sha256,
            idempotency_key=idempotency_key,
        )

    def status(self, job_id: str) -> JobRecord:
        """Read one durable job within the configured workspace."""
        return self._existing_registry().status(
            workspace_id=self._workspace_id,
            job_id=job_id,
        )

    def cancel(self, *, job_id: str, expected_revision: int) -> JobRecord:
        """Request revisioned cancellation within the configured workspace."""
        return self._existing_registry().cancel(
            workspace_id=self._workspace_id,
            job_id=job_id,
            expected_revision=expected_revision,
        )

    def next_queued(self) -> JobRecord | None:
        """Read the oldest queued job without creating an empty registry."""
        if not self._database_path.exists():
            return None
        return SqliteJobRegistry(self._database_path).next_queued(
            workspace_id=self._workspace_id,
        )

    def next_expired(self, *, now: datetime) -> JobRecord | None:
        """Read one elapsed lease in this workspace without creating a registry."""
        if not self._database_path.exists():
            return None
        return SqliteJobRegistry(self._database_path).next_expired(
            workspace_id=self._workspace_id,
            now=now,
        )

    def has_pending_terminal(self, job_id: str) -> bool:
        """Check whether a scoped Job has a receipt awaiting verification."""
        if not self._database_path.exists():
            return False
        return SqliteJobRegistry(self._database_path).has_pending_terminal(
            workspace_id=self._workspace_id,
            job_id=job_id,
        )

    def claim_and_prepare(
        self,
        *,
        job_id: str,
        worker_id: str,
        now: datetime,
        lease_until: datetime,
        checkpoint_store: CheckpointStore,
    ) -> tuple[JobRecord, BatchPlan]:
        """Claim a queued job and durably prepare its item checkpoints."""
        registry = self._existing_registry()
        queued = registry.status(workspace_id=self._workspace_id, job_id=job_id)
        if (
            queued.input_root_alias != self._input_root_alias
            or queued.input_root_sha256 != self._input_root_sha256
            or queued.output_root_alias != self._output_root_alias
            or queued.output_root_sha256 != self._output_root_sha256
        ):
            raise ValueError("job configuration changed")
        plan = registry.load_plan(
            workspace_id=self._workspace_id,
            plan_id=queued.plan_id,
            input_root=self._input_root,
            input_root_alias=self._input_root_alias,
            input_root_sha256=self._input_root_sha256,
        )
        _, _, operation_digest = encode_plan_receipt(plan, self._input_root)
        if plan.operation != self._operation or queued.operation_fingerprint != operation_digest:
            raise ValueError("job operation changed")
        claimed = registry.claim(
            workspace_id=self._workspace_id,
            job_id=job_id,
            worker_id=worker_id,
            now=now,
            lease_until=lease_until,
        )
        checkpoint_store.prepare(plan)
        prepared = self.mark_prepared(
            job_id=job_id,
            worker_id=worker_id,
            expected_revision=claimed.revision,
            now=now,
            checkpoint_store=checkpoint_store,
        )
        return prepared, plan

    def recover_expired(
        self,
        *,
        job_id: str,
        now: datetime,
        checkpoint_store: CheckpointStore,
    ) -> JobRecord:
        """Release an expired lease only for matching durable item checkpoints."""
        registry = self._existing_registry()

        def verify(record: JobRecord) -> bool:
            return self._checkpoints_match(
                registry,
                record,
                checkpoint_store,
                allow_absent=not record.checkpoint_prepared,
            )

        return registry.recover_expired(
            workspace_id=self._workspace_id,
            job_id=job_id,
            now=now,
            verify_checkpoint=verify,
        )

    def resume(
        self,
        *,
        job_id: str,
        expected_revision: int,
        checkpoint_store: CheckpointStore,
    ) -> JobRecord:
        """Requeue one interrupted job after checkpoint and source verification."""
        registry = self._existing_registry()

        def verify(record: JobRecord) -> None:
            if not self._checkpoints_match(
                registry,
                record,
                checkpoint_store,
                allow_absent=not record.checkpoint_prepared,
            ):
                raise JobTransitionError("checkpoint verification failed")
            plan = registry.load_plan(
                workspace_id=self._workspace_id,
                plan_id=record.plan_id,
                input_root=self._input_root,
                input_root_alias=self._input_root_alias,
                input_root_sha256=self._input_root_sha256,
            )
            for planned in plan.items:
                item = planned.item
                try:
                    observed = _observe_regular_file(
                        item.source_path, expected_snapshot=item.snapshot
                    )
                except DirectoryDiscoveryError as error:
                    raise ValueError("source_changed") from error
                if observed != (item.sha256, item.snapshot):
                    raise ValueError("source_changed")

        return registry.resume(
            workspace_id=self._workspace_id,
            job_id=job_id,
            expected_revision=expected_revision,
            now=datetime.now(UTC),
            verify_resume=verify,
        )

    def mark_prepared(
        self,
        *,
        job_id: str,
        worker_id: str,
        expected_revision: int,
        now: datetime,
        checkpoint_store: CheckpointStore,
    ) -> JobRecord:
        """Require complete checkpoints before permitting item execution."""
        registry = self._existing_registry()
        record = registry.status(workspace_id=self._workspace_id, job_id=job_id)
        if not self._checkpoints_match(registry, record, checkpoint_store, allow_absent=False):
            raise JobTransitionError("checkpoint verification failed")
        return registry.mark_prepared(
            workspace_id=self._workspace_id,
            job_id=job_id,
            worker_id=worker_id,
            expected_revision=expected_revision,
            now=now,
        )

    def renew_lease(
        self,
        *,
        job_id: str,
        worker_id: str,
        expected_revision: int,
        now: datetime,
        lease_until: datetime,
    ) -> JobRecord:
        """Extend a live worker lease without changing the selected authority."""
        return self._existing_registry().renew_lease(
            workspace_id=self._workspace_id,
            job_id=job_id,
            worker_id=worker_id,
            expected_revision=expected_revision,
            now=now,
            lease_until=lease_until,
        )

    def recover_cancelled(
        self,
        *,
        job_id: str,
        now: datetime,
        checkpoint_store: CheckpointStore,
    ) -> JobRecord:
        """Finish an expired cooperative cancellation without rerunning providers."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        registry = self._existing_registry()
        record = registry.status(workspace_id=self._workspace_id, job_id=job_id)
        if record.state != "cancelling" or record.lease_until is None:
            raise JobTransitionError("job is not cancelling")
        if now < record.lease_until:
            raise JobTransitionError("job lease active")
        if not self._checkpoints_match(
            registry,
            record,
            checkpoint_store,
            allow_absent=not record.checkpoint_prepared,
        ):
            raise JobTransitionError("checkpoint verification failed")
        plan = registry.load_plan(
            workspace_id=self._workspace_id,
            plan_id=record.plan_id,
            input_root=self._input_root,
            input_root_alias=self._input_root_alias,
            input_root_sha256=self._input_root_sha256,
        )
        if not record.checkpoint_prepared:
            checkpoint_store.prepare(plan)
            if not self._checkpoints_match(
                registry,
                record,
                checkpoint_store,
                allow_absent=False,
            ):
                raise JobTransitionError("checkpoint verification failed")
        for checkpoint in checkpoint_store.list(plan.plan_id):
            if checkpoint.state in {BatchState.PENDING, BatchState.RUNNING}:
                checkpoint_store.update(
                    plan.plan_id,
                    checkpoint.item_id,
                    expected_states=frozenset({checkpoint.state}),
                    state=BatchState.SKIPPED,
                    attempts=checkpoint.attempts,
                )
        self.stage_terminal(
            job_id=job_id,
            worker_id=None,
            expected_revision=record.revision,
            now=now,
            checkpoint_store=checkpoint_store,
        )
        return self.finalize_terminal(
            job_id=job_id,
            worker_id=None,
            now=now,
            checkpoint_store=checkpoint_store,
        )

    def stage_terminal(
        self,
        *,
        job_id: str,
        worker_id: str | None,
        expected_revision: int,
        now: datetime,
        checkpoint_store: CheckpointStore,
    ) -> TerminalReceipt:
        """Stage a complete checkpoint summary without terminal visibility."""
        registry = self._existing_registry()
        record = registry.status(workspace_id=self._workspace_id, job_id=job_id)
        if not self._checkpoints_match(registry, record, checkpoint_store, allow_absent=False):
            raise JobTransitionError("checkpoint verification failed")
        plan = registry.load_plan(
            workspace_id=self._workspace_id,
            plan_id=record.plan_id,
            input_root=self._input_root,
            input_root_alias=self._input_root_alias,
            input_root_sha256=self._input_root_sha256,
        )
        receipt = build_terminal_receipt(job_id, plan, checkpoint_store.list(plan.plan_id))
        return registry.stage_terminal(
            workspace_id=self._workspace_id,
            job_id=job_id,
            worker_id=worker_id,
            expected_revision=expected_revision,
            now=now,
            receipt=receipt,
        )

    def finalize_terminal(
        self,
        *,
        job_id: str,
        now: datetime,
        checkpoint_store: CheckpointStore,
        worker_id: str | None = None,
    ) -> JobRecord:
        """Verify a pending receipt against checkpoints before publication."""
        registry = self._existing_registry()

        def verify(receipt: TerminalReceipt) -> bool:
            record = registry.status(workspace_id=self._workspace_id, job_id=job_id)
            if not self._checkpoints_match(registry, record, checkpoint_store, allow_absent=False):
                return False
            plan = registry.load_plan(
                workspace_id=self._workspace_id,
                plan_id=record.plan_id,
                input_root=self._input_root,
                input_root_alias=self._input_root_alias,
                input_root_sha256=self._input_root_sha256,
            )
            try:
                current = build_terminal_receipt(
                    job_id,
                    plan,
                    checkpoint_store.list(plan.plan_id),
                )
            except ValueError:
                return False
            return current == receipt

        return registry.finalize_terminal(
            workspace_id=self._workspace_id,
            job_id=job_id,
            now=now,
            worker_id=worker_id,
            verify_receipt=verify,
        )

    def _existing_registry(self) -> SqliteJobRegistry:
        """Open scheduling state only after an authorized enqueue created it."""
        if not self._database_path.exists():
            raise KeyError("job is unavailable")
        return SqliteJobRegistry(self._database_path)

    def _checkpoints_match(
        self,
        registry: SqliteJobRegistry,
        record: JobRecord,
        checkpoint_store: CheckpointStore,
        *,
        allow_absent: bool,
    ) -> bool:
        if (
            record.input_root_alias != self._input_root_alias
            or record.input_root_sha256 != self._input_root_sha256
            or record.output_root_alias != self._output_root_alias
            or record.output_root_sha256 != self._output_root_sha256
        ):
            return False
        plan = registry.load_plan(
            workspace_id=self._workspace_id,
            plan_id=record.plan_id,
            input_root=self._input_root,
            input_root_alias=self._input_root_alias,
            input_root_sha256=self._input_root_sha256,
        )
        _, _, operation_digest = encode_plan_receipt(plan, self._input_root)
        if plan.operation != self._operation or record.operation_fingerprint != operation_digest:
            return False
        expected = {
            planned.item.item_id: (planned.cache_key, planned.item.sha256) for planned in plan.items
        }
        try:
            observed = checkpoint_store.list(plan.plan_id)
            if allow_absent and not observed:
                return True
            return len(observed) == len(expected) and all(
                expected.get(checkpoint.item_id) == (checkpoint.cache_key, checkpoint.source_sha256)
                and bool(history := checkpoint_store.history(plan.plan_id, checkpoint.item_id))
                and history[-1] == checkpoint
                for checkpoint in observed
            )
        except (TypeError, ValueError):
            return False
