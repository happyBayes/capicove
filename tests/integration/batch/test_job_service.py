"""Operator batch planning must remain read-only until authorized enqueue."""

import os
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.batch._adapters.sqlite import SqliteBatchStore
from capicove.batch.checkpoint import BatchCheckpoint, CheckpointTransitionError
from capicove.batch.executor import BatchExecutor
from capicove.batch.job_checkpoint_store import JobCheckpointStores
from capicove.batch.job_registry import JobTransitionError, SqliteJobRegistry
from capicove.batch.job_service import BatchJobService
from capicove.batch.job_worker import JobWorker
from capicove.batch.models import (
    BatchFailure,
    BatchItem,
    BatchModality,
    BatchState,
    OperationFingerprint,
)
from capicove.batch.retry import RetryPolicy


def _service(tmp_path: Path) -> BatchJobService:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    (input_root / "meeting.wav").write_bytes(b"original recording")
    return BatchJobService(
        database_path=tmp_path / "jobs.sqlite3",
        workspace_id="workspace-a",
        input_root=input_root,
        input_root_alias="inputs",
        input_root_sha256="a" * 64,
        output_root_alias="outputs",
        output_root_sha256="b" * 64,
        modality=BatchModality.AUDIO,
        media_types={".wav": "audio/wav"},
        operation=OperationFingerprint(
            operation_id="capicove.audio.transcribe",
            operation_version="1",
            modalities=frozenset({BatchModality.AUDIO}),
            parameters_sha256="c" * 64,
            provider_fingerprint="d" * 64,
        ),
    )


def test_plan_is_read_only_and_enqueue_persists_current_plan(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()

    assert len(planned.items) == 1
    assert not (tmp_path / "jobs.sqlite3").exists()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    assert queued.state == "queued"
    assert queued.plan_id == planned.plan_id
    assert service.status(queued.job_id) == queued
    assert (tmp_path / "jobs.sqlite3").exists()


def test_service_cancel_queued_job_with_revisioned_workspace_authority(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="cancel-one")

    cancelled = service.cancel(job_id=queued.job_id, expected_revision=queued.revision)

    assert cancelled.job_id == queued.job_id
    assert cancelled.state == "cancelled"
    assert cancelled.revision == queued.revision + 1
    assert service.status(queued.job_id) == cancelled
    with pytest.raises(JobTransitionError, match="revision_conflict"):
        service.cancel(job_id=queued.job_id, expected_revision=queued.revision)


def test_service_cancel_unknown_job_does_not_create_registry(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(KeyError, match="job is unavailable"):
        service.cancel(job_id="job:" + "0" * 24, expected_revision=1)

    assert not (tmp_path / "jobs.sqlite3").exists()


def test_service_resume_unknown_job_does_not_create_registry(tmp_path: Path) -> None:
    service = _service(tmp_path)
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    with pytest.raises(KeyError, match="job is unavailable"):
        service.resume(
            job_id="job:" + "0" * 24,
            expected_revision=1,
            checkpoint_store=checkpoints,
        )

    assert not (tmp_path / "jobs.sqlite3").exists()


@pytest.mark.parametrize(
    "invoke",
    [
        lambda service, checkpoints: service.claim_and_prepare(
            job_id="job:" + "0" * 24,
            worker_id="worker-one",
            now=datetime(2026, 1, 1, tzinfo=UTC),
            lease_until=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30),
            checkpoint_store=checkpoints,
        ),
        lambda service, checkpoints: service.recover_expired(
            job_id="job:" + "0" * 24,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=checkpoints,
        ),
        lambda service, checkpoints: service.mark_prepared(
            job_id="job:" + "0" * 24,
            worker_id="worker-one",
            expected_revision=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=checkpoints,
        ),
        lambda service, _checkpoints: service.renew_lease(
            job_id="job:" + "0" * 24,
            worker_id="worker-one",
            expected_revision=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            lease_until=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=30),
        ),
        lambda service, checkpoints: service.recover_cancelled(
            job_id="job:" + "0" * 24,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=checkpoints,
        ),
        lambda service, checkpoints: service.stage_terminal(
            job_id="job:" + "0" * 24,
            worker_id="worker-one",
            expected_revision=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=checkpoints,
        ),
        lambda service, checkpoints: service.finalize_terminal(
            job_id="job:" + "0" * 24,
            worker_id="worker-one",
            now=datetime(2026, 1, 1, tzinfo=UTC),
            checkpoint_store=checkpoints,
        ),
    ],
    ids=[
        "claim-and-prepare",
        "recover-expired",
        "mark-prepared",
        "renew-lease",
        "recover-cancelled",
        "stage-terminal",
        "finalize-terminal",
    ],
)
def test_unknown_job_operations_do_not_initialize_registry(
    tmp_path: Path, invoke: Callable[[BatchJobService, SqliteBatchStore], object]
) -> None:
    service = _service(tmp_path)
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    with pytest.raises(KeyError, match="job is unavailable"):
        invoke(service, checkpoints)

    assert not (tmp_path / "jobs.sqlite3").exists()


def test_service_cancel_cannot_cross_workspace(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="owned-job")
    foreign = BatchJobService(
        database_path=tmp_path / "jobs.sqlite3",
        workspace_id="foreign-workspace",
        input_root=tmp_path / "inputs",
        input_root_alias="inputs",
        input_root_sha256="a" * 64,
        output_root_alias="outputs",
        output_root_sha256="b" * 64,
        modality=BatchModality.AUDIO,
        media_types={".wav": "audio/wav"},
        operation=plan.operation,
    )

    with pytest.raises(KeyError, match="job is unavailable"):
        foreign.cancel(job_id=queued.job_id, expected_revision=queued.revision)

    assert service.status(queued.job_id) == queued


def test_enqueue_rejects_changed_source_without_creating_registry(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    (tmp_path / "inputs" / "meeting.wav").write_bytes(b"changed recording")

    with pytest.raises(ValueError, match="plan_changed"):
        service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    assert not (tmp_path / "jobs.sqlite3").exists()


def test_same_key_retry_returns_existing_job_after_metadata_only_change(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    first = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")
    source = tmp_path / "inputs" / "meeting.wav"
    previous = source.stat()
    os.utime(source, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))

    retried = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    assert retried == first


def test_expired_lease_requires_matching_real_checkpoints(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    claimed = registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    checkpoints.prepare(plan)
    prepared = service.mark_prepared(
        job_id=queued.job_id,
        worker_id="worker-a",
        expected_revision=claimed.revision,
        now=now + timedelta(seconds=1),
        checkpoint_store=checkpoints,
    )
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        connection.execute(
            "UPDATE batch_items SET source_sha256 = ? WHERE plan_id = ?",
            ("f" * 64, plan.plan_id),
        )
    with pytest.raises(JobTransitionError, match="checkpoint"):
        service.recover_expired(
            job_id=queued.job_id,
            now=now + timedelta(seconds=31),
            checkpoint_store=checkpoints,
        )
    assert service.status(queued.job_id) == prepared

    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        connection.execute(
            "UPDATE batch_items SET source_sha256 = ? WHERE plan_id = ?",
            (plan.items[0].item.sha256, plan.plan_id),
        )
    recovered = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )
    assert recovered.state == "interrupted"
    assert recovered.revision == prepared.revision + 1
    assert service.status(queued.job_id) == recovered


def test_interrupted_job_resumes_with_same_identity_and_verified_checkpoints(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    cache_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    checkpoints = JobCheckpointStores(cache_store).for_job(queued.job_id)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    prepared, _ = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    interrupted = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )

    resumed = service.resume(
        job_id=queued.job_id,
        expected_revision=interrupted.revision,
        checkpoint_store=checkpoints,
    )

    assert prepared.state == "running"
    assert resumed.job_id == queued.job_id
    assert resumed.state == "queued"
    assert resumed.revision == interrupted.revision + 1

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    completed = JobWorker(
        service,
        cache_store,
        Operation(),
        worker_id="worker-new",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).run_next()
    assert completed is not None
    assert completed.job_id == queued.job_id
    assert completed.state == "succeeded"


def test_resume_rejects_changed_source_and_leaves_job_interrupted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    SqliteJobRegistry(tmp_path / "jobs.sqlite3").claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    interrupted = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )
    (tmp_path / "inputs" / "meeting.wav").write_bytes(b"changed recording")

    with pytest.raises(ValueError, match="source_changed"):
        service.resume(
            job_id=queued.job_id,
            expected_revision=interrupted.revision,
            checkpoint_store=checkpoints,
        )

    assert service.status(queued.job_id) == interrupted


def test_resume_rejects_tampered_checkpoint_and_leaves_job_interrupted(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    interrupted = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        connection.execute(
            "UPDATE batch_items SET source_sha256 = ? WHERE plan_id = ?",
            ("f" * 64, plan.plan_id),
        )

    with pytest.raises(JobTransitionError, match="checkpoint"):
        service.resume(
            job_id=queued.job_id,
            expected_revision=interrupted.revision,
            checkpoint_store=checkpoints,
        )

    assert service.status(queued.job_id) == interrupted


def test_resume_rejects_checkpoint_state_changed_without_history(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    cache_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    checkpoints = JobCheckpointStores(cache_store).for_job(queued.job_id)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    interrupted = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )
    with sqlite3.connect(checkpoints.database_path) as connection:
        connection.execute(
            "UPDATE batch_items SET state = ? WHERE plan_id = ?",
            (BatchState.SKIPPED.value, plan.plan_id),
        )

    with pytest.raises(JobTransitionError, match="checkpoint"):
        service.resume(
            job_id=queued.job_id,
            expected_revision=interrupted.revision,
            checkpoint_store=checkpoints,
        )

    assert service.status(queued.job_id) == interrupted


def test_resume_reports_corrupt_checkpoint_history_without_advancing_job(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    cache_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    checkpoints = JobCheckpointStores(cache_store).for_job(queued.job_id)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    interrupted = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )
    with sqlite3.connect(checkpoints.database_path) as connection:
        connection.execute(
            "UPDATE batch_events SET state = ? WHERE plan_id = ?",
            ("not-a-state", plan.plan_id),
        )

    with pytest.raises(JobTransitionError, match="checkpoint verification failed"):
        service.resume(
            job_id=queued.job_id,
            expected_revision=interrupted.revision,
            checkpoint_store=checkpoints,
        )

    assert service.status(queued.job_id) == interrupted


def test_crash_before_checkpoint_preparation_can_recover_safely(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    claimed = registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    recovered = service.recover_expired(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )

    assert claimed.state == "running"
    assert recovered.state == "interrupted"
    assert recovered.revision == claimed.revision + 1
    assert checkpoints.list(plan.plan_id) == ()


def test_preparation_marker_requires_complete_checkpoints_before_execution(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = SqliteJobRegistry(tmp_path / "jobs.sqlite3").claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    with pytest.raises(JobTransitionError, match="checkpoint"):
        service.mark_prepared(
            job_id=queued.job_id,
            worker_id="worker-a",
            expected_revision=claimed.revision,
            now=now + timedelta(seconds=1),
            checkpoint_store=checkpoints,
        )

    checkpoints.prepare(plan)
    prepared = service.mark_prepared(
        job_id=queued.job_id,
        worker_id="worker-a",
        expected_revision=claimed.revision,
        now=now + timedelta(seconds=1),
        checkpoint_store=checkpoints,
    )
    assert prepared.checkpoint_prepared is True
    assert prepared.revision == claimed.revision + 1
    assert service.status(queued.job_id) == prepared


def test_expired_cancelling_job_preserves_completed_checkpoint_and_publishes_cancelled(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    (tmp_path / "inputs" / "other.wav").write_bytes(b"second recording")
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime.now(UTC)
    prepared, _ = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    first = plan.items[0]
    checkpoints.update(
        plan.plan_id,
        first.item.item_id,
        expected_states=frozenset({BatchState.PENDING}),
        state=BatchState.SUCCEEDED,
        attempts=1,
    )
    cancelling = SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=prepared.revision,
    )

    recovered = service.recover_cancelled(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )

    assert cancelling.state == "cancelling"
    assert recovered.state == "cancelled"
    assert recovered.terminal_manifest_sha256 is not None
    assert recovered.completed_items == 2
    assert service.status(queued.job_id) == recovered
    assert sorted(checkpoint.state for checkpoint in checkpoints.list(plan.plan_id)) == [
        BatchState.SKIPPED,
        BatchState.SUCCEEDED,
    ]


def test_cancellation_recovery_rejects_active_lease_and_tampered_checkpoints(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime.now(UTC)
    prepared, _ = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    cancelling = SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=prepared.revision,
    )

    with pytest.raises(JobTransitionError, match="lease active"):
        service.recover_cancelled(
            job_id=queued.job_id,
            now=now + timedelta(seconds=1),
            checkpoint_store=checkpoints,
        )
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        connection.execute(
            "UPDATE batch_items SET source_sha256 = ? WHERE plan_id = ?",
            ("f" * 64, plan.plan_id),
        )
    with pytest.raises(JobTransitionError, match="checkpoint"):
        service.recover_cancelled(
            job_id=queued.job_id,
            now=now + timedelta(seconds=31),
            checkpoint_store=checkpoints,
        )
    assert service.status(queued.job_id) == cancelling
    assert checkpoints.list(plan.plan_id)[0].state is BatchState.PENDING


def test_cancellation_recovery_after_crash_before_checkpoint_preparation(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    now = datetime.now(UTC)
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    claimed = registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    registry.cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=claimed.revision,
    )
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    assert checkpoints.list(plan.plan_id) == ()

    recovered = service.recover_cancelled(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )

    assert recovered.state == "cancelled"
    assert recovered.terminal_manifest_sha256 is not None
    assert [checkpoint.state for checkpoint in checkpoints.list(plan.plan_id)] == [
        BatchState.SKIPPED
    ]


def test_cancellation_recovery_finalizes_matching_pending_receipt(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime.now(UTC)
    prepared, _ = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    item = plan.items[0].item
    checkpoints.update(
        plan.plan_id,
        item.item_id,
        expected_states=frozenset({BatchState.PENDING}),
        state=BatchState.SKIPPED,
        attempts=0,
    )
    cancelling = SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=prepared.revision,
    )
    service.stage_terminal(
        job_id=queued.job_id,
        worker_id="worker-lost",
        expected_revision=cancelling.revision,
        now=now + timedelta(seconds=1),
        checkpoint_store=checkpoints,
    )
    assert service.status(queued.job_id).state == "cancelling"

    recovered = service.recover_cancelled(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )

    assert recovered.state == "cancelled"
    assert recovered.terminal_manifest_sha256 is not None
    assert service.status(queued.job_id) == recovered


def test_cancellation_recovery_preserves_concurrent_completed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime.now(UTC)
    prepared, _ = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    cancelling = SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=prepared.revision,
    )
    original_update = checkpoints.update
    competed = False

    def competing_update(
        plan_id: str,
        item_id: str,
        *,
        expected_states: frozenset[BatchState],
        state: BatchState,
        attempts: int,
        failure: BatchFailure | None = None,
    ) -> BatchCheckpoint:
        nonlocal competed
        if state is BatchState.SKIPPED and not competed:
            competed = True
            original_update(
                plan_id,
                item_id,
                expected_states=frozenset({BatchState.PENDING}),
                state=BatchState.SUCCEEDED,
                attempts=1,
            )
        return original_update(
            plan_id,
            item_id,
            expected_states=expected_states,
            state=state,
            attempts=attempts,
            failure=failure,
        )

    monkeypatch.setattr(checkpoints, "update", competing_update)
    with pytest.raises(CheckpointTransitionError, match="checkpoint state"):
        service.recover_cancelled(
            job_id=queued.job_id,
            now=now + timedelta(seconds=31),
            checkpoint_store=checkpoints,
        )
    assert service.status(queued.job_id) == cancelling
    assert checkpoints.list(plan.plan_id)[0].state is BatchState.SUCCEEDED

    monkeypatch.setattr(checkpoints, "update", original_update)
    recovered = service.recover_cancelled(
        job_id=queued.job_id,
        now=now + timedelta(seconds=32),
        checkpoint_store=checkpoints,
    )
    assert recovered.state == "cancelled"
    assert checkpoints.list(plan.plan_id)[0].state is BatchState.SUCCEEDED


def test_worker_claim_prepares_durable_plan_before_execution(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=UTC)

    prepared, restored_plan = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )

    assert prepared.state == "running"
    assert prepared.checkpoint_prepared is True
    assert prepared.revision == 3
    assert restored_plan == planned
    assert tuple(checkpoint.item_id for checkpoint in checkpoints.list(planned.plan_id)) == (
        planned.items[0].item.item_id,
    )


def test_completed_batch_needs_verified_terminal_receipt_after_worker_loss(tmp_path: Path) -> None:
    service = _service(tmp_path)
    queued = service.enqueue(plan_id=service.plan().plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    prepared, plan = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )

    class TranscribeOperation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"transcript": "approved", "item_id": item.item_id}

    execution = BatchExecutor(
        checkpoints,
        checkpoints,
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).execute(plan, TranscribeOperation())
    assert execution.items[0].result is not None

    pending = service.stage_terminal(
        job_id=queued.job_id,
        worker_id="worker-a",
        expected_revision=prepared.revision,
        now=now + timedelta(seconds=1),
        checkpoint_store=checkpoints,
    )
    assert len(pending.checkpoint_sha256) == 64
    assert service.status(queued.job_id).state == "running"
    assert service.status(queued.job_id).terminal_manifest_sha256 is None

    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        connection.execute(
            "UPDATE batch_items SET cache_key = ? WHERE plan_id = ?",
            ("cache:" + "f" * 64, plan.plan_id),
        )
    with pytest.raises(ValueError, match="terminal receipt verification"):
        service.finalize_terminal(
            job_id=queued.job_id,
            now=now + timedelta(seconds=31),
            checkpoint_store=checkpoints,
        )
    assert service.status(queued.job_id).state == "running"
    with sqlite3.connect(tmp_path / "checkpoints.sqlite3") as connection:
        connection.execute(
            "UPDATE batch_items SET cache_key = ? WHERE plan_id = ?",
            (plan.items[0].cache_key, plan.plan_id),
        )

    completed = service.finalize_terminal(
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        checkpoint_store=checkpoints,
    )
    assert completed.state == "succeeded"
    assert completed.completed_items == 1
    assert completed.terminal_manifest_sha256 is not None
    assert (
        SqliteJobRegistry(tmp_path / "jobs.sqlite3").status(
            workspace_id="workspace-a", job_id=queued.job_id
        )
        == completed
    )
    assert b"approved" not in (tmp_path / "jobs.sqlite3").read_bytes()

    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute(
            "UPDATE job_terminals SET receipt_json = ? WHERE job_id = ?",
            ("{}", queued.job_id),
        )
    with pytest.raises(ValueError, match="terminal receipt"):
        SqliteJobRegistry(tmp_path / "jobs.sqlite3").status(
            workspace_id="workspace-a", job_id=queued.job_id
        )


def test_worker_source_replacement_finishes_failed_without_running_provider(tmp_path: Path) -> None:
    service = _service(tmp_path)
    queued = service.enqueue(plan_id=service.plan().plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    prepared, plan = service.claim_and_prepare(
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
        checkpoint_store=checkpoints,
    )
    (tmp_path / "inputs" / "meeting.wav").write_bytes(b"replacement recording")
    calls: list[str] = []

    class RecordingOperation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.item_id)
            return {"transcript": "must not run"}

    execution = BatchExecutor(
        checkpoints,
        checkpoints,
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).execute(plan, RecordingOperation())
    assert execution.items[0].state.value == "source_changed"
    assert calls == []

    service.stage_terminal(
        job_id=queued.job_id,
        worker_id="worker-a",
        expected_revision=prepared.revision,
        now=now + timedelta(seconds=1),
        checkpoint_store=checkpoints,
    )
    failed = service.finalize_terminal(
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now + timedelta(seconds=2),
        checkpoint_store=checkpoints,
    )
    assert failed.state == "failed"
    assert failed.error_code == "ITEM_FAILURE"
    assert failed.completed_items == 1
