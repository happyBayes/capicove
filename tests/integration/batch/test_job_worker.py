"""A local worker executes queued jobs independently of client sessions."""

import sqlite3
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import inf, nan
from multiprocessing import get_context
from pathlib import Path
from threading import Barrier, Event, Thread
from time import monotonic, sleep
from typing import Any

import pytest

import capicove.batch.job_registry as job_registry
from capicove.batch._adapters.sqlite import SqliteBatchStore
from capicove.batch.executor import BatchExecutor
from capicove.batch.job_checkpoint_store import JobCheckpointStores
from capicove.batch.job_registry import JobTransitionError, SqliteJobRegistry
from capicove.batch.job_service import BatchJobService
from capicove.batch.job_terminal import decode_terminal_receipt, encode_terminal_receipt
from capicove.batch.job_worker import JobWorker
from capicove.batch.models import BatchItem, BatchModality, BatchState, OperationFingerprint
from capicove.batch.retry import RetryPolicy
from tests.integration.batch.test_job_service import _service


def _reopen_process_service(root: Path) -> BatchJobService:
    return BatchJobService(
        database_path=root / "jobs.sqlite3",
        workspace_id="workspace-a",
        input_root=root / "inputs",
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


def _run_spawned_worker(root: Path, started: Any, release: Any, stop: Any) -> None:
    service = _reopen_process_service(root)

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return service.operation_fingerprint

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            started.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test operation was not released")
            return {"item_id": item.item_id, "observed": item.source_path.read_text()}

    JobWorker(
        service,
        SqliteBatchStore(root / "checkpoints.sqlite3"),
        Operation(),
        worker_id="worker-process",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).run_until_stopped(stop, poll_seconds=0.02)


def test_spawned_worker_finishes_after_observer_is_reopened(tmp_path: Path) -> None:
    """A fresh worker interpreter must use durable state, not a caller session."""
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="process-one")
    assert service.status(queued.job_id).state == "queued"
    assert not (tmp_path / "checkpoints-jobs").exists()

    context = get_context("spawn")
    started = context.Event()
    release = context.Event()
    stop = context.Event()
    process = context.Process(
        target=_run_spawned_worker,
        args=(tmp_path, started, release, stop),
    )
    process.start()
    try:
        assert started.wait(timeout=10)
        del service
        reopened = _reopen_process_service(tmp_path)
        assert reopened.status(queued.job_id).state == "running"
        release.set()
        deadline = monotonic() + 10
        while reopened.status(queued.job_id).state != "succeeded" and monotonic() < deadline:
            sleep(0.02)
        completed = reopened.status(queued.job_id)
    finally:
        release.set()
        stop.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert process.exitcode == 0
    assert completed.state == "succeeded"
    assert completed.completed_items == 1
    assert completed.terminal_manifest_sha256 is not None


def test_spawned_worker_observes_external_cancellation(tmp_path: Path) -> None:
    """A separately started worker must honor a durable cancellation request."""
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="process-cancel")
    context = get_context("spawn")
    started = context.Event()
    release = context.Event()
    stop = context.Event()
    process = context.Process(
        target=_run_spawned_worker,
        args=(tmp_path, started, release, stop),
    )
    process.start()
    try:
        assert started.wait(timeout=10)
        observer = _reopen_process_service(tmp_path)
        running = observer.status(queued.job_id)
        assert running.state == "running"
        requested = observer.cancel(
            job_id=queued.job_id,
            expected_revision=running.revision,
        )
        assert requested.state == "cancelling"
        release.set()
        deadline = monotonic() + 10
        while observer.status(queued.job_id).state != "cancelled" and monotonic() < deadline:
            sleep(0.02)
        completed = observer.status(queued.job_id)
    finally:
        release.set()
        stop.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert process.exitcode == 0
    assert completed.state == "cancelled"
    assert completed.job_id == queued.job_id


def test_worker_runs_approved_batch_and_publishes_verified_terminal_job(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")
    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"transcript": "local result", "item_id": item.item_id}

    worker = JobWorker(
        service,
        store,
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
        lease_seconds=30,
    )
    completed = worker.run_one(queued.job_id)

    assert completed.state == "succeeded"
    assert completed.completed_items == 1
    assert completed.terminal_manifest_sha256 is not None
    assert service.status(queued.job_id) == completed
    assert b"local result" not in (tmp_path / "jobs.sqlite3").read_bytes()


@pytest.mark.parametrize(
    "forged_state", ["failed", "running", "queued", "interrupted", "queued_without_manifest"]
)
def test_status_rejects_state_inconsistent_with_committed_receipt(
    tmp_path: Path, forged_state: str
) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    JobWorker(
        service,
        SqliteBatchStore(tmp_path / "checkpoints.sqlite3"),
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).run_one(queued.job_id)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        if forged_state == "queued_without_manifest":
            connection.execute(
                "UPDATE jobs SET state = 'queued', terminal_manifest_sha256 = NULL "
                "WHERE job_id = ?",
                (queued.job_id,),
            )
        else:
            connection.execute(
                "UPDATE jobs SET state = ? WHERE job_id = ?",
                (forged_state, queued.job_id),
            )

    with pytest.raises(ValueError, match="job terminal receipt changed"):
        service.status(queued.job_id)


@pytest.mark.parametrize("tamper", ["job_id", "plan_id", "completed_items", "total_items"])
def test_status_rejects_receipt_identity_or_count_mismatch(tmp_path: Path, tamper: str) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    JobWorker(
        service,
        SqliteBatchStore(tmp_path / "checkpoints.sqlite3"),
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).run_one(queued.job_id)
    database = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        if tamper in {"job_id", "plan_id"}:
            raw = connection.execute(
                "SELECT receipt_json FROM job_terminals WHERE job_id = ?", (queued.job_id,)
            ).fetchone()[0]
            receipt = decode_terminal_receipt(raw)
            other_receipt = (
                replace(receipt, job_id="job:" + "f" * 24)
                if tamper == "job_id"
                else replace(receipt, plan_id="plan:" + "f" * 64)
            )
            encoded, receipt_hash = encode_terminal_receipt(other_receipt)
            connection.execute(
                "UPDATE job_terminals SET receipt_json = ?, receipt_sha256 = ? WHERE job_id = ?",
                (encoded, receipt_hash, queued.job_id),
            )
            connection.execute(
                "UPDATE jobs SET terminal_manifest_sha256 = ? WHERE job_id = ?",
                (receipt_hash, queued.job_id),
            )
        else:
            connection.execute(
                f"UPDATE jobs SET {tamper} = ? WHERE job_id = ?",
                (0 if tamper == "completed_items" else 2, queued.job_id),
            )

    with pytest.raises(ValueError, match="job terminal receipt changed"):
        service.status(queued.job_id)


def test_status_reads_job_and_terminal_from_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    JobWorker(
        service,
        SqliteBatchStore(tmp_path / "checkpoints.sqlite3"),
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).run_one(queued.job_id)
    database = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        receipt_hash = connection.execute(
            "SELECT receipt_sha256 FROM job_terminals WHERE job_id = ?", (queued.job_id,)
        ).fetchone()[0]
        connection.execute(
            "UPDATE jobs SET state = 'running', worker_id = 'worker-a', "
            "lease_until = ?, terminal_manifest_sha256 = NULL WHERE job_id = ?",
            (datetime(2027, 1, 1, tzinfo=UTC).isoformat(), queued.job_id),
        )
        connection.execute(
            "UPDATE job_terminals SET committed_at = NULL WHERE job_id = ?",
            (queued.job_id,),
        )

    original = job_registry._job_record
    switched: list[bool] = []

    def commit_after_job_decode(row: sqlite3.Row) -> job_registry.JobRecord:
        record = original(row)
        switched.append(True)
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE jobs SET state = 'succeeded', worker_id = NULL, "
                "lease_until = NULL, terminal_manifest_sha256 = ? WHERE job_id = ?",
                (receipt_hash, queued.job_id),
            )
            connection.execute(
                "UPDATE job_terminals SET committed_at = ? WHERE job_id = ?",
                (datetime(2026, 1, 1, tzinfo=UTC).isoformat(), queued.job_id),
            )
        return record

    monkeypatch.setattr(job_registry, "_job_record", commit_after_job_decode)
    observed = service.status(queued.job_id)
    monkeypatch.setattr(job_registry, "_job_record", original)

    assert switched == [True]
    assert observed.state == "running"
    assert service.status(queued.job_id).state == "succeeded"


def test_worker_rejects_unknown_job_before_creating_checkpoint_store(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    service.enqueue(plan_id=planned.plan_id, idempotency_key="known")
    cache_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        cache_store,
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )

    with pytest.raises(KeyError, match="job is unavailable"):
        worker.run_one("job:" + "f" * 24)

    assert not (tmp_path / "checkpoints-jobs").exists()


def test_worker_runs_oldest_queued_job_and_leaves_next_job_queued(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    first = service.enqueue(plan_id=planned.plan_id, idempotency_key="first")
    second = service.enqueue(plan_id=planned.plan_id, idempotency_key="second")
    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        store,
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )

    assert service.status(first.job_id).state == "queued"
    assert service.status(second.job_id).state == "queued"
    first_result = worker.run_next()
    assert first_result is not None
    assert first_result.job_id == first.job_id
    assert service.status(first.job_id).state == "succeeded"
    assert service.status(second.job_id).state == "queued"
    second_result = worker.run_next()
    assert second_result is not None
    assert second_result.job_id == second.job_id
    assert service.status(second.job_id).state == "succeeded"
    assert worker.run_next() is None


def test_worker_loop_waits_for_job_and_stops_after_signal(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    stop = Event()
    outcomes: list[BaseException] = []

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        store,
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )
    run_until_stopped = worker.run_until_stopped

    def run_loop() -> None:
        try:
            run_until_stopped(stop, poll_seconds=0.02)
        except BaseException as error:
            outcomes.append(error)

    thread = Thread(target=run_loop)
    thread.start()
    try:
        queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="later")
        deadline = monotonic() + 5
        while service.status(queued.job_id).state != "succeeded" and monotonic() < deadline:
            sleep(0.02)
        assert service.status(queued.job_id).state == "succeeded"
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcomes == []


def test_worker_recover_next_marks_expired_prepared_job_interrupted(tmp_path: Path) -> None:
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
    calls: list[str] = []

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.item_id)
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        cache_store,
        Operation(),
        worker_id="worker-new",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )

    recovered = worker.recover_next(now=now + timedelta(seconds=31))

    assert recovered is not None
    assert recovered.job_id == queued.job_id
    assert recovered.state == "interrupted"
    assert service.status(queued.job_id) == recovered
    assert calls == []
    assert worker.recover_next(now=now + timedelta(seconds=32)) is None


def test_worker_recovery_does_not_recreate_missing_prepared_checkpoints(tmp_path: Path) -> None:
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
    checkpoint_file = tmp_path / "checkpoints-jobs" / f"{queued.job_id[4:]}.sqlite3"
    checkpoint_file.unlink()

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        cache_store,
        Operation(),
        worker_id="worker-new",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )

    with pytest.raises(JobTransitionError, match="checkpoint"):
        worker.recover_next(now=now + timedelta(seconds=31))

    assert not checkpoint_file.exists()
    assert service.status(queued.job_id).state == "running"


def test_worker_recover_next_finalizes_pending_verified_receipt(tmp_path: Path) -> None:
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
    calls: list[str] = []

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.item_id)
            return {"item_id": item.item_id}

    BatchExecutor(
        checkpoints,
        cache_store,
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).execute(plan, Operation())
    service.stage_terminal(
        job_id=queued.job_id,
        worker_id="worker-lost",
        expected_revision=prepared.revision,
        now=now + timedelta(seconds=1),
        checkpoint_store=checkpoints,
    )
    assert service.status(queued.job_id).state == "running"
    assert len(calls) == 1
    worker = JobWorker(
        service,
        cache_store,
        Operation(),
        worker_id="worker-new",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )

    recovered = worker.recover_next(now=now + timedelta(seconds=31))

    assert recovered is not None
    assert recovered.state == "succeeded"
    assert service.status(queued.job_id) == recovered
    assert len(calls) == 1


def test_worker_loop_recovers_expired_job_without_rerunning_provider(tmp_path: Path) -> None:
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
    calls: list[str] = []

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.item_id)
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        cache_store,
        Operation(),
        worker_id="worker-new",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )
    stop = Event()
    errors: list[Exception] = []

    def run() -> None:
        try:
            worker.run_until_stopped(stop, poll_seconds=0.02)
        except Exception as error:
            errors.append(error)

    thread = Thread(target=run)
    thread.start()
    try:
        deadline = monotonic() + 2
        while service.status(queued.job_id).state == "running" and monotonic() < deadline:
            sleep(0.02)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert service.status(queued.job_id).state == "interrupted"
    assert calls == []


def test_competing_workers_do_not_fail_when_one_recovers_expired_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    recovery_time = now + timedelta(seconds=31)
    selected = Barrier(2)
    real_next_expired = service.next_expired
    outcomes: list[object] = []

    def gated_next_expired(*, now: datetime) -> object:
        record = real_next_expired(now=now)
        selected.wait(timeout=5)
        return record

    monkeypatch.setattr(service, "next_expired", gated_next_expired)

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    def recover(worker_id: str) -> None:
        worker = JobWorker(
            service,
            cache_store,
            Operation(),
            worker_id=worker_id,
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
        try:
            outcomes.append(worker.recover_next(now=recovery_time))
        except Exception as error:
            outcomes.append(error)

    threads = [
        Thread(target=recover, args=("worker-a",)),
        Thread(target=recover, args=("worker-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(outcomes) == 2
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(getattr(outcome, "state", None) == "interrupted" for outcome in outcomes) == 1


@pytest.mark.parametrize("poll_seconds", [0, -1, nan, inf, True, "1", None])
def test_worker_loop_rejects_unbounded_polling_interval(
    tmp_path: Path, poll_seconds: float
) -> None:
    service = _service(tmp_path)
    planned = service.plan()

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        SqliteBatchStore(tmp_path / "checkpoints.sqlite3"),
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )
    stop = Event()
    stop.set()

    with pytest.raises(ValueError, match="poll_seconds"):
        worker.run_until_stopped(stop, poll_seconds=poll_seconds)


@pytest.mark.parametrize("lease_seconds", [True, "1", None])
def test_worker_rejects_invalid_lease_interval(tmp_path: Path, lease_seconds: float) -> None:
    service = _service(tmp_path)
    planned = service.plan()

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    with pytest.raises(ValueError, match="lease_seconds"):
        JobWorker(
            service,
            SqliteBatchStore(tmp_path / "checkpoints.sqlite3"),
            Operation(),
            worker_id="worker-a",
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
            lease_seconds=lease_seconds,
        )


@pytest.mark.parametrize("max_workers", [0, True, 65, "1", None])
def test_worker_rejects_invalid_concurrency_before_claim(tmp_path: Path, max_workers: int) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    with pytest.raises(ValueError, match="max_workers"):
        JobWorker(
            service,
            store,
            Operation(),
            worker_id="worker-a",
            max_workers=max_workers,
            retry_policy=RetryPolicy(max_retries=0),
        )
    assert service.status(queued.job_id).state == "queued"
    assert not (tmp_path / "checkpoints-jobs").exists()


@pytest.mark.parametrize("worker_id", ["", "../other", "1worker", "w" * 129])
def test_worker_rejects_invalid_identity_before_claim(tmp_path: Path, worker_id: str) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    with pytest.raises(ValueError, match="worker_id"):
        JobWorker(
            service,
            store,
            Operation(),
            worker_id=worker_id,
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
    assert service.status(queued.job_id).state == "queued"
    assert not (tmp_path / "checkpoints-jobs").exists()


def test_worker_rejects_wrong_provider_before_claim(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")
    wrong = replace(planned.operation, provider_fingerprint="f" * 64)

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return wrong

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    with pytest.raises(ValueError, match="operation fingerprint"):
        JobWorker(
            service,
            store,
            Operation(),
            worker_id="worker-a",
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
    assert service.status(queued.job_id).state == "queued"
    assert not (tmp_path / "checkpoints-jobs").exists()


def test_worker_rechecks_provider_identity_before_claim(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")

    class Operation:
        fingerprint = planned.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    operation = Operation()
    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    worker = JobWorker(
        service,
        store,
        operation,
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )
    operation.fingerprint = replace(planned.operation, provider_fingerprint="f" * 64)

    with pytest.raises(ValueError, match="operation fingerprint"):
        worker.run_one(queued.job_id)

    assert service.status(queued.job_id).state == "queued"
    assert not (tmp_path / "checkpoints-jobs").exists()


def test_competing_workers_do_not_fail_when_one_claims_selected_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    selection_barrier = Barrier(2)
    real_next_queued = service.next_queued
    outcomes: list[object] = []

    def gated_next_queued() -> object:
        selected = real_next_queued()
        selection_barrier.wait(timeout=5)
        return selected

    monkeypatch.setattr(service, "next_queued", gated_next_queued)

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            return {"item_id": item.item_id}

    def run(worker_id: str) -> None:
        worker = JobWorker(
            service,
            checkpoints,
            Operation(),
            worker_id=worker_id,
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
        try:
            outcomes.append(worker.run_next())
        except Exception as error:
            outcomes.append(error)

    workers = [Thread(target=run, args=("worker-a",)), Thread(target=run, args=("worker-b",))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert sum(outcome is None for outcome in outcomes) == 1
    assert (
        sum(
            getattr(outcome, "job_id", None) == queued.job_id
            and getattr(outcome, "state", None) == "succeeded"
            for outcome in outcomes
        )
        == 1
    )


def test_worker_renews_lease_while_operation_is_running(tmp_path: Path) -> None:
    service = _service(tmp_path)
    planned = service.plan()
    queued = service.enqueue(plan_id=planned.plan_id, idempotency_key="request-one")
    store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    started = Event()
    release = Event()
    outcomes: list[str] = []

    class SlowOperation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return planned.operation

        def execute(self, _item: BatchItem, /) -> Mapping[str, object]:
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test operation was not released")
            return {"transcript": "complete"}

    worker = JobWorker(
        service,
        store,
        SlowOperation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
        lease_seconds=1.5,
    )
    thread = Thread(target=lambda: outcomes.append(worker.run_one(queued.job_id).state))
    thread.start()
    try:
        assert started.wait(timeout=3)
        deadline = monotonic() + 3
        while service.status(queued.job_id).revision < 4 and monotonic() < deadline:
            sleep(0.05)
        observed = service.status(queued.job_id)
        assert observed.state == "running"
        assert observed.revision >= 4
        assert observed.lease_until is not None
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcomes == ["succeeded"]


def test_worker_stops_new_provider_calls_after_heartbeat_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _service(tmp_path)
    (tmp_path / "inputs" / "z-other.wav").write_bytes(b"second recording")
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    started = Event()
    release = Event()
    heartbeat_stopped = Event()
    calls: list[str] = []
    outcomes: list[object] = []

    class SlowOperation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.relative_path.as_posix())
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test operation was not released")
            return {"item_id": item.item_id}

    def lose_heartbeat(**_kwargs: object) -> None:
        raise JobTransitionError("simulated lost lease")

    monkeypatch.setattr(service, "renew_lease", lose_heartbeat)
    monkeypatch.setattr("capicove.batch.job_worker.Event", lambda: heartbeat_stopped)
    checkpoint_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    worker = JobWorker(
        service,
        checkpoint_store,
        SlowOperation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
        lease_seconds=0.3,
    )

    def run() -> None:
        try:
            outcomes.append(worker.run_one(queued.job_id))
        except Exception as error:
            outcomes.append(error)

    thread = Thread(target=run)
    thread.start()
    try:
        assert started.wait(timeout=3)
        assert heartbeat_stopped.wait(timeout=3)
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert calls == ["meeting.wav"]
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], JobTransitionError)
    assert service.status(queued.job_id).state == "running"
    owned = JobCheckpointStores(checkpoint_store).for_job(queued.job_id)
    states = {
        planned.item.relative_path.as_posix(): owned.load(plan.plan_id, planned.item.item_id).state
        for planned in plan.items
    }
    assert states == {"meeting.wav": BatchState.SUCCEEDED, "z-other.wav": BatchState.PENDING}
    recovered = worker.recover_next(now=datetime.now(UTC) + timedelta(seconds=1))
    assert recovered is not None and recovered.state == "interrupted"
    resumed = service.resume(
        job_id=queued.job_id,
        expected_revision=recovered.revision,
        checkpoint_store=owned,
    )
    assert resumed.state == "queued"
    monkeypatch.undo()
    completed = JobWorker(
        service,
        checkpoint_store,
        SlowOperation(),
        worker_id="worker-b",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    ).run_one(queued.job_id)
    assert completed.state == "succeeded"
    assert calls == ["meeting.wav", "z-other.wav"]


def test_running_cancellation_preserves_completed_item_and_skips_remaining(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    (tmp_path / "inputs" / "other.wav").write_bytes(b"second recording")
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    started = Event()
    release = Event()
    calls: list[str] = []
    outcomes: list[object] = []

    class SlowOperation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.relative_path.as_posix())
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test operation was not released")
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        checkpoints,
        SlowOperation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
        lease_seconds=3,
    )

    def run() -> None:
        try:
            outcomes.append(worker.run_one(queued.job_id))
        except Exception as error:
            outcomes.append(error)

    thread = Thread(target=run)
    thread.start()
    try:
        assert started.wait(timeout=3)
        running = service.status(queued.job_id)
        cancelling = SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            expected_revision=running.revision,
        )
        assert cancelling.state == "cancelling"
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(outcomes) == 1
    completed = outcomes[0]
    assert getattr(completed, "state", None) == "cancelled"
    assert service.status(queued.job_id) == completed
    assert calls == ["meeting.wav"]
    owned = JobCheckpointStores(checkpoints).for_job(queued.job_id)
    assert sorted(checkpoint.state for checkpoint in owned.list(plan.plan_id)) == [
        BatchState.SKIPPED,
        BatchState.SUCCEEDED,
    ]


def test_distinct_jobs_with_same_plan_do_not_share_cancelled_checkpoints(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    (tmp_path / "inputs" / "other.wav").write_bytes(b"second recording")
    plan = service.plan()
    first = service.enqueue(plan_id=plan.plan_id, idempotency_key="first")
    second = service.enqueue(plan_id=plan.plan_id, idempotency_key="second")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    started = Event()
    release = Event()
    calls: list[str] = []
    outcomes: list[object] = []

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.relative_path.as_posix())
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test operation was not released")
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        checkpoints,
        Operation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )
    thread = Thread(target=lambda: outcomes.append(worker.run_one(first.job_id)))
    thread.start()
    try:
        assert started.wait(timeout=3)
        running = service.status(first.job_id)
        SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
            workspace_id="workspace-a",
            job_id=first.job_id,
            expected_revision=running.revision,
        )
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(outcomes) == 1
    assert getattr(outcomes[0], "state", None) == "cancelled"
    assert calls == ["meeting.wav"]

    restarted_cache = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    restarted_worker = JobWorker(
        service,
        restarted_cache,
        Operation(),
        worker_id="worker-restarted",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
    )
    second_result = restarted_worker.run_one(second.job_id)

    assert second_result.state == "succeeded"
    assert "other.wav" in calls
    first_states = [
        checkpoint.state
        for checkpoint in JobCheckpointStores(restarted_cache)
        .for_job(first.job_id)
        .list(plan.plan_id)
    ]
    second_states = [
        checkpoint.state
        for checkpoint in JobCheckpointStores(restarted_cache)
        .for_job(second.job_id)
        .list(plan.plan_id)
    ]
    assert BatchState.SKIPPED in first_states
    assert second_states == [BatchState.SUCCEEDED, BatchState.SUCCEEDED]


def test_concurrent_jobs_with_same_plan_keep_independent_checkpoints(tmp_path: Path) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    first = service.enqueue(plan_id=plan.plan_id, idempotency_key="first")
    second = service.enqueue(plan_id=plan.plan_id, idempotency_key="second")
    cache_store = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    executing = Barrier(2)
    outcomes: list[object] = []
    calls: list[str] = []

    class Operation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            calls.append(item.item_id)
            executing.wait(timeout=5)
            return {"item_id": item.item_id}

    def run(job_id: str, worker_id: str) -> None:
        worker = JobWorker(
            service,
            cache_store,
            Operation(),
            worker_id=worker_id,
            max_workers=1,
            retry_policy=RetryPolicy(max_retries=0),
        )
        try:
            outcomes.append(worker.run_one(job_id))
        except Exception as error:
            outcomes.append(error)

    threads = [
        Thread(target=run, args=(first.job_id, "worker-a")),
        Thread(target=run, args=(second.job_id, "worker-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(outcomes) == 2
    assert {getattr(outcome, "state", None) for outcome in outcomes} == {"succeeded"}
    assert len(calls) == 2
    owned = JobCheckpointStores(cache_store)
    assert owned.for_job(first.job_id).list(plan.plan_id)[0].state is BatchState.SUCCEEDED
    assert owned.for_job(second.job_id).list(plan.plan_id)[0].state is BatchState.SUCCEEDED


def test_worker_renews_lease_after_running_cancellation_revision_change(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    plan = service.plan()
    queued = service.enqueue(plan_id=plan.plan_id, idempotency_key="request-one")
    checkpoints = SqliteBatchStore(tmp_path / "checkpoints.sqlite3")
    started = Event()
    release = Event()
    outcomes: list[object] = []

    class SlowOperation:
        @property
        def fingerprint(self) -> OperationFingerprint:
            return plan.operation

        def execute(self, item: BatchItem, /) -> Mapping[str, object]:
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test operation was not released")
            return {"item_id": item.item_id}

    worker = JobWorker(
        service,
        checkpoints,
        SlowOperation(),
        worker_id="worker-a",
        max_workers=1,
        retry_policy=RetryPolicy(max_retries=0),
        lease_seconds=1.5,
    )

    def run() -> None:
        try:
            outcomes.append(worker.run_one(queued.job_id))
        except Exception as error:
            outcomes.append(error)

    thread = Thread(target=run)
    thread.start()
    try:
        assert started.wait(timeout=3)
        running = service.status(queued.job_id)
        cancelling = SqliteJobRegistry(tmp_path / "jobs.sqlite3").cancel(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            expected_revision=running.revision,
        )
        deadline = monotonic() + 3
        while service.status(queued.job_id).revision == cancelling.revision:
            if monotonic() >= deadline:
                break
            sleep(0.02)
        observed = service.status(queued.job_id)
        assert observed.state == "cancelling"
        assert observed.revision > cancelling.revision
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(outcomes) == 1
    assert getattr(outcomes[0], "state", None) == "cancelled"
