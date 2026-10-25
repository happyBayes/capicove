"""Durable operator jobs remain visible independently of MCP sessions."""

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Literal

import pytest

from capicove.batch.discovery import DirectoryDiscoverer
from capicove.batch.job_registry import (
    JobConflictError,
    JobRecord,
    JobTransitionError,
    SqliteJobRegistry,
)
from capicove.batch.models import BatchModality, BatchPlan, OperationFingerprint
from capicove.batch.planner import BatchPlanner


def _plan(root: Path, *, item_count: int = 1, parameters_sha256: str = "a" * 64) -> BatchPlan:
    root.mkdir(exist_ok=True)
    for index in range(item_count):
        (root / f"meeting-{index}.wav").write_bytes(f"private-source-{index}".encode())
    items = DirectoryDiscoverer({".wav": "audio/wav"}).discover(root, BatchModality.AUDIO)
    operation = OperationFingerprint(
        operation_id="capicove.audio.transcribe",
        operation_version="1",
        modalities=frozenset({BatchModality.AUDIO}),
        parameters_sha256=parameters_sha256,
        provider_fingerprint="b" * 64,
    )
    return BatchPlanner().plan(items, operation)


def _enqueue(
    registry: SqliteJobRegistry,
    root: Path,
    *,
    workspace_id: str = "workspace-a",
    plan: BatchPlan | None = None,
    idempotency_key: str = "one-request",
) -> JobRecord:
    selected = plan if plan is not None else _plan(root)
    return registry.enqueue_plan(
        workspace_id=workspace_id,
        plan=selected,
        input_root=root,
        input_root_alias="inputs",
        input_root_sha256="b" * 64,
        output_root_alias="outputs",
        output_root_sha256="c" * 64,
        idempotency_key=idempotency_key,
    )


def test_enqueue_without_worker_survives_reopen_and_same_key_retry(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    root = tmp_path / "inputs"
    plan = _plan(root)
    first = _enqueue(SqliteJobRegistry(database), root, plan=plan)

    reopened = SqliteJobRegistry(database)
    observed = reopened.status(workspace_id="workspace-a", job_id=first.job_id)
    retried = _enqueue(reopened, root, plan=plan)

    assert first.job_id.startswith("job:")
    assert len(first.job_id) == 28
    assert observed == first == retried
    assert observed.state == "queued"
    assert observed.revision == 1
    assert observed.worker_id is None


@pytest.mark.parametrize("forged_state", ["succeeded", "failed"])
def test_status_rejects_terminal_job_without_receipt(tmp_path: Path, forged_state: str) -> None:
    database = tmp_path / "jobs.sqlite3"
    registry = SqliteJobRegistry(database)
    created = _enqueue(registry, tmp_path / "inputs")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE jobs SET state = ? WHERE job_id = ?",
            (forged_state, created.job_id),
        )

    with pytest.raises(ValueError, match="job terminal receipt changed"):
        SqliteJobRegistry(database).status(workspace_id="workspace-a", job_id=created.job_id)


def test_new_enqueue_rejects_source_changed_after_plan_without_a_job(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    registry = SqliteJobRegistry(database)
    root = tmp_path / "inputs"
    planned = _plan(root)
    (root / "meeting-0.wav").write_bytes(b"changed after planning")

    with pytest.raises(ValueError, match="source_changed"):
        _enqueue(registry, root, plan=planned)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM job_plans").fetchone()[0] == 0


def test_enqueue_rejects_replaced_parent_before_foreign_source_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "jobs.sqlite3"
    registry = SqliteJobRegistry(database)
    root = tmp_path / "inputs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "meeting.wav").write_bytes(b"authorized recording")
    planned = _plan(root, item_count=0)
    nested.rename(root / "original-nested")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "meeting.wav").write_bytes(b"private foreign recording")
    try:
        nested.symlink_to(foreign, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this OS")

    opened_foreign: list[Path] = []
    original_open = Path.open

    def guarded_open(path: Path, mode: Literal["rb"] = "rb") -> BinaryIO:
        if path.resolve().is_relative_to(foreign):
            opened_foreign.append(path)
            raise AssertionError("foreign source was opened")
        return original_open(path, mode)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="source_changed"):
        _enqueue(registry, root, plan=planned)
    assert opened_foreign == []


def test_same_key_cannot_rebind_plan_or_cross_workspace(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    root = tmp_path / "inputs"
    created = _enqueue(registry, root)

    with pytest.raises(JobConflictError, match="idempotency_conflict"):
        _enqueue(registry, root, plan=_plan(root, parameters_sha256="d" * 64))
    with pytest.raises(KeyError, match="job is unavailable"):
        registry.status(workspace_id="workspace-b", job_id=created.job_id)
    assert registry.status(workspace_id="workspace-a", job_id=created.job_id) == created


def test_next_queued_is_oldest_workspace_job_and_skips_cancelled(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    root = tmp_path / "inputs"
    planned = _plan(root)
    first = _enqueue(registry, root, plan=planned, idempotency_key="first")
    second = _enqueue(registry, root, plan=planned, idempotency_key="second")

    assert registry.next_queued(workspace_id="workspace-a") == first
    assert registry.next_queued(workspace_id="workspace-b") is None

    registry.cancel(workspace_id="workspace-a", job_id=first.job_id, expected_revision=1)
    assert registry.next_queued(workspace_id="workspace-a") == second


def test_queued_jobs_lists_claim_order_within_one_workspace(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    root = tmp_path / "inputs"
    planned = _plan(root)
    first = _enqueue(registry, root, plan=planned, idempotency_key="first")
    second = _enqueue(registry, root, plan=planned, idempotency_key="second")
    _enqueue(registry, root, workspace_id="workspace-b", plan=planned, idempotency_key="other")

    assert registry.queued_jobs(workspace_id="workspace-a", limit=10) == (first, second)
    assert registry.queued_jobs(workspace_id="workspace-a", limit=1) == (first,)
    registry.cancel(workspace_id="workspace-a", job_id=first.job_id, expected_revision=1)
    assert registry.queued_jobs(workspace_id="workspace-a", limit=10) == (second,)
    with pytest.raises(ValueError, match="limit"):
        registry.queued_jobs(workspace_id="workspace-a", limit=0)


def test_next_expired_selects_only_workspace_scoped_elapsed_leases(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    root = tmp_path / "inputs"
    plan = _plan(root)
    first = _enqueue(registry, root, plan=plan, idempotency_key="first")
    second = _enqueue(registry, root, plan=plan, idempotency_key="second")
    foreign = _enqueue(
        registry,
        root,
        workspace_id="workspace-b",
        plan=plan,
        idempotency_key="foreign",
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first_running = registry.claim(
        workspace_id="workspace-a",
        job_id=first.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=10),
    )
    registry.claim(
        workspace_id="workspace-a",
        job_id=second.job_id,
        worker_id="worker-b",
        now=now,
        lease_until=now + timedelta(seconds=20),
    )
    registry.claim(
        workspace_id="workspace-b",
        job_id=foreign.job_id,
        worker_id="worker-foreign",
        now=now,
        lease_until=now + timedelta(seconds=5),
    )

    assert registry.next_expired(workspace_id="workspace-a", now=now + timedelta(seconds=9)) is None
    assert (
        registry.next_expired(workspace_id="workspace-a", now=now + timedelta(seconds=15))
        == first_running
    )
    foreign_expired = registry.next_expired(
        workspace_id="workspace-b", now=now + timedelta(seconds=15)
    )
    assert foreign_expired is not None
    registry.recover_expired(
        workspace_id="workspace-a",
        job_id=first.job_id,
        now=now + timedelta(seconds=15),
        verify_checkpoint=lambda _record: True,
    )
    remaining = registry.next_expired(workspace_id="workspace-a", now=now + timedelta(seconds=15))
    assert remaining is None
    second_expired = registry.next_expired(
        workspace_id="workspace-a", now=now + timedelta(seconds=25)
    )
    assert second_expired is not None
    assert second_expired.job_id == second.job_id


def test_registry_rejects_symlinked_parent_before_file_creation(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this OS")

    with pytest.raises(ValueError, match=r"symlink|reparse"):
        SqliteJobRegistry(alias / "jobs.sqlite3")
    assert not (actual / "jobs.sqlite3").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits are not Windows ACLs")
def test_registry_requires_private_parent_before_file_creation(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(ValueError, match="private"):
        SqliteJobRegistry(shared / "jobs.sqlite3")
    assert not (shared / "jobs.sqlite3").exists()


def test_queued_cancel_is_durable_revision_checked_and_terminal(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    created = _enqueue(registry, tmp_path / "inputs")

    cancelled = registry.cancel(
        workspace_id="workspace-a", job_id=created.job_id, expected_revision=1
    )
    assert cancelled.state == "cancelled"
    assert cancelled.revision == 2
    assert (
        SqliteJobRegistry(tmp_path / "jobs.sqlite3").status(
            workspace_id="workspace-a", job_id=created.job_id
        )
        == cancelled
    )
    with pytest.raises(JobTransitionError, match="revision_conflict"):
        registry.cancel(workspace_id="workspace-a", job_id=created.job_id, expected_revision=1)
    with pytest.raises(JobTransitionError, match="terminal"):
        registry.cancel(workspace_id="workspace-a", job_id=created.job_id, expected_revision=2)


def test_resume_requires_interrupted_state_and_current_revision(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    queued = _enqueue(registry, tmp_path / "inputs")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-lost",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    interrupted = registry.recover_expired(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        verify_checkpoint=lambda _record: True,
    )

    with pytest.raises(JobTransitionError, match="revision_conflict"):
        registry.resume(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            expected_revision=queued.revision,
            now=now + timedelta(seconds=32),
            verify_resume=lambda _record: None,
        )
    resumed = registry.resume(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=interrupted.revision,
        now=now + timedelta(seconds=32),
        verify_resume=lambda _record: None,
    )
    assert resumed.state == "queued"
    assert resumed.job_id == queued.job_id
    assert resumed.revision == interrupted.revision + 1
    assert resumed.checkpoint_prepared is False
    with pytest.raises(JobTransitionError, match="interrupted"):
        registry.resume(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            expected_revision=resumed.revision,
            now=now + timedelta(seconds=33),
            verify_resume=lambda _record: None,
        )


def test_running_cancellation_retry_is_idempotent_at_current_revision(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    queued = _enqueue(registry, tmp_path / "inputs")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    running = registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )
    cancelling = registry.cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=running.revision,
    )
    assert cancelling.updated_at > running.updated_at

    retried = registry.cancel(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        expected_revision=cancelling.revision,
    )

    assert retried == cancelling
    with pytest.raises(JobTransitionError, match="revision_conflict"):
        registry.cancel(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            expected_revision=running.revision,
        )


def test_registry_preserves_host_owned_utf8_workspace_identity(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    created = _enqueue(registry, tmp_path / "inputs", workspace_id="研究空间")

    assert registry.status(workspace_id="研究空间", job_id=created.job_id) == created
    with pytest.raises(KeyError, match="job is unavailable"):
        registry.status(workspace_id="workspace-a", job_id=created.job_id)


def test_claim_queued_job_is_durable_and_exclusive(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    first = SqliteJobRegistry(database)
    queued = _enqueue(first, tmp_path / "inputs")
    now = datetime.now(UTC)
    lease_until = now + timedelta(seconds=30)

    claimed = first.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=lease_until,
    )
    assert claimed.state == "running"
    assert claimed.revision == 2
    assert claimed.worker_id == "worker-a"
    assert claimed.lease_until == lease_until
    assert claimed.updated_at == now
    second = SqliteJobRegistry(database)
    with pytest.raises(JobTransitionError, match="not queued"):
        second.claim(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            worker_id="worker-b",
            now=now,
            lease_until=lease_until,
        )
    assert second.status(workspace_id="workspace-a", job_id=queued.job_id) == claimed


def test_expired_lease_requires_checkpoint_verification_before_release(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    queued = _enqueue(registry, tmp_path / "inputs")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )

    with pytest.raises(JobTransitionError, match="lease active"):
        registry.recover_expired(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            now=now + timedelta(seconds=29),
            verify_checkpoint=lambda _: True,
        )
    with pytest.raises(JobTransitionError, match="checkpoint"):
        registry.recover_expired(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            now=now + timedelta(seconds=31),
            verify_checkpoint=lambda _: False,
        )
    assert registry.status(workspace_id="workspace-a", job_id=queued.job_id) == claimed

    interrupted = registry.recover_expired(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        now=now + timedelta(seconds=31),
        verify_checkpoint=lambda record: record.plan_id == queued.plan_id,
    )
    assert interrupted.state == "interrupted"
    assert interrupted.revision == 3
    assert interrupted.worker_id is None
    assert interrupted.lease_until is None
    assert (
        SqliteJobRegistry(tmp_path / "jobs.sqlite3").status(
            workspace_id="workspace-a", job_id=queued.job_id
        )
        == interrupted
    )


def test_enqueue_persists_bounded_job_audit_without_item_payload(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    registry = SqliteJobRegistry(database)
    root = tmp_path / "inputs"
    plan = _plan(root, item_count=3)
    before = datetime.now(UTC)
    created = _enqueue(registry, root, plan=plan)
    after = datetime.now(UTC)

    assert before <= created.created_at <= after
    assert created.updated_at == created.created_at
    assert created.total_items == 3
    assert created.completed_items == 0
    assert created.terminal_manifest_sha256 is None
    assert created.error_code is None
    assert (
        SqliteJobRegistry(database).status(workspace_id="workspace-a", job_id=created.job_id)
        == created
    )


def test_reopen_rejects_incomplete_version_one_schema(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 1")
    database.chmod(0o600)

    with pytest.raises(ValueError, match="incompatible job database schema"):
        SqliteJobRegistry(database)
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(jobs)")]
        added_plan_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'job_plans'"
        ).fetchone()
    assert columns == ["job_id"]
    assert added_plan_table is None


def test_only_current_worker_can_renew_unexpired_lease(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    queued = _enqueue(registry, tmp_path / "inputs")
    now = datetime.now(UTC)
    claimed = registry.claim(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        now=now,
        lease_until=now + timedelta(seconds=30),
    )

    with pytest.raises(JobTransitionError, match="worker mismatch"):
        registry.renew_lease(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            worker_id="worker-b",
            expected_revision=claimed.revision,
            now=now + timedelta(seconds=10),
            lease_until=now + timedelta(seconds=60),
        )
    renewed = registry.renew_lease(
        workspace_id="workspace-a",
        job_id=queued.job_id,
        worker_id="worker-a",
        expected_revision=claimed.revision,
        now=now + timedelta(seconds=10),
        lease_until=now + timedelta(seconds=60),
    )
    assert renewed.revision == 3
    assert renewed.state == "running"
    assert renewed.lease_until == now + timedelta(seconds=60)
    assert renewed.updated_at == now + timedelta(seconds=10)
    with pytest.raises(JobTransitionError, match="revision_conflict"):
        registry.renew_lease(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            worker_id="worker-a",
            expected_revision=claimed.revision,
            now=now + timedelta(seconds=11),
            lease_until=now + timedelta(seconds=70),
        )
    with pytest.raises(JobTransitionError, match="lease active"):
        registry.recover_expired(
            workspace_id="workspace-a",
            job_id=queued.job_id,
            now=now + timedelta(seconds=31),
            verify_checkpoint=lambda _: True,
        )


def test_enqueue_plan_persists_reconstructable_receipt_without_source_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inputs"
    root.mkdir()
    source = root / "meeting.wav"
    source.write_bytes(b"synthetic-private-source")
    items = DirectoryDiscoverer({".wav": "audio/wav"}).discover(root, BatchModality.AUDIO)
    operation = OperationFingerprint(
        operation_id="capicove.audio.transcribe",
        operation_version="1",
        modalities=frozenset({BatchModality.AUDIO}),
        parameters_sha256="a" * 64,
        provider_fingerprint="b" * 64,
    )
    plan = BatchPlanner().plan(items, operation)
    database = tmp_path / "jobs.sqlite3"
    registry = SqliteJobRegistry(database)

    job = registry.enqueue_plan(
        workspace_id="workspace-a",
        plan=plan,
        input_root=root,
        input_root_alias="inputs",
        input_root_sha256="c" * 64,
        output_root_alias="outputs",
        output_root_sha256="d" * 64,
        idempotency_key="one-request",
    )
    reopened = SqliteJobRegistry(database)
    restored = reopened.load_plan(
        workspace_id="workspace-a",
        plan_id=job.plan_id,
        input_root=root,
        input_root_alias="inputs",
        input_root_sha256="c" * 64,
    )

    assert restored == plan
    assert reopened.status(workspace_id="workspace-a", job_id=job.job_id) == job
    database_bytes = database.read_bytes()
    assert str(root).encode() not in database_bytes
    assert b"synthetic-private-source" not in database_bytes


def test_enqueue_rejects_unregistered_plan_id(tmp_path: Path) -> None:
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    with pytest.raises(ValueError, match="registered plan"):
        registry.enqueue(
            workspace_id="workspace-a",
            plan_id=f"plan:{'a' * 64}",
            operation_fingerprint="operation-v1",
            input_root_alias="inputs",
            input_root_sha256="b" * 64,
            output_root_alias="outputs",
            output_root_sha256="c" * 64,
            item_count=1,
            idempotency_key="one-request",
        )


def test_load_plan_rejects_tampered_item_count(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    root = tmp_path / "inputs"
    registry = SqliteJobRegistry(database)
    job = _enqueue(registry, root)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE job_plans SET item_count = 2 WHERE workspace_id = ? AND plan_id = ?",
            ("workspace-a", job.plan_id),
        )

    with pytest.raises(ValueError, match="plan receipt metadata changed"):
        registry.load_plan(
            workspace_id="workspace-a",
            plan_id=job.plan_id,
            input_root=root,
            input_root_alias="inputs",
            input_root_sha256="b" * 64,
        )


def test_load_plan_rejects_unreviewed_receipt_field_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    database = tmp_path / "jobs.sqlite3"
    root = tmp_path / "inputs"
    registry = SqliteJobRegistry(database)
    job = _enqueue(registry, root)
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT receipt_json FROM job_plans WHERE workspace_id = ? AND plan_id = ?",
            ("workspace-a", job.plan_id),
        ).fetchone()
        assert row is not None
        receipt = json.loads(row[0])
        receipt["operation"]["unreviewed_provider"] = "other"
        changed = json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        connection.execute(
            "UPDATE job_plans SET receipt_json = ?, receipt_sha256 = ? "
            "WHERE workspace_id = ? AND plan_id = ?",
            (
                changed,
                hashlib.sha256(changed.encode()).hexdigest(),
                "workspace-a",
                job.plan_id,
            ),
        )

    with pytest.raises(ValueError, match="invalid plan receipt"):
        registry.load_plan(
            workspace_id="workspace-a",
            plan_id=job.plan_id,
            input_root=root,
            input_root_alias="inputs",
            input_root_sha256="b" * 64,
        )


def test_load_plan_rejects_symlinked_input_root(tmp_path: Path) -> None:
    root = tmp_path / "inputs"
    registry = SqliteJobRegistry(tmp_path / "jobs.sqlite3")
    job = _enqueue(registry, root)
    alias = tmp_path / "input-alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this OS")

    with pytest.raises(ValueError, match=r"symlink|reparse"):
        registry.load_plan(
            workspace_id="workspace-a",
            plan_id=job.plan_id,
            input_root=alias,
            input_root_alias="inputs",
            input_root_sha256="b" * 64,
        )
