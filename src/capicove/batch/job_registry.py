"""Durable scheduling records for workspace-owned batch jobs."""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import stat
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from capicove.batch.discovery import DirectoryDiscoveryError, _observe_regular_file
from capicove.batch.job_plan_receipt import (
    decode_plan_receipt,
    encode_plan_receipt,
    receipt_sha256,
)
from capicove.batch.job_terminal import (
    TerminalReceipt,
    decode_terminal_receipt,
    encode_terminal_receipt,
)
from capicove.batch.models import BatchPlan


_PLAN_ID = re.compile(r"plan:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,127}\Z")
_JOB_COLUMNS = (
    "job_id",
    "workspace_id",
    "plan_id",
    "operation_fingerprint",
    "input_root_alias",
    "input_root_sha256",
    "output_root_alias",
    "output_root_sha256",
    "idempotency_key",
    "state",
    "revision",
    "worker_id",
    "lease_until",
    "created_at",
    "updated_at",
    "total_items",
    "completed_items",
    "checkpoint_prepared",
    "terminal_manifest_sha256",
    "error_code",
)
_PLAN_COLUMNS = (
    "workspace_id",
    "plan_id",
    "input_root_alias",
    "input_root_sha256",
    "operation_fingerprint",
    "item_count",
    "receipt_sha256",
    "receipt_json",
    "created_at",
)
_TERMINAL_COLUMNS = ("job_id", "receipt_json", "receipt_sha256", "committed_at")


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Describe one durable job without source content or item results."""

    job_id: str
    workspace_id: str
    plan_id: str
    operation_fingerprint: str
    input_root_alias: str
    input_root_sha256: str
    output_root_alias: str
    output_root_sha256: str
    state: str
    revision: int
    worker_id: str | None
    lease_until: datetime | None
    created_at: datetime
    updated_at: datetime
    total_items: int
    completed_items: int
    checkpoint_prepared: bool
    terminal_manifest_sha256: str | None
    error_code: str | None


class JobConflictError(ValueError):
    """Reject a reused idempotency key for a different job plan."""


class JobTransitionError(ValueError):
    """Reject a stale or forbidden job state transition."""


class JobClaimLostError(JobTransitionError):
    """Report that another actor removed a selected job from the queue."""


class SqliteJobRegistry:
    """Persist operator jobs separately from per-item batch checkpoints."""

    def __init__(self, database_path: Path) -> None:
        if not database_path.is_absolute():
            raise ValueError("job database path must be absolute")
        if os.name == "nt":
            from capicove._windows_files import validate_windows_file_target

            validate_windows_file_target(database_path)
        elif any(parent.is_symlink() for parent in database_path.parents):
            raise ValueError("job database parent must not contain symlinks")
        if not database_path.parent.is_dir():
            raise ValueError("job database parent must exist")
        if os.name == "posix":
            parent = database_path.parent.stat()
            if parent.st_uid != os.geteuid() or parent.st_mode & 0o077:
                raise ValueError("job database parent must be owner-private")
        self._database_path = database_path
        self._prepare_file()
        with self._connection() as connection, connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1}:
                raise ValueError("unsupported job database version")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    operation_fingerprint TEXT NOT NULL,
                    input_root_alias TEXT NOT NULL,
                    input_root_sha256 TEXT NOT NULL,
                    output_root_alias TEXT NOT NULL,
                    output_root_sha256 TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    worker_id TEXT,
                    lease_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    total_items INTEGER NOT NULL,
                    completed_items INTEGER NOT NULL,
                    checkpoint_prepared INTEGER NOT NULL DEFAULT 0
                        CHECK (checkpoint_prepared IN (0, 1)),
                    terminal_manifest_sha256 TEXT,
                    error_code TEXT,
                    UNIQUE(workspace_id, idempotency_key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_plans (
                    workspace_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    input_root_alias TEXT NOT NULL,
                    input_root_sha256 TEXT NOT NULL,
                    operation_fingerprint TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, plan_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_terminals (
                    job_id TEXT PRIMARY KEY,
                    receipt_json TEXT NOT NULL,
                    receipt_sha256 TEXT NOT NULL,
                    committed_at TEXT
                )
                """
            )
            columns = tuple(row[1] for row in connection.execute("PRAGMA table_info(jobs)"))
            if columns != _JOB_COLUMNS:
                raise ValueError("incompatible job database schema")
            plan_columns = tuple(
                row[1] for row in connection.execute("PRAGMA table_info(job_plans)")
            )
            if plan_columns != _PLAN_COLUMNS:
                raise ValueError("incompatible job plan database schema")
            terminal_columns = tuple(
                row[1] for row in connection.execute("PRAGMA table_info(job_terminals)")
            )
            if terminal_columns != _TERMINAL_COLUMNS:
                raise ValueError("incompatible job terminal database schema")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_queue_order "
                "ON jobs (workspace_id, state, created_at, job_id)"
            )
            connection.execute("PRAGMA user_version = 1")

    def enqueue(
        self,
        *,
        workspace_id: str,
        plan_id: str,
        operation_fingerprint: str,
        input_root_alias: str,
        input_root_sha256: str,
        output_root_alias: str,
        output_root_sha256: str,
        item_count: int,
        idempotency_key: str,
    ) -> JobRecord:
        """Durably queue one plan or return its identical idempotent retry."""
        if (
            not isinstance(workspace_id, str)
            or not workspace_id.strip()
            or len(workspace_id.encode("utf-8")) > 128
        ):
            raise ValueError("workspace_id must be nonblank and at most 128 UTF-8 bytes")
        if _PLAN_ID.fullmatch(plan_id) is None:
            raise ValueError("plan_id must be a plan-prefixed SHA-256")
        if not operation_fingerprint or len(operation_fingerprint) > 128:
            raise ValueError("operation_fingerprint must be 1 to 128 characters")
        if any(
            _IDENTIFIER.fullmatch(alias) is None for alias in (input_root_alias, output_root_alias)
        ):
            raise ValueError("root aliases must be bounded identifiers")
        if any(
            _SHA256.fullmatch(value) is None for value in (input_root_sha256, output_root_sha256)
        ):
            raise ValueError("root hashes must be lowercase SHA-256 values")
        if not idempotency_key or len(idempotency_key.encode("utf-8")) > 128:
            raise ValueError("idempotency_key must be 1 to 128 UTF-8 bytes")
        if type(item_count) is not int or not 0 <= item_count <= 1_000_000:
            raise ValueError("item_count must be between 0 and 1,000,000")
        identity = (
            workspace_id,
            plan_id,
            operation_fingerprint,
            input_root_alias,
            input_root_sha256,
            output_root_alias,
            output_root_sha256,
            item_count,
            idempotency_key,
        )
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            plan = connection.execute(
                "SELECT * FROM job_plans WHERE workspace_id = ? AND plan_id = ?",
                (workspace_id, plan_id),
            ).fetchone()
            if plan is None or (
                str(plan["input_root_alias"]) != input_root_alias
                or str(plan["input_root_sha256"]) != input_root_sha256
                or str(plan["operation_fingerprint"]) != operation_fingerprint
                or int(plan["item_count"]) != item_count
                or receipt_sha256(str(plan["receipt_json"])) != str(plan["receipt_sha256"])
            ):
                raise ValueError("enqueue requires a matching registered plan")
            return _enqueue_record(connection, identity)

    def enqueue_plan(
        self,
        *,
        workspace_id: str,
        plan: BatchPlan,
        input_root: Path,
        input_root_alias: str,
        input_root_sha256: str,
        output_root_alias: str,
        output_root_sha256: str,
        idempotency_key: str,
    ) -> JobRecord:
        """Atomically persist an immutable plan receipt with its queued job."""
        if not workspace_id.strip() or len(workspace_id.encode("utf-8")) > 128:
            raise ValueError("workspace_id must be nonblank and at most 128 UTF-8 bytes")
        if any(
            _IDENTIFIER.fullmatch(alias) is None for alias in (input_root_alias, output_root_alias)
        ) or any(
            _SHA256.fullmatch(value) is None for value in (input_root_sha256, output_root_sha256)
        ):
            raise ValueError("job roots require bounded aliases and SHA-256 identities")
        if not idempotency_key or len(idempotency_key.encode("utf-8")) > 128:
            raise ValueError("idempotency_key must be 1 to 128 UTF-8 bytes")
        receipt, digest, operation_digest = encode_plan_receipt(plan, input_root)
        identity = (
            workspace_id,
            plan.plan_id,
            operation_digest,
            input_root_alias,
            input_root_sha256,
            output_root_alias,
            output_root_sha256,
            len(plan.items),
            idempotency_key,
        )
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM job_plans WHERE workspace_id = ? AND plan_id = ?",
                (workspace_id, plan.plan_id),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO job_plans (workspace_id, plan_id, input_root_alias, "
                    "input_root_sha256, operation_fingerprint, item_count, "
                    "receipt_sha256, receipt_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        workspace_id,
                        plan.plan_id,
                        input_root_alias,
                        input_root_sha256,
                        operation_digest,
                        len(plan.items),
                        digest,
                        receipt,
                        datetime.now(UTC).isoformat(),
                    ),
                )
            elif (
                str(existing["input_root_alias"]) != input_root_alias
                or str(existing["input_root_sha256"]) != input_root_sha256
                or str(existing["operation_fingerprint"]) != operation_digest
                or int(existing["item_count"]) != len(plan.items)
                or receipt_sha256(str(existing["receipt_json"])) != str(existing["receipt_sha256"])
            ):
                raise JobConflictError("plan_conflict")
            elif str(existing["receipt_json"]) != receipt:
                previous = connection.execute(
                    "SELECT * FROM jobs WHERE workspace_id = ? AND idempotency_key = ?",
                    (workspace_id, idempotency_key),
                ).fetchone()
                if previous is None:
                    raise JobConflictError("plan_conflict")
                if _job_identity(previous) != identity:
                    raise JobConflictError("idempotency_conflict")
                decode_plan_receipt(str(existing["receipt_json"]), plan.plan_id, input_root)
                return _job_record(previous)
            previous = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND idempotency_key = ?",
                (workspace_id, idempotency_key),
            ).fetchone()
            if previous is None:
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
            return _enqueue_record(connection, identity)

    def load_plan(
        self,
        *,
        workspace_id: str,
        plan_id: str,
        input_root: Path,
        input_root_alias: str,
        input_root_sha256: str,
    ) -> BatchPlan:
        """Reconstruct a verified plan beneath an operator-configured root."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM job_plans WHERE workspace_id = ? AND plan_id = ?",
                (workspace_id, plan_id),
            ).fetchone()
        if row is None:
            raise KeyError("job plan is unavailable")
        if (
            str(row["input_root_alias"]) != input_root_alias
            or str(row["input_root_sha256"]) != input_root_sha256
        ):
            raise ValueError("job plan root identity changed")
        receipt = str(row["receipt_json"])
        if receipt_sha256(receipt) != str(row["receipt_sha256"]):
            raise ValueError("job plan receipt changed")
        plan = decode_plan_receipt(receipt, plan_id, input_root)
        _, _, operation_digest = encode_plan_receipt(plan, input_root)
        if len(plan.items) != int(row["item_count"]) or operation_digest != str(
            row["operation_fingerprint"]
        ):
            raise ValueError("job plan receipt metadata changed")
        return plan

    def status(self, *, workspace_id: str, job_id: str) -> JobRecord:
        """Return one scoped job without changing its state."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT jobs.*, job_terminals.receipt_json AS terminal_receipt_json, "
                "job_terminals.receipt_sha256 AS terminal_receipt_sha256, "
                "job_terminals.committed_at AS terminal_committed_at "
                "FROM jobs LEFT JOIN job_terminals USING (job_id) "
                "WHERE jobs.workspace_id = ? AND jobs.job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
        if row is None:
            raise KeyError("job is unavailable")
        record = _job_record(row)
        if record.state in {"succeeded", "failed"} and record.terminal_manifest_sha256 is None:
            raise ValueError("job terminal receipt changed")
        terminal_receipt = row["terminal_receipt_json"]
        if (
            record.terminal_manifest_sha256 is None
            and terminal_receipt is not None
            and (
                row["terminal_committed_at"] is not None
                or record.state not in {"running", "cancelling"}
            )
        ):
            raise ValueError("job terminal receipt changed")
        if record.terminal_manifest_sha256 is not None:
            if (
                terminal_receipt is None
                or row["terminal_committed_at"] is None
                or str(row["terminal_receipt_sha256"]) != record.terminal_manifest_sha256
            ):
                raise ValueError("job terminal receipt changed")
            receipt = decode_terminal_receipt(str(terminal_receipt))
            if encode_terminal_receipt(receipt)[1] != record.terminal_manifest_sha256 or (
                record.state not in {"succeeded", "failed", "cancelled"}
                or receipt.job_id != record.job_id
                or receipt.plan_id != record.plan_id
                or receipt.total_items != record.total_items
                or receipt.completed_items != record.completed_items
                or (
                    record.state in {"succeeded", "failed"}
                    and (record.state == "succeeded") != (receipt.failed_items == 0)
                )
            ):
                raise ValueError("job terminal receipt changed")
        return record

    def next_queued(self, *, workspace_id: str) -> JobRecord | None:
        """Return the oldest queued job in one workspace without claiming it."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND state = 'queued' "
                "ORDER BY created_at, job_id LIMIT 1",
                (workspace_id,),
            ).fetchone()
        return _job_record(row) if row is not None else None

    def queued_jobs(self, *, workspace_id: str, limit: int) -> tuple[JobRecord, ...]:
        """Return up to ``limit`` queued jobs in claim order without claiming them."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND state = 'queued' "
                "ORDER BY created_at, job_id LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
        return tuple(_job_record(row) for row in rows)

    def next_expired(self, *, workspace_id: str, now: datetime) -> JobRecord | None:
        """Return one elapsed running or cancelling lease in this workspace."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? "
                "AND state IN ('running', 'cancelling') "
                "AND lease_until <= ? "
                "ORDER BY lease_until, created_at, job_id LIMIT 1",
                (workspace_id, now.astimezone(UTC).isoformat()),
            ).fetchone()
        return _job_record(row) if row is not None else None

    def has_pending_terminal(self, *, workspace_id: str, job_id: str) -> bool:
        """Report an uncommitted receipt for a Job in the selected workspace."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM jobs JOIN job_terminals USING (job_id) "
                "WHERE jobs.workspace_id = ? AND jobs.job_id = ? "
                "AND job_terminals.committed_at IS NULL",
                (workspace_id, job_id),
            ).fetchone()
        return row is not None

    def cancel(self, *, workspace_id: str, job_id: str, expected_revision: int) -> JobRecord:
        """Request cancellation with compare-and-swap revision authority."""
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            if int(row["revision"]) != expected_revision:
                raise JobTransitionError("revision_conflict")
            state = str(row["state"])
            if state not in {"queued", "running", "cancelling"}:
                raise JobTransitionError("terminal job cannot be cancelled")
            if state == "cancelling":
                return _job_record(row)
            next_state = "cancelled" if state == "queued" else "cancelling"
            connection.execute(
                "UPDATE jobs SET state = ?, revision = revision + 1, updated_at = ? "
                "WHERE workspace_id = ? AND job_id = ? AND revision = ?",
                (
                    next_state,
                    datetime.now(UTC).isoformat(),
                    workspace_id,
                    job_id,
                    expected_revision,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert updated is not None
            return _job_record(updated)

    def claim(
        self,
        *,
        workspace_id: str,
        job_id: str,
        worker_id: str,
        now: datetime,
        lease_until: datetime,
    ) -> JobRecord:
        """Acquire an exclusive worker lease for a queued job."""
        if _IDENTIFIER.fullmatch(worker_id) is None:
            raise ValueError("worker_id must be a bounded identifier")
        if now.tzinfo is None or lease_until.tzinfo is None or lease_until <= now:
            raise ValueError("worker lease must have a future timezone-aware deadline")
        encoded_until = lease_until.astimezone(UTC).isoformat()
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            if str(row["state"]) != "queued":
                raise JobClaimLostError("job is not queued")
            updated = connection.execute(
                "UPDATE jobs SET state = 'running', revision = revision + 1, "
                "worker_id = ?, lease_until = ?, updated_at = ? "
                "WHERE workspace_id = ? AND job_id = ? AND state = 'queued' AND revision = ?",
                (
                    worker_id,
                    encoded_until,
                    now.astimezone(UTC).isoformat(),
                    workspace_id,
                    job_id,
                    int(row["revision"]),
                ),
            )
            if updated.rowcount != 1:
                raise JobClaimLostError("job lease changed concurrently")
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert claimed is not None
            return _job_record(claimed)

    def recover_expired(
        self,
        *,
        workspace_id: str,
        job_id: str,
        now: datetime,
        verify_checkpoint: Callable[[JobRecord], bool],
    ) -> JobRecord:
        """Release an expired lease only after checkpoint verification."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            record = _job_record(row)
            if record.state != "running" or record.lease_until is None:
                raise JobTransitionError("job has no running lease")
            if now < record.lease_until:
                raise JobTransitionError("job lease active")
            pending = connection.execute(
                "SELECT committed_at FROM job_terminals WHERE job_id = ?", (job_id,)
            ).fetchone()
            if pending is not None and pending["committed_at"] is None:
                raise JobTransitionError("pending terminal receipt requires recovery")
            if not verify_checkpoint(record):
                raise JobTransitionError("checkpoint verification failed")
            updated = connection.execute(
                "UPDATE jobs SET state = 'interrupted', revision = revision + 1, "
                "worker_id = NULL, lease_until = NULL "
                "WHERE workspace_id = ? AND job_id = ? AND state = 'running' AND revision = ?",
                (workspace_id, job_id, record.revision),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("job lease changed concurrently")
            interrupted = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert interrupted is not None
            return _job_record(interrupted)

    def resume(
        self,
        *,
        workspace_id: str,
        job_id: str,
        expected_revision: int,
        now: datetime,
        verify_resume: Callable[[JobRecord], None],
    ) -> JobRecord:
        """Requeue an interrupted job only after scoped recovery verification."""
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            record = _job_record(row)
            if record.revision != expected_revision:
                raise JobTransitionError("revision_conflict")
            if record.state != "interrupted":
                raise JobTransitionError("job is not interrupted")
            terminal = connection.execute(
                "SELECT 1 FROM job_terminals WHERE job_id = ?", (job_id,)
            ).fetchone()
            if terminal is not None:
                raise JobTransitionError("terminal receipt prevents resume")
            verify_resume(record)
            updated = connection.execute(
                "UPDATE jobs SET state = 'queued', revision = revision + 1, "
                "checkpoint_prepared = 0, worker_id = NULL, lease_until = NULL, "
                "updated_at = ? WHERE workspace_id = ? AND job_id = ? "
                "AND state = 'interrupted' AND revision = ?",
                (now.astimezone(UTC).isoformat(), workspace_id, job_id, expected_revision),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("job resume changed concurrently")
            resumed = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert resumed is not None
            return _job_record(resumed)

    def mark_prepared(
        self,
        *,
        workspace_id: str,
        job_id: str,
        worker_id: str,
        expected_revision: int,
        now: datetime,
    ) -> JobRecord:
        """Mark verified checkpoints ready before item execution begins."""
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            record = _job_record(row)
            if record.revision != expected_revision:
                raise JobTransitionError("revision_conflict")
            if record.worker_id != worker_id or record.state != "running":
                raise JobTransitionError("worker lease is unavailable")
            if record.lease_until is None or now >= record.lease_until:
                raise JobTransitionError("worker lease expired")
            if record.checkpoint_prepared:
                raise JobTransitionError("checkpoints already prepared")
            updated = connection.execute(
                "UPDATE jobs SET checkpoint_prepared = 1, revision = revision + 1, "
                "updated_at = ? WHERE workspace_id = ? AND job_id = ? "
                "AND worker_id = ? AND revision = ? AND checkpoint_prepared = 0",
                (
                    now.astimezone(UTC).isoformat(),
                    workspace_id,
                    job_id,
                    worker_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("job preparation changed concurrently")
            prepared = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert prepared is not None
            return _job_record(prepared)

    def stage_terminal(
        self,
        *,
        workspace_id: str,
        job_id: str,
        worker_id: str | None,
        expected_revision: int,
        now: datetime,
        receipt: TerminalReceipt,
    ) -> TerminalReceipt:
        """Persist a pending content-free receipt without reporting success."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        encoded, digest = encode_terminal_receipt(receipt)
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            record = _job_record(row)
            live_worker = (
                worker_id is not None
                and record.worker_id == worker_id
                and record.lease_until is not None
                and now < record.lease_until
                and record.checkpoint_prepared
            )
            expired_cancellation = (
                worker_id is None
                and record.state == "cancelling"
                and record.lease_until is not None
                and now >= record.lease_until
            )
            if (
                record.job_id != receipt.job_id
                or record.plan_id != receipt.plan_id
                or record.total_items != receipt.total_items
                or record.revision != expected_revision
                or record.state not in {"running", "cancelling"}
                or not (live_worker or expired_cancellation)
            ):
                raise JobTransitionError("job lease or terminal identity changed")
            existing = connection.execute(
                "SELECT * FROM job_terminals WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["receipt_json"]) != encoded
                    or str(existing["receipt_sha256"]) != digest
                    or existing["committed_at"] is not None
                ):
                    raise JobConflictError("terminal_receipt_conflict")
                return receipt
            connection.execute(
                "INSERT INTO job_terminals (job_id, receipt_json, receipt_sha256, committed_at) "
                "VALUES (?, ?, ?, NULL)",
                (job_id, encoded, digest),
            )
        return receipt

    def finalize_terminal(
        self,
        *,
        workspace_id: str,
        job_id: str,
        now: datetime,
        worker_id: str | None,
        verify_receipt: Callable[[TerminalReceipt], bool],
    ) -> JobRecord:
        """Commit a verified pending receipt under a live or expired lease."""
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            record = _job_record(row)
            if record.state not in {"running", "cancelling"} or (
                not record.checkpoint_prepared
                and not (record.state == "cancelling" and worker_id is None)
            ):
                raise JobTransitionError("job is not ready for terminal publication")
            if record.lease_until is None or (worker_id is None and now < record.lease_until):
                raise JobTransitionError("job lease is active")
            if worker_id is not None and (
                record.worker_id != worker_id or now >= record.lease_until
            ):
                raise JobTransitionError("worker lease is unavailable")
            terminal = connection.execute(
                "SELECT * FROM job_terminals WHERE job_id = ?", (job_id,)
            ).fetchone()
            if terminal is None or terminal["committed_at"] is not None:
                raise JobTransitionError("pending terminal receipt is unavailable")
            encoded = str(terminal["receipt_json"])
            digest = str(terminal["receipt_sha256"])
            receipt = decode_terminal_receipt(encoded)
            if (
                encode_terminal_receipt(receipt)[1] != digest
                or receipt.job_id != record.job_id
                or receipt.plan_id != record.plan_id
                or receipt.total_items != record.total_items
                or receipt.completed_items != record.total_items
                or not verify_receipt(receipt)
            ):
                raise ValueError("job terminal receipt verification failed")
            state = (
                "cancelled"
                if record.state == "cancelling"
                else "failed"
                if receipt.failed_items
                else "succeeded"
            )
            timestamp = now.astimezone(UTC).isoformat()
            updated = connection.execute(
                "UPDATE jobs SET state = ?, revision = revision + 1, worker_id = NULL, "
                "lease_until = NULL, updated_at = ?, completed_items = ?, "
                "terminal_manifest_sha256 = ?, error_code = ? "
                "WHERE workspace_id = ? AND job_id = ? AND state = ? AND revision = ?",
                (
                    state,
                    timestamp,
                    receipt.completed_items,
                    digest,
                    "ITEM_FAILURE" if state == "failed" else None,
                    workspace_id,
                    job_id,
                    record.state,
                    record.revision,
                ),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("job changed during terminal publication")
            connection.execute(
                "UPDATE job_terminals SET committed_at = ? WHERE job_id = ?",
                (timestamp, job_id),
            )
            committed = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert committed is not None
            return _job_record(committed)

    def renew_lease(
        self,
        *,
        workspace_id: str,
        job_id: str,
        worker_id: str,
        expected_revision: int,
        now: datetime,
        lease_until: datetime,
    ) -> JobRecord:
        """Extend an unexpired lease held by the same worker."""
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if now.tzinfo is None or lease_until.tzinfo is None or lease_until <= now:
            raise ValueError("worker lease must have a future timezone-aware deadline")
        with self._connection() as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job is unavailable")
            record = _job_record(row)
            if record.revision != expected_revision:
                raise JobTransitionError("revision_conflict")
            if record.worker_id != worker_id:
                raise JobTransitionError("worker mismatch")
            if record.state not in {"running", "cancelling"} or record.lease_until is None:
                raise JobTransitionError("job has no renewable lease")
            if now >= record.lease_until:
                raise JobTransitionError("job lease expired")
            if lease_until <= record.lease_until:
                raise ValueError("new worker lease must extend the current deadline")
            updated = connection.execute(
                "UPDATE jobs SET revision = revision + 1, lease_until = ?, updated_at = ? "
                "WHERE workspace_id = ? AND job_id = ? AND worker_id = ? AND revision = ?",
                (
                    lease_until.astimezone(UTC).isoformat(),
                    now.astimezone(UTC).isoformat(),
                    workspace_id,
                    job_id,
                    worker_id,
                    expected_revision,
                ),
            )
            if updated.rowcount != 1:
                raise JobTransitionError("job lease changed concurrently")
            renewed = connection.execute(
                "SELECT * FROM jobs WHERE workspace_id = ? AND job_id = ?",
                (workspace_id, job_id),
            ).fetchone()
            assert renewed is not None
            return _job_record(renewed)

    def _prepare_file(self) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._database_path, flags, 0o600)
        except FileExistsError:
            metadata = self._database_path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise ValueError("job database must be a private regular file") from None
        else:
            os.close(descriptor)

    def _connection(self) -> closing[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return closing(connection)


def _job_identity(row: sqlite3.Row) -> tuple[object, ...]:
    return tuple(
        int(row[name]) if name == "total_items" else str(row[name])
        for name in (
            "workspace_id",
            "plan_id",
            "operation_fingerprint",
            "input_root_alias",
            "input_root_sha256",
            "output_root_alias",
            "output_root_sha256",
            "total_items",
            "idempotency_key",
        )
    )


def _enqueue_record(connection: sqlite3.Connection, identity: tuple[object, ...]) -> JobRecord:
    workspace_id = str(identity[0])
    idempotency_key = str(identity[-1])
    row = connection.execute(
        "SELECT * FROM jobs WHERE workspace_id = ? AND idempotency_key = ?",
        (workspace_id, idempotency_key),
    ).fetchone()
    if row is not None:
        if _job_identity(row) != identity:
            raise JobConflictError("idempotency_conflict")
        return _job_record(row)
    job_id = f"job:{secrets.token_hex(12)}"
    created_at = datetime.now(UTC).isoformat()
    connection.execute(
        """
        INSERT INTO jobs (
            job_id, workspace_id, plan_id, operation_fingerprint,
            input_root_alias, input_root_sha256, output_root_alias,
            output_root_sha256, total_items, idempotency_key, state,
            revision, worker_id, lease_until, created_at, updated_at,
            completed_items, terminal_manifest_sha256, error_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 1, NULL,
            NULL, ?, ?, 0, NULL, NULL)
        """,
        (job_id, *identity, created_at, created_at),
    )
    created = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert created is not None
    return _job_record(created)


def _job_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=str(row["job_id"]),
        workspace_id=str(row["workspace_id"]),
        plan_id=str(row["plan_id"]),
        operation_fingerprint=str(row["operation_fingerprint"]),
        input_root_alias=str(row["input_root_alias"]),
        input_root_sha256=str(row["input_root_sha256"]),
        output_root_alias=str(row["output_root_alias"]),
        output_root_sha256=str(row["output_root_sha256"]),
        state=str(row["state"]),
        revision=int(row["revision"]),
        worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
        lease_until=(
            datetime.fromisoformat(str(row["lease_until"]))
            if row["lease_until"] is not None
            else None
        ),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        total_items=int(row["total_items"]),
        completed_items=int(row["completed_items"]),
        checkpoint_prepared=bool(row["checkpoint_prepared"]),
        terminal_manifest_sha256=(
            str(row["terminal_manifest_sha256"])
            if row["terminal_manifest_sha256"] is not None
            else None
        ),
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
    )
