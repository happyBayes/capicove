"""Closed workflow and durable operational identities for graph builds."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from capicove.contracts.data import DatasetSnapshotRef
from capicove.contracts.graph_build import GraphScope
from capicove.graph_build._adapters.sqlite import SqliteSnapshotStore
from capicove.graph_build.codec import _dataset_payload, _dataset_record, canonical_bytes
from capicove.graph_build.snapshots import SnapshotAccessGate


_RUN_ID = re.compile(r"build:[0-9a-f]{32}")
_RUN_STATUSES = frozenset({"pending", "running", "succeeded", "partial", "failed", "cancelled"})
_TERMINAL_STATUSES = frozenset({"succeeded", "partial", "failed", "cancelled"})
_RESULT_FIELDS = {
    "version",
    "run_id",
    "status",
    "graph_ref",
    "completeness",
    "coverage",
    "issue_codes",
    "retrieval_binding_ref",
}
_RUN_SCHEMA = (
    """CREATE TABLE build_runs (
        run_id TEXT PRIMARY KEY, workspace TEXT NOT NULL, scope TEXT NOT NULL,
        request BLOB NOT NULL, request_digest TEXT NOT NULL,
        profile_digest TEXT NOT NULL, input_digest TEXT NOT NULL,
        baseline_digest TEXT, deadline_at REAL NOT NULL,
        runtime_digest TEXT, pin_digest TEXT NOT NULL,
        status TEXT NOT NULL, cancelled INTEGER NOT NULL,
        terminal_prior_status TEXT, terminal_prior_cancelled INTEGER,
        terminal_digest TEXT)""",
    """CREATE TABLE build_stages (
        run_id TEXT NOT NULL, stage TEXT NOT NULL, artifact_ref TEXT NOT NULL,
        record_digest TEXT NOT NULL,
        PRIMARY KEY (run_id, stage))""",
    """CREATE TABLE build_results (
        run_id TEXT PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL)""",
    """CREATE TABLE build_runtime (
        run_id TEXT PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL)""",
    """CREATE TRIGGER immutable_build_terminal
        BEFORE UPDATE OF status, cancelled, terminal_prior_status,
        terminal_prior_cancelled, terminal_digest ON build_runs
        WHEN OLD.terminal_digest IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'immutable build terminal'); END""",
)


class _BuildRunSqliteStore(SqliteSnapshotStore):
    """Use the capability's private-path safeguards with an exact run schema."""

    @staticmethod
    def _schema(connection: sqlite3.Connection, initialize: bool) -> None:
        objects = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master"
        ).fetchall()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if initialize and version == 0 and not objects:
            for statement in _RUN_SCHEMA:
                connection.execute(statement)
            connection.execute("PRAGMA user_version=2")
            version = 2
            objects = connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master"
            ).fetchall()
        expected: set[tuple[str, str, str, str | None]] = set()
        for statement in _RUN_SCHEMA:
            name = statement.split()[2]
            if statement.lstrip().startswith("CREATE TABLE"):
                expected.add(("table", name, name, statement))
                expected.add(("index", f"sqlite_autoindex_{name}_1", name, None))
            else:
                expected.add(("trigger", name, "build_runs", statement))
        if version != 2 or {tuple(row) for row in objects} != expected:
            raise ValueError("build run database schema is unsupported or corrupt")


def stage_dependencies() -> Mapping[str, tuple[str, ...]]:
    """Return a fresh immutable view of the only accepted workflow."""
    return MappingProxyType(
        {
            "representations": (),
            "text_units": ("representations",),
            "extraction": ("text_units",),
            "identity_embeddings": ("extraction",),
            "resolution": ("extraction", "identity_embeddings"),
            "alignment": ("resolution",),
            "aggregation": ("alignment", "extraction"),
            "diagnostics": ("aggregation",),
            "communities": ("aggregation",),
            "reports": ("communities", "diagnostics"),
            "publication": ("reports",),
            "retrieval": ("publication",),
        }
    )


def validate_dependencies(graph: Mapping[str, tuple[str, ...]]) -> None:
    """Reject cycles, unknown stages, omissions and every workflow alteration."""
    if dict(graph) != dict(stage_dependencies()):
        raise ValueError("unsupported_build_workflow")


def _text(value: object) -> None:
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ValueError("invalid_build_reference")
    value.encode("utf-8")


def _digest(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid_build_digest")


def _run_id(value: object) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise ValueError("invalid_build_run_id")
    return value


def _deadline(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError("invalid_build_deadline")
    return value


def _pin_digest(
    run_id: str,
    scope: GraphScope,
    request_digest: str,
    profile_digest: str,
    input_digest: str,
    baseline_digest: str | None,
    deadline_at: float,
    runtime_digest: str | None,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "version": 1,
                "run_id": run_id,
                "workspace": scope.workspace_ref,
                "scope": scope.scope_ref,
                "request_digest": request_digest,
                "profile_digest": profile_digest,
                "input_digest": input_digest,
                "baseline_digest": baseline_digest,
                "deadline_at": deadline_at,
                "runtime_digest": runtime_digest,
            }
        )
    ).hexdigest()


def _stage_digest(run_id: str, stage: str, artifact_ref: str) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "version": 1,
                "run_id": run_id,
                "stage": stage,
                "artifact_ref": artifact_ref,
            }
        )
    ).hexdigest()


def _terminal_digest(
    pin_digest: str,
    prior_status: str,
    prior_cancelled: bool,
    result_digest: str,
    status: str,
    cancelled: bool,
) -> str:
    """Commit one terminal transition independently from its mutable result row."""
    return hashlib.sha256(
        canonical_bytes(
            {
                "version": 1,
                "pin_digest": pin_digest,
                "prior_status": prior_status,
                "prior_cancelled": prior_cancelled,
                "result_digest": result_digest,
                "status": status,
                "cancelled": cancelled,
            }
        )
    ).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate_build_json_field")
        result[key] = value
    return result


def _json(payload: bytes, error_code: str) -> Any:
    if type(payload) is not bytes:
        raise ValueError(error_code)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_pairs)
        if canonical_bytes(value) != payload:
            raise ValueError(error_code)
        return value
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(error_code) from error


@dataclass(frozen=True, slots=True)
class GraphBuildRequest:
    """Pin one source inventory, scope, accepted profile and optional baseline."""

    input_snapshot_ref: DatasetSnapshotRef
    scope: GraphScope
    profile_id: str
    baseline_snapshot_id: str | None
    mode: str

    def __post_init__(self) -> None:
        if (
            type(self.input_snapshot_ref) is not DatasetSnapshotRef
            or type(self.scope) is not GraphScope
        ):
            raise TypeError("build_requires_typed_input_and_scope")
        _text(self.profile_id)
        if self.baseline_snapshot_id is not None:
            _text(self.baseline_snapshot_id)
        if type(self.mode) is not str or self.mode not in {"reuse", "replay", "regenerate"}:
            raise ValueError("unsupported_build_mode")


def request_bytes(request: GraphBuildRequest) -> bytes:
    """Encode the complete closed request without operational state."""
    return canonical_bytes(
        {
            "version": 1,
            "input_snapshot_ref": _dataset_payload(request.input_snapshot_ref),
            "scope": asdict(request.scope),
            "profile_id": request.profile_id,
            "baseline_snapshot_id": request.baseline_snapshot_id,
            "mode": request.mode,
        }
    )


def _request(payload: bytes) -> GraphBuildRequest:
    try:
        value = _json(payload, "invalid_persisted_build_request")
        if type(value) is not dict or set(value) != {
            "version",
            "input_snapshot_ref",
            "scope",
            "profile_id",
            "baseline_snapshot_id",
            "mode",
        }:
            raise ValueError("invalid_persisted_build_request")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("invalid_persisted_build_request")
        if type(value["scope"]) is not dict or set(value["scope"]) != {
            "workspace_ref",
            "scope_ref",
        }:
            raise ValueError("invalid_persisted_build_request")
        result = GraphBuildRequest(
            _dataset_record(value["input_snapshot_ref"]),
            GraphScope(**value["scope"]),
            value["profile_id"],
            value["baseline_snapshot_id"],
            value["mode"],
        )
        if request_bytes(result) != payload:
            raise ValueError("noncanonical_build_request")
        return result
    except (KeyError, TypeError, ValueError, RecursionError) as error:
        raise ValueError("invalid_persisted_build_request") from error


def _result_identity(payload: bytes, run_id: str, status: str) -> None:
    value = _json(payload, "invalid_persisted_build_result")
    if (
        type(value) is not dict
        or set(value) != _RESULT_FIELDS
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["run_id"] != run_id
        or value["status"] != status
        or type(value["coverage"]) is not list
        or len(value["coverage"]) > len(stage_dependencies())
        or type(value["issue_codes"]) is not list
        or len(value["issue_codes"]) > 128
        or value["completeness"] not in {"complete", "partial"}
    ):
        raise ValueError("invalid_persisted_build_result")
    if value["graph_ref"] is not None:
        _text(value["graph_ref"])
    if value["retrieval_binding_ref"] is not None:
        _text(value["retrieval_binding_ref"])
    for stage in value["coverage"]:
        _text(stage)
    for issue in value["issue_codes"]:
        _text(issue)


@dataclass(frozen=True, slots=True)
class BuildRun:
    """Durable operational state; separate from immutable graph identity."""

    run_id: str
    request: GraphBuildRequest
    profile_fingerprint: str
    input_digest: str
    baseline_digest: str | None
    deadline_at: float
    status: str
    cancelled: bool


class BuildRunStore:
    """Private scoped SQLite records with immutable requests and explicit cancel."""

    def __init__(self, path: Path, gate: SnapshotAccessGate) -> None:
        self.path = path
        self.gate = gate
        self._storage = _BuildRunSqliteStore(path)

    def create(
        self,
        request: GraphBuildRequest,
        *,
        profile_fingerprint: str,
        input_digest: str,
        deadline_at: float,
        baseline_digest: str | None = None,
        runtime_payload: bytes | None = None,
    ) -> BuildRun:
        """Create a fresh auditable run even for an identical semantic request."""
        self.gate.require_scope(request.scope)
        _digest(profile_fingerprint)
        _digest(input_digest)
        _deadline(deadline_at)
        if baseline_digest is not None:
            _digest(baseline_digest)
        if (request.baseline_snapshot_id is None) != (baseline_digest is None):
            raise ValueError("build_baseline_pin_mismatch")
        if runtime_payload is not None:
            self._validate_runtime(runtime_payload)
        payload = request_bytes(request)
        request_digest = hashlib.sha256(payload).hexdigest()
        runtime_digest = (
            hashlib.sha256(runtime_payload).hexdigest() if runtime_payload is not None else None
        )
        run_id = "build:" + uuid.uuid4().hex
        pin_digest = _pin_digest(
            run_id,
            request.scope,
            request_digest,
            profile_fingerprint,
            input_digest,
            baseline_digest,
            deadline_at,
            runtime_digest,
        )
        revision = self.gate.revision
        with self._storage.transaction() as connection:
            with self.gate.guard(revision):
                self.gate.require_scope(request.scope)
                connection.execute(
                    "INSERT INTO build_runs VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, NULL, NULL, NULL)",
                    (
                        run_id,
                        request.scope.workspace_ref,
                        request.scope.scope_ref,
                        payload,
                        request_digest,
                        profile_fingerprint,
                        input_digest,
                        baseline_digest,
                        deadline_at,
                        runtime_digest,
                        pin_digest,
                    ),
                )
                if runtime_payload is not None:
                    connection.execute(
                        "INSERT INTO build_runtime VALUES (?, ?, ?)",
                        (
                            run_id,
                            runtime_payload,
                            hashlib.sha256(runtime_payload).hexdigest(),
                        ),
                    )
        return self.get(run_id, request.scope)

    def _row(self, connection: sqlite3.Connection, run_id: str, scope: GraphScope) -> sqlite3.Row:
        self.gate.require_scope(scope)
        _run_id(run_id)
        row = connection.execute(
            "SELECT * FROM build_runs WHERE run_id = ? AND workspace = ? AND scope = ?",
            (run_id, scope.workspace_ref, scope.scope_ref),
        ).fetchone()
        if row is None:
            raise PermissionError("build_run_unavailable")
        return cast(sqlite3.Row, row)

    def _run(self, row: sqlite3.Row, run_id: str, scope: GraphScope) -> BuildRun:
        if row["run_id"] != run_id:
            raise ValueError("build_run_identity_failure")
        _run_id(row["run_id"])
        if (
            type(row["workspace"]) is not str
            or type(row["scope"]) is not str
            or row["workspace"] != scope.workspace_ref
            or row["scope"] != scope.scope_ref
        ):
            raise ValueError("build_scope_integrity_failure")
        if type(row["request"]) is not bytes:
            raise ValueError("build_request_integrity_failure")
        payload = row["request"]
        _digest(row["request_digest"])
        if hashlib.sha256(payload).hexdigest() != row["request_digest"]:
            raise ValueError("build_request_integrity_failure")
        request = _request(payload)
        if request.scope != scope:
            raise ValueError("build_scope_integrity_failure")
        _digest(row["profile_digest"])
        _digest(row["input_digest"])
        baseline_digest = row["baseline_digest"]
        if baseline_digest is not None:
            _digest(baseline_digest)
        if (request.baseline_snapshot_id is None) != (baseline_digest is None):
            raise ValueError("build_baseline_pin_mismatch")
        deadline_at = _deadline(row["deadline_at"])
        runtime_digest = row["runtime_digest"]
        if runtime_digest is not None:
            _digest(runtime_digest)
        _digest(row["pin_digest"])
        expected_pin = _pin_digest(
            run_id,
            scope,
            row["request_digest"],
            row["profile_digest"],
            row["input_digest"],
            baseline_digest,
            deadline_at,
            runtime_digest,
        )
        if row["pin_digest"] != expected_pin:
            raise ValueError("build_pin_integrity_failure")
        status = row["status"]
        cancelled = row["cancelled"]
        if type(status) is not str or status not in _RUN_STATUSES:
            raise ValueError("invalid_persisted_build_status")
        if type(cancelled) is not int or cancelled not in (0, 1):
            raise ValueError("invalid_persisted_build_cancelled")
        prior_status = row["terminal_prior_status"]
        prior_cancelled = row["terminal_prior_cancelled"]
        terminal_digest = row["terminal_digest"]
        if status in _TERMINAL_STATUSES:
            if (
                type(prior_status) is not str
                or prior_status not in {"pending", "running"}
                or type(prior_cancelled) is not int
                or prior_cancelled not in (0, 1)
                or (bool(prior_cancelled) and status != "cancelled")
                or terminal_digest is None
            ):
                raise ValueError("build_terminal_commitment_failure")
            _digest(terminal_digest)
        elif any(value is not None for value in (prior_status, prior_cancelled, terminal_digest)):
            raise ValueError("build_terminal_commitment_failure")
        return BuildRun(
            run_id,
            request,
            row["profile_digest"],
            row["input_digest"],
            baseline_digest,
            deadline_at,
            status,
            bool(cancelled),
        )

    def get(self, run_id: str, scope: GraphScope) -> BuildRun:
        """Authorize exact scope and verify canonical request before use."""
        with self._storage.transaction() as connection:
            row = self._row(connection, run_id, scope)
            run = self._run(row, run_id, scope)
            self._runtime_for_run(connection, row, run_id)
            result = connection.execute(
                "SELECT payload, digest FROM build_results WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run.status in _TERMINAL_STATUSES:
                if result is None:
                    raise ValueError("terminal_build_result_missing")
                self._result(result, run, row)
            elif result is not None:
                raise ValueError("nonterminal_build_has_result")
            return run

    def cancel(self, run_id: str, scope: GraphScope) -> None:
        """Idempotently prevent new work without deleting any completed effects."""
        revision = self.gate.revision
        with self._storage.transaction() as connection:
            with self.gate.guard(revision):
                run = self._run(self._row(connection, run_id, scope), run_id, scope)
                if run.status in _TERMINAL_STATUSES or run.cancelled:
                    return
                connection.execute(
                    "UPDATE build_runs SET cancelled = 1 WHERE run_id = ?", (run_id,)
                )

    @staticmethod
    def _validate_runtime(payload: bytes) -> None:
        value = _json(payload, "invalid_build_runtime")
        if type(value) is not dict:
            raise ValueError("invalid_build_runtime")

    @classmethod
    def _runtime(cls, row: sqlite3.Row, expected_digest: str) -> bytes:
        payload, digest = row["payload"], row["digest"]
        if type(payload) is not bytes:
            raise ValueError("build_runtime_integrity_failure")
        _digest(digest)
        if digest != expected_digest or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("build_runtime_integrity_failure")
        cls._validate_runtime(payload)
        return payload

    @classmethod
    def _runtime_for_run(
        cls, connection: sqlite3.Connection, build_row: sqlite3.Row, run_id: str
    ) -> bytes | None:
        row = connection.execute(
            "SELECT payload, digest FROM build_runtime WHERE run_id = ?", (run_id,)
        ).fetchone()
        expected = build_row["runtime_digest"]
        if (row is None) != (expected is None):
            raise ValueError("build_runtime_integrity_failure")
        return None if row is None else cls._runtime(row, expected)

    def runtime(self, run_id: str, scope: GraphScope) -> bytes | None:
        """Return the exact authorized adapter/policy identity pinned at start."""
        with self._storage.transaction() as connection:
            build_row = self._row(connection, run_id, scope)
            self._run(build_row, run_id, scope)
            return self._runtime_for_run(connection, build_row, run_id)

    def pin_runtime(self, run_id: str, scope: GraphScope, payload: bytes) -> None:
        """Compatibly pin runtime identity before effects; exact repeats are safe."""
        self._validate_runtime(payload)
        revision = self.gate.revision
        with self._storage.transaction() as connection:
            with self.gate.guard(revision):
                build_row = self._row(connection, run_id, scope)
                run = self._run(build_row, run_id, scope)
                row = connection.execute(
                    "SELECT payload, digest FROM build_runtime WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is not None:
                    expected = build_row["runtime_digest"]
                    if expected is None or self._runtime(row, expected) != payload:
                        raise ValueError("build_runtime_conflict")
                    return
                if (
                    run.status in _TERMINAL_STATUSES
                    or run.cancelled
                    or self._stage_rows(connection, run_id)
                ):
                    raise ValueError("build_runtime_must_precede_effects")
                connection.execute(
                    "INSERT INTO build_runtime VALUES (?, ?, ?)",
                    (run_id, payload, hashlib.sha256(payload).hexdigest()),
                )
                runtime_digest = hashlib.sha256(payload).hexdigest()
                pin_digest = _pin_digest(
                    run_id,
                    scope,
                    build_row["request_digest"],
                    build_row["profile_digest"],
                    build_row["input_digest"],
                    build_row["baseline_digest"],
                    build_row["deadline_at"],
                    runtime_digest,
                )
                connection.execute(
                    "UPDATE build_runs SET runtime_digest = ?, pin_digest = ? WHERE run_id = ?",
                    (runtime_digest, pin_digest, run_id),
                )

    @staticmethod
    def _stage_rows(connection: sqlite3.Connection, run_id: str) -> tuple[tuple[str, str], ...]:
        raw = connection.execute(
            "SELECT stage, artifact_ref, record_digest FROM build_stages WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        rows: dict[str, str] = {}
        for row in raw:
            stage, artifact_ref = row["stage"], row["artifact_ref"]
            if type(stage) is not str or stage not in stage_dependencies():
                raise ValueError("unknown_persisted_stage")
            if stage in rows:
                raise ValueError("duplicate_persisted_stage")
            _text(artifact_ref)
            _digest(row["record_digest"])
            if row["record_digest"] != _stage_digest(run_id, stage, artifact_ref):
                raise ValueError("persisted_stage_integrity_failure")
            rows[stage] = artifact_ref
        ordered = tuple(stage_dependencies())
        present = tuple(stage for stage in ordered if stage in rows)
        if present != ordered[: len(present)]:
            raise ValueError("persisted_stages_are_not_a_workflow_prefix")
        for stage in present:
            if not set(stage_dependencies()[stage]) <= rows.keys():
                raise ValueError("persisted_stage_dependencies_missing")
        return tuple((stage, rows[stage]) for stage in present)

    def stages(self, run_id: str, scope: GraphScope) -> tuple[tuple[str, str], ...]:
        """Return committed stage records in the fixed dependency order."""
        with self._storage.transaction() as connection:
            build_row = self._row(connection, run_id, scope)
            self._run(build_row, run_id, scope)
            self._runtime_for_run(connection, build_row, run_id)
            return self._stage_rows(connection, run_id)

    def record_stage(self, run_id: str, scope: GraphScope, stage: str, artifact_ref: str) -> None:
        """Persist ordered exact completion; never replace another stage output."""
        if stage not in stage_dependencies():
            raise ValueError("unknown_build_stage")
        _text(artifact_ref)
        revision = self.gate.revision
        with self._storage.transaction() as connection:
            with self.gate.guard(revision):
                build_row = self._row(connection, run_id, scope)
                run = self._run(build_row, run_id, scope)
                existing = dict(self._stage_rows(connection, run_id))
                if stage in existing:
                    if existing[stage] != artifact_ref:
                        raise ValueError("committed_stage_conflict")
                    return
                if run.status in _TERMINAL_STATUSES:
                    raise ValueError("terminal_build_is_immutable")
                if run.cancelled:
                    raise InterruptedError("build_cancelled")
                if self._runtime_for_run(connection, build_row, run_id) is None:
                    raise ValueError("build_runtime_not_pinned")
                ordered = tuple(stage_dependencies())
                if stage != ordered[len(existing)]:
                    raise ValueError("stage_dependencies_not_committed")
                connection.execute(
                    "INSERT INTO build_stages VALUES (?, ?, ?, ?)",
                    (run_id, stage, artifact_ref, _stage_digest(run_id, stage, artifact_ref)),
                )
                connection.execute(
                    "UPDATE build_runs SET status = 'running' WHERE run_id = ?",
                    (run_id,),
                )

    @staticmethod
    def _result(row: sqlite3.Row, run: BuildRun, build_row: sqlite3.Row) -> bytes:
        payload, digest = row["payload"], row["digest"]
        if type(payload) is not bytes:
            raise ValueError("build_result_integrity_failure")
        _digest(digest)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("build_result_integrity_failure")
        if run.status not in _TERMINAL_STATUSES:
            raise ValueError("nonterminal_build_has_result")
        _result_identity(payload, run.run_id, run.status)
        expected = _terminal_digest(
            build_row["pin_digest"],
            build_row["terminal_prior_status"],
            bool(build_row["terminal_prior_cancelled"]),
            digest,
            run.status,
            run.cancelled,
        )
        if build_row["terminal_digest"] != expected:
            raise ValueError("build_terminal_commitment_failure")
        return payload

    def result(self, run_id: str, scope: GraphScope) -> bytes | None:
        """Read a terminal result after verifying immutable result bytes."""
        with self._storage.transaction() as connection:
            build_row = self._row(connection, run_id, scope)
            run = self._run(build_row, run_id, scope)
            self._runtime_for_run(connection, build_row, run_id)
            row = connection.execute(
                "SELECT payload, digest FROM build_results WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                if run.status in _TERMINAL_STATUSES:
                    raise ValueError("terminal_build_result_missing")
                return None
            return self._result(row, run, build_row)

    def finish(
        self,
        run_id: str,
        scope: GraphScope,
        status: str,
        payload: bytes,
        *,
        expected_revision: int,
        evidence_refs: tuple[str, ...] = (),
    ) -> None:
        """Retain a terminal result atomically; exact acknowledged repeats are safe."""
        if type(status) is not str or status not in _TERMINAL_STATUSES:
            raise ValueError("invalid_build_status")
        _run_id(run_id)
        _result_identity(payload, run_id, status)
        if type(evidence_refs) is not tuple or any(
            type(ref) is not str or not ref.strip() for ref in evidence_refs
        ):
            raise ValueError("invalid_terminal_evidence_refs")
        revision = expected_revision
        if type(revision) is not int or revision < 0:
            raise ValueError("invalid_expected_policy_revision")
        with self._storage.transaction() as connection:
            with self.gate.guard(revision):
                self.gate.require_scope(scope)
                self.gate.require_access(evidence_refs, scope)
                if self.gate.revision != revision:
                    raise PermissionError("build_policy_revision_changed")
                build_row = self._row(connection, run_id, scope)
                run = self._run(build_row, run_id, scope)
                if self._runtime_for_run(connection, build_row, run_id) is None:
                    raise ValueError("build_runtime_not_pinned")
                row = connection.execute(
                    "SELECT payload, digest FROM build_results WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is not None:
                    existing = self._result(row, run, build_row)
                    if run.status != status or existing != payload:
                        raise ValueError("build_result_conflict")
                    return
                if run.status in _TERMINAL_STATUSES:
                    raise ValueError("terminal_build_result_missing")
                if run.cancelled and status != "cancelled":
                    raise InterruptedError("build_cancelled")
                result_digest = hashlib.sha256(payload).hexdigest()
                final_cancelled = status == "cancelled"
                terminal_digest = _terminal_digest(
                    build_row["pin_digest"],
                    run.status,
                    run.cancelled,
                    result_digest,
                    status,
                    final_cancelled,
                )
                connection.execute(
                    "INSERT INTO build_results VALUES (?, ?, ?)",
                    (run_id, payload, result_digest),
                )
                connection.execute(
                    "UPDATE build_runs SET status = ?, cancelled = ?, "
                    "terminal_prior_status = ?, terminal_prior_cancelled = ?, "
                    "terminal_digest = ? WHERE run_id = ?",
                    (
                        status,
                        int(final_cancelled),
                        run.status,
                        int(run.cancelled),
                        terminal_digest,
                        run_id,
                    ),
                )
                self.gate.require_access(evidence_refs, scope)
                if self.gate.revision != revision:
                    raise PermissionError("build_policy_revision_changed")
