"""Administrative plan, distinct human review and single reserved execution (R3.03).

A plan binds the exact target Tool/version, canonical input and the host's current
policy, model, egress and pointer pins. A review is accepted only from a verified
human on the administrative audience with ``approval:review``, recent
multi-factor authentication, and a different person than the requester
(issuer, tenant and subject, independent of the client application). Execution
re-validates the pins, then reserves the approval, the idempotency key and the
operation in one control-store transaction. Approvals move through
pending -> approved -> reserved -> consumed or to void, and are never recycled.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from capicove.contracts.remote import ApprovalRef, AuthorityKey, OperationRef
from capicove.runtime.remote_operations import OperationError, OperationService
from capicove.runtime.remote_recovery import TombstoneJournal


REVIEW_WINDOW_SECONDS = 24 * 3600
MFA_MAX_AGE_SECONDS = 300
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS approvals (approval_id TEXT PRIMARY KEY,"
    " requester TEXT NOT NULL, requester_person TEXT NOT NULL, tool_id TEXT NOT NULL,"
    " tool_version INTEGER NOT NULL, arguments TEXT NOT NULL, pins TEXT NOT NULL,"
    " plan_sha256 TEXT NOT NULL, state TEXT NOT NULL, reviewer TEXT, created REAL NOT NULL,"
    " expires REAL NOT NULL, idempotency_key TEXT, operation_id TEXT, executor TEXT)",
    "CREATE TABLE IF NOT EXISTS approval_history (sequence INTEGER PRIMARY KEY AUTOINCREMENT,"
    " approval_id TEXT NOT NULL, state TEXT NOT NULL, at REAL NOT NULL, actor TEXT)",
)


@dataclass(frozen=True, slots=True)
class ReviewerClaims:
    """Verified token facts about the reviewer's authentication, never request input."""

    auth_time: float
    methods: frozenset[str]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fields(authority: AuthorityKey) -> list[str]:
    key = authority.principal_key
    return [
        key.issuer,
        key.tenant_id,
        key.application_id,
        key.subject_kind,
        key.subject,
        authority.workspace_binding_id,
        authority.grant_id,
        authority.audience,
    ]


def _person(authority: AuthorityKey) -> str:
    key = authority.principal_key
    return _canonical([key.issuer, key.tenant_id, key.subject])


class ApprovalService:
    """Plan/review/execute over the same control store as operation admission."""

    def __init__(
        self,
        operations: OperationService,
        *,
        admin_audience: str,
        current_pins: Callable[[], Mapping[str, str]],
        clock: Callable[[], float] = time.time,
        tombstones: TombstoneJournal | None = None,
    ) -> None:
        self._operations = operations
        self._admin_audience = admin_audience
        self._pins = current_pins
        self._clock = clock
        self._tombstones = tombstones
        with operations.transaction() as connection:
            for statement in _SCHEMA:
                connection.execute(statement)
            if tombstones is not None:
                # Restoring an older control store must not revive voided plans.
                for approval_id in sorted(tombstones.entries("approval")):
                    row = connection.execute(
                        "SELECT state FROM approvals WHERE approval_id = ?", (approval_id,)
                    ).fetchone()
                    if row is not None and row[0] in {"pending", "approved"}:
                        self._set(connection, approval_id, "void", "restore-gate")

    def void(self, approval_id: str, *, reason: str) -> ApprovalRef:
        """Irreversibly void an unexecuted plan; executed plans keep their history."""
        if not isinstance(reason, str) or not reason.strip():
            raise OperationError("invalid_input", "a void reason is required")
        with self._operations.transaction() as connection:
            row = self._row(connection, approval_id)
            if row["state"] in {"reserved", "consumed"}:
                raise OperationError("conflict", "an executed plan cannot be voided")
            if self._tombstones is not None:
                self._tombstones.record("approval", approval_id)
            if row["state"] != "void":
                self._set(connection, approval_id, "void", "operator: " + reason[:200])
            return self._reference({**row, "state": "void"})

    def _record(self, connection: Any, approval_id: str, state: str, actor: str | None) -> None:
        connection.execute(
            "INSERT INTO approval_history(approval_id, state, at, actor) VALUES (?, ?, ?, ?)",
            (approval_id, state, self._clock(), actor),
        )

    def _set(self, connection: Any, approval_id: str, state: str, actor: str | None) -> None:
        connection.execute(
            "UPDATE approvals SET state = ? WHERE approval_id = ?", (state, approval_id)
        )
        self._record(connection, approval_id, state, actor)

    def _row(self, connection: Any, approval_id: str) -> dict[str, Any]:
        cursor = connection.execute(
            "SELECT * FROM approvals WHERE approval_id = ?",
            (approval_id if isinstance(approval_id, str) else "",),
        )
        row = cursor.fetchone()
        if row is None:
            raise OperationError("not_found")
        return dict(zip((column[0] for column in cursor.description), row, strict=True))

    @staticmethod
    def _digest(tool_id: str, version: int, arguments: str, pins: str, requester: str) -> str:
        return hashlib.sha256(
            _canonical([tool_id, version, arguments, pins, requester]).encode()
        ).hexdigest()

    def _reference(self, row: Mapping[str, Any]) -> ApprovalRef:
        return ApprovalRef(
            row["approval_id"],
            row["state"],
            row["plan_sha256"],
            row["reviewer"],
            row["operation_id"],
        )

    def plan(
        self,
        authority: AuthorityKey,
        permissions: frozenset[str],
        tool_id: str,
        tool_version: int,
        arguments: Mapping[str, Any],
    ) -> ApprovalRef:
        """Validate a target exactly as admission would and record a pending plan."""
        admission = self._operations.prepare(
            authority, permissions, tool_id, tool_version, arguments, "plan"
        )
        pins = _canonical(dict(self._pins()))
        requester = _canonical(_fields(authority))
        digest = self._digest(tool_id, tool_version, admission.canonical, pins, requester)
        approval_id = "approval:" + secrets.token_hex(16)
        now = self._clock()
        with self._operations.transaction() as connection:
            connection.execute(
                "INSERT INTO approvals(approval_id, requester, requester_person, tool_id,"
                " tool_version, arguments, pins, plan_sha256, state, created, expires)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    approval_id,
                    requester,
                    _person(authority),
                    tool_id,
                    tool_version,
                    admission.canonical,
                    pins,
                    digest,
                    now,
                    now + REVIEW_WINDOW_SECONDS,
                ),
            )
            self._record(connection, approval_id, "pending", authority.principal_key.subject)
        return ApprovalRef(approval_id, "pending", digest, None, None)

    def review(
        self,
        reviewer: AuthorityKey,
        permissions: frozenset[str],
        claims: ReviewerClaims,
        approval_id: str,
        plan_sha256: str,
        *,
        approve: bool,
    ) -> ApprovalRef:
        """Accept a decision only from a distinct, freshly authenticated human."""
        now = self._clock()
        if (
            not isinstance(reviewer, AuthorityKey)
            or reviewer.audience != self._admin_audience
            or "approval:review" not in permissions
            or reviewer.principal_key.subject_kind != "human"
        ):
            raise OperationError("permission_denied")
        if (
            not isinstance(claims, ReviewerClaims)
            or "mfa" not in claims.methods
            or not now - MFA_MAX_AGE_SECONDS <= claims.auth_time <= now
        ):
            raise OperationError("recent_mfa_required")
        with self._operations.transaction() as connection:
            row = self._row(connection, approval_id)
            if row["requester_person"] == _person(reviewer):
                raise OperationError("self_approval")
            if row["state"] != "pending":
                raise OperationError("conflict", "plan is not pending")
            if now >= row["expires"]:
                raise OperationError("approval_expired")
            if plan_sha256 != row["plan_sha256"]:
                raise OperationError("approval_stale", "reviewed plan digest differs")
            if _canonical(dict(self._pins())) != row["pins"]:
                self._set(connection, approval_id, "void", reviewer.principal_key.subject)
                return self._reference({**row, "state": "void"})
            subject = reviewer.principal_key.subject
            if approve:
                connection.execute(
                    "UPDATE approvals SET reviewer = ? WHERE approval_id = ?",
                    (subject, approval_id),
                )
                self._set(connection, approval_id, "approved", subject)
                return self._reference({**row, "state": "approved", "reviewer": subject})
            self._set(connection, approval_id, "void", subject)
            return self._reference({**row, "state": "void"})

    def execute(
        self,
        executor: AuthorityKey,
        permissions: frozenset[str],
        approval_id: str,
        idempotency_key: str,
    ) -> OperationRef:
        """Reserve the approval, idempotency key and operation in one transaction."""
        with self._operations.transaction() as connection:
            row = self._row(connection, approval_id)
        executor_fields = _canonical(_fields(executor))
        if executor_fields != row["requester"] and executor.principal_key.subject_kind != "service":
            raise OperationError("permission_denied")
        admission = self._operations.prepare(
            executor,
            permissions,
            row["tool_id"],
            row["tool_version"],
            json.loads(row["arguments"]),
            idempotency_key,
        )
        with self._operations.transaction() as connection:
            row = self._row(connection, approval_id)
            if row["state"] in {"reserved", "consumed"}:
                if row["idempotency_key"] == idempotency_key and row["executor"] == executor_fields:
                    return self._operations.admit(connection, admission)
                raise OperationError("approval_consumed")
            if row["state"] != "approved":
                raise OperationError("conflict", "plan is not approved")
            if self._clock() >= row["expires"] or _canonical(dict(self._pins())) != row["pins"]:
                self._set(connection, approval_id, "void", executor.principal_key.subject)
                stale = True
            else:
                stale = False
                connection.execute(
                    "UPDATE approvals SET idempotency_key = ?, executor = ? WHERE approval_id = ?",
                    (idempotency_key, executor_fields, approval_id),
                )
                self._set(connection, approval_id, "reserved", executor.principal_key.subject)
                reference = self._operations.admit(connection, admission)
                connection.execute(
                    "UPDATE approvals SET operation_id = ? WHERE approval_id = ?",
                    (reference.operation_id, approval_id),
                )
                self._set(connection, approval_id, "consumed", executor.principal_key.subject)
        if stale:
            raise OperationError("approval_stale")
        return reference

    def status(self, approval_id: str) -> ApprovalRef:
        """Return the current approval state for operator inspection."""
        with self._operations.transaction() as connection:
            return self._reference(self._row(connection, approval_id))

    def history(self, approval_id: str) -> list[dict[str, Any]]:
        """Return the append-only approval history."""
        with self._operations.transaction() as connection:
            rows = connection.execute(
                "SELECT state, at, actor FROM approval_history WHERE approval_id = ?"
                " ORDER BY sequence",
                (approval_id,),
            ).fetchall()
        return [{"state": state, "at": at, "actor": actor} for state, at, actor in rows]
