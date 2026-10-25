"""Durable remote operations: idempotent admission, fenced leases and recovery (R3.02).

Implements the RFC-0028 section 9 state machine over the workspace control store.
Admission reserves the ``(AuthorityKey, tool, version, kind, idempotency key)``
namespace together with the operation and an audit outbox row in one transaction.
Same key and same canonical input returns the same operation; changed input is a
stable conflict. Only registered targets are admissible; operation, approval,
policy and remote-control Tools, unknown Tools and audience-mismatched admin
targets are reported as ``unavailable``. Workers claim queued work with an
expiring lease and a fencing token, the live authority is rechecked at start,
and a stale fence can never publish. Recovery marks every expired lease
``outcome_unknown``: it never replays work that may have started, and only an
explicit reconciliation appends the resolved terminal outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from capicove.contracts.remote import AuthorityKey, OperationRef, PrincipalKey
from capicove.runtime._remote.control_store import ControlStore
from capicove.runtime.remote_audit import AuditLog


MAX_ARGUMENT_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
_CONTROL_PREFIXES = (
    "capicove.operation.",
    "capicove.approval.",
    "capicove.policy.",
    "capicove.remote.",
)
_TERMINAL = frozenset({"succeeded", "failed", "cancelled", "timed_out"})
_SCHEMA_KEYS = frozenset(
    {"type", "required", "properties", "additionalProperties", "maxLength", "minimum", "maximum"}
)


class OperationError(RuntimeError):
    """A typed admission or lifecycle failure with a stable public ``code``."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class OperationTarget:
    """One registered bounded operation and its closed input schema."""

    tool_id: str
    tool_version: int
    kind: str
    input_schema: Mapping[str, Any]
    permissions: frozenset[str]
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    admin: bool = False
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if self.tool_id.startswith(_CONTROL_PREFIXES):
            raise ValueError("control-plane Tools cannot be operation targets")
        _check_schema(self.input_schema)


@dataclass(frozen=True)
class Admission:
    """A fully validated, not yet reserved admission request."""

    authority: AuthorityKey
    target: OperationTarget
    idempotency_key: str
    canonical: str = field(repr=False)
    input_sha256: str
    ceiling: frozenset[str]


@dataclass(frozen=True)
class Claim:
    """A leased execution attempt bound to its fencing token."""

    operation_id: str
    fence: int
    tool_id: str
    tool_version: int
    arguments: Mapping[str, Any] = field(repr=False)


def _check_schema(schema: Mapping[str, Any]) -> None:
    if not isinstance(schema, Mapping) or not set(schema) <= _SCHEMA_KEYS:
        raise ValueError("operation schemas use a closed keyword subset")
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError("object schemas must be closed")
        for child in schema.get("properties", {}).values():
            _check_schema(child)


def _validate(value: Any, schema: Mapping[str, Any]) -> None:
    kind = schema.get("type")
    checks: dict[str, Callable[[Any], bool]] = {
        "object": lambda item: isinstance(item, dict),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "number": lambda item: type(item) in {int, float} and math.isfinite(item),
        "boolean": lambda item: type(item) is bool,
    }
    if kind not in checks or not checks[kind](value):
        raise OperationError("invalid_input", "argument type does not match the schema")
    if kind == "object":
        properties = schema.get("properties", {})
        if not set(value) <= set(properties) or not set(schema.get("required", ())) <= set(value):
            raise OperationError("invalid_input", "arguments must match the closed schema")
        for name, item in value.items():
            _validate(item, properties[name])
    if kind == "string" and len(value) > schema.get("maxLength", len(value)):
        raise OperationError("invalid_input", "string exceeds its schema bound")
    if kind in {"integer", "number"} and not (
        schema.get("minimum", value) <= value <= schema.get("maximum", value)
    ):
        raise OperationError("invalid_input", "number outside its schema bound")


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise OperationError("invalid_input", "arguments are not canonical JSON") from error


def _authority_fields(authority: AuthorityKey) -> list[str]:
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


def _digest(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical(list(values)).encode()).hexdigest()


def _authority_from(raw: str) -> AuthorityKey:
    values = json.loads(raw)
    return AuthorityKey(
        PrincipalKey(values[0], values[1], values[2], values[3], values[4]),
        values[5],
        values[6],
        values[7],
    )


class OperationService:
    """Admission, leased execution and recovery over one control store."""

    def __init__(
        self,
        path: Path,
        targets: Sequence[OperationTarget],
        *,
        authorize: Callable[[AuthorityKey], None] | None = None,
        admin_audience: str | None = None,
        max_queued_per_application: int = 20,
        max_active_per_application: int = 2,
        max_queued_per_tenant: int = 100,
        audit: AuditLog | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = ControlStore(path)
        self._targets = {(target.tool_id, target.tool_version): target for target in targets}
        if len(self._targets) != len(targets):
            raise ValueError("duplicate operation target")
        self._authorize = authorize or (lambda _authority: None)
        self._admin_audience = admin_audience
        self._max_queued = max_queued_per_application
        self._max_active = max_active_per_application
        self._max_tenant = max_queued_per_tenant
        self._audit = audit
        self._clock = clock

    def _live(self, authority: AuthorityKey) -> None:
        if not isinstance(authority, AuthorityKey):
            raise OperationError("invalid_input", "authority is required")
        try:
            self._authorize(authority)
        except PermissionError as error:
            raise OperationError("permission_denied") from error

    def _target(self, authority: AuthorityKey, tool_id: object, version: object) -> OperationTarget:
        target = (
            self._targets.get((tool_id, version))
            if isinstance(tool_id, str) and type(version) is int
            else None
        )
        if target is None or str(tool_id).startswith(_CONTROL_PREFIXES):
            raise OperationError("unavailable")
        on_admin_audience = (
            self._admin_audience is not None and authority.audience == self._admin_audience
        )
        if target.admin != on_admin_audience:
            raise OperationError("unavailable")
        return target

    def _record(self, connection: Any, operation_id: str, state: str, reason: str | None) -> None:
        now = self._clock()
        connection.execute(
            "INSERT INTO history(operation_id, state, at, reason) VALUES (?, ?, ?, ?)",
            (operation_id, state, now, reason),
        )
        connection.execute(
            "INSERT INTO outbox(operation_id, event, at) VALUES (?, ?, ?)",
            (operation_id, state, now),
        )

    def _transition(
        self, connection: Any, operation_id: str, before: Sequence[str], after: str, reason: str
    ) -> bool:
        placeholders = ",".join("?" for _ in before)
        changed = connection.execute(
            f"UPDATE operations SET state = ? WHERE operation_id = ? AND state IN ({placeholders})",
            (after, operation_id, *before),
        ).rowcount
        if changed:
            self._record(connection, operation_id, after, reason)
        return bool(changed)

    def prepare(
        self,
        authority: AuthorityKey,
        permissions: frozenset[str],
        tool_id: str,
        tool_version: int,
        arguments: Mapping[str, Any],
        idempotency_key: str,
    ) -> Admission:
        """Validate target, permissions, schema and live authority before any write."""
        target = self._target(authority, tool_id, tool_version)
        if "operation:submit" not in permissions or not target.permissions <= permissions:
            raise OperationError("permission_denied")
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128:
            raise OperationError("invalid_input", "idempotency key must be 1-128 characters")
        _validate(arguments, {"type": "object", **target.input_schema})
        canonical = _canonical(arguments)
        if len(canonical.encode()) > MAX_ARGUMENT_BYTES:
            raise OperationError("invalid_input", "arguments exceed the admission bound")
        self._live(authority)
        return Admission(
            authority,
            target,
            idempotency_key,
            canonical,
            hashlib.sha256(canonical.encode()).hexdigest(),
            frozenset(permissions & target.permissions),
        )

    def transaction(self) -> Any:
        """Open one control-store transaction for an atomic composite admission."""
        return self._store.transaction()

    def admit(self, connection: Any, admission: Admission) -> OperationRef:
        """Reserve the idempotency namespace and the operation inside ``connection``."""
        target = admission.target
        fields = _authority_fields(admission.authority)
        owner = _digest(fields)
        application = _digest(fields[:3])
        tenant = _digest(fields[:2])
        if self._draining(connection):
            raise OperationError("draining")
        existing = connection.execute(
            "SELECT operation_id, state, input_sha256 FROM operations WHERE owner = ?"
            " AND tool_id = ? AND tool_version = ? AND kind = ? AND idempotency_key = ?",
            (owner, target.tool_id, target.tool_version, target.kind, admission.idempotency_key),
        ).fetchone()
        if existing is not None:
            if existing[2] != admission.input_sha256:
                raise OperationError("idempotency_conflict")
            return OperationRef(
                existing[0], existing[1], target.tool_id, target.tool_version, existing[2]
            )
        (queued,) = connection.execute(
            "SELECT COUNT(*) FROM operations WHERE application = ?"
            " AND state IN ('accepted', 'queued')",
            (application,),
        ).fetchone()
        if queued >= self._max_queued:
            raise OperationError("quota_exceeded", "application queue is full")
        (tenant_queued,) = connection.execute(
            "SELECT COUNT(*) FROM operations WHERE tenant = ? AND state IN ('accepted', 'queued')",
            (tenant,),
        ).fetchone()
        if tenant_queued >= self._max_tenant:
            raise OperationError("quota_exceeded", "tenant queue is full")
        operation_id = "operation:" + secrets.token_hex(16)
        connection.execute(
            "INSERT INTO operations(operation_id, owner, application, tenant, authority, tool_id,"
            " tool_version, kind, idempotency_key, input_sha256, arguments, ceiling, state,"
            " created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted', ?)",
            (
                operation_id,
                owner,
                application,
                tenant,
                _canonical(fields),
                target.tool_id,
                target.tool_version,
                target.kind,
                admission.idempotency_key,
                admission.input_sha256,
                admission.canonical,
                _canonical(sorted(admission.ceiling)),
                self._clock(),
            ),
        )
        self._record(connection, operation_id, "accepted", None)
        self._transition(connection, operation_id, ("accepted",), "queued", "admitted")
        return OperationRef(
            operation_id, "queued", target.tool_id, target.tool_version, admission.input_sha256
        )

    def submit(
        self,
        authority: AuthorityKey,
        permissions: frozenset[str],
        tool_id: str,
        tool_version: int,
        arguments: Mapping[str, Any],
        idempotency_key: str,
    ) -> OperationRef:
        """Admit one operation atomically, or return the identical prior admission."""
        admission = self.prepare(
            authority, permissions, tool_id, tool_version, arguments, idempotency_key
        )
        if admission.target.requires_approval:
            raise OperationError("approval_required")
        self.require_audit()
        with self._store.transaction() as connection:
            reference = self.admit(connection, admission)
        self._try_drain()
        return reference

    def require_audit(self) -> None:
        """Refuse mutations unless the audit backlog can be exported right now."""
        if self._audit is None:
            return
        try:
            self.drain_audit()
        except OSError as error:
            raise OperationError("audit_unavailable") from error
        if not self._audit.writable():
            raise OperationError("audit_unavailable")

    def _try_drain(self) -> None:
        if self._audit is not None:
            try:
                self.drain_audit()
            except OSError:
                # The transactional outbox keeps the events; the next mutation refuses.
                pass

    def drain_audit(self) -> int:
        """Export undrained outbox events to the audit chain in order."""
        if self._audit is None:
            return 0
        exported = 0
        with self._store.transaction() as connection:
            rows = connection.execute(
                "SELECT sequence, operation_id, event, at FROM outbox WHERE drained = 0"
                " ORDER BY sequence"
            ).fetchall()
            for sequence, operation_id, event, at in rows:
                self._audit.append(
                    {"outbox": sequence, "operation_id": operation_id, "event": event, "at": at}
                )
                connection.execute("UPDATE outbox SET drained = 1 WHERE sequence = ?", (sequence,))
                exported += 1
        return exported

    def _draining(self, connection: Any) -> bool:
        row = connection.execute("SELECT value FROM settings WHERE name = 'draining'").fetchone()
        return row is not None and row[0] == "1"

    def drain(self, *, timeout_seconds: float = 60.0) -> dict[str, int]:
        """Stop admitting and starting work, then wait up to the budget for leases."""
        with self._store.transaction() as connection:
            connection.execute(
                "INSERT INTO settings(name, value) VALUES ('draining', '1')"
                " ON CONFLICT(name) DO UPDATE SET value = '1'"
            )
        deadline = time.monotonic() + timeout_seconds
        while True:
            with self._store.transaction() as connection:
                counts = dict(
                    connection.execute(
                        "SELECT state, COUNT(*) FROM operations WHERE state IN"
                        " ('running', 'cancellation_requested', 'queued') GROUP BY state"
                    ).fetchall()
                )
            active = counts.get("running", 0) + counts.get("cancellation_requested", 0)
            if active == 0 or time.monotonic() >= deadline:
                return {"active": active, "queued": counts.get("queued", 0)}
            time.sleep(0.05)

    def resume_admission(self) -> None:
        """Leave drain mode after recovery."""
        with self._store.transaction() as connection:
            connection.execute("DELETE FROM settings WHERE name = 'draining'")

    def claim(self, worker: str, *, lease_seconds: float) -> Claim | None:
        """Lease the oldest startable queued operation; revoked work is cancelled."""
        now = self._clock()
        with self._store.transaction() as connection:
            if self._draining(connection):
                return None
            rows = connection.execute(
                "SELECT operation_id, application, authority, tool_id, tool_version, arguments,"
                " fence FROM operations WHERE state = 'queued' ORDER BY rowid"
            ).fetchall()
            for operation_id, application, authority, tool_id, version, arguments, fence in rows:
                (active,) = connection.execute(
                    "SELECT COUNT(*) FROM operations WHERE application = ?"
                    " AND state IN ('running', 'cancellation_requested')",
                    (application,),
                ).fetchone()
                if active >= self._max_active:
                    continue
                try:
                    self._authorize(_authority_from(authority))
                except PermissionError:
                    self._transition(
                        connection, operation_id, ("queued",), "cancelled", "grant revoked"
                    )
                    continue
                connection.execute(
                    "UPDATE operations SET state = 'running', fence = ?, worker = ?,"
                    " lease_until = ? WHERE operation_id = ? AND state = 'queued'",
                    (fence + 1, worker, now + lease_seconds, operation_id),
                )
                self._record(connection, operation_id, "running", worker)
                return Claim(operation_id, fence + 1, tool_id, version, json.loads(arguments))
        return None

    def execute(self, claim: Claim) -> dict[str, Any]:
        """Run the registered handler for a claim; callers isolate it as required."""
        target = self._targets[(claim.tool_id, claim.tool_version)]
        return dict(target.handler(claim.arguments))

    def complete(self, operation_id: str, fence: int, output: Mapping[str, Any]) -> OperationRef:
        """Publish a terminal success only under the current fence and live lease."""
        encoded = _canonical(dict(output))
        if len(encoded.encode()) > MAX_OUTPUT_BYTES:
            raise OperationError("output_too_large")
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT state, fence, lease_until, tool_id, tool_version, input_sha256"
                " FROM operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationError("not_found")
            state, current, lease_until, tool_id, version, digest = row
            if (
                state not in {"running", "cancellation_requested"}
                or current != fence
                or lease_until is None
                or lease_until <= self._clock()
            ):
                raise OperationError("stale_fence")
            connection.execute(
                "UPDATE operations SET output = ? WHERE operation_id = ?", (encoded, operation_id)
            )
            self._transition(connection, operation_id, (state,), "succeeded", "completed")
        return OperationRef(operation_id, "succeeded", tool_id, version, digest)

    def fail(self, operation_id: str, fence: int, reason: str) -> OperationRef:
        """Record a known failure under the current fence."""
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT state, fence, tool_id, tool_version, input_sha256 FROM operations"
                " WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationError("not_found")
            if row[0] not in {"running", "cancellation_requested"} or row[1] != fence:
                raise OperationError("stale_fence")
            self._transition(connection, operation_id, (row[0],), "failed", reason[:512])
        return OperationRef(operation_id, "failed", row[2], row[3], row[4])

    def mark_unknown(self, operation_id: str, fence: int, reason: str) -> OperationRef:
        """Record that a leased attempt ended without a trustworthy outcome."""
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT state, fence, tool_id, tool_version, input_sha256 FROM operations"
                " WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationError("not_found")
            if row[0] not in {"running", "cancellation_requested"} or row[1] != fence:
                raise OperationError("stale_fence")
            self._transition(connection, operation_id, (row[0],), "outcome_unknown", reason[:512])
        return OperationRef(operation_id, "outcome_unknown", row[2], row[3], row[4])

    def recover(self) -> dict[str, int]:
        """Requeue never-started work; expired leases become ``outcome_unknown``."""
        now = self._clock()
        with self._store.transaction() as connection:
            accepted = connection.execute(
                "SELECT operation_id FROM operations WHERE state = 'accepted'"
            ).fetchall()
            for (operation_id,) in accepted:
                self._transition(connection, operation_id, ("accepted",), "queued", "recovered")
            expired = connection.execute(
                "SELECT operation_id, state FROM operations WHERE state IN"
                " ('running', 'cancellation_requested') AND lease_until <= ?",
                (now,),
            ).fetchall()
            for operation_id, state in expired:
                self._transition(
                    connection, operation_id, (state,), "outcome_unknown", "lease expired"
                )
        return {"requeued": len(accepted), "outcome_unknown": len(expired)}

    def _owned(
        self, connection: Any, authority: AuthorityKey, operation_id: str
    ) -> tuple[Any, ...]:
        row = connection.execute(
            "SELECT operation_id, state, tool_id, tool_version, input_sha256, output"
            " FROM operations WHERE operation_id = ? AND owner = ?",
            (operation_id, _digest(_authority_fields(authority))),
        ).fetchone()
        if row is None:
            raise OperationError("not_found")
        return cast(tuple[Any, ...], row)

    def status(
        self, authority: AuthorityKey, permissions: frozenset[str], operation_id: str
    ) -> OperationRef:
        """Return the owner's current state after a fresh authorization."""
        self._live(authority)
        if "operation:submit" not in permissions:
            raise OperationError("permission_denied")
        with self._store.transaction() as connection:
            row = self._owned(connection, authority, operation_id)
        return OperationRef(row[0], row[1], row[2], row[3], row[4])

    def result(
        self, authority: AuthorityKey, permissions: frozenset[str], operation_id: str
    ) -> dict[str, Any]:
        """Return the terminal output only to the live owner."""
        reference = self.status(authority, permissions, operation_id)
        with self._store.transaction() as connection:
            row = self._owned(connection, authority, operation_id)
        output = json.loads(row[5]) if reference.state == "succeeded" and row[5] else None
        return {"state": reference.state, "output": output}

    def cancel(
        self, authority: AuthorityKey, permissions: frozenset[str], operation_id: str
    ) -> OperationRef:
        """Cancel queued work, request cancellation of running work, report terminals."""
        self.status(authority, permissions, operation_id)
        with self._store.transaction() as connection:
            row = self._owned(connection, authority, operation_id)
            state = row[1]
            if state in {"accepted", "queued"}:
                self._transition(connection, operation_id, (state,), "cancelled", "owner")
                state = "cancelled"
            elif state == "running":
                self._transition(
                    connection, operation_id, ("running",), "cancellation_requested", "owner"
                )
                state = "cancellation_requested"
        # Cancellation never waits on audit export; the outbox keeps the event.
        self._try_drain()
        return OperationRef(row[0], state, row[2], row[3], row[4])

    def reconcile(
        self, operation_id: str, state: str, output: Mapping[str, Any] | None, *, evidence: str
    ) -> OperationRef:
        """Append an operator-verified terminal outcome to an unknown operation."""
        if state not in _TERMINAL or not isinstance(evidence, str) or not evidence.strip():
            raise OperationError(
                "invalid_input", "reconciliation needs a terminal state and evidence"
            )
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT state, tool_id, tool_version, input_sha256 FROM operations"
                " WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if row is None:
                raise OperationError("not_found")
            if row[0] != "outcome_unknown":
                raise OperationError("conflict", "only unknown outcomes are reconciled")
            if output is not None:
                connection.execute(
                    "UPDATE operations SET output = ? WHERE operation_id = ?",
                    (_canonical(dict(output)), operation_id),
                )
            self._transition(
                connection,
                operation_id,
                ("outcome_unknown",),
                state,
                "reconciled: " + evidence[:400],
            )
        return OperationRef(operation_id, state, row[1], row[2], row[3])  # type: ignore[arg-type]

    def history(self, operation_id: str) -> list[dict[str, Any]]:
        """Return the append-only state history for operator inspection."""
        with self._store.transaction() as connection:
            rows = connection.execute(
                "SELECT state, at, reason FROM history WHERE operation_id = ? ORDER BY sequence",
                (operation_id,),
            ).fetchall()
        return [{"state": state, "at": at, "reason": reason} for state, at, reason in rows]

    def count(self) -> int:
        """Return the number of admitted operations."""
        with self._store.transaction() as connection:
            (total,) = connection.execute("SELECT COUNT(*) FROM operations").fetchone()
        return int(total)
