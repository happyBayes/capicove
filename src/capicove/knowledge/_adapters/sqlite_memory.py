"""One authoritative WAL ledger and synchronous, replayable projections."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty, LifoQueue
from threading import RLock
from typing import Any
from uuid import uuid4

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import digest, micros
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge._adapters.sqlite_diagnostics import _verify as sqlite_diagnostics__verify
from capicove.knowledge._adapters.sqlite_jobs import claim_job as sqlite_jobs_claim_job
from capicove.knowledge._adapters.sqlite_projection import _project as sqlite_projection__project
from capicove.knowledge._adapters.sqlite_projection import (
    _write_index as sqlite_projection__write_index,
)
from capicove.knowledge._adapters.sqlite_projection import rebuild as sqlite_projection_rebuild
from capicove.knowledge._adapters.sqlite_queries import candidates as sqlite_queries_candidates
from capicove.knowledge._adapters.sqlite_queries import (
    entity_assertions_for_merge as sqlite_queries_entity_assertions_for_merge,
)
from capicove.knowledge._adapters.sqlite_queries import (
    memory_policy_for as sqlite_queries_memory_policy_for,
)
from capicove.knowledge.errors import MemoryError

from .sqlite_codec import _json, backend_errors

# SQL identifiers and clauses in this private adapter come only from closed
# module constants. Caller data is always bound using SQLite parameters.
from .sqlite_schema import _RECORD_TABLES, _SCHEMA, migrate


class SQLiteMemoryRepository:
    """Commit canonical events and projections together; never copy source bodies into events.

    FTS is a derived index. Projection recovery uses stored projection deltas;
    immutable source text is loaded only through the source port during reindexing.
    """

    @backend_errors
    def __init__(self, path: Path) -> None:
        if sqlite3.sqlite_version_info < (3, 42, 0):
            raise MemoryError("sqlite_version_unsupported")
        self.path = path
        if path.is_symlink():
            raise ValueError("unsafe memory path")
        if not path.exists():
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        if path.stat().st_mode & 0o077:
            raise ValueError("memory store must be private")
        self._corrupt = False
        self._lock = RLock()
        self._db = sqlite3.connect(
            self.path, timeout=5, isolation_level=None, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        tables = {
            row[0] for row in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if tables and (
            self._db.execute("PRAGMA application_id").fetchone()[0] != 1128353869
            or "schema_migrations" not in tables
        ):
            self._db.close()
            raise MemoryError("unsupported_schema")
        if tables and [
            row[0]
            for row in self._db.execute("SELECT version FROM schema_migrations ORDER BY version")
        ] not in ([1], [1, 2]):
            self._db.close()
            raise MemoryError("unsupported_schema")
        self._db.execute("PRAGMA application_id=1128353869")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA secure_delete=ON")
        self._data_version = 0
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            migrate(db)
            if sqlite3.sqlite_version_info >= (3, 42, 0):
                for table in (
                    "episode_fts",
                    "fact_fts",
                    "entity_fts",
                    "episode_words",
                    "fact_words",
                ):
                    db.execute(f"INSERT INTO {table}({table},rank) VALUES ('secure-delete',1)")  # nosec B608

        self.verify()
        self._readers: LifoQueue[sqlite3.Connection] = LifoQueue(maxsize=10)
        self._reader_connections: list[sqlite3.Connection] = []
        for _ in range(10):
            reader = sqlite3.connect(
                self.path.absolute().as_uri() + "?mode=ro",
                uri=True,
                timeout=5,
                isolation_level=None,
                check_same_thread=False,
            )
            reader.row_factory = sqlite3.Row
            reader.execute("PRAGMA query_only=ON")
            self._reader_connections.append(reader)
            self._readers.put(reader)

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        """Bounded independent WAL readers; never keep a writer mutex during search."""
        try:
            reader = self._readers.get(timeout=5)
        except Empty as exc:
            raise MemoryError("read_capacity_unavailable") from exc
        try:
            yield reader
        finally:
            if reader.in_transaction:
                reader.rollback()
            self._readers.put(reader)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            yield self._db

    def close(self) -> None:
        with self._lock:
            for reader in self._reader_connections:
                reader.close()
            self._db.close()

    def __del__(self) -> None:
        for reader in getattr(self, "_reader_connections", ()):
            reader.close()
        if hasattr(self, "_db"):
            self._db.close()

    def verify(self) -> bool:
        with self._connection() as db:
            version = db.execute("PRAGMA data_version").fetchone()[0]
            own_transaction = not db.in_transaction
            if own_transaction:
                db.execute("BEGIN")
            try:
                result = self._verify()
                self._data_version = version
                return result
            except (ValueError, TypeError, sqlite3.Error):
                self._corrupt = True
                return False
            finally:
                if own_transaction:
                    db.rollback()

    def _verify(self) -> bool:
        # Stream in ledger order and retain only operation metadata. No projection
        # rebuild, payload decoding, or optional provider is involved in recovery.
        return sqlite_diagnostics__verify(self)

    def doctor(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.verify() else "read_only_safe",
            "write_enabled": not self._corrupt,
            "backend": "sqlite",
            "journal_mode": "wal",
        }

    def _guard(self) -> None:
        with self._connection() as db:
            changed = db.execute("PRAGMA data_version").fetchone()[0] != self._data_version
            if changed:
                self.verify()
        if self._corrupt:
            raise MemoryError("corrupt_event_log")

    @backend_errors
    def head(self, workspace: str) -> tuple[int, datetime]:
        self._guard()
        with self._connection() as db:
            row = db.execute(
                "SELECT revision,recorded FROM workspace_heads WHERE workspace=?", (workspace,)
            ).fetchone()
        now = micros(datetime.now(UTC))
        value = now if row is None else max(now, row["recorded"] + 1)
        return (
            0 if row is None else row["revision"],
            datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=value),
        )

    @backend_errors
    def replay_receipt(self, workspace: str, key: str, request: object) -> dict[str, Any] | None:
        self._guard()
        with self._connection() as db:
            row = db.execute(
                "SELECT request_hash,receipt FROM memory_operations WHERE workspace=? AND key=?",
                (workspace, key),
            ).fetchone()
        if row is None:
            return None
        if row["request_hash"] != digest(request):
            raise MemoryError("write_conflict")
        return json.loads(row["receipt"])  # type: ignore[no-any-return]

    @backend_errors
    def records(
        self, workspace: str, scope: str, kind: str, identity: str | None = None
    ) -> list[dict[str, Any]]:
        self._guard()
        table = _RECORD_TABLES[kind]
        with self._connection() as db:
            if identity is None:
                rows = db.execute(
                    f"SELECT body FROM {table} WHERE workspace=? AND scope=?",  # nosec B608
                    (workspace, scope),
                ).fetchall()
            else:
                rows = db.execute(
                    f"SELECT body FROM {table} WHERE workspace=? AND scope=? AND identity=?",  # nosec B608
                    (workspace, scope, identity),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    @backend_errors
    def facts(
        self,
        workspace: str,
        scope: str,
        *,
        slot: str | None = None,
        fact: str | None = None,
        assertion: str | None = None,
        current: bool = False,
    ) -> list[dict[str, Any]]:
        self._guard()
        terms, values = ["workspace=?", "scope=?"], [workspace, scope]
        if slot is not None:
            terms.append("slot=?")
            values.append(slot)
        if fact is not None:
            terms.append("fact=?")
            values.append(fact)
        if assertion is not None:
            terms.append("assertion=?")
            values.append(assertion)
        if current:
            terms.append("recorded_to IS NULL")
        with self._connection() as db:
            rows = db.execute(
                "SELECT body,recorded_to FROM fact_versions WHERE " + " AND ".join(terms),  # nosec B608
                values,
            ).fetchall()
        result = []
        for row in rows:
            body = json.loads(row[0])
            body["recorded_to"] = row[1]
            with self._connection() as links:
                child = links.execute(
                    "SELECT version FROM fact_versions WHERE workspace=? AND scope=? AND fact=? AND json_extract(body,'$.parent_version_id')=?",
                    (workspace, scope, body["id"], body["version"]),
                ).fetchone()
            body["superseding_version_id"] = (
                body.get("superseding_version_id") if child is None else child[0]
            )
            result.append(body)
        return result

    @backend_errors
    def commit(
        self,
        *,
        workspace: str,
        scope: str,
        key: str,
        request: object,
        expected_revision: int,
        recorded: datetime,
        events: list[tuple[str, dict[str, Any]]],
        receipt: dict[str, Any],
        indexes: list[tuple[str, str, str]] | None = None,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._guard()
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                self._guard()
                prior = db.execute(
                    "SELECT request_hash,receipt FROM memory_operations WHERE workspace=? AND key=?",
                    (workspace, key),
                ).fetchone()
                if prior:
                    if prior["request_hash"] != digest(request):
                        raise MemoryError("write_conflict")
                    db.rollback()
                    return json.loads(prior["receipt"])  # type: ignore[no-any-return]
                if job is not None:
                    leased = db.execute(
                        "SELECT token,state,lease_until FROM outbox_jobs WHERE workspace=? AND scope=? AND identity=?",
                        (workspace, scope, job["identity"]),
                    ).fetchone()
                    if (
                        leased is None
                        or leased["state"] != "leased"
                        or leased["token"] != job["token"]
                        or leased["lease_until"] <= micros(datetime.now(UTC))
                    ):
                        raise MemoryError("lease_lost")
                db.execute("INSERT OR IGNORE INTO workspace_heads VALUES (?,0,'',0)", (workspace,))
                head = db.execute(
                    "SELECT * FROM workspace_heads WHERE workspace=?", (workspace,)
                ).fetchone()
                if head["revision"] != expected_revision:
                    raise MemoryError("concurrent_write")
                if not events or micros(recorded) <= head["recorded"]:
                    raise ValueError("invalid operation timestamp or empty operation")
                revision, previous = expected_revision, head["hash"]
                operation_id = uuid4().hex
                result = dict(receipt, committed_revision=expected_revision + len(events))
                request_hash, receipt_hash = digest(request), digest(result)
                for kind, body in events:
                    revision += 1
                    event_id = uuid4().hex
                    payload_hash = digest(body)
                    aggregate_type = (
                        kind.split("Accepted")[0] if kind.endswith("Accepted") else kind
                    )
                    aggregate_id = digest([kind, body.get("id", scope)])
                    event_hash = digest(
                        [
                            workspace,
                            scope,
                            revision,
                            event_id,
                            kind,
                            micros(recorded),
                            payload_hash,
                            previous,
                            operation_id,
                            1,
                            aggregate_type,
                            aggregate_id,
                            request_hash,
                            receipt_hash,
                        ]
                    )
                    db.execute(
                        "INSERT INTO memory_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            workspace,
                            revision,
                            scope,
                            event_id,
                            kind,
                            micros(recorded),
                            _json(body),
                            payload_hash,
                            previous,
                            event_hash,
                            operation_id,
                            1,
                            aggregate_type,
                            aggregate_id,
                        ),
                    )
                    self._project(db, workspace, scope, kind, body, micros(recorded))
                    previous = event_hash
                if any(kind == "MemoryErased" for kind, _ in events):
                    db.execute(
                        "UPDATE memory_events SET payload=NULL WHERE workspace=? AND scope=? AND event_type!='MemoryErased'",
                        (workspace, scope),
                    )
                    db.execute(
                        "UPDATE memory_operations SET key='erased:'||operation_id,receipt=? WHERE workspace=? AND scope=?",
                        (_json({"state": "erased"}), workspace, scope),
                    )
                for kind, identity, content in indexes or []:
                    self._write_index(db, workspace, scope, kind, identity, content)
                db.execute(
                    "UPDATE workspace_heads SET revision=?,hash=?,recorded=? WHERE workspace=?",
                    (revision, previous, micros(recorded), workspace),
                )
                db.execute(
                    "INSERT OR REPLACE INTO projection_offsets VALUES (?,?)", (workspace, revision)
                )
                if job is not None:
                    db.execute(
                        "UPDATE outbox_jobs SET state=?,token=NULL,lease_until=NULL,next_attempt=? WHERE workspace=? AND scope=? AND identity=?",
                        (
                            job["state"],
                            micros(recorded) + 2 ** job["attempts"] * 1000000,
                            workspace,
                            scope,
                            job["identity"],
                        ),
                    )
                result = dict(receipt, committed_revision=revision)
                db.execute(
                    "INSERT INTO memory_operations VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        workspace,
                        key,
                        scope,
                        digest(request),
                        _json(result),
                        receipt_hash,
                        operation_id,
                        expected_revision + 1,
                        revision,
                    ),
                )
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    def _project(
        self,
        db: sqlite3.Connection,
        workspace: str,
        scope: str,
        kind: str,
        body: dict[str, Any],
        recorded: int,
    ) -> None:
        return sqlite_projection__project(self, db, workspace, scope, kind, body, recorded)

    @backend_errors
    def lexical(self, workspace: str, scope: str, query: str) -> tuple[str, ...]:
        self._guard()
        if not query.strip():
            return ()
        # LIKE supports literal short/CJK substrings; trigram accelerates longer text.
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        identities: list[str] = []
        with self._connection() as db:
            for table in ("episode_fts", "fact_fts"):
                rows = db.execute(
                    f"SELECT identity FROM {table} WHERE workspace=? AND scope=? AND text LIKE ? ESCAPE '\\' ORDER BY identity LIMIT 100",  # nosec B608
                    (workspace, scope, "%" + escaped + "%"),
                ).fetchall()
                identities.extend(row[0] for row in rows)
        return tuple(dict.fromkeys(identities))[:100]

    @backend_errors
    def rebuild(self, read_text: Callable[[str, str, EvidenceSpan], str]) -> None:
        return sqlite_projection_rebuild(self, read_text)

    @backend_errors
    def candidates(
        self, query: MemoryQuery, *, lane: str, identities: tuple[str, ...] = ()
    ) -> list[dict[str, Any]]:
        return sqlite_queries_candidates(self, query, lane=lane, identities=identities)

    @backend_errors
    def checkpoint(self) -> None:
        with self._connection() as db:
            result = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result[0]:
                raise MemoryError("erasure_checkpoint_pending")

    @backend_errors
    def claim_job(
        self,
        workspace: str,
        scope: str,
        identity: str,
        *,
        now: datetime,
        lease_seconds: int,
        force: bool = False,
    ) -> dict[str, Any] | None:
        return sqlite_jobs_claim_job(
            self, workspace, scope, identity, now=now, lease_seconds=lease_seconds, force=force
        )

    @backend_errors
    def entity_assertions_for_merge(
        self, workspace: str, scope: str, entity_id: str
    ) -> list[dict[str, Any]]:
        return sqlite_queries_entity_assertions_for_merge(self, workspace, scope, entity_id)

    @backend_errors
    def memory_policy_for(self, workspace: str, scope: str, target: str) -> dict[str, Any] | None:
        return sqlite_queries_memory_policy_for(self, workspace, scope, target)

    def _write_index(
        self,
        db: sqlite3.Connection,
        workspace: str,
        scope: str,
        kind: str,
        identity: str,
        content: str,
    ) -> None:
        return sqlite_projection__write_index(self, db, workspace, scope, kind, identity, content)

    @backend_errors
    def due_jobs(self, workspace: str, scope: str, *, now: datetime, limit: int) -> tuple[str, ...]:
        self._guard()
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid worker limit")
        with self._connection() as db:
            rows = db.execute(
                "SELECT identity FROM outbox_jobs WHERE workspace=? AND scope=? AND ((state IN ('pending','retryable') AND next_attempt<=?) OR (state='leased' AND lease_until<=?)) ORDER BY next_attempt,identity LIMIT ?",
                (workspace, scope, micros(now), micros(now), limit),
            ).fetchall()
        return tuple(row[0] for row in rows)

    @backend_errors
    def similar_entities(
        self, workspace: str, scope: str, name: str, entity_type: str
    ) -> list[dict[str, Any]]:
        self._guard()
        grams = tuple(dict.fromkeys(name[i : i + 3] for i in range(len(name) - 2)))
        if not grams:
            return []
        expression = " OR ".join('"' + gram.replace('"', '""') + '"' for gram in grams)
        with self._connection() as db:
            rows = db.execute(
                "SELECT e.body FROM entity_fts f JOIN entities e ON e.workspace=f.workspace AND e.scope=f.scope AND e.identity=f.identity WHERE f.text MATCH ? AND e.workspace=? AND e.scope=? AND json_extract(e.body,'$.entity_type')=? ORDER BY f.rank,e.identity LIMIT 100",
                (expression, workspace, scope, entity_type),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
