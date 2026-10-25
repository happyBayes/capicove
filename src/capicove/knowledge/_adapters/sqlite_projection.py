"""Sqlite projection operations for temporal memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from capicove.contracts.evidence import EvidenceRef, EvidenceSpan
from capicove.knowledge.errors import MemoryError

from .sqlite_codec import _json
from .sqlite_schema import _PROJECTION_TABLES, _RECORD_TABLES


if TYPE_CHECKING:
    from capicove.knowledge._adapters.sqlite_memory import SQLiteMemoryRepository


def _project(
    self: SQLiteMemoryRepository,
    db: sqlite3.Connection,
    workspace: str,
    scope: str,
    kind: str,
    body: dict[str, Any],
    recorded: int,
) -> None:
    record_kinds = {
        "EpisodeAccepted": "episode",
        "EntityResolved": "entity",
        "EntityResolutionPending": "resolution",
        "EntityAliasAdded": "alias",
        "EntityRedirected": "redirect",
        "MemoryAdmitted": "memory",
        "MemoryArchived": "archive",
        "ExtractionCompleted": "extraction",
        "ExtractionPartiallyCompleted": "extraction",
        "EpisodeIndexed": "extraction",
        "ExtractionFailed": "extraction",
    }
    if kind in record_kinds:
        db.execute(
            f"INSERT OR REPLACE INTO {_RECORD_TABLES[record_kinds[kind]]} VALUES (?,?,?,?)",
            (workspace, scope, body["id"], _json(body)),
        )
        if kind == "EntityResolved":
            db.execute(
                "DELETE FROM entity_fts WHERE workspace=? AND scope=? AND identity=?",
                (workspace, scope, body["id"]),
            )
            db.execute(
                "INSERT INTO entity_fts(workspace,scope,identity,text) VALUES (?,?,?,?)",
                (workspace, scope, body["id"], body["name"]),
            )
        if kind in {"ExtractionCompleted", "ExtractionFailed", "ExtractionPartiallyCompleted"}:
            db.execute(
                "UPDATE outbox_jobs SET state=?,attempts=?,token=NULL,lease_until=NULL,next_attempt=? WHERE workspace=? AND scope=? AND identity=?",
                (
                    body["state"],
                    body["attempts"],
                    recorded + 2 ** body["attempts"] * 1000000,
                    workspace,
                    scope,
                    body["id"],
                ),
            )
    elif kind in {"FactAccepted", "FactDisputed", "FactRejected"}:
        parent = body.get("parent_version_id")
        if parent:
            db.execute(
                "UPDATE fact_versions SET recorded_to=? WHERE workspace=? AND scope=? AND version=? AND recorded_to IS NULL",
                (recorded, workspace, scope, parent),
            )
        db.execute(
            "INSERT INTO fact_versions (workspace,scope,version,fact,slot,assertion,recorded_from,recorded_to,body,valid_from,valid_to) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                workspace,
                scope,
                body["version"],
                body["id"],
                body["conflict_key"],
                body["assertion_hash"],
                recorded,
                None,
                _json(body),
                body["validity"][0],
                body["validity"][1],
            ),
        )
        for span in body["evidence"]:
            db.execute(
                "INSERT INTO fact_evidence VALUES (?,?,?,?,?,?)",
                (workspace, scope, body["version"], span["digest"], span["start"], span["end"]),
            )
        db.execute(
            "INSERT INTO fact_conflicts VALUES (?,?,?,?,?)",
            (workspace, scope, body["version"], body["conflict_key"], body["state"]),
        )
    elif kind == "ExtractionQueued":
        existing = db.execute(
            "SELECT state,lease_until,body FROM outbox_jobs WHERE workspace=? AND scope=? AND identity=?",
            (workspace, scope, body["id"]),
        ).fetchone()
        if existing and body.get("fingerprint") != json.loads(existing["body"]).get("fingerprint"):
            if existing["state"] == "leased" and existing["lease_until"] > recorded:
                raise MemoryError("extraction_unavailable")
            db.execute(
                "DELETE FROM outbox_jobs WHERE workspace=? AND scope=? AND identity=?",
                (workspace, scope, body["id"]),
            )
        db.execute(
            "INSERT OR IGNORE INTO outbox_jobs VALUES (?,?,?,'pending',NULL,NULL,0,?,0)",
            (workspace, scope, body["id"], _json(body)),
        )
    elif kind == "MemoryErased":
        for table in _PROJECTION_TABLES:
            if table == "projection_offsets":
                continue
            db.execute(f"DELETE FROM {table} WHERE workspace=? AND scope=?", (workspace, scope))  # nosec B608


def rebuild(
    self: SQLiteMemoryRepository, read_text: Callable[[str, str, EvidenceSpan], str]
) -> None:
    """Replay verified ledger deltas atomically, without extraction/model execution."""
    if not self.verify():
        raise MemoryError("corrupt_event_log")
    with self._connection() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            for table in _PROJECTION_TABLES:
                db.execute(f"DELETE FROM {table}")  # nosec B608
            for row in db.execute(
                "SELECT * FROM memory_events ORDER BY workspace,revision"
            ).fetchall():
                if row["payload"] is None:
                    continue
                body = json.loads(row["payload"])
                self._project(
                    db, row["workspace"], row["scope"], row["event_type"], body, row["recorded"]
                )
            for row in db.execute("SELECT * FROM episodes").fetchall():
                body = json.loads(row["body"])
                content = read_text(
                    row["workspace"],
                    row["scope"],
                    EvidenceSpan(EvidenceRef(body["content_sha256"]), 0, body["content_length"]),
                )
                self._write_index(
                    db, row["workspace"], row["scope"], "episode", row["identity"], content
                )
            for row in db.execute(
                "SELECT * FROM fact_versions WHERE recorded_to IS NULL"
            ).fetchall():
                body = json.loads(row["body"])
                label = db.execute(
                    "SELECT body FROM entities WHERE workspace=? AND scope=? AND identity=?",
                    (row["workspace"], row["scope"], body["subject"]),
                ).fetchone()
                content = (
                    f"{json.loads(label[0])['name']} {body['predicate']} {body['object']['value']}"
                )
                self._write_index(db, row["workspace"], row["scope"], "fact", row["fact"], content)
            db.execute(
                "INSERT INTO projection_offsets SELECT workspace,revision FROM workspace_heads"
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise


def _write_index(
    self: SQLiteMemoryRepository,
    db: sqlite3.Connection,
    workspace: str,
    scope: str,
    kind: str,
    identity: str,
    content: str,
) -> None:
    table = {"episode": "episode_fts", "fact": "fact_fts"}[kind]
    existing = db.execute(
        "SELECT docid FROM fts_entries WHERE workspace=? AND scope=? AND kind=? AND identity=?",
        (workspace, scope, kind, identity),
    ).fetchone()
    if existing:
        docid = existing[0]
        db.execute(f"UPDATE {table} SET text=? WHERE rowid=?", (content, docid))  # nosec B608
    else:
        cursor = db.execute(
            f"INSERT INTO {table}(workspace,scope,identity,text) VALUES (?,?,?,?)",  # nosec B608
            (workspace, scope, identity, content),
        )
        docid = cursor.lastrowid
        db.execute(
            "INSERT INTO fts_entries VALUES (?,?,?,?,?)",
            (workspace, scope, kind, identity, docid),
        )
    words = {"episode": "episode_words", "fact": "fact_words"}[kind]
    db.execute(
        f"INSERT OR REPLACE INTO {words}(rowid,workspace,scope,identity,text) VALUES (?,?,?,?,?)",
        (docid, workspace, scope, identity, content),
    )
