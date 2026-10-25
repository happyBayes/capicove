"""Transactional single-owner local registry for immutable skill versions.

Review identity names the local operator; it is not remote authentication. The host
must keep this administration API outside Agent-callable tools and protect its files.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import TracebackType
from typing import Any, Self
from uuid import uuid4

from capicove.contracts import CandidateState
from capicove.evolution.models import EvaluationReport, PairedResult, SkillVersion
from capicove.evolution.validation import content_digest, validate_skill


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, version INTEGER NOT NULL,
    parent_id TEXT, content TEXT NOT NULL, hash TEXT NOT NULL, sources TEXT NOT NULL,
    state TEXT NOT NULL, source_batch_id TEXT, UNIQUE(name, version));
CREATE TABLE IF NOT EXISTS active (name TEXT PRIMARY KEY, candidate_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
    candidate_id TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL,
    action TEXT NOT NULL, reviewer TEXT NOT NULL, reason TEXT NOT NULL,
    previous_id TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS usage (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL,
    episode_id TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, payload TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS immutable_candidate BEFORE UPDATE OF
    id,name,version,parent_id,content,hash,sources,source_batch_id ON candidates
    BEGIN SELECT RAISE(ABORT, 'immutable candidate'); END;
CREATE TRIGGER IF NOT EXISTS retain_candidate BEFORE DELETE ON candidates
    BEGIN SELECT RAISE(ABORT, 'retained candidate'); END;
"""


class _Unspecified:
    """Distinguish omission from explicitly expecting no active version."""


_UNSPECIFIED = _Unspecified()


class EvolutionRegistry:
    """Persist candidates, evidence, evaluations, reviews and actual skill loads."""

    def __init__(self, root: Path) -> None:
        if os.name != "posix" or not hasattr(os, "getuid"):
            raise ValueError("evolution storage requires POSIX ownership support")
        root = root.absolute()
        if root.is_symlink():
            raise ValueError("registry root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_stat = root.stat()
        if root_stat.st_uid != os.getuid() or stat.S_IMODE(root_stat.st_mode) & 0o077:
            raise ValueError("registry root must be private to its local owner (0700)")
        self.path = root / "evolution.sqlite3"
        existed = self.path.exists() or self.path.is_symlink()
        for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            if path.exists() or path.is_symlink():
                info = path.lstat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077
                ):
                    raise ValueError(
                        "registry files must be private regular owner files without links"
                    )
        if existed:
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as check:
                application_id = check.execute("PRAGMA application_id").fetchone()[0]
                version = check.execute("PRAGMA user_version").fetchone()[0]
                if application_id != 1128353875 or version != 1:
                    raise ValueError("unknown registry schema or unrelated database")
        else:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        self._db = sqlite3.connect(self.path, isolation_level=None, timeout=10)
        self._db.row_factory = sqlite3.Row
        if not existed:
            self._db.execute("PRAGMA application_id=1128353875")
            self._db.execute("PRAGMA user_version=1")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        for table in ("evaluations", "reviews", "usage", "batches"):
            for action in ("UPDATE", "DELETE"):
                self._db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} "
                    f"BEFORE {action} ON {table} "
                    "BEGIN SELECT RAISE(ABORT, 'append-only record'); END"
                )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the local SQLite handle."""
        self._db.close()

    def _current_id(self, name: str) -> str | None:
        row = self._db.execute("SELECT candidate_id FROM active WHERE name=?", (name,)).fetchone()
        return str(row[0]) if row else None

    def _check_current(self, name: str, expected: str | None) -> None:
        if self._current_id(name) != expected:
            raise ValueError("stale active baseline; reread and review again")

    def get_candidate(self, candidate_id: str) -> SkillVersion:
        """Read retained immutable content and verify its identity."""
        row = self._db.execute("SELECT * FROM candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        if content_digest(row["content"]) != row["hash"]:
            raise ValueError("candidate content hash mismatch")
        if validate_skill(row["name"], row["content"]) != row["content"]:
            raise ValueError("candidate content is not canonical")
        self._validate_batch_sources(row["source_batch_id"], tuple(json.loads(row["sources"])))
        return SkillVersion(
            row["id"],
            row["name"],
            row["version"],
            row["parent_id"],
            row["content"],
            row["hash"],
            tuple(json.loads(row["sources"])),
            CandidateState(row["state"]),
            row["source_batch_id"],
        )

    def list_candidates(self) -> tuple[SkillVersion, ...]:
        """Return retained versions in stable asset/version order."""
        rows = self._db.execute("SELECT id FROM candidates ORDER BY name,version").fetchall()
        return tuple(self.get_candidate(row[0]) for row in rows)

    def active(self, name: str) -> SkillVersion | None:
        """Resolve only the current verified active version."""
        candidate_id = self._current_id(name)
        if candidate_id is None:
            return None
        candidate = self.get_candidate(candidate_id)
        if candidate.state != CandidateState.ACTIVE or candidate.name != name:
            raise ValueError("active pointer integrity failure")
        return candidate

    def create_candidate(
        self,
        name: str,
        content: str,
        source_episode_ids: tuple[str, ...],
        *,
        expected_active: str | None,
        source_batch_id: str | None = None,
    ) -> SkillVersion:
        """Create a retained draft without changing runtime behavior."""
        content = validate_skill(name, content)
        if not source_episode_ids or any(not source.strip() for source in source_episode_ids):
            raise ValueError("candidate requires source episode identities")
        if len(set(source_episode_ids)) != len(source_episode_ids):
            raise ValueError("duplicate source episodes")
        candidate_id = str(uuid4())
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._check_current(name, expected_active)
            self._validate_batch_sources(source_batch_id, source_episode_ids)
            if expected_active is not None:
                self.active(name)
            existing = self._db.execute(
                "SELECT id FROM candidates WHERE name=? AND hash=? AND sources=? "
                "AND parent_id IS ? AND source_batch_id IS ? ORDER BY version DESC LIMIT 1",
                (
                    name,
                    content_digest(content),
                    json.dumps(source_episode_ids),
                    expected_active,
                    source_batch_id,
                ),
            ).fetchone()
            if existing is not None:
                result = self.get_candidate(existing[0])
                self._db.commit()
                return result
            version = self._db.execute(
                "SELECT COALESCE(MAX(version),0)+1 FROM candidates WHERE name=?", (name,)
            ).fetchone()[0]
            self._db.execute(
                "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    candidate_id,
                    name,
                    version,
                    expected_active,
                    content,
                    content_digest(content),
                    json.dumps(source_episode_ids),
                    CandidateState.DRAFT,
                    source_batch_id,
                ),
            )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise
        return self.get_candidate(candidate_id)

    def record_evaluation(
        self, report: EvaluationReport, *, expected_latest_id: str | None = None
    ) -> EvaluationReport:
        """Append bound observations and derive whether review is available."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            if expected_latest_id is not None:
                latest = self.latest_evaluation(report.candidate_id)
                if latest is None or latest.report_id != expected_latest_id:
                    raise ValueError("evaluation attempt was superseded")
            candidate = self.get_candidate(report.candidate_id)
            if candidate.state not in {CandidateState.DRAFT, CandidateState.AWAITING_APPROVAL}:
                raise ValueError("candidate state does not allow evaluation")
            self._validate_binding(candidate, report)
            payload = json.dumps(asdict(report), sort_keys=True, separators=(",", ":"))
            self._db.execute(
                "INSERT INTO evaluations(id,candidate_id,payload,hash) VALUES (?,?,?,?)",
                (report.report_id, report.candidate_id, payload, content_digest(payload)),
            )
            self._db.execute(
                "UPDATE candidates SET state=? WHERE id=?",
                (
                    CandidateState.AWAITING_APPROVAL if report.passed else CandidateState.DRAFT,
                    candidate.candidate_id,
                ),
            )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise
        return report

    def _validate_binding(self, candidate: SkillVersion, report: EvaluationReport) -> None:
        if (
            report.content_hash != candidate.content_hash
            or report.baseline_id != candidate.parent_id
        ):
            raise ValueError("evaluation content or baseline binding mismatch")
        exposed_sources = set(candidate.source_episode_ids)
        if candidate.source_batch_id is not None:
            self._validate_batch_sources(candidate.source_batch_id, candidate.source_episode_ids)
            batch = json.loads(self.get_batch(candidate.source_batch_id))
            exposed_sources = set(batch["source_episode_ids"])
        if set(report.synthesis_episode_ids) != exposed_sources:
            raise ValueError("evaluation synthesis sources do not match candidate")
        splits = [
            report.synthesis_episode_ids,
            report.replay_episode_ids,
            report.held_out_episode_ids,
        ]
        flattened = [item for split in splits for item in split]
        if any(not item.strip() for item in flattened) or len(set(flattened)) != len(flattened):
            raise ValueError("evaluation splits must be disjoint with unique identities")
        if not report.report_id.strip() or not report.evaluator_id.strip():
            raise ValueError("evaluation requires report and evaluator identities")

    def latest_evaluation(self, candidate_id: str) -> EvaluationReport | None:
        """Read the latest immutable evaluation, verifying stored bytes."""
        row = self._db.execute(
            "SELECT payload,hash FROM evaluations WHERE candidate_id=? ORDER BY sequence DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        if content_digest(row[0]) != row[1]:
            raise ValueError("evaluation hash mismatch")
        data = json.loads(row[0])
        data["results"] = tuple(PairedResult(**item) for item in data["results"])
        for key in ("synthesis_episode_ids", "replay_episode_ids", "held_out_episode_ids"):
            data[key] = tuple(data[key])
        return EvaluationReport(**data)

    def _review(
        self, candidate_id: str, action: str, reviewer: str, reason: str, previous_id: str | None
    ) -> None:
        if not reviewer.strip() or not reason.strip():
            raise ValueError("reviewer identity and review reason are required")
        self._db.execute(
            "INSERT INTO reviews(candidate_id,action,reviewer,reason,previous_id) VALUES (?,?,?,?,?)",
            (candidate_id, action, reviewer, reason, previous_id),
        )

    def approve(
        self, candidate_id: str, reviewer: str, reason: str, *, expected_active: str | None
    ) -> SkillVersion:
        """Atomically review and activate a passing candidate at its exact baseline."""
        return self._activate(candidate_id, reviewer, reason, expected_active, rollback_name=None)

    def rollback(
        self,
        name: str,
        candidate_id: str,
        reviewer: str,
        reason: str,
        *,
        expected_active: str | None,
    ) -> SkillVersion:
        """Restore a previously approved version of the same asset with explicit review."""
        return self._activate(candidate_id, reviewer, reason, expected_active, rollback_name=name)

    def _activate(
        self,
        candidate_id: str,
        reviewer: str,
        reason: str,
        expected_active: str | None,
        rollback_name: str | None,
    ) -> SkillVersion:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            candidate = self.get_candidate(candidate_id)
            self._check_current(candidate.name, expected_active)
            if rollback_name is not None:
                approved = self._db.execute(
                    "SELECT 1 FROM reviews WHERE candidate_id=? AND action='approve'",
                    (candidate_id,),
                ).fetchone()
                if (
                    rollback_name != candidate.name
                    or not approved
                    or expected_active is None
                    or candidate.state
                    not in {CandidateState.SUPERSEDED, CandidateState.ROLLED_BACK}
                ):
                    raise ValueError(
                        "rollback requires a retained previously approved same-asset version"
                    )
            else:
                report = self.latest_evaluation(candidate_id)
                if report is None or not report.passed:
                    raise ValueError("passing evaluation required for approval")
                self._validate_binding(candidate, report)
                if candidate.parent_id != expected_active:
                    raise ValueError("stale candidate baseline")
                if candidate.state != CandidateState.AWAITING_APPROVAL:
                    raise ValueError("candidate is not awaiting approval")
            self._review(
                candidate_id,
                "rollback" if rollback_name else "approve",
                reviewer,
                reason,
                expected_active,
            )
            if expected_active is not None:
                self.active(candidate.name)
                self._db.execute(
                    "UPDATE candidates SET state=? WHERE id=?",
                    (
                        CandidateState.ROLLED_BACK if rollback_name else CandidateState.SUPERSEDED,
                        expected_active,
                    ),
                )
            self._db.execute(
                "UPDATE candidates SET state=? WHERE id=?", (CandidateState.ACTIVE, candidate_id)
            )
            self._db.execute(
                "INSERT INTO active VALUES (?,?) ON CONFLICT(name) DO UPDATE SET candidate_id=excluded.candidate_id",
                (candidate.name, candidate_id),
            )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise
        return self.get_candidate(candidate_id)

    def reject(
        self,
        candidate_id: str,
        reviewer: str,
        reason: str,
        *,
        expected_active: str | _Unspecified | None = _UNSPECIFIED,
    ) -> SkillVersion:
        """Retain a rejected candidate and its explicit review."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            candidate = self.get_candidate(candidate_id)
            if not isinstance(expected_active, _Unspecified):
                self._check_current(candidate.name, expected_active)
            if candidate.state not in {CandidateState.DRAFT, CandidateState.AWAITING_APPROVAL}:
                raise ValueError("only draft or awaiting-approval candidates may be rejected")
            self._review(candidate_id, "reject", reviewer, reason, self._current_id(candidate.name))
            self._db.execute(
                "UPDATE candidates SET state=? WHERE id=?", (CandidateState.REJECTED, candidate_id)
            )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise
        return self.get_candidate(candidate_id)

    def record_use(self, candidate_id: str, episode_id: str) -> None:
        """Record the exact verified active version loaded for one episode."""
        if not episode_id.strip():
            raise ValueError("usage requires an episode identity")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            candidate = self.get_candidate(candidate_id)
            if self.active(candidate.name) != candidate:
                raise ValueError("only an active version can be loaded")
            self._db.execute(
                "INSERT INTO usage(candidate_id,episode_id) VALUES (?,?)",
                (candidate_id, episode_id),
            )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise

    def store_batch(self, batch_id: str, payload: str) -> None:
        """Store immutable sanitized evidence metadata under a caller's stable identity."""
        if not batch_id.strip():
            raise ValueError("batch identity is required")
        encoded = json.dumps(
            json.loads(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self._db.execute("BEGIN IMMEDIATE")
        try:
            existing = self._db.execute(
                "SELECT payload FROM batches WHERE id=?", (batch_id,)
            ).fetchone()
            if existing is not None:
                if self.get_batch(batch_id) != encoded:
                    raise ValueError("batch identity conflicts with retained content")
            else:
                self._db.execute(
                    "INSERT INTO batches VALUES (?,?,?)",
                    (batch_id, encoded, content_digest(encoded)),
                )
            self._db.commit()
        except BaseException:
            self._db.rollback()
            raise

    def get_batch(self, batch_id: str) -> str:
        """Verify and read a retained evidence batch."""
        row = self._db.execute(
            "SELECT payload,hash FROM batches WHERE id=?", (batch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        if content_digest(row[0]) != row[1]:
            raise ValueError("batch hash mismatch")
        return str(row[0])

    def list_batches(self) -> tuple[str, ...]:
        """List retained evidence identities in stable order."""
        return tuple(str(row[0]) for row in self._db.execute("SELECT id FROM batches ORDER BY id"))

    def evaluation_report(self, candidate_id: str) -> EvaluationReport | None:
        """Read the latest report for administration views."""
        return self.latest_evaluation(candidate_id)

    def history(self, name: str) -> tuple[dict[str, Any], ...]:
        """Read immutable review history for one asset."""
        rows = self._db.execute(
            "SELECT reviews.* FROM reviews JOIN candidates ON reviews.candidate_id=candidates.id "
            "WHERE candidates.name=? ORDER BY reviews.sequence",
            (name,),
        )
        return tuple(dict(row) for row in rows)

    def usage(self, candidate_id: str) -> tuple[dict[str, Any], ...]:
        """Read actual load records for one retained version."""
        return tuple(
            dict(row)
            for row in self._db.execute(
                "SELECT * FROM usage WHERE candidate_id=? ORDER BY sequence", (candidate_id,)
            )
        )

    def _validate_batch_sources(self, batch_id: str | None, sources: tuple[str, ...]) -> None:
        """Verify an exact retained snapshot and its source membership."""
        if batch_id is None:
            return
        payload = json.loads(self.get_batch(batch_id))
        batch_sources = payload.get("source_episode_ids") if isinstance(payload, dict) else None
        if (
            not isinstance(batch_sources, list)
            or any(not isinstance(item, str) or not item.strip() for item in batch_sources)
            or not set(sources).issubset(set(batch_sources))
        ):
            raise ValueError("candidate sources do not belong to retained evidence batch")
