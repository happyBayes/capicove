"""Source-owned content-addressed payloads with a separate privacy deny authority.

The privacy directory is non-rollback state, excluded from ordinary backups. An
 erased scope is permanently denied; use a new host-authorized scope for new data.
"""

import hashlib
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from capicove.contracts.evidence import EvidenceRef, EvidenceSpan
from capicove.contracts.knowledge import text as validate_text
from capicove.source.errors import EvidenceUnavailable
from capicove.source.models import evidence_errors, validate_text_payload


class SQLiteTextEvidenceStore:
    @evidence_errors
    def __init__(
        self, path: Path, privacy_dir: Path, *, authorize: Callable[[str, str], bool] | None = None
    ) -> None:
        self._authorize = authorize
        if path.is_symlink() or privacy_dir.is_symlink():
            raise ValueError("unsafe Evidence path")
        privacy_dir.mkdir(mode=0o700, exist_ok=True)
        if privacy_dir.stat().st_mode & 0o077:
            raise ValueError("privacy directory must be private")
        self._path, self._privacy = path.absolute(), privacy_dir.absolute()
        # Open the deny authority first and retain it across payload restoration.
        self._deny = self._privacy / "erased-scopes.sqlite3"
        existing = path.exists() and path.stat().st_size > 0
        if existing and not self._deny.exists():
            raise EvidenceUnavailable("privacy_authority_unavailable")
        for name in (path, self._deny):
            if name.is_symlink():
                raise ValueError("unsafe Evidence path")
            if not name.exists():
                fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
            if name.stat().st_mode & 0o077:
                raise ValueError("Evidence store must be private")
        with closing(sqlite3.connect(self._deny)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS authority (id TEXT PRIMARY KEY)")
            authority = db.execute("SELECT id FROM authority").fetchone()
            if authority is None:
                if existing:
                    raise EvidenceUnavailable("privacy_authority_unavailable")
                self._authority = uuid4().hex
                db.execute("INSERT INTO authority VALUES (?)", (self._authority,))
            else:
                self._authority = authority[0]
            db.execute(
                "CREATE TABLE IF NOT EXISTS erased (workspace TEXT, scope TEXT, PRIMARY KEY(workspace,scope))"
            )
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA secure_delete=ON")
            db.execute("CREATE TABLE IF NOT EXISTS privacy_binding (id TEXT PRIMARY KEY)")
            binding = db.execute("SELECT id FROM privacy_binding").fetchone()
            if binding is None:
                if existing:
                    raise EvidenceUnavailable("privacy_authority_unavailable")
                db.execute("INSERT INTO privacy_binding VALUES (?)", (self._authority,))
            elif binding[0] != self._authority:
                raise EvidenceUnavailable("privacy_authority_unavailable")
            db.execute(
                "CREATE TABLE IF NOT EXISTS text_evidence (workspace TEXT, scope TEXT, digest TEXT, body TEXT NOT NULL, PRIMARY KEY(workspace,scope,digest))"
            )

    @contextmanager
    def _authority_connection(self) -> Iterator[sqlite3.Connection]:
        try:
            db = sqlite3.connect(self._deny.as_uri() + "?mode=rw", uri=True, timeout=5)
        except sqlite3.Error:
            raise EvidenceUnavailable("privacy_authority_unavailable") from None
        with closing(db):
            try:
                row = db.execute("SELECT id FROM authority").fetchone()
            except sqlite3.Error:
                raise EvidenceUnavailable("privacy_authority_unavailable") from None
            if row is None or row[0] != self._authority:
                raise EvidenceUnavailable("privacy_authority_unavailable")
            yield db

    @contextmanager
    def _guard(self, workspace: str, scope: str) -> Iterator[sqlite3.Connection]:
        self._check_access(workspace, scope)
        with self._authority_connection() as guard, guard:
            guard.execute("BEGIN")
            if guard.execute(
                "SELECT 1 FROM erased WHERE workspace=? AND scope=?", (workspace, scope)
            ).fetchone():
                raise EvidenceUnavailable("evidence_unavailable")
            yield guard

    @evidence_errors
    def is_erased(self, *, workspace: str, scope: str) -> bool:
        self._check_access(workspace, scope)
        with self._authority_connection() as db:
            return (
                db.execute(
                    "SELECT 1 FROM erased WHERE workspace=? AND scope=?", (workspace, scope)
                ).fetchone()
                is not None
            )

    @evidence_errors
    def accept_text(self, *, workspace: str, scope: str, text: str) -> EvidenceRef:
        validate_text_payload(text)
        ref = EvidenceRef(hashlib.sha256(text.encode()).hexdigest())
        with self._guard(workspace, scope), closing(sqlite3.connect(self._path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO text_evidence VALUES (?,?,?,?)",
                (workspace, scope, ref.content_sha256, text),
            )
        return ref

    @evidence_errors
    def read_text(self, *, workspace: str, scope: str, span: EvidenceSpan) -> str:
        return self.read_spans(workspace=workspace, scope=scope, spans=(span,))[0]

    @evidence_errors
    def read_spans(
        self, *, workspace: str, scope: str, spans: tuple[EvidenceSpan, ...]
    ) -> tuple[str, ...]:
        if not isinstance(spans, tuple) or any(
            not isinstance(span, EvidenceSpan) for span in spans
        ):
            raise ValueError("invalid spans")
        with self._guard(workspace, scope), closing(sqlite3.connect(self._path)) as db:
            bodies: dict[str, str] = {}
            results = []
            for span in spans:
                digest = span.reference.content_sha256
                if digest not in bodies:
                    row = db.execute(
                        "SELECT body FROM text_evidence WHERE workspace=? AND scope=? AND digest=?",
                        (workspace, scope, digest),
                    ).fetchone()
                    if row is None or hashlib.sha256(row[0].encode()).hexdigest() != digest:
                        raise EvidenceUnavailable("evidence_unavailable")
                    bodies[digest] = row[0]
                body = bodies[digest]
                if span.end > len(body):
                    raise EvidenceUnavailable("evidence_unavailable")
                results.append(body[span.start : span.end])
            return tuple(results)

    @evidence_errors
    def erase_scope(self, *, workspace: str, scope: str) -> None:
        self._check_access(workspace, scope)
        with self._authority_connection() as deny, deny:
            deny.execute("BEGIN IMMEDIATE")
            deny.execute("INSERT OR IGNORE INTO erased VALUES (?,?)", (workspace, scope))
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute(
                "DELETE FROM text_evidence WHERE workspace=? AND scope=?", (workspace, scope)
            )
        with closing(sqlite3.connect(self._path)) as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    @contextmanager
    def authorize_scope(self, *, workspace: str, scope: str) -> Iterator[None]:
        """Serialize a derived commit against the source-owned erasure decision."""
        with self._guard(workspace, scope):
            yield

    def _check_access(self, workspace: str, scope: str) -> None:
        validate_text(workspace)
        validate_text(scope)
        if self._authorize is not None and self._authorize(workspace, scope) is not True:
            raise EvidenceUnavailable("permission_denied")
