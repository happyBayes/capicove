"""SQLite publication, preview activation, and stable query leases."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, RLock, Timer
from typing import Protocol, Self


SQLITE_GENERATION_APPLICATION_ID = 0x43415047  # ``CAPG``
SQLITE_GENERATION_SCHEMA_VERSION = 2
MAX_LEASE_INTERVAL = timedelta(minutes=10)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ConcurrentActivationError(RuntimeError):
    """Signal that an activation alias changed since the caller observed it."""


class GenerationQualificationError(RuntimeError):
    """Signal that a publication lacks a matching trusted preview qualification."""


@dataclass(frozen=True, slots=True)
class PublishedGenerationRecord:
    """Record one immutable filesystem generation registered after publication."""

    activation_set_id: str
    manifest_sha256: str
    published_root: Path
    workspace_id: str
    corpus_ref: str
    retrieval_profile_ref: str
    published_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "activation_set_id",
            "workspace_id",
            "corpus_ref",
            "retrieval_profile_ref",
        ):
            _non_blank(name, getattr(self, name))
        _sha256("manifest_sha256", self.manifest_sha256)
        if (
            not isinstance(self.published_root, Path)
            or not self.published_root.is_absolute()
            or not self.published_root.is_dir()
            or self.published_root.is_symlink()
        ):
            raise ValueError("published_root must be an existing absolute non-symlink directory")
        _aware("published_at", self.published_at)


@dataclass(frozen=True, slots=True)
class PreviewActivationQualification:
    """Locate a trusted stored qualification; possession conveys no authority."""

    qualification_id: str

    def __post_init__(self) -> None:
        _non_blank("qualification_id", self.qualification_id)


@dataclass(frozen=True, slots=True)
class VerifiedPreviewQualification:
    """Contain the verifier-produced identity for one preview activation only."""

    qualification_id: str
    activation_set_id: str
    manifest_sha256: str
    profile_ref: str
    report_sha256: str
    policy_sha256: str
    verification_receipt_sha256: str
    verified_at: datetime

    def __post_init__(self) -> None:
        for name in ("qualification_id", "activation_set_id", "profile_ref"):
            _non_blank(name, getattr(self, name))
        for name in (
            "manifest_sha256",
            "report_sha256",
            "policy_sha256",
            "verification_receipt_sha256",
        ):
            _sha256(name, getattr(self, name))
        _aware("verified_at", self.verified_at)


class PreviewActivationVerifier(Protocol):
    """Reopen and verify trusted local qualification evidence."""

    def verify(
        self,
        qualification: PreviewActivationQualification,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> VerifiedPreviewQualification: ...


@dataclass(frozen=True, slots=True)
class GenerationAliasSnapshot:
    """Pin the full atomic alias identity, including requalification version."""

    workspace_id: str
    profile_ref: str
    activation_set_id: str
    manifest_sha256: str
    qualification_id: str
    qualification_tier: str
    version: int

    def __post_init__(self) -> None:
        for name in (
            "workspace_id",
            "profile_ref",
            "activation_set_id",
            "qualification_id",
        ):
            _non_blank(name, getattr(self, name))
        _sha256("manifest_sha256", self.manifest_sha256)
        if self.qualification_tier != "preview":
            raise ValueError("the implemented lifecycle supports preview qualification only")
        _positive_int("version", self.version)


@dataclass(frozen=True, slots=True)
class GenerationLeaseRecord:
    """Persist one stable alias snapshot independently from later activation."""

    lease_id: str
    owner_id: str
    owner_session_epoch: str
    acquired_at: datetime
    expires_at: datetime
    released_at: datetime | None
    alias_snapshot: GenerationAliasSnapshot

    def __post_init__(self) -> None:
        for name in ("lease_id", "owner_id", "owner_session_epoch"):
            _non_blank(name, getattr(self, name))
        _aware("acquired_at", self.acquired_at)
        _aware("expires_at", self.expires_at)
        if self.expires_at <= self.acquired_at:
            raise ValueError("expires_at must be later than acquired_at")
        if self.released_at is not None:
            _aware("released_at", self.released_at)
        if not isinstance(self.alias_snapshot, GenerationAliasSnapshot):
            raise TypeError("alias_snapshot must be a GenerationAliasSnapshot")


@dataclass(frozen=True, slots=True)
class PublishedCandidateLeaseRecord:
    """Persist an exact published-set lease without qualification or alias state."""

    lease_id: str
    activation_set_id: str
    manifest_sha256: str
    profile_ref: str
    owner_id: str
    owner_session_epoch: str
    acquired_at: datetime
    expires_at: datetime
    released_at: datetime | None

    def __post_init__(self) -> None:
        for name in (
            "lease_id",
            "activation_set_id",
            "profile_ref",
            "owner_id",
            "owner_session_epoch",
        ):
            _non_blank(name, getattr(self, name))
        _sha256("manifest_sha256", self.manifest_sha256)
        _lease_interval(self.acquired_at, self.expires_at)
        if self.released_at is not None:
            _aware("released_at", self.released_at)


class OwnerSessionRegistry(Protocol):
    """Provide external proof that one exact process/session epoch is dead."""

    def prove_dead(
        self,
        owner_id: str,
        owner_session_epoch: str,
        *,
        deadline: datetime,
    ) -> bool: ...


_SCHEMA = """
CREATE TABLE generation_sets (
  activation_set_id TEXT PRIMARY KEY,
  manifest_sha256 TEXT NOT NULL,
  published_root TEXT NOT NULL,
  workspace_id TEXT NOT NULL,
  corpus_ref TEXT NOT NULL,
  retrieval_profile_ref TEXT NOT NULL,
  published_at TEXT NOT NULL
);

CREATE TABLE verified_qualifications (
  qualification_id TEXT PRIMARY KEY,
  activation_set_id TEXT NOT NULL REFERENCES generation_sets(activation_set_id),
  manifest_sha256 TEXT NOT NULL,
  profile_ref TEXT NOT NULL,
  report_sha256 TEXT NOT NULL,
  policy_sha256 TEXT NOT NULL,
  verification_receipt_sha256 TEXT NOT NULL UNIQUE,
  qualification_tier TEXT NOT NULL CHECK (qualification_tier = 'preview'),
  verified_at TEXT NOT NULL,
  record_json TEXT NOT NULL,
  UNIQUE (qualification_id, activation_set_id, profile_ref, qualification_tier)
);

CREATE TABLE active_aliases (
  workspace_id TEXT NOT NULL,
  profile_ref TEXT NOT NULL,
  activation_set_id TEXT NOT NULL REFERENCES generation_sets(activation_set_id),
  manifest_sha256 TEXT NOT NULL,
  qualification_id TEXT NOT NULL,
  qualification_tier TEXT NOT NULL CHECK (qualification_tier = 'preview'),
  version INTEGER NOT NULL CHECK (version > 0),
  PRIMARY KEY (workspace_id, profile_ref),
  FOREIGN KEY (qualification_id, activation_set_id, profile_ref, qualification_tier)
    REFERENCES verified_qualifications (
      qualification_id, activation_set_id, profile_ref, qualification_tier
    )
);

CREATE TABLE query_leases (
  lease_id TEXT PRIMARY KEY,
  workspace_id TEXT NOT NULL,
  profile_ref TEXT NOT NULL,
  activation_set_id TEXT NOT NULL REFERENCES generation_sets(activation_set_id),
  manifest_sha256 TEXT NOT NULL,
  qualification_id TEXT NOT NULL,
  qualification_tier TEXT NOT NULL CHECK (qualification_tier = 'preview'),
  alias_version INTEGER NOT NULL CHECK (alias_version > 0),
  owner_id TEXT NOT NULL,
  owner_session_epoch TEXT NOT NULL,
  acquired_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  released_at TEXT,
  FOREIGN KEY (qualification_id, activation_set_id, profile_ref, qualification_tier)
    REFERENCES verified_qualifications (
      qualification_id, activation_set_id, profile_ref, qualification_tier
    )
);

CREATE TABLE generation_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT,
  activation_set_id TEXT NOT NULL REFERENCES generation_sets(activation_set_id),
  event_type TEXT NOT NULL,
  occurred_at TEXT NOT NULL,
  details_json TEXT NOT NULL
);

CREATE TRIGGER generation_sets_no_update BEFORE UPDATE ON generation_sets
BEGIN SELECT RAISE(ABORT, 'generation_sets are append-only'); END;
CREATE TRIGGER generation_sets_no_delete BEFORE DELETE ON generation_sets
BEGIN SELECT RAISE(ABORT, 'generation_sets are append-only'); END;
CREATE TRIGGER qualifications_no_update BEFORE UPDATE ON verified_qualifications
BEGIN SELECT RAISE(ABORT, 'verified_qualifications are append-only'); END;
CREATE TRIGGER qualifications_no_delete BEFORE DELETE ON verified_qualifications
BEGIN SELECT RAISE(ABORT, 'verified_qualifications are append-only'); END;
CREATE TRIGGER generation_events_no_update BEFORE UPDATE ON generation_events
BEGIN SELECT RAISE(ABORT, 'generation_events are append-only'); END;
CREATE TRIGGER generation_events_no_delete BEFORE DELETE ON generation_events
BEGIN SELECT RAISE(ABORT, 'generation_events are append-only'); END;
"""

_GENERATION_SETS_CANDIDATE_INDEX_SQL = """
CREATE UNIQUE INDEX IF NOT EXISTS generation_sets_candidate_identity
ON generation_sets (activation_set_id, manifest_sha256, retrieval_profile_ref)
"""

_PUBLISHED_CANDIDATE_LEASES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS published_candidate_leases (
  lease_id TEXT PRIMARY KEY,
  activation_set_id TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  profile_ref TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  owner_session_epoch TEXT NOT NULL,
  acquired_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  released_at TEXT,
  FOREIGN KEY (activation_set_id, manifest_sha256, profile_ref)
    REFERENCES generation_sets (
      activation_set_id, manifest_sha256, retrieval_profile_ref
    )
)
"""

_PUBLISHED_CANDIDATE_QUERY_ID_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS published_candidate_leases_query_id_unique
BEFORE INSERT ON published_candidate_leases
WHEN EXISTS (SELECT 1 FROM query_leases WHERE lease_id = NEW.lease_id)
BEGIN SELECT RAISE(ABORT, 'lease_id already exists'); END
"""

_QUERY_CANDIDATE_ID_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS query_leases_candidate_id_unique
BEFORE INSERT ON query_leases
WHEN EXISTS (SELECT 1 FROM published_candidate_leases WHERE lease_id = NEW.lease_id)
BEGIN SELECT RAISE(ABORT, 'lease_id already exists'); END
"""

_QUERY_LEASE_IDENTITY_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS query_leases_identity_immutable
BEFORE UPDATE ON query_leases
WHEN NEW.lease_id IS NOT OLD.lease_id
  OR NEW.workspace_id IS NOT OLD.workspace_id
  OR NEW.profile_ref IS NOT OLD.profile_ref
  OR NEW.activation_set_id IS NOT OLD.activation_set_id
  OR NEW.manifest_sha256 IS NOT OLD.manifest_sha256
  OR NEW.qualification_id IS NOT OLD.qualification_id
  OR NEW.qualification_tier IS NOT OLD.qualification_tier
  OR NEW.alias_version IS NOT OLD.alias_version
  OR NEW.owner_id IS NOT OLD.owner_id
  OR NEW.owner_session_epoch IS NOT OLD.owner_session_epoch
  OR NEW.acquired_at IS NOT OLD.acquired_at
BEGIN SELECT RAISE(ABORT, 'query lease identity is immutable'); END
"""

_PUBLISHED_CANDIDATE_IDENTITY_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS published_candidate_leases_identity_immutable
BEFORE UPDATE ON published_candidate_leases
WHEN NEW.lease_id IS NOT OLD.lease_id
  OR NEW.activation_set_id IS NOT OLD.activation_set_id
  OR NEW.manifest_sha256 IS NOT OLD.manifest_sha256
  OR NEW.profile_ref IS NOT OLD.profile_ref
  OR NEW.owner_id IS NOT OLD.owner_id
  OR NEW.owner_session_epoch IS NOT OLD.owner_session_epoch
  OR NEW.acquired_at IS NOT OLD.acquired_at
  OR NEW.expires_at IS NOT OLD.expires_at
  OR OLD.released_at IS NOT NULL
  OR NEW.released_at IS NULL
BEGIN SELECT RAISE(ABORT, 'published candidate lease identity is immutable'); END
"""

_V2_OBJECT_SQL = (
    ("index", "generation_sets_candidate_identity", _GENERATION_SETS_CANDIDATE_INDEX_SQL),
    ("table", "published_candidate_leases", _PUBLISHED_CANDIDATE_LEASES_TABLE_SQL),
    (
        "trigger",
        "published_candidate_leases_query_id_unique",
        _PUBLISHED_CANDIDATE_QUERY_ID_TRIGGER_SQL,
    ),
    ("trigger", "query_leases_candidate_id_unique", _QUERY_CANDIDATE_ID_TRIGGER_SQL),
    ("trigger", "query_leases_identity_immutable", _QUERY_LEASE_IDENTITY_TRIGGER_SQL),
    (
        "trigger",
        "published_candidate_leases_identity_immutable",
        _PUBLISHED_CANDIDATE_IDENTITY_TRIGGER_SQL,
    ),
)

_V2_SCHEMA = ";\n\n".join(sql.strip() for _, _, sql in _V2_OBJECT_SQL)
_SCHEMA_MIGRATION_BUSY_TIMEOUT_MS = 1_000

_CANDIDATE_LEASE_TABLE_INFO = (
    ("lease_id", "TEXT", 0, None, 1),
    ("activation_set_id", "TEXT", 1, None, 0),
    ("manifest_sha256", "TEXT", 1, None, 0),
    ("profile_ref", "TEXT", 1, None, 0),
    ("owner_id", "TEXT", 1, None, 0),
    ("owner_session_epoch", "TEXT", 1, None, 0),
    ("acquired_at", "TEXT", 1, None, 0),
    ("expires_at", "TEXT", 1, None, 0),
    ("released_at", "TEXT", 0, None, 0),
)

_CANDIDATE_LEASE_FOREIGN_KEYS = (
    (0, "generation_sets", "activation_set_id", "activation_set_id", "NO ACTION", "NO ACTION"),
    (1, "generation_sets", "manifest_sha256", "manifest_sha256", "NO ACTION", "NO ACTION"),
    (2, "generation_sets", "profile_ref", "retrieval_profile_ref", "NO ACTION", "NO ACTION"),
)


class SqliteGenerationCatalog:
    """Persist preview activation and query leases with atomic SQLite CAS."""

    def __init__(
        self,
        database_path: Path,
        connection: sqlite3.Connection,
        verifier: PreviewActivationVerifier,
        owner_session_registry: OwnerSessionRegistry | None,
    ) -> None:
        self.database_path = database_path
        self._connection = connection
        self._verifier = verifier
        self._owner_session_registry = owner_session_registry
        self._connection_lock = RLock()
        self._closed = False

    @classmethod
    def open(
        cls,
        database_path: Path,
        *,
        verifier: PreviewActivationVerifier,
        owner_session_registry: OwnerSessionRegistry | None = None,
    ) -> Self:
        if not isinstance(database_path, Path) or not database_path.is_absolute():
            raise ValueError("database_path must be an absolute Path")
        if owner_session_registry is not None and not callable(
            getattr(owner_session_registry, "prove_dead", None)
        ):
            raise TypeError("owner_session_registry must provide prove_dead")
        database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        database_path.parent.chmod(0o700)
        is_new = not database_path.exists() or database_path.stat().st_size == 0
        connection = sqlite3.connect(
            database_path,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            )
            if is_new and not tables and application_id == 0 and user_version == 0:
                cls._initialize(connection)
            elif application_id != SQLITE_GENERATION_APPLICATION_ID:
                raise ValueError("unsupported generation lifecycle database identity or version")
            elif user_version == 1:
                cls._migrate_v1_to_v2(connection)
            elif user_version == SQLITE_GENERATION_SCHEMA_VERSION:
                cls._validate_current_schema(connection)
            else:
                raise ValueError("unsupported generation lifecycle database identity or version")
            connection.execute("PRAGMA query_only=OFF")
            database_path.chmod(0o600)
            return cls(database_path, connection, verifier, owner_session_registry)
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        try:
            connection.executescript(f"BEGIN EXCLUSIVE;\n{_SCHEMA}\n{_V2_SCHEMA}")
            connection.execute(f"PRAGMA application_id={SQLITE_GENERATION_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={SQLITE_GENERATION_SCHEMA_VERSION}")
            SqliteGenerationCatalog._validate_v2_schema(connection)
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        """Migrate only the exact legacy v1 schema in one exclusive transaction."""
        try:
            connection.execute(f"PRAGMA busy_timeout={_SCHEMA_MIGRATION_BUSY_TIMEOUT_MS}")
            connection.execute("BEGIN EXCLUSIVE")
            SqliteGenerationCatalog._validate_database_identity(connection, user_version=1)
            SqliteGenerationCatalog._validate_schema_objects(
                connection,
                include_v2=False,
            )
            SqliteGenerationCatalog._validate_database_integrity(connection)
            for _object_type, _name, statement in _V2_OBJECT_SQL:
                connection.execute(statement)
            SqliteGenerationCatalog._validate_v2_schema(connection)
            connection.execute(f"PRAGMA user_version={SQLITE_GENERATION_SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except sqlite3.OperationalError as error:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise TimeoutError("generation lifecycle schema migration remained busy") from error
            raise
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _validate_current_schema(connection: sqlite3.Connection) -> None:
        """Validate v2 without creating, replacing, or repairing schema objects."""
        try:
            connection.execute(f"PRAGMA busy_timeout={_SCHEMA_MIGRATION_BUSY_TIMEOUT_MS}")
            connection.execute("BEGIN")
            SqliteGenerationCatalog._validate_database_identity(
                connection,
                user_version=SQLITE_GENERATION_SCHEMA_VERSION,
            )
            SqliteGenerationCatalog._validate_v2_schema(connection)
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _validate_v2_schema(connection: sqlite3.Connection) -> None:
        SqliteGenerationCatalog._validate_schema_objects(connection, include_v2=True)
        for object_type, name, expected_sql in _V2_OBJECT_SQL:
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, name),
            ).fetchone()
            if row is None or _canonical_schema_sql(str(row[0])) != _canonical_schema_sql(
                expected_sql
            ):
                raise ValueError(f"generation lifecycle schema object {name!r} is incompatible")

        table_info = tuple(
            (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
            for row in connection.execute("PRAGMA table_info(published_candidate_leases)")
        )
        if table_info != _CANDIDATE_LEASE_TABLE_INFO:
            raise ValueError("generation lifecycle candidate lease table is incompatible")

        foreign_keys = tuple(
            (
                int(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
                str(row[6]),
            )
            for row in connection.execute("PRAGMA foreign_key_list(published_candidate_leases)")
        )
        if foreign_keys != _CANDIDATE_LEASE_FOREIGN_KEYS:
            raise ValueError("generation lifecycle candidate lease foreign keys are incompatible")

        candidate_indexes = tuple(
            (str(row[1]), int(row[2]), str(row[3]), int(row[4]))
            for row in connection.execute("PRAGMA index_list(published_candidate_leases)")
        )
        if len(candidate_indexes) != 1 or candidate_indexes[0][1:] != (1, "pk", 0):
            raise ValueError("generation lifecycle candidate lease primary key is incompatible")
        candidate_index_columns = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM pragma_index_info(?)",
                (candidate_indexes[0][0],),
            )
        )
        if candidate_index_columns != ("lease_id",):
            raise ValueError("generation lifecycle candidate lease primary key is incompatible")

        generation_indexes = {
            str(row[1]): (int(row[2]), str(row[3]), int(row[4]))
            for row in connection.execute("PRAGMA index_list(generation_sets)")
        }
        if generation_indexes.get("generation_sets_candidate_identity") != (1, "c", 0):
            raise ValueError("generation lifecycle candidate identity index is incompatible")
        candidate_identity_columns = tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM pragma_index_info(?)",
                ("generation_sets_candidate_identity",),
            )
        )
        if candidate_identity_columns != (
            "activation_set_id",
            "manifest_sha256",
            "retrieval_profile_ref",
        ):
            raise ValueError("generation lifecycle candidate identity index is incompatible")

        duplicate_lease_id = connection.execute(
            """
            SELECT query_leases.lease_id
              FROM query_leases
              JOIN published_candidate_leases USING (lease_id)
             LIMIT 1
            """
        ).fetchone()
        if duplicate_lease_id is not None:
            raise ValueError("generation lifecycle lease identity exists in multiple tables")
        SqliteGenerationCatalog._validate_database_integrity(connection)

    @staticmethod
    def _validate_schema_objects(
        connection: sqlite3.Connection,
        *,
        include_v2: bool,
    ) -> None:
        expected = _expected_schema_objects(include_v2=include_v2)
        observed = {
            (str(row[0]), str(row[1])): _canonical_schema_sql(str(row[2]))
            for row in connection.execute(
                """
                SELECT type, name, sql FROM sqlite_master
                 WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        if observed != expected:
            version = "v2" if include_v2 else "v1"
            missing = sorted(f"{kind}:{name}" for kind, name in expected.keys() - observed.keys())
            unexpected = sorted(
                f"{kind}:{name}" for kind, name in observed.keys() - expected.keys()
            )
            changed = sorted(
                f"{kind}:{name}"
                for kind, name in expected.keys() & observed.keys()
                if expected[(kind, name)] != observed[(kind, name)]
            )
            raise ValueError(
                f"generation lifecycle {version} schema is incompatible; "
                f"missing={missing}, unexpected={unexpected}, changed={changed}"
            )

    @staticmethod
    def _validate_database_integrity(connection: sqlite3.Connection) -> None:
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("generation lifecycle schema has invalid foreign keys")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("generation lifecycle schema failed integrity_check")

    @staticmethod
    def _validate_database_identity(
        connection: sqlite3.Connection,
        *,
        user_version: int,
    ) -> None:
        observed_application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        observed_user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if (
            observed_application_id != SQLITE_GENERATION_APPLICATION_ID
            or observed_user_version != user_version
        ):
            raise ValueError("generation lifecycle database identity changed while opening")

    def close(self) -> None:
        with self._connection_lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def register_published(
        self,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> None:
        with self._connection_lock:
            self._register_published(published, deadline=deadline)

    def _register_published(
        self,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> None:
        if not isinstance(published, PublishedGenerationRecord):
            raise TypeError("published must be a PublishedGenerationRecord")
        self._begin(deadline)
        try:
            current = self._fetch_published_row(published.activation_set_id)
            if current is not None:
                if current != published:
                    if current.manifest_sha256 != published.manifest_sha256:
                        raise ValueError("activation set already has a different manifest")
                    raise ValueError("activation set publication identity conflicts")
                self._connection.execute("COMMIT")
                return
            self._connection.execute(
                """
                INSERT INTO generation_sets (
                  activation_set_id, manifest_sha256, published_root, workspace_id,
                  corpus_ref, retrieval_profile_ref, published_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    published.activation_set_id,
                    published.manifest_sha256,
                    str(published.published_root),
                    published.workspace_id,
                    published.corpus_ref,
                    published.retrieval_profile_ref,
                    _encode_datetime(published.published_at),
                ),
            )
            self._append_event(
                published.activation_set_id,
                "published",
                published.published_at,
                {"manifest_sha256": published.manifest_sha256},
            )
            self._connection.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def resolve_published(
        self,
        activation_set_id: str,
        *,
        deadline: datetime,
    ) -> PublishedGenerationRecord | None:
        with self._connection_lock:
            return self._resolve_published(activation_set_id, deadline=deadline)

    def _resolve_published(
        self,
        activation_set_id: str,
        *,
        deadline: datetime,
    ) -> PublishedGenerationRecord | None:
        _non_blank("activation_set_id", activation_set_id)
        self._configure_deadline(deadline)
        return self._fetch_published_row(activation_set_id)

    def acquire_published(
        self,
        *,
        activation_set_id: str,
        expected_manifest_sha256: str,
        profile_ref: str,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        acquired_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> PublishedCandidateLeaseRecord:
        """Lease one exact published candidate without consulting an active alias."""
        with self._connection_lock:
            return self._acquire_published(
                activation_set_id=activation_set_id,
                expected_manifest_sha256=expected_manifest_sha256,
                profile_ref=profile_ref,
                lease_id=lease_id,
                owner_id=owner_id,
                owner_session_epoch=owner_session_epoch,
                acquired_at=acquired_at,
                expires_at=expires_at,
                deadline=deadline,
            )

    def _acquire_published(
        self,
        *,
        activation_set_id: str,
        expected_manifest_sha256: str,
        profile_ref: str,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        acquired_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> PublishedCandidateLeaseRecord:
        for name, value in (
            ("activation_set_id", activation_set_id),
            ("profile_ref", profile_ref),
            ("lease_id", lease_id),
            ("owner_id", owner_id),
            ("owner_session_epoch", owner_session_epoch),
        ):
            _non_blank(name, value)
        _sha256("expected_manifest_sha256", expected_manifest_sha256)
        _lease_interval(acquired_at, expires_at)
        self._begin(deadline)
        try:
            self._configure_deadline(deadline)
            published = self._fetch_published_row(activation_set_id)
            self._configure_deadline(deadline)
            if (
                published is None
                or published.manifest_sha256 != expected_manifest_sha256
                or published.retrieval_profile_ref != profile_ref
            ):
                raise GenerationQualificationError("published candidate identity mismatch")
            self._configure_deadline(deadline)
            self._connection.execute(
                """
                INSERT INTO published_candidate_leases (
                  lease_id, activation_set_id, manifest_sha256, profile_ref, owner_id,
                  owner_session_epoch, acquired_at, expires_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    lease_id,
                    activation_set_id,
                    expected_manifest_sha256,
                    profile_ref,
                    owner_id,
                    owner_session_epoch,
                    _encode_datetime(acquired_at),
                    _encode_datetime(expires_at),
                ),
            )
            self._configure_deadline(deadline)
            self._commit_before_deadline(deadline)
            return PublishedCandidateLeaseRecord(
                lease_id=lease_id,
                activation_set_id=activation_set_id,
                manifest_sha256=expected_manifest_sha256,
                profile_ref=profile_ref,
                owner_id=owner_id,
                owner_session_epoch=owner_session_epoch,
                acquired_at=acquired_at,
                expires_at=expires_at,
                released_at=None,
            )
        except BaseException:
            self._rollback()
            raise

    def resolve_active(
        self,
        workspace_id: str,
        profile_ref: str,
        *,
        deadline: datetime,
    ) -> GenerationAliasSnapshot | None:
        with self._connection_lock:
            return self._resolve_active(workspace_id, profile_ref, deadline=deadline)

    def _resolve_active(
        self,
        workspace_id: str,
        profile_ref: str,
        *,
        deadline: datetime,
    ) -> GenerationAliasSnapshot | None:
        _non_blank("workspace_id", workspace_id)
        _non_blank("profile_ref", profile_ref)
        self._configure_deadline(deadline)
        row = self._connection.execute(
            """
            SELECT workspace_id, profile_ref, activation_set_id, manifest_sha256,
                   qualification_id, qualification_tier, version
              FROM active_aliases WHERE workspace_id = ? AND profile_ref = ?
            """,
            (workspace_id, profile_ref),
        ).fetchone()
        return None if row is None else _alias_from_row(row)

    def activate(
        self,
        *,
        workspace_id: str,
        profile_ref: str,
        activation_set_id: str,
        expected_previous_snapshot: GenerationAliasSnapshot | None,
        qualification: PreviewActivationQualification,
        activated_at: datetime,
        deadline: datetime,
    ) -> GenerationAliasSnapshot:
        with self._connection_lock:
            return self._activate(
                workspace_id=workspace_id,
                profile_ref=profile_ref,
                activation_set_id=activation_set_id,
                expected_previous_snapshot=expected_previous_snapshot,
                qualification=qualification,
                activated_at=activated_at,
                deadline=deadline,
            )

    def _activate(
        self,
        *,
        workspace_id: str,
        profile_ref: str,
        activation_set_id: str,
        expected_previous_snapshot: GenerationAliasSnapshot | None,
        qualification: PreviewActivationQualification,
        activated_at: datetime,
        deadline: datetime,
    ) -> GenerationAliasSnapshot:
        for name, value in (
            ("workspace_id", workspace_id),
            ("profile_ref", profile_ref),
            ("activation_set_id", activation_set_id),
        ):
            _non_blank(name, value)
        if expected_previous_snapshot is not None and not isinstance(
            expected_previous_snapshot,
            GenerationAliasSnapshot,
        ):
            raise TypeError("expected_previous_snapshot must be a GenerationAliasSnapshot")
        if not isinstance(qualification, PreviewActivationQualification):
            raise TypeError("qualification must be a PreviewActivationQualification")
        _aware("activated_at", activated_at)
        self._configure_deadline(deadline)
        published = self._fetch_published_row(activation_set_id)
        if published is None:
            raise GenerationQualificationError("activation set is not published")
        if published.workspace_id != workspace_id or published.retrieval_profile_ref != profile_ref:
            raise GenerationQualificationError("publication does not match alias scope")
        verified = self._verifier.verify(qualification, published, deadline=deadline)
        self._validate_verified(qualification, published, verified)

        self._begin(deadline)
        try:
            current_published = self._fetch_published_row(activation_set_id)
            if current_published != published:
                raise GenerationQualificationError("publication changed during verification")
            self._record_verified(verified)
            current_row = self._connection.execute(
                """
                SELECT workspace_id, profile_ref, activation_set_id, manifest_sha256,
                       qualification_id, qualification_tier, version
                  FROM active_aliases WHERE workspace_id = ? AND profile_ref = ?
                """,
                (workspace_id, profile_ref),
            ).fetchone()
            current = None if current_row is None else _alias_from_row(current_row)
            if current != expected_previous_snapshot:
                raise ConcurrentActivationError("active alias changed before activation")
            version = 1 if current is None else current.version + 1
            snapshot = GenerationAliasSnapshot(
                workspace_id=workspace_id,
                profile_ref=profile_ref,
                activation_set_id=activation_set_id,
                manifest_sha256=published.manifest_sha256,
                qualification_id=verified.qualification_id,
                qualification_tier="preview",
                version=version,
            )
            if current is None:
                self._connection.execute(
                    """
                    INSERT INTO active_aliases (
                      workspace_id, profile_ref, activation_set_id, manifest_sha256,
                      qualification_id, qualification_tier, version
                    ) VALUES (?, ?, ?, ?, ?, 'preview', ?)
                    """,
                    (
                        workspace_id,
                        profile_ref,
                        activation_set_id,
                        published.manifest_sha256,
                        verified.qualification_id,
                        version,
                    ),
                )
            else:
                updated = self._connection.execute(
                    """
                    UPDATE active_aliases
                       SET activation_set_id = ?, manifest_sha256 = ?, qualification_id = ?,
                           qualification_tier = 'preview', version = ?
                     WHERE workspace_id = ? AND profile_ref = ? AND version = ?
                    """,
                    (
                        activation_set_id,
                        published.manifest_sha256,
                        verified.qualification_id,
                        version,
                        workspace_id,
                        profile_ref,
                        current.version,
                    ),
                )
                if updated.rowcount != 1:
                    raise ConcurrentActivationError("active alias changed during activation")
            event_type = (
                "activated"
                if current is None
                else "requalified"
                if current.activation_set_id == activation_set_id
                else "superseded"
            )
            self._append_event(
                activation_set_id,
                event_type,
                activated_at,
                {
                    "qualification_id": verified.qualification_id,
                    "version": version,
                    "previous_activation_set_id": (
                        None if current is None else current.activation_set_id
                    ),
                },
            )
            self._connection.execute("COMMIT")
            return snapshot
        except BaseException:
            self._rollback()
            raise

    def acquire(
        self,
        *,
        workspace_id: str,
        profile_ref: str,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        acquired_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> GenerationLeaseRecord:
        with self._connection_lock:
            return self._acquire(
                workspace_id=workspace_id,
                profile_ref=profile_ref,
                lease_id=lease_id,
                owner_id=owner_id,
                owner_session_epoch=owner_session_epoch,
                acquired_at=acquired_at,
                expires_at=expires_at,
                deadline=deadline,
            )

    def _acquire(
        self,
        *,
        workspace_id: str,
        profile_ref: str,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        acquired_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> GenerationLeaseRecord:
        for name, value in (
            ("workspace_id", workspace_id),
            ("profile_ref", profile_ref),
            ("lease_id", lease_id),
            ("owner_id", owner_id),
            ("owner_session_epoch", owner_session_epoch),
        ):
            _non_blank(name, value)
        _lease_interval(acquired_at, expires_at)
        self._begin(deadline)
        try:
            row = self._connection.execute(
                """
                SELECT workspace_id, profile_ref, activation_set_id, manifest_sha256,
                       qualification_id, qualification_tier, version
                  FROM active_aliases WHERE workspace_id = ? AND profile_ref = ?
                """,
                (workspace_id, profile_ref),
            ).fetchone()
            if row is None:
                raise GenerationQualificationError("no qualified active generation")
            alias = _alias_from_row(row)
            self._connection.execute(
                """
                INSERT INTO query_leases (
                  lease_id, workspace_id, profile_ref, activation_set_id, manifest_sha256,
                  qualification_id, qualification_tier, alias_version, owner_id,
                  owner_session_epoch, acquired_at, expires_at, released_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'preview', ?, ?, ?, ?, ?, NULL)
                """,
                (
                    lease_id,
                    workspace_id,
                    profile_ref,
                    alias.activation_set_id,
                    alias.manifest_sha256,
                    alias.qualification_id,
                    alias.version,
                    owner_id,
                    owner_session_epoch,
                    _encode_datetime(acquired_at),
                    _encode_datetime(expires_at),
                ),
            )
            self._connection.execute("COMMIT")
            return GenerationLeaseRecord(
                lease_id=lease_id,
                owner_id=owner_id,
                owner_session_epoch=owner_session_epoch,
                acquired_at=acquired_at,
                expires_at=expires_at,
                released_at=None,
                alias_snapshot=alias,
            )
        except BaseException:
            self._rollback()
            raise

    def renew(
        self,
        *,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        expected_expires_at: datetime,
        renewed_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> GenerationLeaseRecord:
        with self._connection_lock:
            return self._renew(
                lease_id=lease_id,
                owner_id=owner_id,
                owner_session_epoch=owner_session_epoch,
                expected_expires_at=expected_expires_at,
                renewed_at=renewed_at,
                expires_at=expires_at,
                deadline=deadline,
            )

    def _renew(
        self,
        *,
        lease_id: str,
        owner_id: str,
        owner_session_epoch: str,
        expected_expires_at: datetime,
        renewed_at: datetime,
        expires_at: datetime,
        deadline: datetime,
    ) -> GenerationLeaseRecord:
        for name, value in (
            ("lease_id", lease_id),
            ("owner_id", owner_id),
            ("owner_session_epoch", owner_session_epoch),
        ):
            _non_blank(name, value)
        _aware("expected_expires_at", expected_expires_at)
        _lease_interval(renewed_at, expires_at)
        if expires_at <= expected_expires_at:
            raise ValueError("lease renewal must advance expiry")
        self._begin(deadline)
        try:
            updated = self._connection.execute(
                """
                UPDATE query_leases SET expires_at = ?
                 WHERE lease_id = ? AND owner_id = ? AND owner_session_epoch = ?
                   AND expires_at = ? AND released_at IS NULL
                """,
                (
                    _encode_datetime(expires_at),
                    lease_id,
                    owner_id,
                    owner_session_epoch,
                    _encode_datetime(expected_expires_at),
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("lease renewal compare-and-swap failed")
            row = self._connection.execute(
                "SELECT * FROM query_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            self._connection.execute("COMMIT")
            if row is None:  # pragma: no cover - same transaction invariant
                raise ValueError("lease renewal row disappeared")
            return _lease_from_row(row)
        except BaseException:
            self._rollback()
            raise

    def release(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        with self._connection_lock:
            self._release(lease_id, released_at=released_at, deadline=deadline)

    def _release(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        _non_blank("lease_id", lease_id)
        _aware("released_at", released_at)
        self._begin(deadline)
        try:
            row = self._connection.execute(
                "SELECT released_at FROM query_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            if row is None:
                raise ValueError("lease does not exist")
            if row[0] is None:
                self._connection.execute(
                    "UPDATE query_leases SET released_at = ? WHERE lease_id = ?",
                    (_encode_datetime(released_at), lease_id),
                )
            self._connection.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def release_published(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        """Release only one exact published-candidate lease."""
        with self._connection_lock:
            self._release_published(
                lease_id,
                released_at=released_at,
                deadline=deadline,
            )

    def _release_published(
        self,
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        _non_blank("lease_id", lease_id)
        _aware("released_at", released_at)
        self._begin(deadline)
        try:
            self._configure_deadline(deadline)
            row = self._connection.execute(
                "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
                (lease_id,),
            ).fetchone()
            self._configure_deadline(deadline)
            if row is None:
                raise ValueError("published candidate lease does not exist")
            if row[0] is None:
                self._connection.execute(
                    "UPDATE published_candidate_leases SET released_at = ? WHERE lease_id = ?",
                    (_encode_datetime(released_at), lease_id),
                )
            self._configure_deadline(deadline)
            self._commit_before_deadline(deadline)
        except BaseException:
            self._rollback()
            raise

    def recover_stale_published(
        self,
        *,
        owner_id: str,
        owner_session_epoch: str,
        recovered_at: datetime,
        deadline: datetime,
    ) -> int:
        """Release candidate leases only after external proof of exact owner death."""
        with self._connection_lock:
            return self._recover_stale_published(
                owner_id=owner_id,
                owner_session_epoch=owner_session_epoch,
                recovered_at=recovered_at,
                deadline=deadline,
            )

    def _recover_stale_published(
        self,
        *,
        owner_id: str,
        owner_session_epoch: str,
        recovered_at: datetime,
        deadline: datetime,
    ) -> int:
        _non_blank("owner_id", owner_id)
        _non_blank("owner_session_epoch", owner_session_epoch)
        _aware("recovered_at", recovered_at)
        self._configure_deadline(deadline)
        registry = self._owner_session_registry
        if registry is None:
            raise RuntimeError("candidate owner session registry is not configured")
        proven_dead = registry.prove_dead(owner_id, owner_session_epoch, deadline=deadline)
        self._configure_deadline(deadline)
        if proven_dead is not True:
            raise GenerationQualificationError("candidate owner session is not proven dead")
        self._begin(deadline)
        try:
            self._configure_deadline(deadline)
            updated = self._connection.execute(
                """
                UPDATE published_candidate_leases SET released_at = ?
                 WHERE owner_id = ? AND owner_session_epoch = ? AND released_at IS NULL
                """,
                (_encode_datetime(recovered_at), owner_id, owner_session_epoch),
            )
            self._configure_deadline(deadline)
            self._commit_before_deadline(deadline)
            return updated.rowcount
        except BaseException:
            self._rollback()
            raise

    def _validate_verified(
        self,
        qualification: PreviewActivationQualification,
        published: PublishedGenerationRecord,
        verified: VerifiedPreviewQualification,
    ) -> None:
        if not isinstance(verified, VerifiedPreviewQualification):
            raise GenerationQualificationError("verifier returned an invalid record")
        if (
            verified.qualification_id != qualification.qualification_id
            or verified.activation_set_id != published.activation_set_id
            or verified.manifest_sha256 != published.manifest_sha256
            or verified.profile_ref != published.retrieval_profile_ref
        ):
            raise GenerationQualificationError("verified qualification identity mismatch")

    def _record_verified(self, verified: VerifiedPreviewQualification) -> None:
        encoded = _canonical_json(
            {
                **asdict(verified),
                "verified_at": _encode_datetime(verified.verified_at),
                "qualification_tier": "preview",
            }
        ).decode("utf-8")
        current = self._connection.execute(
            "SELECT record_json FROM verified_qualifications WHERE qualification_id = ?",
            (verified.qualification_id,),
        ).fetchone()
        if current is not None:
            if current[0] != encoded:
                raise GenerationQualificationError("qualification ID conflicts")
            return
        self._connection.execute(
            """
            INSERT INTO verified_qualifications (
              qualification_id, activation_set_id, manifest_sha256, profile_ref,
              report_sha256, policy_sha256, verification_receipt_sha256,
              qualification_tier, verified_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'preview', ?, ?)
            """,
            (
                verified.qualification_id,
                verified.activation_set_id,
                verified.manifest_sha256,
                verified.profile_ref,
                verified.report_sha256,
                verified.policy_sha256,
                verified.verification_receipt_sha256,
                _encode_datetime(verified.verified_at),
                encoded,
            ),
        )

    def _fetch_published_row(self, activation_set_id: str) -> PublishedGenerationRecord | None:
        row = self._connection.execute(
            """
            SELECT activation_set_id, manifest_sha256, published_root, workspace_id,
                   corpus_ref, retrieval_profile_ref, published_at
              FROM generation_sets WHERE activation_set_id = ?
            """,
            (activation_set_id,),
        ).fetchone()
        if row is None:
            return None
        return PublishedGenerationRecord(
            activation_set_id=row[0],
            manifest_sha256=row[1],
            published_root=Path(row[2]),
            workspace_id=row[3],
            corpus_ref=row[4],
            retrieval_profile_ref=row[5],
            published_at=_decode_datetime(row[6]),
        )

    def _append_event(
        self,
        activation_set_id: str,
        event_type: str,
        occurred_at: datetime,
        details: dict[str, object],
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO generation_events (
              activation_set_id, event_type, occurred_at, details_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                activation_set_id,
                event_type,
                _encode_datetime(occurred_at),
                _canonical_json(details).decode("utf-8"),
            ),
        )

    def _configure_deadline(self, deadline: datetime) -> None:
        _aware("deadline", deadline)
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("deadline exceeded")
        busy_ms = max(1, min(5_000, int(remaining * 1_000)))
        self._connection.execute(f"PRAGMA busy_timeout={busy_ms}")

    def _begin(self, deadline: datetime) -> None:
        self._configure_deadline(deadline)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise TimeoutError("generation lifecycle database remained busy") from error
            raise

    def _commit_before_deadline(self, deadline: datetime) -> None:
        """Interrupt this transaction's COMMIT at its deadline without a stray timer."""
        self._configure_deadline(deadline)
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("deadline exceeded")
        interrupted = Event()

        def interrupt_commit() -> None:
            interrupted.set()
            self._connection.interrupt()

        timer = Timer(remaining, interrupt_commit)
        timer.daemon = True
        timer.start()
        try:
            self._connection.execute("COMMIT")
            # A successful SQLite COMMIT is authoritative. Never report a timeout
            # after durability may already be visible to another connection.
        except sqlite3.OperationalError as error:
            if (
                interrupted.is_set()
                and getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT
                and self._connection.in_transaction
            ):
                raise TimeoutError("deadline exceeded during commit") from error
            raise
        finally:
            timer.cancel()
            timer.join()

    def _rollback(self) -> None:
        if self._connection.in_transaction:
            self._connection.execute("ROLLBACK")


def _alias_from_row(row: tuple[object, ...]) -> GenerationAliasSnapshot:
    return GenerationAliasSnapshot(
        workspace_id=str(row[0]),
        profile_ref=str(row[1]),
        activation_set_id=str(row[2]),
        manifest_sha256=str(row[3]),
        qualification_id=str(row[4]),
        qualification_tier=str(row[5]),
        version=_stored_int(row[6]),
    )


def _lease_from_row(row: tuple[object, ...]) -> GenerationLeaseRecord:
    alias = GenerationAliasSnapshot(
        workspace_id=str(row[1]),
        profile_ref=str(row[2]),
        activation_set_id=str(row[3]),
        manifest_sha256=str(row[4]),
        qualification_id=str(row[5]),
        qualification_tier=str(row[6]),
        version=_stored_int(row[7]),
    )
    return GenerationLeaseRecord(
        lease_id=str(row[0]),
        owner_id=str(row[8]),
        owner_session_epoch=str(row[9]),
        acquired_at=_decode_datetime(str(row[10])),
        expires_at=_decode_datetime(str(row[11])),
        released_at=(None if row[12] is None else _decode_datetime(str(row[12]))),
        alias_snapshot=alias,
    )


def _canonical_schema_sql(value: str) -> str:
    without_condition = re.sub(r"\bIF\s+NOT\s+EXISTS\b", "", value, flags=re.IGNORECASE)
    return " ".join(without_condition.rstrip("; \n\t").split()).casefold()


def _expected_schema_objects(*, include_v2: bool) -> dict[tuple[str, str], str]:
    """Build the exact committed schema shape in an isolated in-memory catalog."""
    expected = sqlite3.connect(":memory:", isolation_level=None)
    try:
        expected.executescript(_SCHEMA)
        if include_v2:
            for _object_type, _name, statement in _V2_OBJECT_SQL:
                expected.execute(statement)
        return {
            (str(row[0]), str(row[1])): _canonical_schema_sql(str(row[2]))
            for row in expected.execute(
                """
                SELECT type, name, sql FROM sqlite_master
                 WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
                """
            )
        }
    finally:
        expected.close()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _lease_interval(acquired_at: datetime, expires_at: datetime) -> None:
    _aware("acquired_at", acquired_at)
    _aware("expires_at", expires_at)
    if expires_at <= acquired_at or expires_at - acquired_at > MAX_LEASE_INTERVAL:
        raise ValueError("lease interval must be positive and no greater than 10 minutes")


def _encode_datetime(value: datetime) -> str:
    _aware("datetime", value)
    return value.isoformat()


def _decode_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    _aware("stored datetime", parsed)
    return parsed


def _stored_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("stored integer has an invalid type")
    return value


def _non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


def _sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _aware(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "SQLITE_GENERATION_APPLICATION_ID",
    "SQLITE_GENERATION_SCHEMA_VERSION",
    "ConcurrentActivationError",
    "GenerationAliasSnapshot",
    "GenerationLeaseRecord",
    "GenerationQualificationError",
    "PreviewActivationQualification",
    "PreviewActivationVerifier",
    "PublishedGenerationRecord",
    "SqliteGenerationCatalog",
    "VerifiedPreviewQualification",
]
