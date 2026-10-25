"""Integration tests for the preview-only personal generation lifecycle."""

from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from capicove.contracts.retrieval import SearchFailureCode
from capicove.retrieval._adapters import sqlite_image_generations as sqlite_generations_module
from capicove.retrieval._adapters.sqlite_image_generations import (
    ConcurrentActivationError,
    GenerationAliasSnapshot,
    GenerationQualificationError,
    PreviewActivationQualification,
    PublishedCandidateLeaseRecord,
    PublishedGenerationRecord,
    SqliteGenerationCatalog,
    VerifiedPreviewQualification,
)
from capicove.retrieval.search_models import (
    DeferredCleanupSupervisor,
    RetrievalLane,
    SearchFailure,
)
from capicove.runtime.retrieval import GenerationSetManager, OpenedGenerationHandles


NOW = datetime.now(UTC)
FUTURE = NOW + timedelta(hours=1)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


@dataclass
class _Verifier:
    records: dict[str, VerifiedPreviewQualification]

    def verify(
        self,
        qualification: PreviewActivationQualification,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> VerifiedPreviewQualification:
        assert deadline is FUTURE
        try:
            verified = self.records[qualification.qualification_id]
        except KeyError as error:
            raise GenerationQualificationError("qualification is not trusted") from error
        if (
            verified.activation_set_id != published.activation_set_id
            or verified.manifest_sha256 != published.manifest_sha256
            or verified.profile_ref != published.retrieval_profile_ref
        ):
            raise GenerationQualificationError("qualification does not match publication")
        return verified


@dataclass
class _OwnerSessionRegistry:
    dead: set[tuple[str, str]]
    calls: list[tuple[str, str]]

    def prove_dead(
        self,
        owner_id: str,
        owner_session_epoch: str,
        *,
        deadline: datetime,
    ) -> bool:
        assert deadline > datetime.now(UTC)
        identity = (owner_id, owner_session_epoch)
        self.calls.append(identity)
        return identity in self.dead


def _published(tmp_path: Path, set_id: str, digest: str) -> PublishedGenerationRecord:
    root = tmp_path / "published" / set_id
    root.mkdir(parents=True)
    return PublishedGenerationRecord(
        activation_set_id=set_id,
        manifest_sha256=digest,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="personal-photos",
        retrieval_profile_ref="personal-lite",
        published_at=NOW,
    )


def _verified(set_id: str, digest: str) -> VerifiedPreviewQualification:
    return VerifiedPreviewQualification(
        qualification_id=f"qualification-{set_id}",
        activation_set_id=set_id,
        manifest_sha256=digest,
        profile_ref="personal-lite",
        report_sha256=SHA_C,
        policy_sha256="d" * 64,
        verification_receipt_sha256=("e" if set_id == "set-a" else "f") * 64,
        verified_at=NOW,
    )


def _catalogs(tmp_path: Path) -> tuple[SqliteGenerationCatalog, SqliteGenerationCatalog]:
    verifier = _Verifier(
        {
            "qualification-set-a": _verified("set-a", SHA_A),
            "qualification-set-b": _verified("set-b", SHA_B),
        }
    )
    database = tmp_path / "lifecycle.sqlite3"
    return (
        SqliteGenerationCatalog.open(database, verifier=verifier),
        SqliteGenerationCatalog.open(database, verifier=verifier),
    )


def _alias_bytes(value: GenerationAliasSnapshot) -> bytes:
    return json.dumps(
        asdict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _downgrade_to_v1_schema(connection: sqlite3.Connection) -> None:
    for trigger in (
        "published_candidate_leases_query_id_unique",
        "query_leases_candidate_id_unique",
        "query_leases_identity_immutable",
        "published_candidate_leases_identity_immutable",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    connection.execute("DROP TABLE IF EXISTS published_candidate_leases")
    connection.execute("DROP INDEX IF EXISTS generation_sets_candidate_identity")
    connection.execute("PRAGMA user_version=1")


def test_published_candidate_lease_is_exact_and_never_reads_or_changes_alias(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)
    catalog.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)
    active = catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )
    before = _alias_bytes(active)
    statements: list[str] = []
    catalog._connection.set_trace_callback(statements.append)

    candidate = catalog.acquire_published(
        activation_set_id="set-b",
        expected_manifest_sha256=SHA_B,
        profile_ref="personal-lite",
        lease_id="candidate-1",
        owner_id="admin-worker",
        owner_session_epoch="admin-session-1",
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=2),
        deadline=FUTURE,
    )

    assert isinstance(candidate, PublishedCandidateLeaseRecord)
    assert candidate.activation_set_id == "set-b"
    assert candidate.manifest_sha256 == SHA_B
    assert not hasattr(candidate, "qualification_id")
    assert not hasattr(candidate, "qualification_tier")
    assert all("active_aliases" not in statement.lower() for statement in statements)
    assert all("verified_qualifications" not in statement.lower() for statement in statements)
    with pytest.raises(sqlite3.IntegrityError, match="lease_id already exists"):
        catalog.acquire(
            workspace_id="workspace-1",
            profile_ref="personal-lite",
            lease_id="candidate-1",
            owner_id="preview-worker",
            owner_session_epoch="preview-session",
            acquired_at=NOW,
            expires_at=NOW + timedelta(minutes=1),
            deadline=FUTURE,
        )
    during = catalog.resolve_active("workspace-1", "personal-lite", deadline=FUTURE)
    assert during is not None
    assert _alias_bytes(during) == before

    with pytest.raises(TypeError, match="PreviewActivationQualification"):
        catalog.activate(
            workspace_id="workspace-1",
            profile_ref="personal-lite",
            activation_set_id="set-b",
            expected_previous_snapshot=active,
            qualification=cast(Any, candidate),
            activated_at=NOW + timedelta(seconds=1),
            deadline=FUTURE,
        )
    catalog.release_published(
        "candidate-1",
        released_at=NOW + timedelta(seconds=2),
        deadline=FUTURE,
    )
    catalog.release_published(
        "candidate-1",
        released_at=NOW + timedelta(seconds=3),
        deadline=FUTURE,
    )
    after = catalog.resolve_active("workspace-1", "personal-lite", deadline=FUTURE)
    assert after is not None
    assert _alias_bytes(after) == before


@pytest.mark.parametrize(
    ("set_id", "digest", "profile"),
    (
        ("missing", SHA_B, "personal-lite"),
        ("set-b", SHA_A, "personal-lite"),
        ("set-b", SHA_B, "enterprise-disk"),
    ),
)
def test_published_candidate_rejects_wrong_exact_identity(
    tmp_path: Path,
    set_id: str,
    digest: str,
    profile: str,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)

    with pytest.raises(GenerationQualificationError, match="candidate"):
        catalog.acquire_published(
            activation_set_id=set_id,
            expected_manifest_sha256=digest,
            profile_ref=profile,
            lease_id="candidate-wrong",
            owner_id="admin-worker",
            owner_session_epoch="admin-session-1",
            acquired_at=NOW,
            expires_at=NOW + timedelta(minutes=2),
            deadline=FUTURE,
        )

    assert catalog._connection.execute(
        "SELECT COUNT(*) FROM published_candidate_leases"
    ).fetchone() == (0,)


def test_candidate_duplicate_deadline_and_transaction_failure_roll_back(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)

    def acquire(
        lease_id: str,
        *,
        deadline: datetime = FUTURE,
    ) -> PublishedCandidateLeaseRecord:
        return catalog.acquire_published(
            activation_set_id="set-b",
            expected_manifest_sha256=SHA_B,
            profile_ref="personal-lite",
            lease_id=lease_id,
            owner_id="admin-worker",
            owner_session_epoch="admin-session-1",
            acquired_at=NOW,
            expires_at=NOW + timedelta(minutes=2),
            deadline=deadline,
        )

    with pytest.raises(TimeoutError, match="deadline"):
        acquire("candidate-expired", deadline=NOW - timedelta(seconds=1))
    acquire("candidate-duplicate")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint"):
        acquire("candidate-duplicate")
    with pytest.raises(TimeoutError, match="deadline"):
        catalog.release_published(
            "candidate-duplicate",
            released_at=NOW + timedelta(seconds=1),
            deadline=NOW - timedelta(seconds=1),
        )
    assert catalog._connection.execute(
        "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-duplicate",),
    ).fetchone() == (None,)
    with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
        catalog._connection.execute(
            "UPDATE published_candidate_leases SET profile_ref = ? WHERE lease_id = ?",
            ("enterprise-disk", "candidate-duplicate"),
        )

    catalog._connection.execute(
        """
        CREATE TRIGGER fail_candidate_insert BEFORE INSERT ON published_candidate_leases
        BEGIN SELECT RAISE(ABORT, 'injected candidate transaction failure'); END
        """
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        acquire("candidate-rolled-back")
    assert catalog._connection.execute(
        "SELECT COUNT(*) FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-rolled-back",),
    ).fetchone() == (0,)


def test_candidate_mutations_roll_back_when_sql_crosses_deadline(tmp_path: Path) -> None:
    registry = _OwnerSessionRegistry(dead={("admin-worker", "admin-session-1")}, calls=[])
    catalog = SqliteGenerationCatalog.open(
        tmp_path / "lifecycle.sqlite3",
        verifier=_Verifier({}),
        owner_session_registry=registry,
    )
    catalog.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)

    delayed: list[str] = []

    def delay_insert(statement: str) -> None:
        if "INSERT INTO published_candidate_leases" in statement:
            delayed.append(statement)
            time.sleep(0.12)

    acquired_at = datetime.now(UTC)
    catalog._connection.set_trace_callback(delay_insert)
    with pytest.raises(TimeoutError, match="deadline"):
        catalog.acquire_published(
            activation_set_id="set-b",
            expected_manifest_sha256=SHA_B,
            profile_ref="personal-lite",
            lease_id="candidate-late-insert",
            owner_id="admin-worker",
            owner_session_epoch="admin-session-1",
            acquired_at=acquired_at,
            expires_at=acquired_at + timedelta(minutes=1),
            deadline=datetime.now(UTC) + timedelta(milliseconds=40),
        )
    assert delayed
    assert catalog._connection.execute(
        "SELECT COUNT(*) FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-late-insert",),
    ).fetchone() == (0,)

    catalog._connection.set_trace_callback(None)
    catalog.acquire_published(
        activation_set_id="set-b",
        expected_manifest_sha256=SHA_B,
        profile_ref="personal-lite",
        lease_id="candidate-late-release",
        owner_id="admin-worker",
        owner_session_epoch="admin-session-1",
        acquired_at=acquired_at,
        expires_at=acquired_at + timedelta(minutes=1),
        deadline=FUTURE,
    )

    delayed.clear()

    def delay_update(statement: str) -> None:
        if "UPDATE published_candidate_leases SET released_at" in statement:
            delayed.append(statement)
            time.sleep(0.12)

    catalog._connection.set_trace_callback(delay_update)
    with pytest.raises(TimeoutError, match="deadline"):
        catalog.release_published(
            "candidate-late-release",
            released_at=datetime.now(UTC),
            deadline=datetime.now(UTC) + timedelta(milliseconds=40),
        )
    assert delayed
    assert catalog._connection.execute(
        "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-late-release",),
    ).fetchone() == (None,)

    delayed.clear()
    with pytest.raises(TimeoutError, match="deadline"):
        catalog.recover_stale_published(
            owner_id="admin-worker",
            owner_session_epoch="admin-session-1",
            recovered_at=datetime.now(UTC),
            deadline=datetime.now(UTC) + timedelta(milliseconds=40),
        )
    assert delayed
    assert catalog._connection.execute(
        "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-late-release",),
    ).fetchone() == (None,)
    catalog._connection.set_trace_callback(None)
    catalog.close()


def test_candidate_commits_are_interrupted_before_late_persistence(tmp_path: Path) -> None:
    registry = _OwnerSessionRegistry(dead={("admin-worker", "admin-session-1")}, calls=[])
    catalog = SqliteGenerationCatalog.open(
        tmp_path / "lifecycle.sqlite3",
        verifier=_Verifier({}),
        owner_session_registry=registry,
    )
    catalog.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)
    commit_delays: list[str] = []

    def delay_commit(statement: str) -> None:
        if statement.strip().casefold() == "commit":
            commit_delays.append(statement)
            time.sleep(0.12)

    acquired_at = datetime.now(UTC)
    catalog._connection.set_trace_callback(delay_commit)
    with pytest.raises(TimeoutError, match="deadline exceeded during commit"):
        catalog.acquire_published(
            activation_set_id="set-b",
            expected_manifest_sha256=SHA_B,
            profile_ref="personal-lite",
            lease_id="candidate-late-commit",
            owner_id="admin-worker",
            owner_session_epoch="admin-session-1",
            acquired_at=acquired_at,
            expires_at=acquired_at + timedelta(minutes=1),
            deadline=datetime.now(UTC) + timedelta(milliseconds=40),
        )
    assert len(commit_delays) == 1
    assert catalog._connection.execute(
        "SELECT COUNT(*) FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-late-commit",),
    ).fetchone() == (0,)

    catalog._connection.set_trace_callback(None)
    catalog.acquire_published(
        activation_set_id="set-b",
        expected_manifest_sha256=SHA_B,
        profile_ref="personal-lite",
        lease_id="candidate-late-release-commit",
        owner_id="admin-worker",
        owner_session_epoch="admin-session-1",
        acquired_at=acquired_at,
        expires_at=acquired_at + timedelta(minutes=1),
        deadline=FUTURE,
    )
    commit_delays.clear()
    catalog._connection.set_trace_callback(delay_commit)
    with pytest.raises(TimeoutError, match="deadline exceeded during commit"):
        catalog.release_published(
            "candidate-late-release-commit",
            released_at=datetime.now(UTC),
            deadline=datetime.now(UTC) + timedelta(milliseconds=40),
        )
    assert len(commit_delays) == 1
    assert catalog._connection.execute(
        "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-late-release-commit",),
    ).fetchone() == (None,)

    commit_delays.clear()
    with pytest.raises(TimeoutError, match="deadline exceeded during commit"):
        catalog.recover_stale_published(
            owner_id="admin-worker",
            owner_session_epoch="admin-session-1",
            recovered_at=datetime.now(UTC),
            deadline=datetime.now(UTC) + timedelta(milliseconds=40),
        )
    assert len(commit_delays) == 1
    assert catalog._connection.execute(
        "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
        ("candidate-late-release-commit",),
    ).fetchone() == (None,)
    catalog._connection.set_trace_callback(None)
    catalog.close()


@pytest.mark.parametrize(
    ("sqlite_errorcode", "in_transaction"),
    (
        (None, True),
        (sqlite3.SQLITE_IOERR, True),
        (sqlite3.SQLITE_INTERRUPT, False),
    ),
)
def test_commit_timer_classifies_only_confirmed_active_interrupts(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_errorcode: int | None,
    in_transaction: bool,
) -> None:
    timers: list[Any] = []

    class ImmediateTimer:
        daemon = False

        def __init__(self, _interval: float, callback: Any) -> None:
            self._callback = callback
            self.cancelled = False
            self.joined = False
            timers.append(self)

        def start(self) -> None:
            self._callback()

        def cancel(self) -> None:
            self.cancelled = True

        def join(self) -> None:
            self.joined = True

    class CommitConnection:
        def __init__(self) -> None:
            self.in_transaction = in_transaction
            self.commit_calls = 0
            self.interrupt_calls = 0

        def execute(self, statement: str) -> None:
            if statement == "COMMIT":
                self.commit_calls += 1
            if statement == "COMMIT" and sqlite_errorcode is not None:
                error = sqlite3.OperationalError("unrelated operational failure")
                error.sqlite_errorcode = sqlite_errorcode
                raise error

        def interrupt(self) -> None:
            self.interrupt_calls += 1

    monkeypatch.setattr(sqlite_generations_module, "Timer", ImmediateTimer)
    catalog = cast(Any, object.__new__(SqliteGenerationCatalog))
    connection = CommitConnection()
    catalog._connection = connection

    if sqlite_errorcode is None:
        catalog._commit_before_deadline(datetime.now(UTC) + timedelta(seconds=1))
    else:
        with pytest.raises(sqlite3.OperationalError, match="unrelated operational failure"):
            catalog._commit_before_deadline(datetime.now(UTC) + timedelta(seconds=1))

    assert connection.commit_calls == 1
    assert connection.interrupt_calls == 1
    assert len(timers) == 1
    assert timers[0].cancelled
    assert timers[0].joined


def test_v1_to_v2_migration_preserves_existing_preview_rows(tmp_path: Path) -> None:
    catalog, second_catalog = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)
    catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )
    catalog.acquire(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        lease_id="preview-before-extension",
        owner_id="worker-1",
        owner_session_epoch="session-1",
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
        deadline=FUTURE,
    )
    catalog.close()
    second_catalog.close()
    database = tmp_path / "lifecycle.sqlite3"
    with sqlite3.connect(database) as connection:
        _downgrade_to_v1_schema(connection)
        before_rows = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in (
                "generation_sets",
                "verified_qualifications",
                "active_aliases",
                "query_leases",
                "generation_events",
            )
        }

    reopened = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    reopened.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("published_candidate_leases",),
        ).fetchone() == ("published_candidate_leases",)
        assert {
            table: connection.execute(f"SELECT * FROM {table}").fetchall() for table in before_rows
        } == before_rows


def test_v2_open_rejects_schema_mutation_without_repair(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER query_leases_identity_immutable")

    with pytest.raises(ValueError, match="v2 schema is incompatible"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                ("query_leases_identity_immutable",),
            ).fetchone()
            is None
        )


def test_v1_migration_rejects_forged_extension_and_rolls_back(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    with sqlite3.connect(database) as connection:
        _downgrade_to_v1_schema(connection)
        connection.execute(
            """
            CREATE TABLE published_candidate_leases (
              lease_id TEXT,
              activation_set_id TEXT,
              manifest_sha256 TEXT,
              profile_ref TEXT,
              owner_id TEXT,
              owner_session_epoch TEXT,
              acquired_at TEXT,
              expires_at TEXT,
              released_at TEXT
            )
            """
        )

    with pytest.raises(ValueError, match="v1 schema is incompatible"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert (
            connection.execute("PRAGMA foreign_key_list(published_candidate_leases)").fetchall()
            == []
        )
        assert connection.execute("PRAGMA index_list(published_candidate_leases)").fetchall() == []
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                ("generation_sets_candidate_identity",),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                ("query_leases_identity_immutable",),
            ).fetchone()
            is None
        )


def test_v1_migration_rolls_back_created_objects_before_version_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)
    catalog.close()
    with sqlite3.connect(database) as connection:
        _downgrade_to_v1_schema(connection)

    def reject_created_v2(connection: sqlite3.Connection) -> None:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("published_candidate_leases",),
        ).fetchone() == ("published_candidate_leases",)
        raise ValueError("injected post-create validation failure")

    monkeypatch.setattr(
        SqliteGenerationCatalog,
        "_validate_v2_schema",
        staticmethod(reject_created_v2),
    )

    with pytest.raises(ValueError, match="post-create validation failure"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute("SELECT activation_set_id FROM generation_sets").fetchall() == [
            ("set-a",)
        ]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                ("published_candidate_leases",),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                ("generation_sets_candidate_identity",),
            ).fetchone()
            is None
        )


def test_v2_schema_cannot_masquerade_as_v1(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=1")

    with pytest.raises(ValueError, match="v1 schema is incompatible"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("published_candidate_leases",),
        ).fetchone() == ("published_candidate_leases",)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            ("query_leases_identity_immutable",),
        ).fetchone() == ("query_leases_identity_immutable",)


def test_v1_migration_rechecks_identity_after_open_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    with sqlite3.connect(database) as connection:
        _downgrade_to_v1_schema(connection)
    original_migrate = SqliteGenerationCatalog._migrate_v1_to_v2

    def drift_then_migrate(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA application_id=123")
        original_migrate(connection)

    monkeypatch.setattr(
        SqliteGenerationCatalog,
        "_migrate_v1_to_v2",
        staticmethod(drift_then_migrate),
    )

    with pytest.raises(ValueError, match="identity changed while opening"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA application_id").fetchone() == (123,)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                ("published_candidate_leases",),
            ).fetchone()
            is None
        )


def test_v2_validation_rechecks_version_after_open_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    original_validate = SqliteGenerationCatalog._validate_current_schema

    def drift_then_validate(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA user_version=1")
        original_validate(connection)

    monkeypatch.setattr(
        SqliteGenerationCatalog,
        "_validate_current_schema",
        staticmethod(drift_then_validate),
    )

    with pytest.raises(ValueError, match="identity changed while opening"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("published_candidate_leases",),
        ).fetchone() == ("published_candidate_leases",)


def test_v2_schema_validation_coexists_with_reader_transaction(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN")
    blocker.execute("SELECT * FROM generation_sets").fetchall()
    started = time.monotonic()

    reopened = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    assert time.monotonic() - started < 1.0
    reopened.close()
    blocker.execute("ROLLBACK")
    blocker.close()


@pytest.mark.slow
def test_real_v1_to_v2_migration_wait_is_bounded_and_atomic(tmp_path: Path) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    catalog = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    catalog.close()
    with sqlite3.connect(database) as connection:
        _downgrade_to_v1_schema(connection)
    blocker = sqlite3.connect(database, isolation_level=None)
    blocker.execute("BEGIN")
    blocker.execute("SELECT * FROM generation_sets").fetchall()
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="migration remained busy"):
        SqliteGenerationCatalog.open(database, verifier=_Verifier({}))

    assert time.monotonic() - started < 2.0
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                ("published_candidate_leases",),
            ).fetchone()
            is None
        )
    blocker.execute("ROLLBACK")
    blocker.close()
    reopened = SqliteGenerationCatalog.open(database, verifier=_Verifier({}))
    reopened.close()
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)


def test_restart_recovers_candidate_only_after_owner_session_death_proof(
    tmp_path: Path,
) -> None:
    database = tmp_path / "lifecycle.sqlite3"
    verifier = _Verifier({})
    first = SqliteGenerationCatalog.open(database, verifier=verifier)
    first.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)
    first.acquire_published(
        activation_set_id="set-b",
        expected_manifest_sha256=SHA_B,
        profile_ref="personal-lite",
        lease_id="candidate-from-dead-process",
        owner_id="admin-worker",
        owner_session_epoch="dead-session",
        acquired_at=NOW - timedelta(minutes=2),
        expires_at=NOW - timedelta(minutes=1),
        deadline=FUTURE,
    )
    first.close()

    registry = _OwnerSessionRegistry(dead=set(), calls=[])
    restarted = SqliteGenerationCatalog.open(
        database,
        verifier=verifier,
        owner_session_registry=registry,
    )
    with pytest.raises(GenerationQualificationError, match="not proven dead"):
        restarted.recover_stale_published(
            owner_id="admin-worker",
            owner_session_epoch="dead-session",
            recovered_at=NOW,
            deadline=FUTURE,
        )
    registry.dead.add(("admin-worker", "dead-session"))
    assert (
        restarted.recover_stale_published(
            owner_id="admin-worker",
            owner_session_epoch="dead-session",
            recovered_at=NOW,
            deadline=FUTURE,
        )
        == 1
    )
    assert (
        restarted.recover_stale_published(
            owner_id="admin-worker",
            owner_session_epoch="dead-session",
            recovered_at=NOW,
            deadline=FUTURE,
        )
        == 0
    )
    assert registry.calls == [
        ("admin-worker", "dead-session"),
        ("admin-worker", "dead-session"),
        ("admin-worker", "dead-session"),
    ]


def test_activation_is_atomic_compare_and_swap_and_lease_is_stable(tmp_path: Path) -> None:
    catalog_a, catalog_b = _catalogs(tmp_path)
    catalog_a.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)
    catalog_a.register_published(_published(tmp_path, "set-b", SHA_B), deadline=FUTURE)

    first = catalog_a.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )
    with pytest.raises(ConcurrentActivationError):
        catalog_b.activate(
            workspace_id="workspace-1",
            profile_ref="personal-lite",
            activation_set_id="set-b",
            expected_previous_snapshot=None,
            qualification=PreviewActivationQualification("qualification-set-b"),
            activated_at=NOW,
            deadline=FUTURE,
        )

    lease_a = catalog_a.acquire(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        lease_id="lease-a",
        owner_id="worker-a",
        owner_session_epoch="session-a",
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=2),
        deadline=FUTURE,
    )
    second = catalog_b.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-b",
        expected_previous_snapshot=first,
        qualification=PreviewActivationQualification("qualification-set-b"),
        activated_at=NOW + timedelta(seconds=1),
        deadline=FUTURE,
    )
    lease_b = catalog_b.acquire(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        lease_id="lease-b",
        owner_id="worker-b",
        owner_session_epoch="session-b",
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=2),
        deadline=FUTURE,
    )

    assert first.version == 1
    assert second.version == 2
    assert lease_a.alias_snapshot.activation_set_id == "set-a"
    assert lease_b.alias_snapshot.activation_set_id == "set-b"
    assert catalog_a.resolve_active("workspace-1", "personal-lite", deadline=FUTURE) == second


def test_publication_is_idempotent_but_conflicts_and_untrusted_activation_fail(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    published = _published(tmp_path, "set-a", SHA_A)
    catalog.register_published(published, deadline=FUTURE)
    catalog.register_published(published, deadline=FUTURE)

    with pytest.raises(ValueError, match="different manifest"):
        catalog.register_published(
            PublishedGenerationRecord(
                activation_set_id="set-a",
                manifest_sha256=SHA_B,
                published_root=published.published_root,
                workspace_id=published.workspace_id,
                corpus_ref=published.corpus_ref,
                retrieval_profile_ref=published.retrieval_profile_ref,
                published_at=published.published_at,
            ),
            deadline=FUTURE,
        )
    with pytest.raises(GenerationQualificationError):
        catalog.activate(
            workspace_id="workspace-1",
            profile_ref="personal-lite",
            activation_set_id="set-a",
            expected_previous_snapshot=None,
            qualification=PreviewActivationQualification("unknown"),
            activated_at=NOW,
            deadline=FUTURE,
        )
    assert catalog.resolve_active("workspace-1", "personal-lite", deadline=FUTURE) is None


def test_release_is_idempotent_and_renewal_requires_full_owner_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)
    catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )
    expires = NOW + timedelta(minutes=1)
    catalog.acquire(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        lease_id="lease-a",
        owner_id="worker-a",
        owner_session_epoch="session-a",
        acquired_at=NOW,
        expires_at=expires,
        deadline=FUTURE,
    )
    with pytest.raises(sqlite3.IntegrityError, match="lease_id already exists"):
        catalog.acquire_published(
            activation_set_id="set-a",
            expected_manifest_sha256=SHA_A,
            profile_ref="personal-lite",
            lease_id="lease-a",
            owner_id="admin-worker",
            owner_session_epoch="admin-session",
            acquired_at=NOW,
            expires_at=NOW + timedelta(minutes=1),
            deadline=FUTURE,
        )
    catalog.acquire_published(
        activation_set_id="set-a",
        expected_manifest_sha256=SHA_A,
        profile_ref="personal-lite",
        lease_id="candidate-other",
        owner_id="admin-worker",
        owner_session_epoch="admin-session",
        acquired_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
        deadline=FUTURE,
    )
    with pytest.raises(sqlite3.IntegrityError, match="query lease identity is immutable"):
        catalog._connection.execute(
            "UPDATE query_leases SET lease_id = ? WHERE lease_id = ?",
            ("candidate-other", "lease-a"),
        )
    renewed = catalog.renew(
        lease_id="lease-a",
        owner_id="worker-a",
        owner_session_epoch="session-a",
        expected_expires_at=expires,
        renewed_at=NOW + timedelta(seconds=30),
        expires_at=NOW + timedelta(minutes=2),
        deadline=FUTURE,
    )
    assert renewed.expires_at == NOW + timedelta(minutes=2)

    with pytest.raises(ValueError, match="lease renewal"):
        catalog.renew(
            lease_id="lease-a",
            owner_id="another-worker",
            owner_session_epoch="session-a",
            expected_expires_at=renewed.expires_at,
            renewed_at=NOW + timedelta(minutes=1),
            expires_at=NOW + timedelta(minutes=3),
            deadline=FUTURE,
        )
    release_statements: list[str] = []
    catalog._connection.set_trace_callback(release_statements.append)
    monkeypatch.setattr(
        catalog,
        "_commit_before_deadline",
        lambda _deadline: pytest.fail("active release used candidate COMMIT timer"),
    )
    catalog.release("lease-a", released_at=NOW + timedelta(minutes=1), deadline=FUTURE)
    catalog.release("lease-a", released_at=NOW + timedelta(minutes=1), deadline=FUTURE)
    catalog._connection.set_trace_callback(None)
    assert all(
        "published_candidate_leases" not in statement.lower() for statement in release_statements
    )


def test_schema_identity_and_append_only_rows_are_enforced(tmp_path: Path) -> None:
    catalog, _ = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)

    connection = sqlite3.connect(tmp_path / "lifecycle.sqlite3")
    try:
        assert connection.execute("PRAGMA application_id").fetchone() == (0x43415047,)
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE generation_sets SET manifest_sha256 = ? WHERE activation_set_id = ?",
                (SHA_B, "set-a"),
            )
    finally:
        connection.close()


def test_one_catalog_serializes_concurrent_worker_leases(tmp_path: Path) -> None:
    catalog, _ = _catalogs(tmp_path)
    catalog.register_published(_published(tmp_path, "set-a", SHA_A), deadline=FUTURE)
    catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )

    def acquire(index: int):  # type: ignore[no-untyped-def]
        return catalog.acquire(
            workspace_id="workspace-1",
            profile_ref="personal-lite",
            lease_id=f"lease-{index}",
            owner_id=f"worker-{index}",
            owner_session_epoch="session-1",
            acquired_at=NOW,
            expires_at=NOW + timedelta(minutes=1),
            deadline=FUTURE,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        leases = tuple(executor.map(acquire, range(4)))
        tuple(
            executor.map(
                lambda lease: catalog.release(
                    lease.lease_id,
                    released_at=NOW + timedelta(seconds=1),
                    deadline=FUTURE,
                ),
                leases,
            )
        )

    assert {lease.alias_snapshot.activation_set_id for lease in leases} == {"set-a"}


def test_manager_releases_sqlite_lease_after_request_deadline_expires(
    tmp_path: Path,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    published = _published(tmp_path, "set-a", SHA_A)
    catalog.register_published(published, deadline=FUTURE)
    catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )

    class DeadlineExpiringFactory:
        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> None:
            del published
            while datetime.now(UTC) < deadline:
                time.sleep(0.001)
            raise TimeoutError("request deadline expired while opening generation")

    supervisor = DeferredCleanupSupervisor(max_pending=1)
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=catalog,
        handle_factory=cast(Any, DeadlineExpiringFactory()),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: "lease-expired-open",
        clock=lambda: datetime.now(UTC),
        close_timeout=timedelta(seconds=1),
        cleanup_supervisor=supervisor,
    )
    request_deadline = datetime.now(UTC) + timedelta(milliseconds=20)

    with pytest.raises(SearchFailure) as raised:
        manager.acquire("personal-lite", deadline=request_deadline)

    assert raised.value.code is SearchFailureCode.DEADLINE_EXCEEDED
    assert supervisor.drain(timeout_seconds=0).completed
    connection = sqlite3.connect(tmp_path / "lifecycle.sqlite3")
    try:
        released_at = connection.execute(
            "SELECT released_at FROM query_leases WHERE lease_id = ?",
            ("lease-expired-open",),
        ).fetchone()
    finally:
        connection.close()
    assert released_at is not None
    assert released_at[0] is not None


def test_manager_recovers_after_last_sqlite_lease_release_fails_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    published = _published(tmp_path, "set-a", SHA_A)
    catalog.register_published(published, deadline=FUTURE)
    catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )

    release_attempts: list[str] = []
    release = catalog.release

    def fail_first_release(
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        release_attempts.append(lease_id)
        if len(release_attempts) == 1:
            raise sqlite3.OperationalError("injected release failure")
        release(lease_id, released_at=released_at, deadline=deadline)

    monkeypatch.setattr(catalog, "release", fail_first_release)

    @dataclass
    class CloseCounter:
        calls: int = 0

        def close(self) -> None:
            self.calls += 1

    class CountingFactory:
        def __init__(self) -> None:
            self.opened: list[tuple[CloseCounter, ...]] = []

        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del deadline
            handles = tuple(CloseCounter() for _ in range(4))
            self.opened.append(handles)
            return OpenedGenerationHandles(
                manifest=cast(
                    Any,
                    SimpleNamespace(
                        activation_set_id=published.activation_set_id,
                        workspace_id=published.workspace_id,
                        retrieval_profile_ref=published.retrieval_profile_ref,
                    ),
                ),
                unit_catalog=cast(Any, handles[0]),
                readers={
                    RetrievalLane.PERCEPTUAL: cast(Any, handles[1]),
                    RetrievalLane.INSTANCE: cast(Any, handles[2]),
                    RetrievalLane.CONCEPT: cast(Any, handles[3]),
                },
                close_callbacks=tuple(handle.close for handle in handles),
            )

    lease_ids = iter(("lease-first", "lease-second"))
    factory = CountingFactory()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=catalog,
        handle_factory=factory,
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: next(lease_ids),
        clock=lambda: datetime.now(UTC),
    )

    first = manager.acquire("personal-lite", deadline=FUTURE)
    with pytest.raises(SearchFailure) as failed_close:
        first.close()

    assert failed_close.value.code is SearchFailureCode.INTERNAL
    assert release_attempts == ["lease-first"]
    assert [handle.calls for handle in factory.opened[0]] == [1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="generation lease is closed"):
        first.__enter__()

    second = manager.acquire("personal-lite", deadline=FUTURE)
    assert release_attempts == ["lease-first", "lease-first"]
    assert len(factory.opened) == 2

    first.close()
    assert release_attempts == ["lease-first", "lease-first"]
    second.close()

    assert release_attempts == ["lease-first", "lease-first", "lease-second"]
    assert all(
        handle.calls == 1 for generation_handles in factory.opened for handle in generation_handles
    )
    connection = sqlite3.connect(tmp_path / "lifecycle.sqlite3")
    try:
        lease_rows = connection.execute(
            "SELECT lease_id, released_at FROM query_leases ORDER BY lease_id"
        ).fetchall()
    finally:
        connection.close()
    assert [lease_id for lease_id, _ in lease_rows] == ["lease-first", "lease-second"]
    assert all(released_at is not None for _, released_at in lease_rows)


def test_manager_never_reuses_partially_closed_generation_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, _ = _catalogs(tmp_path)
    published = _published(tmp_path, "set-a", SHA_A)
    catalog.register_published(published, deadline=FUTURE)
    catalog.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-a",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-set-a"),
        activated_at=NOW,
        deadline=FUTURE,
    )

    release_attempts: list[str] = []
    release = catalog.release

    def recording_release(
        lease_id: str,
        *,
        released_at: datetime,
        deadline: datetime,
    ) -> None:
        release_attempts.append(lease_id)
        release(lease_id, released_at=released_at, deadline=deadline)

    monkeypatch.setattr(catalog, "release", recording_release)

    @dataclass
    class RetryableClose:
        name: str
        events: list[str]
        fail_once: bool = False
        calls: int = 0

        def close(self) -> None:
            self.calls += 1
            self.events.append(self.name)
            if self.fail_once and self.calls == 1:
                raise OSError("injected handle close failure")

    class FaultInjectingFactory:
        def __init__(self) -> None:
            self.opened: list[tuple[RetryableClose, ...]] = []

        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del deadline
            events: list[str] = []
            generation_number = len(self.opened) + 1
            handles = (
                RetryableClose(f"generation-{generation_number}-last", events),
                RetryableClose(
                    f"generation-{generation_number}-fails-once",
                    events,
                    fail_once=generation_number == 1,
                ),
                RetryableClose(f"generation-{generation_number}-first", events),
            )
            self.opened.append(handles)
            return OpenedGenerationHandles(
                manifest=cast(
                    Any,
                    SimpleNamespace(
                        activation_set_id=published.activation_set_id,
                        workspace_id=published.workspace_id,
                        retrieval_profile_ref=published.retrieval_profile_ref,
                    ),
                ),
                unit_catalog=cast(Any, handles[0]),
                readers={
                    RetrievalLane.PERCEPTUAL: cast(Any, handles[0]),
                    RetrievalLane.INSTANCE: cast(Any, handles[1]),
                    RetrievalLane.CONCEPT: cast(Any, handles[2]),
                },
                close_callbacks=tuple(handle.close for handle in handles),
            )

    lease_ids = iter(("lease-first", "lease-second"))
    factory = FaultInjectingFactory()
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=catalog,
        handle_factory=factory,
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: next(lease_ids),
        clock=lambda: datetime.now(UTC),
    )

    first = manager.acquire("personal-lite", deadline=FUTURE)
    with pytest.raises(SearchFailure) as failed_close:
        first.close()

    assert failed_close.value.code is SearchFailureCode.INTERNAL
    assert [handle.calls for handle in factory.opened[0]] == [0, 1, 1]
    assert release_attempts == []

    second = manager.acquire("personal-lite", deadline=FUTURE)
    assert len(factory.opened) == 2
    assert [handle.calls for handle in factory.opened[0]] == [1, 2, 1]
    assert release_attempts == ["lease-first"]

    first.close()
    second.close()
    assert [handle.calls for handle in factory.opened[1]] == [1, 1, 1]
    assert release_attempts == ["lease-first", "lease-second"]

    connection = sqlite3.connect(tmp_path / "lifecycle.sqlite3")
    try:
        unreleased_count = connection.execute(
            "SELECT COUNT(*) FROM query_leases WHERE released_at IS NULL"
        ).fetchone()
    finally:
        connection.close()
    assert unreleased_count == (0,)
