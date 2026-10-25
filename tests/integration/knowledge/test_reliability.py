"""Failures, concurrent retries, large histories, and closed predicate policy."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import Authority, Predicate, TimeInterval, TypedLiteral
from capicove.contracts.memory import MemoryQuery, MemoryRecord
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


def service(root: Path) -> TemporalKnowledgeService:
    return TemporalKnowledgeService(
        SQLiteMemoryRepository(root / "memory.db"),
        SQLiteTextEvidenceStore(root / "source.db", root / "privacy"),
        ontology_version="v1",
        predicates=(
            Predicate("flag", ("person",), "literal", ("boolean",), "many", "static"),
            Predicate("city", ("person",), "literal", ("string",), "one", "interval"),
        ),
    )


def test_concurrent_same_key_returns_one_episode_and_receipt(tmp_path: Path) -> None:
    s = service(tmp_path)
    now = datetime.now(UTC)

    def write(_: int) -> dict[str, Any]:
        return s.remember_episode(
            workspace="w", scope="s", content="same input", event_time=now, idempotency_key="key"
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(write, range(12)))
    assert all(r == receipts[0] for r in receipts)
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM memory_operations").fetchone()[0] == 1
    assert s.doctor()["status"] == "ok"


@pytest.mark.slow
def test_more_than_one_thousand_operations_do_not_rebuild_or_cap(tmp_path: Path) -> None:
    s = service(tmp_path)
    now = datetime.now(UTC)
    first = None
    for i in range(1005):
        receipt = s.remember_episode(
            workspace="w",
            scope="s",
            content=f"authored episode {i}",
            event_time=now,
            idempotency_key=f"key{i}",
        )
        if first is None:
            first = receipt
    assert first is not None
    assert s.get_episode(workspace="w", scope="s", episode_id=first["episode_id"])
    assert s.doctor()["status"] == "ok"
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1005
        assert db.execute("SELECT COUNT(*) FROM episode_fts").fetchone()[0] == 1005


def test_many_typed_values_coexist_and_higher_authority_wins(tmp_path: Path) -> None:
    s = service(tmp_path)
    now = datetime.now(UTC)
    for i, boolean in enumerate((True, False)):
        s.assert_fact(
            workspace="w",
            scope="s",
            subject_name="Alice",
            subject_type="person",
            predicate="flag",
            object=TypedLiteral("boolean", boolean),
            validity=TimeInterval(),
            event_time=now,
            evidence_text=f"flag {boolean}",
            idempotency_key=f"flag{i}",
        )
    entity = s.find_entity(workspace="w", scope="s", name="Alice")
    assert entity is not None
    assert (
        len(s.current_facts(workspace="w", scope="s", subject=entity.entity_id, predicate="flag"))
        == 2
    )
    for i, (city, authority) in enumerate((("Paris", Authority.MODEL), ("London", Authority.USER))):
        receipt = s.assert_fact(
            workspace="w",
            scope="s",
            subject_name="Alice",
            subject_type="person",
            predicate="city",
            object=TypedLiteral("string", city),
            validity=TimeInterval(),
            event_time=now,
            evidence_text=city,
            idempotency_key=f"city{i}",
            authority=authority,
        )
        s.record_memory(
            MemoryRecord(
                f"city-memory{i}",
                "w",
                "s",
                receipt["fact_id"],
                "semantic",
                "internal",
                "standard",
                None,
                0.5,
                "fixture",
                now,
            ),
            idempotency_key=f"admit-city{i}",
        )
    facts = s.current_facts(workspace="w", scope="s", subject=entity.entity_id, predicate="city")
    assert len(facts) == 1 and facts[0].object == TypedLiteral("string", "London")
    assert len(s.recall(MemoryQuery("w", "s", types=("semantic",), states=("rejected",))).hits) == 1


def test_projection_failure_rolls_back_ledger_head_and_index(tmp_path: Path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    revision, recorded = repository.head("w")
    with pytest.raises(KeyError):
        repository.commit(
            workspace="w",
            scope="s",
            key="k",
            request={},
            expected_revision=revision,
            recorded=recorded,
            events=[("EntityResolved", {"id": "e", "entity_type": "person", "name": "name"})],
            receipt={},
            indexes=[("invalid", "e", "bad")],
        )
    assert repository.head("w")[0] == 0
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0


def test_unknown_schema_is_not_adopted_or_modified(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE foreign_format(x TEXT)")
    path.chmod(0o600)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="unsupported_schema"):
        SQLiteMemoryRepository(path)
    assert path.read_bytes() == before


@pytest.mark.slow
def test_index_append_work_does_not_scale_with_total_episode_history(tmp_path: Path) -> None:
    s = service(tmp_path)
    repository = s.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    def writes(start: int, stop: int) -> None:
        for i in range(start, stop):
            s.remember_episode(
                workspace="w",
                scope="s",
                content=f"synthetic record {i}",
                event_time=datetime(2026, 1, 1, tzinfo=UTC),
                idempotency_key=f"k{i}",
            )

    repository._db.set_progress_handler(progress, 100)
    writes(0, 50)
    small = steps
    repository._db.set_progress_handler(None, 0)
    writes(50, 950)
    steps = 0
    repository._db.set_progress_handler(progress, 100)
    writes(950, 1000)
    repository._db.set_progress_handler(None, 0)
    assert steps < small * 3, "index append must not scan the lifetime corpus"


@pytest.mark.slow
def test_sparse_graph_lookup_does_not_scan_unrelated_facts(tmp_path: Path) -> None:
    s = service(tmp_path)
    for i in range(1001):
        # These are explicitly distinct authored benchmark identities.
        s.resolve_entity(
            workspace="w",
            scope="s",
            name=f"Person {i}",
            entity_type="person",
            idempotency_key=f"entity{i}",
        )
        AgentMemoryService(s).assert_fact(
            workspace="w",
            scope="s",
            subject_name=f"Person {i}",
            subject_type="person",
            predicate="city",
            object=TypedLiteral("string", "Paris"),
            validity=TimeInterval(),
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
            evidence_text=f"Person {i} lives in Paris",
            idempotency_key=f"f{i}",
        )
    entity = s.find_entity(workspace="w", scope="s", name="Person 0")
    assert entity is not None
    repository = s.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    with repository._read_connection() as db:
        db.set_progress_handler(progress, 100)
    facts = repository.candidates(
        MemoryQuery("w", "s", center_entities=(entity.entity_id,)),
        lane="graph",
        identities=(entity.entity_id,),
    )
    with repository._read_connection() as db:
        db.set_progress_handler(None, 0)
    assert len(facts) == 1
    assert steps < 100, "sparse graph expansion must use the entity indexes"


@pytest.mark.slow
@pytest.mark.parametrize("lane", ("recency", "utility", "lexical"))
def test_episode_lanes_do_not_scan_lifetime_corpus(tmp_path: Path, lane: str) -> None:
    s = service(tmp_path)
    # Utility deliberately covers an empty admission set. The admitted-corpus
    # and unadmitted-noise case is covered separately in test_admission_repair.
    writer = s if lane == "utility" else AgentMemoryService(s)
    for i in range(1000):
        writer.remember_episode(
            workspace="w",
            scope="s",
            content=f"episode {i}",
            event_time=datetime(2026, 1, 1, tzinfo=UTC),
            idempotency_key=f"ep{i}",
        )
    repository = s.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    with repository._read_connection() as db:
        db.set_progress_handler(progress, 100)
    hits = repository.candidates(
        MemoryQuery("w", "s", "episode 999", types=("episodic",)), lane=lane
    )
    with repository._read_connection() as db:
        db.set_progress_handler(None, 0)
    assert len(hits) == {"recency": 50, "utility": 0, "lexical": 1}[lane]
    assert steps < 100, "bounded lanes must not scan the lifetime episode corpus"


def test_recall_read_connection_does_not_hold_writer_lock(tmp_path: Path) -> None:
    from threading import Event

    s = service(tmp_path)
    s.remember_episode(
        workspace="w", scope="s", content="one", event_time=datetime.now(UTC), idempotency_key="one"
    )
    repository = s.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    entered, release = Event(), Event()

    def reader() -> None:
        with repository._read_connection() as db:
            db.execute("BEGIN")
            db.execute("SELECT * FROM episodes").fetchall()
            entered.set()
            assert release.wait(5)
            db.rollback()

    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(reader)
        try:
            assert entered.wait(2)
            writing = pool.submit(
                s.remember_episode,
                workspace="w",
                scope="s",
                content="two",
                event_time=datetime.now(UTC),
                idempotency_key="two",
            )
            assert writing.result(timeout=2)["episode_id"]
        finally:
            release.set()
            reading.result(timeout=2)


def test_old_sqlite_without_secure_fts_erasure_fails_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 40, 0))
    with pytest.raises(RuntimeError, match="sqlite_version_unsupported"):
        SQLiteMemoryRepository(tmp_path / "memory.db")
    assert not (tmp_path / "memory.db").exists()
