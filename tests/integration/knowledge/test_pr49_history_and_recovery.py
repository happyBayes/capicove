"""Challenge both time axes, projection upgrades and current privacy policy."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import EntityRef, Predicate, TimeInterval, TypedLiteral
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


JAN = datetime(2026, 1, 1, tzinfo=UTC)
FEB = datetime(2026, 2, 1, tzinfo=UTC)
MAR = datetime(2026, 3, 1, tzinfo=UTC)


def service(root: Path) -> TemporalKnowledgeService:
    return TemporalKnowledgeService(
        SQLiteMemoryRepository(root / "memory.db"),
        SQLiteTextEvidenceStore(root / "source.db", root / "privacy"),
        ontology_version="v1",
        predicates=(
            Predicate("city", ("person",), "literal", ("string",), "one", "interval"),
            Predicate("knows", ("person",), "entity", ("person",), "one", "interval"),
        ),
    )


def test_graph_filters_world_and_knowledge_before_traversal(tmp_path: Path) -> None:
    s = service(tmp_path)
    alice, boris, camila = [
        s.resolve_entity(
            workspace="w", scope="s", name=name, entity_type="person", idempotency_key=name
        )
        for name in ("Alice", "Boris", "Camila")
    ]
    first = AgentMemoryService(s).assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="knows",
        object=EntityRef(boris.entity_id),
        validity=TimeInterval(),
        event_time=JAN,
        evidence_text="Alice knew Boris.",
        idempotency_key="ab",
    )
    before = s.get_fact_history(workspace="w", scope="s", fact_id=first["fact_id"])[
        0
    ].knowledge.start
    second = AgentMemoryService(s).assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="knows",
        object=EntityRef(camila.entity_id),
        validity=TimeInterval(FEB),
        event_time=FEB,
        evidence_text="Alice now knows Camila.",
        idempotency_key="ac",
    )
    for valid, known, expected in (
        (MAR, before, first["fact_id"]),
        (JAN, None, first["fact_id"]),
        (FEB, None, second["fact_id"]),
    ):
        rows = s.repository.candidates(
            MemoryQuery("w", "s", valid_at=valid, known_at=known, types=("semantic",)),
            lane="graph",
            identities=(alice.entity_id,),
        )
        assert [row["id"] for row in rows] == [expected]
    assert not s.repository.candidates(
        MemoryQuery("w", "other", valid_at=MAR, types=("semantic",)),
        lane="graph",
        identities=(alice.entity_id,),
    )


@pytest.mark.parametrize("existing_index", [False, True])
def test_v1_migration_preserves_ledger_receipts_and_replays_both_axes(
    tmp_path: Path,
    existing_index: bool,
) -> None:
    s = service(tmp_path)
    receipt = s.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "Paris"),
        validity=TimeInterval(JAN, FEB),
        event_time=JAN,
        evidence_text="Alice lived in Paris in January.",
        idempotency_key="a",
    )
    repository = s.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    repository.close()
    # Reconstitute the released PR's v1 projection layout without modifying events.
    with sqlite3.connect(tmp_path / "memory.db") as db:
        before = db.execute("SELECT hash,payload FROM memory_events ORDER BY revision").fetchall()
        operations = db.execute("SELECT receipt FROM memory_operations").fetchall()
        entries = db.execute("SELECT * FROM fts_entries ORDER BY kind,identity").fetchall()
        assert {row[2] for row in entries} == {"episode", "fact"}
        db.execute("DROP INDEX facts_bitemporal")
        db.execute("ALTER TABLE fact_versions DROP COLUMN valid_to")
        db.execute("ALTER TABLE fact_versions DROP COLUMN valid_from")
        db.execute("DELETE FROM schema_migrations WHERE version=2")
        if not existing_index:
            db.execute("DELETE FROM entity_fts")
    restarted = service(tmp_path)
    history = restarted.get_fact_history(workspace="w", scope="s", fact_id=receipt["fact_id"])
    assert history[0].validity == TimeInterval(JAN, FEB)
    # Check before rebuild: replay would otherwise hide a migration duplicate.
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM entity_fts").fetchone()[0] == 1
        assert db.execute("SELECT * FROM fts_entries ORDER BY kind,identity").fetchall() == entries
        for table, kind in (("episode_fts", "episode"), ("fact_fts", "fact")):
            assert (
                db.execute(
                    f"SELECT COUNT(*) FROM fts_entries e JOIN {table} f "
                    "ON e.docid=f.rowid AND e.workspace=f.workspace AND e.scope=f.scope "
                    "AND e.identity=f.identity WHERE e.kind=?",
                    (kind,),
                ).fetchone()[0]
                == 1
            )
    restarted_repository = restarted.repository
    assert isinstance(restarted_repository, SQLiteMemoryRepository)
    restarted_repository.close()
    restarted = service(tmp_path)
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM entity_fts").fetchone()[0] == 1
    restarted.rebuild_projections()
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert (
            db.execute("SELECT hash,payload FROM memory_events ORDER BY revision").fetchall()
            == before
        )
        assert db.execute("SELECT receipt FROM memory_operations").fetchall() == operations
        assert db.execute("SELECT valid_from,valid_to FROM fact_versions").fetchone() == (
            1767225600000000,
            1769904000000000,
        )
        assert db.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall() == [
            (1,),
            (2,),
        ]
    assert restarted.doctor()["status"] == "ok"


def test_archive_and_expiry_use_current_policy_even_for_old_knowledge(tmp_path: Path) -> None:
    agent = AgentMemoryService(service(tmp_path))
    receipt = agent.remember_episode(
        workspace="w",
        scope="s",
        content="Alice private archive marker",
        event_time=JAN,
        idempotency_key="a",
    )
    old = datetime.now(UTC)
    agent.archive_memory(
        workspace="w", scope="s", memory_id=receipt["memory_id"], idempotency_key="archive"
    )
    assert not agent.recall(MemoryQuery("w", "s", "marker", known_at=old)).hits


def test_near_name_requires_explicit_resolution_and_survives_restart(tmp_path: Path) -> None:
    s = service(tmp_path)
    s.resolve_entity(
        workspace="w",
        scope="s",
        name="Alice Johnson",
        entity_type="person",
        idempotency_key="alice",
    )
    arguments = dict(
        workspace="w",
        scope="s",
        subject_name="Alice Jonson",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "Paris"),
        validity=TimeInterval(),
        event_time=JAN,
        evidence_text="Alice Jonson lives in Paris.",
        idempotency_key="fact",
    )
    with pytest.raises(RuntimeError, match="entity_resolution_required"):
        s.assert_fact(**arguments)  # type: ignore[arg-type]
    restarted = service(tmp_path)
    pending = restarted.repository.records("w", "s", "resolution")
    assert len(pending) == 1 and len(pending[0]["candidates"]) == 1
    assert restarted.find_entity(workspace="w", scope="s", name="Alice Jonson") is None
    restarted.resolve_entity(
        workspace="w",
        scope="s",
        name="Alice Jonson",
        entity_type="person",
        idempotency_key="explicit-distinct",
    )
    assert restarted.assert_fact(**arguments)["state"] == "accepted"  # type: ignore[arg-type]


def test_worker_respects_retry_backoff_and_preserves_episode(tmp_path: Path) -> None:
    from capicove.knowledge._adapters.structured_json import StructuredJsonExtractor

    s = service(tmp_path)
    s.extractor = StructuredJsonExtractor()
    agent = AgentMemoryService(s)
    receipt = agent.remember_episode(
        workspace="w",
        scope="s",
        content="unstructured text stays searchable",
        event_time=JAN,
        idempotency_key="a",
        extraction=True,
    )
    outcome = agent.work_extractions(workspace="w", scope="s")
    assert outcome["outcomes"][0]["state"] == "retryable"
    assert agent.work_extractions(workspace="w", scope="s")["processed"] == 0
    assert agent.recall(MemoryQuery("w", "s", "searchable")).hits
    repository = s.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    assert repository.due_jobs(
        "w", "s", now=datetime.now(UTC) + timedelta(seconds=5), limit=10
    ) == (receipt["episode_id"],)
