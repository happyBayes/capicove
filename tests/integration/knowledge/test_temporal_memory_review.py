"""Reviewer regression matrix over the new source-owned Evidence architecture."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import Predicate, TimeInterval, TypedLiteral
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


T0 = datetime(2026, 1, 1, tzinfo=UTC)
T1 = datetime(2026, 2, 1, tzinfo=UTC)
T2 = datetime(2026, 3, 1, tzinfo=UTC)


def service(root: Path) -> TemporalKnowledgeService:
    return TemporalKnowledgeService(
        SQLiteMemoryRepository(root / "memory.sqlite3"),
        SQLiteTextEvidenceStore(root / "source.sqlite3", root / "privacy"),
        ontology_version="v1",
        predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
    )


def assert_city(
    s: TemporalKnowledgeService, value: str, start: datetime, key: str, *, admitted: bool = False
) -> dict[str, Any]:
    writer = AgentMemoryService(s) if admitted else s
    return writer.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", value),
        validity=TimeInterval(start),
        event_time=T2,
        evidence_text="Alice lives in " + value,
        idempotency_key=key,
    )


def test_identity_dedup_history_and_current_dispute_exclusion(tmp_path: Path) -> None:
    s = service(tmp_path)
    a = assert_city(s, "Paris", T0, "a")
    assert assert_city(s, "Paris", T0, "retry-different-key")["fact_id"] == a["fact_id"]
    b = assert_city(s, "London", T1, "b")
    entity = s.find_entity(workspace="w", scope="s", name="alice")
    assert entity is not None
    current = s.current_facts(
        workspace="w", scope="s", subject=entity.entity_id, predicate="city", valid_at=T2
    )
    assert [cast(TypedLiteral, f.object).value for f in current] == ["London"]
    assert (
        s.get_fact_history(workspace="w", scope="s", fact_id=a["fact_id"])[0].knowledge.end
        is not None
    )
    assert s.get_episode(workspace="w", scope="s", episode_id=b["episode_id"]).event_time == T2
    assert_city(s, "Berlin", T1, "c")
    assert (
        s.current_facts(
            workspace="w", scope="s", subject=entity.entity_id, predicate="city", valid_at=T2
        )
        == ()
    )
    assert [
        cast(TypedLiteral, f.object).value
        for f in s.current_facts(
            workspace="w", scope="s", subject=entity.entity_id, predicate="city", valid_at=T0
        )
    ] == ["Paris"]


def test_single_wal_transaction_and_corruption_doctor(tmp_path: Path) -> None:
    s = service(tmp_path)
    assert_city(s, "Paris", T0, "a")
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        events = [r[0] for r in db.execute("SELECT event_type FROM memory_events")]
        assert "EpisodeAccepted" in events and "FactAccepted" in events
        payloads = [r[0] for r in db.execute("SELECT payload FROM memory_events")]
        assert not any("Alice lives in Paris" in p for p in payloads)
        db.execute("UPDATE memory_events SET payload='{}' WHERE revision=1")
    restarted = service(tmp_path)
    assert restarted.doctor()["status"] == "read_only_safe"
    with pytest.raises(RuntimeError, match="corrupt_event_log"):
        assert_city(restarted, "Rome", T1, "b")


def test_normalized_projections_and_replay_without_extractor(tmp_path: Path) -> None:
    s = service(tmp_path)
    receipt = assert_city(s, "Paris", T0, "a")
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "episodes",
            "episode_processing_state",
            "entities",
            "entity_aliases",
            "entity_redirects",
            "fact_evidence",
            "fact_conflicts",
            "memory_records",
            "memory_state",
            "projection_offsets",
        } <= tables
        db.execute("DELETE FROM episodes")
    s.rebuild_projections()
    assert (
        s.get_episode(workspace="w", scope="s", episode_id=receipt["episode_id"]).event_time == T2
    )
    assert s.repository.lexical("w", "s", "Paris")


def test_idempotency_and_min_revision_are_scoped(tmp_path: Path) -> None:
    s = service(tmp_path)
    a = assert_city(s, "Paris", T0, "same")
    assert assert_city(s, "Paris", T0, "same") == a
    with pytest.raises(RuntimeError, match="write_conflict"):
        assert_city(s, "Rome", T0, "same")
    with pytest.raises(RuntimeError, match="consistency_timeout"):
        s.get_fact_history(
            workspace="w", scope="s", fact_id=a["fact_id"], min_revision=a["committed_revision"] + 1
        )
    assert s.get_fact_history(workspace="w", scope="other", fact_id=a["fact_id"]) == ()


def test_recall_filters_both_times_before_ranking_and_cites_source(tmp_path: Path) -> None:
    from capicove.contracts.memory import MemoryQuery

    s = service(tmp_path)
    a = assert_city(s, "Paris", T0, "a", admitted=True)
    known_before = s.get_fact_history(workspace="w", scope="s", fact_id=a["fact_id"])[
        0
    ].knowledge.start
    assert_city(s, "London", T1, "b", admitted=True)
    hits = s.recall(MemoryQuery("w", "s", "Paris", valid_at=T2, types=("semantic",))).hits
    assert all("Paris" not in h.text for h in hits)
    old = s.recall(
        MemoryQuery("w", "s", "Paris", valid_at=T2, known_at=known_before, types=("semantic",))
    )
    assert len(old.hits) == 1
    assert "Paris" in old.hits[0].text and old.hits[0].citations
    assert old.warnings == ()
    assert not s.recall(MemoryQuery("w", "other", "Paris")).hits


def test_chinese_short_literal_recall_and_source_failure(tmp_path: Path) -> None:
    from capicove.contracts.memory import MemoryQuery

    s = service(tmp_path)
    AgentMemoryService(s).remember_episode(
        workspace="w", scope="s", content="小王搬到北京", event_time=T0, idempotency_key="chinese"
    )
    result = s.recall(MemoryQuery("w", "s", "北京"))
    assert result.hits[0].text == "小王搬到北京"
    with sqlite3.connect(tmp_path / "source.sqlite3") as db:
        db.execute("DELETE FROM text_evidence")
    with pytest.raises(RuntimeError, match="evidence_unavailable"):
        s.recall(MemoryQuery("w", "s", "北京"))


def test_malformed_event_enters_safe_mode_and_never_appends(tmp_path: Path) -> None:
    s = service(tmp_path)
    assert_city(s, "Paris", T0, "a")
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        db.execute("UPDATE memory_events SET payload='not-json' WHERE revision=1")
    restarted = service(tmp_path)
    assert restarted.doctor()["status"] == "read_only_safe"


def test_late_backfill_cannot_replace_later_world_fact(tmp_path: Path) -> None:
    s = service(tmp_path)
    assert_city(s, "London", T1, "newer")
    a = assert_city(s, "Paris", T0, "backfill")
    entity = s.find_entity(workspace="w", scope="s", name="ALICE")
    assert entity is not None
    current = s.current_facts(
        workspace="w", scope="s", subject=entity.entity_id, predicate="city", valid_at=T2
    )
    assert cast(TypedLiteral, current[0].object).value == "London"
    assert s.get_fact_history(workspace="w", scope="s", fact_id=a["fact_id"])[-1].validity.end == T1


def test_alias_redirect_preserves_canonical_reads_and_evidence(tmp_path: Path) -> None:
    s = service(tmp_path)
    old = assert_city(s, "Paris", T0, "a")
    alice = s.find_entity(workspace="w", scope="s", name="alice")
    assert alice is not None
    target = s.resolve_entity(
        workspace="w",
        scope="s",
        name="Alice Example",
        entity_type="person",
        idempotency_key="entity",
    )
    s.add_entity_alias(
        workspace="w",
        scope="s",
        entity_id=target.entity_id,
        alias="A. Example",
        idempotency_key="alias",
    )
    s.merge_entities(
        workspace="w",
        scope="s",
        source_id=alice.entity_id,
        target_id=target.entity_id,
        idempotency_key="merge",
    )
    assert s.find_entity(workspace="w", scope="s", name="alice") == target
    assert s.find_entity(workspace="w", scope="s", name="a. example") == target
    facts = s.current_facts(
        workspace="w", scope="s", subject=alice.entity_id, predicate="city", valid_at=T2
    )
    assert len(facts) == 1 and facts[0].subject_ref.entity_id == target.entity_id
    assert (
        s.get_fact_history(workspace="w", scope="s", fact_id=old["fact_id"])[0].knowledge.end
        is not None
    )


def test_ledger_events_link_to_operations_and_schema(tmp_path: Path) -> None:
    s = service(tmp_path)
    receipt = assert_city(s, "Paris", T0, "a")
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        event_columns = {r[1] for r in db.execute("PRAGMA table_info(memory_events)")}
        assert {"operation_id", "schema_version", "aggregate_type", "aggregate_id"} <= event_columns
        row = db.execute(
            "SELECT o.first_revision,o.last_revision,COUNT(e.revision) FROM memory_operations o JOIN memory_events e ON e.operation_id=o.operation_id GROUP BY o.operation_id"
        ).fetchone()
        assert row[0] == 1 and row[1] == receipt["committed_revision"] and row[2] == row[1]
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_live_external_ledger_tampering_blocks_next_write(tmp_path: Path) -> None:
    s = service(tmp_path)
    assert_city(s, "Paris", T0, "a")
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        db.execute("UPDATE memory_events SET payload='{}' WHERE revision=1")
    with pytest.raises(RuntimeError, match="corrupt_event_log"):
        assert_city(s, "London", T1, "b")


def test_explicit_correction_preserves_old_belief_without_current_dispute(tmp_path: Path) -> None:
    s = service(tmp_path)
    old = assert_city(s, "Paris", T0, "old")
    before = s.get_fact_history(workspace="w", scope="s", fact_id=old["fact_id"])[0].knowledge.start
    s.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "London"),
        validity=TimeInterval(T0),
        event_time=T2,
        evidence_text="Correction: Alice lives in London",
        idempotency_key="correct",
        supersedes=old["fact_id"],
    )
    entity = s.find_entity(workspace="w", scope="s", name="Alice")
    assert entity is not None
    current = s.current_facts(
        workspace="w", scope="s", subject=entity.entity_id, predicate="city", valid_at=T2
    )
    previous = s.current_facts(
        workspace="w",
        scope="s",
        subject=entity.entity_id,
        predicate="city",
        valid_at=T2,
        known_at=before,
    )
    assert [cast(TypedLiteral, f.object).value for f in current] == ["London"]
    assert [cast(TypedLiteral, f.object).value for f in previous] == ["Paris"]


def test_tampered_receipt_is_not_replayed_as_success(tmp_path: Path) -> None:
    s = service(tmp_path)
    assert_city(s, "Paris", T0, "a")
    with sqlite3.connect(tmp_path / "memory.sqlite3") as db:
        db.execute("UPDATE memory_operations SET receipt='{}'")
    with pytest.raises(RuntimeError, match="corrupt_event_log"):
        assert_city(s, "Paris", T0, "a")


def test_reasserting_retracted_fact_is_explicit_and_reuses_identity(tmp_path: Path) -> None:
    s = service(tmp_path)
    old = assert_city(s, "Paris", T0, "old")
    s.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "London"),
        validity=TimeInterval(T0),
        event_time=T2,
        evidence_text="corrected London",
        idempotency_key="correct",
        supersedes=old["fact_id"],
    )
    again = assert_city(s, "Paris", T0, "explicit-new-statement")
    assert again["fact_id"] == old["fact_id"] and again["state"] == "disputed"
    s.resolve_entity(
        workspace="w", scope="s", name="Alice", entity_type="person", idempotency_key="resolution"
    )
    with pytest.raises(RuntimeError, match="write_conflict"):
        s.resolve_entity(
            workspace="w", scope="s", name="Bob", entity_type="person", idempotency_key="resolution"
        )
