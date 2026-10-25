from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.memory import MemoryQuery, MemoryRecord
from capicove.contracts.rank_fusion import normalized_rrf
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


def make_service(root: Path) -> TemporalKnowledgeService:
    return TemporalKnowledgeService(
        SQLiteMemoryRepository(root / "memory.db"),
        SQLiteTextEvidenceStore(root / "source.db", root / "privacy"),
        ontology_version="v1",
        predicates=(),
    )


def test_weighted_rrf_is_normalized_and_rejects_duplicate_lane_identity() -> None:
    result = normalized_rrf(((1.0, (("a", 1), ("b", 2))), (0.8, (("b", 1),))))
    assert 0 < result["a"] < result["b"] <= 1
    with pytest.raises(ValueError):
        normalized_rrf(((1.0, (("a", 1), ("a", 2))),))


def test_multiword_fts_and_literal_query_syntax(tmp_path: Path) -> None:
    s = make_service(tmp_path)
    r = AgentMemoryService(s).remember_episode(
        workspace="w",
        scope="s",
        content="Alice currently lives in Paris",
        event_time=datetime.now(UTC),
        idempotency_key="a",
    )
    assert (
        s.repository.candidates(MemoryQuery("w", "s", "Alice Paris"), lane="lexical")[0]["id"]
        == r["episode_id"]
    )
    assert s.repository.candidates(MemoryQuery("w", "s", '" OR scope:*'), lane="lexical") == []


def test_dense_candidates_are_hard_filtered_and_failure_only_warns_when_configured(
    tmp_path: Path,
) -> None:
    s = make_service(tmp_path)
    now = datetime.now(UTC)
    future = AgentMemoryService(s).remember_episode(
        workspace="w",
        scope="s",
        content="future",
        event_time=now + timedelta(days=1),
        idempotency_key="f",
    )
    other = AgentMemoryService(s).remember_episode(
        workspace="w", scope="other", content="other", event_time=now, idempotency_key="o"
    )
    current = AgentMemoryService(s).remember_episode(
        workspace="w", scope="s", content="present", event_time=now, idempotency_key="c"
    )
    s.dense = lambda q: (future["episode_id"], other["episode_id"], current["episode_id"])
    result = s.recall(MemoryQuery("w", "s", "present"))
    assert [hit.target_ref for hit in result.hits] == [current["episode_id"]]
    assert result.warnings == ()

    def failing(q: MemoryQuery) -> tuple[str, ...]:
        from capicove.knowledge.errors import OptionalMemoryLaneUnavailable

        raise OptionalMemoryLaneUnavailable("private provider detail")

    s.dense = failing
    degraded = s.recall(MemoryQuery("w", "s", "present"))
    assert degraded.warnings == ("embedding_unavailable",) and degraded.hits


def test_retention_filters_every_lane(tmp_path: Path) -> None:
    s = make_service(tmp_path)
    now = datetime.now(UTC)
    receipt = s.remember_episode(
        workspace="w", scope="s", content="expired policy", event_time=now, idempotency_key="r"
    )
    record = MemoryRecord(
        "m",
        "w",
        "s",
        receipt["episode_id"],
        "episodic",
        "internal",
        "session",
        now - timedelta(seconds=1),
        0.5,
        "authored",
        now - timedelta(days=1),
    )
    s.record_memory(record, idempotency_key="policy")
    assert s.recall(MemoryQuery("w", "s", "expired")).hits == ()


def test_word_search_and_cjk_literal_index_survive_rebuild(tmp_path: Path) -> None:
    s = make_service(tmp_path)
    r = AgentMemoryService(s).remember_episode(
        workspace="w",
        scope="s",
        content="Alice Paris 中文时序记忆",
        event_time=datetime.now(UTC),
        idempotency_key="r",
    )
    for query in ("Alice Paris", "时序", "文时序记"):
        assert (
            s.repository.candidates(MemoryQuery("w", "s", query), lane="lexical")[0]["id"]
            == r["episode_id"]
        )
    s.rebuild_projections()
    for query in ("Alice Paris", "时序", "文时序记"):
        assert (
            s.repository.candidates(MemoryQuery("w", "s", query), lane="lexical")[0]["id"]
            == r["episode_id"]
        )


def test_two_hop_graph_uses_typed_edges_and_keeps_third_hop_out_of_graph_score(
    tmp_path: Path,
) -> None:
    from capicove.contracts.knowledge import EntityRef, Predicate, TimeInterval

    s = TemporalKnowledgeService(
        SQLiteMemoryRepository(tmp_path / "memory.db"),
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(Predicate("parent", ("person",), "entity", ("person",), "many", "static"),),
    )
    entities = [
        s.resolve_entity(
            workspace="w", scope="s", name=name, entity_type="person", idempotency_key="e" + name
        )
        for name in ("Alice", "Bob", "Carol", "Dave")
    ]
    facts = []
    for i in range(3):
        facts.append(
            AgentMemoryService(s).assert_fact(
                workspace="w",
                scope="s",
                subject_name=entities[i].display_name,
                subject_type="person",
                predicate="parent",
                object=EntityRef(entities[i + 1].entity_id),
                validity=TimeInterval(),
                event_time=datetime.now(UTC),
                evidence_text=f"{entities[i].display_name} parent {entities[i + 1].display_name}",
                idempotency_key=f"f{i}",
            )
        )
    result = s.recall(
        MemoryQuery("w", "s", types=("semantic",), center_entities=(entities[0].entity_id,))
    )
    scores = {hit.target_ref: hit.score for hit in result.hits}
    assert scores[facts[0]["fact_id"]] > scores[facts[2]["fact_id"]]
    assert scores[facts[1]["fact_id"]] > scores[facts[2]["fact_id"]]
    assert (
        s.recall(
            MemoryQuery("w", "other", types=("semantic",), center_entities=(entities[0].entity_id,))
        ).hits
        == ()
    )
