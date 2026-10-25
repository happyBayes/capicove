import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import Predicate
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


def make_agent(root: Path) -> AgentMemoryService:
    knowledge = TemporalKnowledgeService(
        SQLiteMemoryRepository(root / "memory.db"),
        SQLiteTextEvidenceStore(root / "source.db", root / "privacy"),
        ontology_version="v1",
        predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
    )
    return AgentMemoryService(knowledge)


def test_policy_archive_budget_and_erasure(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    now = datetime.now(UTC)
    receipt = agent.remember_episode(
        workspace="w",
        scope="s",
        content="Private source secret value",
        event_time=now,
        idempotency_key="a",
    )
    context = agent.recall(MemoryQuery("w", "s", "Private"), max_characters=10, max_tokens=10)
    assert not context.hits
    assert agent.recall(MemoryQuery("w", "s", "Private")).hits
    agent.archive_memory(
        workspace="w", scope="s", memory_id=receipt["memory_id"], idempotency_key="archive"
    )
    assert not agent.recall(MemoryQuery("w", "s", "Private")).hits
    agent.erase_scope(workspace="w", scope="s", idempotency_key="erase")
    restarted = make_agent(tmp_path)
    assert restarted.knowledge.doctor()["status"] == "ok"
    with pytest.raises(RuntimeError, match="scope_erased"):
        restarted.recall(MemoryQuery("w", "s", "Private"))
    for path in (tmp_path / "memory.db", tmp_path / "source.db"):
        assert b"Private source secret value" not in path.read_bytes()


def test_restricted_pinning_cannot_override_admission_policy(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    with pytest.raises(ValueError, match="sensitivity_rejected"):
        agent.remember_episode(
            workspace="w",
            scope="s",
            content="restricted",
            event_time=datetime.now(UTC),
            idempotency_key="restricted",
            sensitivity="restricted",
            pinned=True,
        )
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0


def test_scope_erasure_scrubs_fts_terms_while_preserving_other_scope(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    for scope, content in [("s", "uniquesecrettermzxy"), ("other", "keepthispublicterm")]:
        agent.remember_episode(
            workspace="w",
            scope=scope,
            content=content,
            event_time=datetime.now(UTC),
            idempotency_key=scope,
        )
    agent.erase_scope(workspace="w", scope="s", idempotency_key="erase")
    assert b"uniquesecrettermzxy" not in (tmp_path / "memory.db").read_bytes()
    assert agent.recall(MemoryQuery("w", "other", "keepthispublicterm")).hits
    agent.knowledge.rebuild_projections()
    assert agent.knowledge.doctor()["status"] == "ok"


def test_erasure_retry_finishes_pending_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(tmp_path)
    agent.remember_episode(
        workspace="w",
        scope="s",
        content="erase me",
        event_time=datetime.now(UTC),
        idempotency_key="r",
    )
    repository = agent.knowledge.repository
    original = repository.checkpoint
    attempts = 0

    def interrupted() -> None:
        from capicove.knowledge.errors import MemoryError

        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MemoryError("erasure_checkpoint_pending")
        original()

    monkeypatch.setattr(repository, "checkpoint", interrupted)
    with pytest.raises(RuntimeError, match="erasure_checkpoint_pending"):
        agent.erase_scope(workspace="w", scope="s", idempotency_key="erase")
    result = agent.erase_scope(workspace="w", scope="s", idempotency_key="erase")
    assert result["state"] == "erased" and attempts == 2


def test_extracted_fact_admission_inherits_episode_policy(tmp_path: Path) -> None:
    from collections.abc import Callable

    from capicove.contracts.evidence import EvidenceSpan
    from capicove.contracts.knowledge import (
        Authority,
        EntityRef,
        FactProposal,
        MemoryEpisode,
        TimeInterval,
        TypedLiteral,
    )

    agent = make_agent(tmp_path)
    entity = agent.knowledge.resolve_entity(
        workspace="w", scope="s", name="Alice", entity_type="person", idempotency_key="e"
    )

    class Extractor:
        extractor_id = "fixture"
        revision = "1"

        def extract(
            self, episode: MemoryEpisode, read_text: Callable[[int, int], str]
        ) -> tuple[FactProposal, ...]:
            return (
                FactProposal(
                    EntityRef(entity.entity_id),
                    "city",
                    TypedLiteral("string", "Paris"),
                    TimeInterval(),
                    (EvidenceSpan(episode.evidence_refs[0], 0, 20),),
                    episode.event_time,
                    Authority.MODEL,
                    0.9,
                ),
            )

    agent.knowledge.extractor = Extractor()
    episode = agent.remember_episode(
        workspace="w",
        scope="s",
        content="Alice lives in Paris",
        event_time=datetime.now(UTC),
        sensitivity="confidential",
        pinned=True,
        extraction=True,
        idempotency_key="ep",
    )
    result = agent.retry_extraction(
        workspace="w", scope="s", episode_id=episode["episode_id"], idempotency_key="x"
    )
    fact_id = result["facts"][0]["fact_id"]
    policy = agent.knowledge.memory_policy_for(workspace="w", scope="s", target=fact_id)
    assert policy is not None
    assert policy["sensitivity"] == "confidential" and policy["retention"] == "pinned"
    assert policy["retention_deadline"] is None
