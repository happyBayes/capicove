import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import FactProposal, MemoryEpisode, Predicate
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


class FailingExtractor:
    extractor_id = "test"
    revision = "1"

    def extract(
        self, episode: MemoryEpisode, read_text: Callable[[int, int], str]
    ) -> tuple[FactProposal, ...]:
        assert read_text(0, 5) == "hello"
        raise RuntimeError("do not persist private exception content")


def test_failed_extraction_preserves_episode_and_reaches_dead_letter(tmp_path: Path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    s = TemporalKnowledgeService(
        repository,
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(),
        extractor=FailingExtractor(),
    )
    receipt = AgentMemoryService(s).remember_episode(
        workspace="w",
        scope="s",
        content="hello evidence",
        event_time=datetime.now(UTC),
        idempotency_key="remember",
        extraction=True,
    )
    for attempt in range(3):
        result = s.retry_extraction(
            workspace="w",
            scope="s",
            episode_id=receipt["episode_id"],
            idempotency_key=f"retry{attempt}",
        )
        assert result["state"] == ("dead_letter" if attempt == 2 else "retryable")
    assert s.get_episode(workspace="w", scope="s", episode_id=receipt["episode_id"])
    assert s.recall(MemoryQuery("w", "s", "hello")).hits
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT state,attempts FROM outbox_jobs").fetchone() == ("dead_letter", 3)
        assert "private exception" not in "\n".join(
            r[0] for r in db.execute("SELECT payload FROM memory_events")
        )


def test_lease_excludes_workers_and_expired_tokens_cannot_complete(tmp_path: Path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    s = TemporalKnowledgeService(
        repository,
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(),
    )
    receipt = s.remember_episode(
        workspace="w",
        scope="s",
        content="hello",
        event_time=datetime.now(UTC),
        idempotency_key="r",
        extraction=True,
    )
    first = repository.claim_job(
        "w", "s", receipt["episode_id"], now=datetime.now(UTC), lease_seconds=1
    )
    assert first is not None
    assert (
        repository.claim_job(
            "w", "s", receipt["episode_id"], now=datetime.now(UTC), lease_seconds=1
        )
        is None
    )
    second = repository.claim_job(
        "w",
        "s",
        receipt["episode_id"],
        now=datetime.now(UTC) + timedelta(seconds=2),
        lease_seconds=30,
    )
    assert second is not None and first["token"] != second["token"]
    revision, recorded = repository.head("w")
    with pytest.raises(RuntimeError, match="lease_lost"):
        repository.commit(
            workspace="w",
            scope="s",
            key="stale-worker",
            request={},
            expected_revision=revision,
            recorded=recorded,
            events=[
                (
                    "ExtractionCompleted",
                    {"id": receipt["episode_id"], "state": "completed", "attempts": 1},
                )
            ],
            receipt={},
            job=dict(first, state="completed"),
        )
    assert repository.head("w")[0] == revision


def test_extracted_facts_use_episode_spans_and_cannot_raise_authority(tmp_path: Path) -> None:
    from capicove.contracts.evidence import EvidenceSpan
    from capicove.contracts.knowledge import Authority, EntityRef, TimeInterval, TypedLiteral

    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    s = TemporalKnowledgeService(
        repository,
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
    )
    entity = s.resolve_entity(
        workspace="w", scope="s", name="Alice", entity_type="person", idempotency_key="entity"
    )

    class Extractor:
        extractor_id = "fixture"
        revision = "1"

        def extract(
            self, episode: MemoryEpisode, read_text: Callable[[int, int], str]
        ) -> tuple[FactProposal, ...]:
            assert read_text(0, 20) == "Alice lives in Paris"
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

    s.extractor = Extractor()
    receipt = s.remember_episode(
        workspace="w",
        scope="s",
        content="Alice lives in Paris",
        event_time=datetime.now(UTC),
        idempotency_key="episode",
        extraction=True,
    )
    result = s.retry_extraction(
        workspace="w", scope="s", episode_id=receipt["episode_id"], idempotency_key="extract"
    )
    assert result["state"] == "completed"
    fact = s.get_fact_history(workspace="w", scope="s", fact_id=result["facts"][0]["fact_id"])[0]
    assert fact.authority is Authority.MODEL and fact.extraction_confidence == 0.9
    repeated = s.retry_extraction(
        workspace="w", scope="s", episode_id=receipt["episode_id"], idempotency_key="extract-again"
    )
    assert repeated["facts"] == result["facts"]


def test_partial_extraction_and_new_provider_revision(tmp_path: Path) -> None:
    from capicove.contracts.evidence import EvidenceSpan
    from capicove.contracts.knowledge import Authority, EntityRef, TimeInterval, TypedLiteral

    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    s = TemporalKnowledgeService(
        repository,
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
    )
    entity = s.resolve_entity(
        workspace="w", scope="s", name="Alice", entity_type="person", idempotency_key="e"
    )

    class Extractor:
        extractor_id = "fixture"
        revision = "1"
        calls = 0

        def extract(
            self, episode: MemoryEpisode, read_text: Callable[[int, int], str]
        ) -> tuple[FactProposal, ...]:
            self.calls += 1
            valid = FactProposal(
                EntityRef(entity.entity_id),
                "city",
                TypedLiteral("string", "Paris"),
                TimeInterval(),
                (EvidenceSpan(episode.evidence_refs[0], 0, 20),),
                episode.event_time,
                Authority.MODEL,
                0.9,
            )
            invalid = FactProposal(
                EntityRef(entity.entity_id),
                "unknown",
                TypedLiteral("string", "Paris"),
                TimeInterval(),
                valid.evidence,
                episode.event_time,
                Authority.MODEL,
                0.9,
            )
            return valid, invalid

    extractor = Extractor()
    s.extractor = extractor
    receipt = s.remember_episode(
        workspace="w",
        scope="s",
        content="Alice lives in Paris",
        event_time=datetime.now(UTC),
        idempotency_key="ep",
        extraction=True,
    )
    first = s.retry_extraction(
        workspace="w", scope="s", episode_id=receipt["episode_id"], idempotency_key="x1"
    )
    assert first["state"] == "partially_completed"
    repeated = s.retry_extraction(
        workspace="w", scope="s", episode_id=receipt["episode_id"], idempotency_key="x2"
    )
    assert repeated["facts"] == first["facts"] and extractor.calls == 1
    extractor.revision = "2"
    updated = s.retry_extraction(
        workspace="w", scope="s", episode_id=receipt["episode_id"], idempotency_key="x3"
    )
    assert updated["facts"][0]["fact_id"] == first["facts"][0]["fact_id"]
    assert extractor.calls == 2
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert db.execute("SELECT state FROM outbox_jobs").fetchone()[0] == "partially_completed"


def test_rebuild_preserves_outbox_retry_deadline(tmp_path: Path) -> None:
    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    s = TemporalKnowledgeService(
        repository,
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(),
        extractor=FailingExtractor(),
    )
    episode = s.remember_episode(
        workspace="w",
        scope="s",
        content="hello",
        event_time=datetime.now(UTC),
        idempotency_key="r",
        extraction=True,
    )
    s.retry_extraction(
        workspace="w", scope="s", episode_id=episode["episode_id"], idempotency_key="x"
    )
    with sqlite3.connect(tmp_path / "memory.db") as db:
        before = db.execute("SELECT state,attempts,next_attempt FROM outbox_jobs").fetchone()
    s.rebuild_projections()
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert (
            db.execute("SELECT state,attempts,next_attempt FROM outbox_jobs").fetchone() == before
        )
