"""Qualify real text-service composition, not embedding-model quality."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.retrieval import (
    StaticTextRetrievalProfileRegistry,
    TextRetrievalProfile,
    TextSearchService,
)
from capicove.retrieval.memory_text_search import MemoryTextSearchLane
from capicove.source import SQLiteTextEvidenceStore
from tests.unit.retrieval.test_text_models import text_embedding_space
from tests.unit.retrieval.test_text_search import (
    Backend,
    Catalog,
    Embedder,
    Lifecycle,
    RecordingLane,
    active_record,
    candidate,
)


def test_memory_uses_text_generation_embedding_and_revalidates_results(tmp_path: Path) -> None:
    memory = TemporalKnowledgeService(
        SQLiteMemoryRepository(tmp_path / "memory.db"),
        SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
        ontology_version="v1",
        predicates=(),
    )
    from capicove.agent import AgentMemoryService

    allowed = AgentMemoryService(memory).remember_episode(
        workspace="w",
        scope="s",
        content="A canine companion named Pepper.",
        event_time=datetime(2020, 1, 1, tzinfo=UTC),
        idempotency_key="allowed",
    )
    hidden = AgentMemoryService(memory).remember_episode(
        workspace="w",
        scope="private",
        content="secret",
        event_time=datetime(2020, 1, 1, tzinfo=UTC),
        idempotency_key="hidden",
    )
    dense = RecordingLane((candidate(hidden["episode_id"], 1), candidate(allowed["episode_id"], 2)))
    profile = TextRetrievalProfile(
        "memory-dense", 0, 1, False, True, 100, 100, text_embedding_space()
    )
    embedder = Embedder()
    text_service = TextSearchService(
        backend=Backend(RecordingLane(), dense, Lifecycle()),  # type: ignore[arg-type]
        catalog=Catalog(active_record()),  # type: ignore[arg-type]
        profiles=StaticTextRetrievalProfileRegistry((profile,)),
        query_embedder=embedder,
        query_id_factory=lambda: "q",
    )
    memory.dense = MemoryTextSearchLane(text_service, profile=profile)
    result = memory.recall(MemoryQuery("w", "s", "dog"))
    assert [h.target_ref for h in result.hits] == [allowed["episode_id"]]
    assert result.hits[0].text == "A canine companion named Pepper."
    assert dense.pins
    embedder.failure = ConnectionError("private provider address")
    degraded = memory.recall(MemoryQuery("w", "s", "canine"))
    assert degraded.hits and degraded.warnings == ("embedding_unavailable",)
    for error in (TypeError, AssertionError, ZeroDivisionError, RuntimeError):
        embedder.failure = error("embedding implementation bug")
        with pytest.raises(error, match="implementation bug"):
            memory.recall(MemoryQuery("w", "s", "dog"))
