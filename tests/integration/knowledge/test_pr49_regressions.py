"""Behavioral acceptance from PR #49, independent of implementation internals."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import Authority, Predicate, TimeInterval, TypedLiteral
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.knowledge.conflicts import Assertion, resolve_slot
from capicove.source import SQLiteTextEvidenceStore


JAN = datetime(2026, 1, 1, tzinfo=UTC)
FEB = datetime(2026, 2, 1, tzinfo=UTC)


def service(root: Path) -> TemporalKnowledgeService:
    return TemporalKnowledgeService(
        SQLiteMemoryRepository(root / "memory.db"),
        SQLiteTextEvidenceStore(root / "source.db", root / "privacy"),
        ontology_version="v1",
        predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_negative_infinity_is_an_ordered_start(reverse: bool) -> None:
    items = (
        Assertion("old", TypedLiteral("string", "Paris"), TimeInterval(), Authority.USER),
        Assertion("new", TypedLiteral("string", "Lyon"), TimeInterval(FEB), Authority.USER),
    )
    result = resolve_slot(
        items[::-1] if reverse else items, cardinality="one", temporal_kind="interval"
    )
    assert result["old"].validity == TimeInterval(None, FEB)
    assert result["old"].state == result["new"].state == "accepted"


@pytest.mark.parametrize("cardinality", ["one", "many"])
def test_static_rejects_bounds_at_conflict_boundary(cardinality: str) -> None:
    with pytest.raises(ValueError, match="static"):
        resolve_slot(
            (Assertion("a", TypedLiteral("string", "Paris"), TimeInterval(JAN), Authority.USER),),
            cardinality=cardinality,
            temporal_kind="static",
        )


def test_agent_semantic_context_contains_readable_evidence(tmp_path: Path) -> None:
    agent = AgentMemoryService(service(tmp_path))
    agent.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "Paris"),
        validity=TimeInterval(JAN),
        event_time=JAN,
        evidence_text="Alice reported moving to Paris.",
        idempotency_key="a",
    )
    result = agent.recall(MemoryQuery("w", "s", "Paris", types=("semantic",)))
    assert len(result.hits) == 1
    assert "Alice reported moving to Paris." in result.hits[0].text
    assert result.hits[0].citations


def test_dense_programming_errors_are_not_optional_outages(tmp_path: Path) -> None:
    s = service(tmp_path)

    def broken(query: MemoryQuery) -> tuple[str, ...]:
        raise TypeError("provider implementation bug")

    s.dense = broken
    with pytest.raises(TypeError, match="implementation bug"):
        s.recall(MemoryQuery("w", "s", "Paris"))


def test_text_search_does_not_fill_unknown_answers_with_recent_memories(tmp_path: Path) -> None:
    s = service(tmp_path)
    AgentMemoryService(s).remember_episode(
        workspace="w",
        scope="s",
        content="Alice lives in Paris",
        event_time=JAN,
        idempotency_key="a",
    )
    assert s.recall(MemoryQuery("w", "s", "astronaut penguin")).hits == ()
    browse = s.recall(MemoryQuery("w", "s"))
    assert browse.hits
    assert "browse_without_query" in browse.warnings


def test_long_episode_yields_a_bounded_exactly_cited_excerpt(tmp_path: Path) -> None:
    agent = AgentMemoryService(service(tmp_path))
    content = "Alice reported moving to Paris. " + "Authored supporting detail. " * 2000
    agent.remember_episode(
        workspace="w", scope="s", content=content, event_time=JAN, idempotency_key="long"
    )
    result = agent.recall(MemoryQuery("w", "s", "Alice"), max_characters=1500, max_tokens=1500)
    assert result.hits
    hit = result.hits[0]
    assert hit.text == content[hit.citations[0].start : hit.citations[0].end]
    assert result.characters <= 1500 and result.estimated_tokens <= 1500
