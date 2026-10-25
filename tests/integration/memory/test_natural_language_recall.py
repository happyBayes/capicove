"""Natural-language recall must retrieve evidence without relaxing policy filters."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import Predicate, TimeInterval, TypedLiteral
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


@pytest.fixture
def agent(tmp_path: Path) -> Iterator[AgentMemoryService]:
    repository = SQLiteMemoryRepository(tmp_path / "memory.db")
    yield AgentMemoryService(
        TemporalKnowledgeService(
            repository,
            SQLiteTextEvidenceStore(tmp_path / "source.db", tmp_path / "privacy"),
            ontology_version="test",
            predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
        )
    )
    repository.close()


def remember(agent: AgentMemoryService, text: str, key: str, scope: str = "s") -> str:
    receipt = agent.remember_episode(
        workspace="w",
        scope=scope,
        content=text,
        event_time=datetime(2025, 1, 1, tzinfo=UTC),
        idempotency_key=key,
    )
    return str(receipt["memory_id"])


def test_question_words_and_punctuation_do_not_hide_relevant_evidence(
    agent: AgentMemoryService,
) -> None:
    remember(agent, "The violin recital took place in October at the community hall.", "a")
    remember(agent, "Today I baked a chocolate cake.", "b")
    context = agent.recall(MemoryQuery("w", "s", "When was my violin recital?"))
    assert len(context.hits) == 1
    assert "October" in context.hits[0].text
    assert context.hits[0].citations


@pytest.mark.parametrize("venue", ["hall", "café"])
def test_question_recall_preserves_scope_time_and_archive(
    agent: AgentMemoryService, venue: str
) -> None:
    memory = remember(agent, f"The violin recital at the {venue} was in October.", "a")
    remember(agent, f"The violin recital at the {venue} was in November.", "other", "other")
    question = f"When was the violin recital at the {venue}?"
    assert len(agent.recall(MemoryQuery("w", "s", question)).hits) == 1
    assert not agent.recall(
        MemoryQuery("w", "s", question, valid_at=datetime(2024, 1, 1, tzinfo=UTC))
    ).hits
    assert not agent.recall(
        MemoryQuery("w", "s", question, known_at=datetime(2024, 1, 1, tzinfo=UTC))
    ).hits
    agent.archive_memory(workspace="w", scope="s", memory_id=memory, idempotency_key="archive")
    assert not agent.recall(MemoryQuery("w", "s", question)).hits


@pytest.mark.parametrize("question", ["What is it?", "Where is the submarine?", '" OR *'])
def test_no_content_match_never_becomes_recency_browse(
    agent: AgentMemoryService, question: str
) -> None:
    remember(agent, "This is the violin recital.", "a")
    assert not agent.recall(MemoryQuery("w", "s", question)).hits


@pytest.mark.parametrize("literal", ["C++", "100%", "中文", "foo_bar"])
def test_symbol_and_cjk_literal_queries_keep_their_meaning(
    agent: AgentMemoryService, literal: str
) -> None:
    remember(agent, "A literal value: " + literal, "a")
    assert agent.recall(MemoryQuery("w", "s", literal)).hits


def test_single_word_name_is_not_mistaken_for_a_question(agent: AgentMemoryService) -> None:
    remember(agent, "The Who performed at the festival.", "a")
    assert agent.recall(MemoryQuery("w", "s", "Who")).hits


@pytest.mark.parametrize("name", ["Aragón", "Arago\u0301n", "München", "São Paulo"])
def test_english_question_with_latin_name_returns_unchanged_evidence(
    agent: AgentMemoryService, name: str
) -> None:
    content = f"The recommended trail in {name} is Cedar Loop."
    remember(agent, content, "a")
    remember(agent, "I baked a chocolate cake.", "b")
    context = agent.recall(MemoryQuery("w", "s", f"What trail in {name} did you recommend?"))
    assert len(context.hits) == 1
    assert context.hits[0].text == content
    span = context.hits[0].citations[0]
    assert (span.start, span.end) == (0, len(content))


@pytest.mark.parametrize("name", ["Aragón", "Arago\u0301n", "München", "São Paulo"])
def test_accented_name_itself_is_searchable_in_an_english_question(
    agent: AgentMemoryService, name: str
) -> None:
    remember(agent, f"{name}: Cedar Loop.", "a")
    remember(agent, "Berlin: Pine Walk.", "b")
    context = agent.recall(MemoryQuery("w", "s", f"What is in {name}?"))
    assert [hit.text for hit in context.hits] == [f"{name}: Cedar Loop."]
    assert not agent.recall(MemoryQuery("w", "s", "What is in Zürich?")).hits


def test_accented_question_searches_semantic_facts(agent: AgentMemoryService) -> None:
    instant = datetime(2025, 1, 1, tzinfo=UTC)
    agent.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "München"),
        validity=TimeInterval(instant),
        event_time=instant,
        evidence_text="Alice moved to München.",
        idempotency_key="fact",
    )
    result = agent.recall(MemoryQuery("w", "s", "Who is in München?", types=("semantic",)))
    assert [hit.text for hit in result.hits] == ["Alice moved to München."]
    assert not agent.recall(
        MemoryQuery("w", "other", "Who is in München?", types=("semantic",))
    ).hits


@pytest.mark.parametrize("name", ["Aragón", "violin"])
def test_terse_queries_still_require_all_terms(agent: AgentMemoryService, name: str) -> None:
    remember(agent, f"{name} cedar", "a")
    remember(agent, f"{name} pine", "b")
    assert [hit.text for hit in agent.recall(MemoryQuery("w", "s", f"{name} cedar")).hits] == [
        f"{name} cedar"
    ]
    assert not agent.recall(MemoryQuery("w", "s", f"{name} missing")).hits


@pytest.mark.parametrize("literal", ["中文", "Москва", "C++", "100%", "foo_bar", "\u212a"])
def test_mixed_questions_with_unsupported_literals_keep_literal_matching(
    agent: AgentMemoryService, literal: str
) -> None:
    remember(agent, f"Only {literal} appears here.", "a")
    assert not agent.recall(MemoryQuery("w", "s", f"What is {literal}?")).hits
