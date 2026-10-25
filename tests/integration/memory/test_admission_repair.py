"""Admission is a current policy gate shared by every recall lane."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pytest

from capicove.agent import AgentMemoryService
from capicove.contracts.knowledge import EntityRef, Predicate, TimeInterval, TypedLiteral
from capicove.contracts.memory import MemoryQuery, MemoryRecord
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


JAN = datetime(2026, 1, 1, tzinfo=UTC)
Kind = Literal["episodic", "semantic"]


def make_agent(root: Path) -> AgentMemoryService:
    return AgentMemoryService(
        TemporalKnowledgeService(
            SQLiteMemoryRepository(root / "memory.db"),
            SQLiteTextEvidenceStore(root / "source.db", root / "privacy"),
            ontology_version="v1",
            predicates=(
                Predicate("city", ("person",), "literal", ("string",), "one", "interval"),
                Predicate("knows", ("person",), "entity", ("person",), "one", "interval"),
            ),
        )
    )


def write_target(agent: AgentMemoryService, kind: Kind) -> str:
    if kind == "episodic":
        return str(
            agent.knowledge.remember_episode(
                workspace="w",
                scope="s",
                content="Alice violin Paris",
                event_time=JAN,
                idempotency_key="base",
            )["episode_id"]
        )
    return str(
        agent.knowledge.assert_fact(
            workspace="w",
            scope="s",
            subject_name="Alice",
            subject_type="person",
            predicate="city",
            object=TypedLiteral("string", "Paris"),
            validity=TimeInterval(),
            event_time=JAN,
            evidence_text="Alice violin Paris",
            idempotency_key="base",
        )["fact_id"]
    )


def admit(
    agent: AgentMemoryService,
    target: str,
    kind: Kind,
    key: str,
    *,
    expired: bool = False,
    pinned: bool = False,
    importance: float = 0.5,
) -> str:
    now = datetime.now(UTC)
    return agent.knowledge.record_memory(
        MemoryRecord(
            key,
            "w",
            "s",
            target,
            kind,
            "internal",
            "pinned" if pinned else "standard",
            now - timedelta(days=1) if expired else None,
            importance,
            "fixture",
            now - timedelta(days=2),
        ),
        idempotency_key=key,
    )


def lane_ids(agent: AgentMemoryService, target: str, kind: Kind, lane: str) -> list[str]:
    entity = agent.knowledge.find_entity(workspace="w", scope="s", name="Alice")
    identities = (entity.entity_id,) if lane == "graph" and entity is not None else (target,)
    return [
        body["id"]
        for body in agent.knowledge.repository.candidates(
            MemoryQuery("w", "s", "Paris", types=(kind,)),
            lane=lane,
            identities=identities,
        )
    ]


CASES = [
    (kind, lane)
    for kind in ("episodic", "semantic")
    for lane in ("lexical", "ids", "recency", "utility", "graph")
    if kind == "semantic" or lane != "graph"
]


@pytest.mark.parametrize(("kind", "lane"), CASES)
def test_unadmitted_target_is_hidden_in_every_lane(tmp_path: Path, kind: Kind, lane: str) -> None:
    agent = make_agent(tmp_path)
    target = write_target(agent, kind)
    assert lane_ids(agent, target, kind, lane) == []
    admit(agent, target, kind, "admission")
    assert lane_ids(agent, target, kind, lane) == [target]


@pytest.mark.parametrize(("kind", "lane"), CASES)
@pytest.mark.parametrize("old_state", ["archived", "expired"])
def test_readmission_uses_latest_policy_and_never_falls_back(
    tmp_path: Path,
    kind: Kind,
    lane: str,
    old_state: str,
) -> None:
    agent = make_agent(tmp_path)
    target = write_target(agent, kind)
    old = admit(agent, target, kind, "old", expired=old_state == "expired")
    if old_state == "archived":
        agent.archive_memory(workspace="w", scope="s", memory_id=old, idempotency_key="archive-old")
    assert lane_ids(agent, target, kind, lane) == []
    latest = admit(agent, target, kind, "new", pinned=True)
    assert lane_ids(agent, target, kind, lane) == [target]
    # Retrying or archiving an old admission cannot replace the newer policy.
    assert admit(agent, target, kind, "old", expired=old_state == "expired") == old
    agent.archive_memory(workspace="w", scope="s", memory_id=old, idempotency_key="archive-again")
    assert lane_ids(agent, target, kind, lane) == [target]
    agent.knowledge.rebuild_projections()
    assert lane_ids(agent, target, kind, lane) == [target]
    agent.archive_memory(workspace="w", scope="s", memory_id=latest, idempotency_key="archive-new")
    assert lane_ids(agent, target, kind, lane) == []
    # A later expired policy must not expose an earlier active/pinned admission.
    admit(agent, target, kind, "active", pinned=True)
    admit(agent, target, kind, "expired-last", expired=True)
    assert lane_ids(agent, target, kind, lane) == []


@pytest.mark.parametrize("kind", ["episodic", "semantic"])
def test_failure_between_base_commit_and_admission_recovers_without_duplicate_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: Kind,
) -> None:
    agent = make_agent(tmp_path)
    arguments: dict[str, Any] = dict(
        workspace="w", scope="s", event_time=JAN, idempotency_key="base"
    )
    if kind == "episodic":
        arguments["content"] = "Alice violin Paris"
    else:
        arguments.update(
            subject_name="Alice",
            subject_type="person",
            predicate="city",
            object=TypedLiteral("string", "Paris"),
            validity=TimeInterval(),
            evidence_text="Alice violin Paris",
        )
    method = agent.remember_episode if kind == "episodic" else agent.assert_fact
    with monkeypatch.context() as patch:

        def interrupted(*args: Any, **kwargs: Any) -> str:
            raise RuntimeError("simulated admission failure")

        patch.setattr(agent.knowledge, "record_memory", interrupted)
        with pytest.raises(RuntimeError, match="simulated admission failure"):
            method(**arguments)
    repository = agent.knowledge.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    repository.close()
    restarted = make_agent(tmp_path)
    rows = restarted.knowledge.repository.records("w", "s", "episode")
    assert len(rows) == 1
    assert not restarted.recall(MemoryQuery("w", "s", "Paris")).hits
    # Admission controls recall, not authorized inspection of committed evidence.
    episode = restarted.get_episode(workspace="w", scope="s", episode_id=rows[0]["id"])
    assert episode.episode_id == rows[0]["id"]
    assert episode.content_length == len("Alice violin Paris")
    assert episode.evidence_refs
    if kind == "semantic":
        facts = restarted.knowledge.repository.facts("w", "s", current=True)
        assert len(facts) == 1
        history = restarted.get_fact_history(workspace="w", scope="s", fact_id=facts[0]["id"])
        assert len(history) == 1
        assert history[0].object == TypedLiteral("string", "Paris")
        assert history[0].state == "accepted"
    method = restarted.remember_episode if kind == "episodic" else restarted.assert_fact
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: method(**arguments), range(2)))
    assert results[0] == results[1]
    assert len(restarted.recall(MemoryQuery("w", "s", "Paris", types=(kind,))).hits) == 1
    with sqlite3.connect(tmp_path / "memory.db") as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM memory_events WHERE event_type='EpisodeAccepted'"
            ).fetchone()[0]
            == 1
        )
        assert (
            db.execute(
                "SELECT COUNT(*) FROM memory_events WHERE event_type='MemoryAdmitted'"
            ).fetchone()[0]
            == 1
        )
    revision = restarted.knowledge.repository.head("w")[0]
    assert method(**arguments) == results[0]
    assert restarted.knowledge.repository.head("w")[0] == revision
    assert restarted.doctor()["status"] == "ok"


def test_dense_provider_cannot_return_unadmitted_content(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    target = write_target(agent, "episodic")
    agent.knowledge.dense = lambda query: (target,)
    assert not agent.recall(MemoryQuery("w", "s", "unrelated")).hits
    admit(agent, target, "episodic", "approved")
    assert [hit.target_ref for hit in agent.recall(MemoryQuery("w", "s", "unrelated")).hits] == [
        target
    ]


def test_incoming_entity_edge_requires_latest_admission(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    boris = agent.resolve_entity(
        workspace="w", scope="s", name="Boris", entity_type="person", idempotency_key="boris"
    )
    target = agent.knowledge.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="knows",
        object=EntityRef(boris.entity_id),
        validity=TimeInterval(),
        event_time=JAN,
        evidence_text="Alice knows Boris",
        idempotency_key="incoming",
    )["fact_id"]

    def incoming_ids() -> list[str]:
        return [
            row["id"]
            for row in agent.knowledge.repository.candidates(
                MemoryQuery("w", "s", types=("semantic",)),
                lane="graph",
                identities=(boris.entity_id,),
            )
        ]

    assert incoming_ids() == []
    policy = admit(agent, target, "semantic", "incoming-admission")
    assert incoming_ids() == [target]
    agent.archive_memory(workspace="w", scope="s", memory_id=policy, idempotency_key="archive")
    assert incoming_ids() == []
    admit(agent, target, "semantic", "incoming-readmission")
    assert incoming_ids() == [target]


@pytest.mark.slow
def test_utility_with_admitted_corpus_does_not_scan_unadmitted_noise(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    targets = []
    for i in range(80):
        target = agent.knowledge.remember_episode(
            workspace="w",
            scope="s",
            content=f"admitted {i}",
            event_time=JAN,
            idempotency_key=f"base{i}",
        )["episode_id"]
        targets.append(target)
        admit(agent, target, "episodic", f"policy{i}", importance=i / 100)
    repository = agent.knowledge.repository
    assert isinstance(repository, SQLiteMemoryRepository)

    def measure() -> tuple[list[str], int]:
        steps = 0

        def progress() -> int:
            nonlocal steps
            steps += 1
            return 0

        # Instrument the whole pool: candidates borrows its own reader.
        for db in repository._reader_connections:
            db.set_progress_handler(progress, 100)
        try:
            rows = repository.candidates(MemoryQuery("w", "s", types=("episodic",)), lane="utility")
        finally:
            for db in repository._reader_connections:
                db.set_progress_handler(None, 0)
        return [row["id"] for row in rows], steps

    expected = list(reversed(targets[-50:]))
    baseline_ids, baseline_steps = measure()
    assert baseline_ids == expected
    assert baseline_steps > 0
    for i in range(1000):
        agent.knowledge.remember_episode(
            workspace="w",
            scope="s",
            content=f"unadmitted {i}",
            event_time=JAN,
            idempotency_key=f"noise{i}",
        )
    noisy_ids, noisy_steps = measure()
    assert noisy_ids == expected
    assert noisy_steps <= baseline_steps * 2, "unadmitted noise must not cause a corpus scan"


def test_utility_uses_latest_importance_not_historical_maximum(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    first = write_target(agent, "episodic")
    admit(agent, first, "episodic", "high", importance=1)
    admit(agent, first, "episodic", "low", importance=0.1)
    second = agent.remember_episode(
        workspace="w", scope="s", content="violin second", event_time=JAN, idempotency_key="second"
    )["episode_id"]
    assert lane_ids(agent, first, "episodic", "utility") == [second, first]


@pytest.mark.parametrize("lane", ["lexical", "recency"])
def test_unadmitted_targets_do_not_consume_candidate_budget(tmp_path: Path, lane: str) -> None:
    agent = make_agent(tmp_path)
    target = write_target(agent, "episodic")
    admit(agent, target, "episodic", "approved")
    for i in range(105):
        agent.knowledge.remember_episode(
            workspace="w",
            scope="s",
            content="Paris",
            event_time=JAN + timedelta(seconds=i + 1),
            idempotency_key=f"unadmitted{i}",
        )
    assert lane_ids(agent, target, "episodic", lane) == [target]


def test_readmission_survives_restart_and_cannot_override_scope_erasure(tmp_path: Path) -> None:
    agent = make_agent(tmp_path)
    target = write_target(agent, "episodic")
    old = admit(agent, target, "episodic", "old")
    agent.archive_memory(workspace="w", scope="s", memory_id=old, idempotency_key="archive")
    admit(agent, target, "episodic", "new", pinned=True)
    repository = agent.knowledge.repository
    assert isinstance(repository, SQLiteMemoryRepository)
    repository.close()
    agent = make_agent(tmp_path)
    old_known = agent.get_episode(workspace="w", scope="s", episode_id=target).recorded_at
    assert [
        hit.target_ref
        for hit in agent.recall(
            MemoryQuery("w", "s", "Paris", known_at=old_known),
        ).hits
    ] == [target]
    agent.erase_scope(workspace="w", scope="s", idempotency_key="erase")
    with pytest.raises(RuntimeError, match="scope_erased"):
        admit(agent, target, "episodic", "after-erase", pinned=True)
    with pytest.raises(RuntimeError, match="scope_erased"):
        agent.recall(MemoryQuery("w", "s", "Paris"))
