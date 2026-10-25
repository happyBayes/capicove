"""Agent-owned admission, retention, and bounded cited context assembly."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

from capicove.agent.context_assembly import assemble_context
from capicove.agent.memory_policy import validate_admission
from capicove.contracts.knowledge import (
    Authority,
    Entity,
    EntityRef,
    MemoryEpisode,
    TemporalFact,
    TimeInterval,
    TypedLiteral,
)
from capicove.contracts.memory import MemoryContext, MemoryQuery, MemoryRecord
from capicove.knowledge import TemporalKnowledgeService


class AgentMemoryService:
    def __init__(self, knowledge: TemporalKnowledgeService) -> None:
        self.knowledge = knowledge

    def get_episode(
        self, *, workspace: str, scope: str, episode_id: str, min_revision: int | None = None
    ) -> MemoryEpisode:
        return self.knowledge.get_episode(
            workspace=workspace, scope=scope, episode_id=episode_id, min_revision=min_revision
        )

    def get_fact_history(
        self, *, workspace: str, scope: str, fact_id: str, min_revision: int | None = None
    ) -> tuple[TemporalFact, ...]:
        return self.knowledge.get_fact_history(
            workspace=workspace, scope=scope, fact_id=fact_id, min_revision=min_revision
        )

    def doctor(self) -> dict[str, Any]:
        return self.knowledge.doctor()

    def _admit(
        self,
        *,
        workspace: str,
        scope: str,
        target: str,
        kind: str,
        sensitivity: str,
        pinned: bool,
        key: str,
        deadline: datetime | None = None,
    ) -> str:
        validate_admission(sensitivity, pinned)
        now = datetime.now(UTC)
        record = MemoryRecord(
            uuid4().hex,
            workspace,
            scope,
            target,
            cast(Any, kind),
            cast(Any, sensitivity),
            "pinned" if pinned else "standard",
            None if pinned else now + timedelta(days=90),
            1.0 if pinned else 0.5,
            "explicit_user" if kind == "episodic" else "accepted_evidence_fact",
            now,
        )
        if deadline is not None:
            record = replace(record, retention_deadline=deadline)
        return self.knowledge.record_memory(record, idempotency_key=key)

    def remember_episode(
        self,
        *,
        workspace: str,
        scope: str,
        content: str,
        event_time: datetime,
        idempotency_key: str,
        sensitivity: str = "internal",
        pinned: bool = False,
        episode_type: str = "message",
        extraction: bool = False,
    ) -> dict[str, Any]:
        validate_admission(sensitivity, pinned)
        result = self.knowledge.remember_episode(
            workspace=workspace,
            scope=scope,
            content=content,
            event_time=event_time,
            idempotency_key=idempotency_key,
            episode_type=episode_type,
            extraction=extraction,
        )
        memory = self._admit(
            workspace=workspace,
            scope=scope,
            target=result["episode_id"],
            kind="episodic",
            sensitivity=sensitivity,
            pinned=pinned,
            key="admission:" + idempotency_key,
        )
        return dict(result, memory_id=memory)

    def assert_fact(
        self,
        *,
        workspace: str,
        scope: str,
        subject_name: str,
        subject_type: str,
        predicate: str,
        object: EntityRef | TypedLiteral,
        validity: TimeInterval,
        event_time: datetime,
        evidence_text: str,
        idempotency_key: str,
        sensitivity: str = "internal",
        pinned: bool = False,
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        validate_admission(sensitivity, pinned)
        result = self.knowledge.assert_fact(
            workspace=workspace,
            scope=scope,
            subject_name=subject_name,
            subject_type=subject_type,
            predicate=predicate,
            object=object,
            validity=validity,
            event_time=event_time,
            evidence_text=evidence_text,
            idempotency_key=idempotency_key,
            authority=Authority.USER,
            supersedes=supersedes,
        )
        if result["state"] == "accepted":
            result = dict(
                result,
                memory_id=self._admit(
                    workspace=workspace,
                    scope=scope,
                    target=result["fact_id"],
                    kind="semantic",
                    sensitivity=sensitivity,
                    pinned=pinned,
                    key="admission:" + idempotency_key,
                ),
            )
        return result

    def recall(
        self, query: MemoryQuery, *, max_characters: int = 12000, max_tokens: int = 4000
    ) -> MemoryContext:
        if (
            type(max_characters) is not int
            or type(max_tokens) is not int
            or not 1 <= max_characters <= 1000000
            or not 1 <= max_tokens <= 1000000
        ):
            raise ValueError("invalid context budgets")
        result = self.knowledge.recall(query)
        return assemble_context(result, max_characters=max_characters, max_tokens=max_tokens)

    def archive_memory(
        self, *, workspace: str, scope: str, memory_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        return self.knowledge.archive_memory(
            workspace=workspace, scope=scope, memory_id=memory_id, idempotency_key=idempotency_key
        )

    def erase_scope(self, *, workspace: str, scope: str, idempotency_key: str) -> dict[str, Any]:
        return self.knowledge.erase_scope(
            workspace=workspace, scope=scope, idempotency_key=idempotency_key
        )

    def retry_extraction(
        self,
        *,
        workspace: str,
        scope: str,
        episode_id: str,
        idempotency_key: str,
        force: bool = True,
    ) -> dict[str, Any]:
        result = self.knowledge.retry_extraction(
            workspace=workspace,
            scope=scope,
            episode_id=episode_id,
            idempotency_key=idempotency_key,
            force=force,
        )
        policy = self.knowledge.memory_policy_for(
            workspace=workspace, scope=scope, target=episode_id
        )
        if policy is None:
            return result
        deadline = (
            None
            if policy["retention_deadline"] is None
            else datetime(1970, 1, 1, tzinfo=UTC)
            + timedelta(microseconds=policy["retention_deadline"])
        )
        if deadline is not None and deadline <= datetime.now(UTC):
            return result
        for fact in result.get("facts", []):
            if fact["state"] == "accepted":
                self._admit(
                    workspace=workspace,
                    scope=scope,
                    target=fact["fact_id"],
                    kind="semantic",
                    sensitivity=policy["sensitivity"],
                    pinned=policy["retention"] == "pinned",
                    deadline=deadline,
                    key="admission:" + idempotency_key + ":" + fact["fact_id"],
                )
        return result

    def work_extractions(self, *, workspace: str, scope: str, limit: int = 10) -> dict[str, Any]:
        outcomes = []
        for episode_id in self.knowledge.due_extractions(
            workspace=workspace, scope=scope, limit=limit
        ):
            try:
                outcomes.append(
                    self.retry_extraction(
                        workspace=workspace,
                        scope=scope,
                        episode_id=episode_id,
                        idempotency_key="worker:" + uuid4().hex,
                        force=False,
                    )
                )
            except RuntimeError as exc:
                if str(exc) not in {"extraction_unavailable", "lease_lost"}:
                    raise
                outcomes.append({"episode_id": episode_id, "state": "claimed_elsewhere"})
        return {"processed": len(outcomes), "outcomes": outcomes}

    def resolve_entity(
        self, *, workspace: str, scope: str, name: str, entity_type: str, idempotency_key: str
    ) -> Entity:
        """Explicit human/host identity decision; automatic extraction never calls this."""
        return self.knowledge.resolve_entity(
            workspace=workspace,
            scope=scope,
            name=name,
            entity_type=entity_type,
            idempotency_key=idempotency_key,
        )
