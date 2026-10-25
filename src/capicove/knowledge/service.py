"""Evidence-backed temporal writes and reads over public repository/source ports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import (
    Authority,
    Entity,
    EntityRef,
    FactProposal,
    MemoryEpisode,
    Predicate,
    TemporalFact,
    TimeInterval,
    TypedLiteral,
    digest,
    micros,
    text,
    utc,
)
from capicove.contracts.memory import MemoryQuery, MemoryRecall, MemoryRecord
from capicove.knowledge.codec import (
    decode_episode,
    decode_fact,
)
from capicove.knowledge.entity_resolution import _entity as entity_resolution__entity
from capicove.knowledge.entity_resolution import (
    add_entity_alias as entity_resolution_add_entity_alias,
)
from capicove.knowledge.entity_resolution import find_entity as entity_resolution_find_entity
from capicove.knowledge.entity_resolution import merge_entities as entity_resolution_merge_entities
from capicove.knowledge.entity_resolution import resolve_entity as entity_resolution_resolve_entity
from capicove.knowledge.errors import MemoryError
from capicove.knowledge.extraction import retry_extraction as extraction_retry_extraction
from capicove.knowledge.facts import _plan_facts as facts__plan_facts
from capicove.knowledge.facts import assert_fact as facts_assert_fact
from capicove.knowledge.memory_search import MemorySearchPlanner
from capicove.knowledge.ports import EntityFactExtractor, TemporalMemoryRepository
from capicove.source import TextEvidenceStore


class TemporalKnowledgeService:
    def __init__(
        self,
        repository: TemporalMemoryRepository,
        evidence: TextEvidenceStore,
        *,
        ontology_version: str,
        predicates: tuple[Predicate, ...],
        extractor: EntityFactExtractor | None = None,
        dense: Callable[[MemoryQuery], tuple[str, ...]] | None = None,
    ) -> None:
        self.repository, self.evidence = repository, evidence
        self.extractor = extractor
        self.dense = dense
        self.ontology_version = text(ontology_version)
        self.predicates = {p.name: p for p in predicates}
        if len(self.predicates) != len(predicates):
            raise ValueError("duplicate predicate")

    def _scope(self, workspace: str, scope: str, min_revision: int | None = None) -> None:
        text(workspace)
        text(scope)
        if self.evidence.is_erased(workspace=workspace, scope=scope):
            raise MemoryError("scope_erased")
        revision, _ = self.repository.head(workspace)
        if min_revision is not None:
            if type(min_revision) is not int or min_revision < 0:
                raise ValueError("invalid min_revision")
            if revision < min_revision:
                raise MemoryError("consistency_timeout")

    def find_entity(self, *, workspace: str, scope: str, name: str) -> Entity | None:
        return entity_resolution_find_entity(self, workspace=workspace, scope=scope, name=name)

    def _entity(self, workspace: str, scope: str, identity: str) -> Entity:
        return entity_resolution__entity(self, workspace, scope, identity)

    def _episode(
        self,
        workspace: str,
        scope: str,
        content: str,
        episode_type: str,
        event_time: datetime,
        recorded: datetime,
        source_ref: str,
    ) -> tuple[dict[str, Any], EvidenceSpan]:
        ref = self.evidence.accept_text(workspace=workspace, scope=scope, text=content)
        body = {
            "id": uuid4().hex,
            "workspace": workspace,
            "scope": scope,
            "episode_type": episode_type,
            "evidence_refs": [ref.content_sha256],
            "content_sha256": ref.content_sha256,
            "event_time": micros(event_time),
            "recorded": micros(recorded),
            "source_ref": source_ref,
            "schema_version": 1,
            "content_length": len(content),
        }
        decode_episode(body)
        return body, EvidenceSpan(ref, 0, len(content))

    def remember_episode(
        self,
        *,
        workspace: str,
        scope: str,
        content: str,
        event_time: datetime,
        idempotency_key: str,
        episode_type: str = "message",
        source_ref: str = "user",
        extraction: bool = False,
    ) -> dict[str, Any]:
        self._scope(workspace, scope)
        text(content, 1024 * 1024)
        text(idempotency_key)
        utc(event_time)
        if episode_type not in {"message", "document", "json", "fact_assertion"}:
            raise ValueError("invalid episode type")
        text(source_ref)
        if type(extraction) is not bool:
            raise ValueError("invalid extraction flag")
        if episode_type == "json":
            parsed = json.loads(content)
            json.dumps(parsed, allow_nan=False)
        request = [
            "remember",
            scope,
            hashlib.sha256(content.encode()).hexdigest(),
            micros(event_time),
            episode_type,
            source_ref,
            extraction,
        ]
        for _ in range(8):
            previous = self.repository.replay_receipt(workspace, idempotency_key, request)
            if previous is not None:
                return previous
            revision, recorded = self.repository.head(workspace)
            body, _ = self._episode(
                workspace, scope, content, episode_type, event_time, recorded, source_ref
            )
            events = [("EpisodeAccepted", body), ("EpisodeIndexed", {"id": body["id"]})]
            if extraction:
                events.append(("ExtractionQueued", {"id": body["id"], "episode_id": body["id"]}))
            try:
                return self._commit(
                    workspace=workspace,
                    scope=scope,
                    key=idempotency_key,
                    request=request,
                    expected_revision=revision,
                    recorded=recorded,
                    events=events,
                    indexes=[("episode", body["id"], content)],
                    receipt={
                        "episode_id": body["id"],
                        "indexing_state": "indexed",
                        "extraction_state": "pending"
                        if extraction and self.extractor is not None
                        else "deferred",
                        "warnings": []
                        if extraction and self.extractor is not None
                        else ["extraction_deferred"],
                    },
                )
            except MemoryError as exc:
                if str(exc) != "concurrent_write":
                    raise
        raise MemoryError("write_conflict")

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
        authority: Authority = Authority.USER,
        supersedes: str | None = None,
    ) -> dict[str, Any]:
        return facts_assert_fact(
            self,
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
            authority=authority,
            supersedes=supersedes,
        )

    def get_episode(
        self, *, workspace: str, scope: str, episode_id: str, min_revision: int | None = None
    ) -> MemoryEpisode:
        self._scope(workspace, scope, min_revision)
        rows = self.repository.records(workspace, scope, "episode", episode_id)
        if not rows:
            raise MemoryError("episode_not_found")
        episode = decode_episode(rows[0])
        self.evidence.read_text(
            workspace=workspace,
            scope=scope,
            span=EvidenceSpan(episode.evidence_refs[0], 0, rows[0]["content_length"]),
        )
        return episode

    def get_fact_history(
        self, *, workspace: str, scope: str, fact_id: str, min_revision: int | None = None
    ) -> tuple[TemporalFact, ...]:
        self._scope(workspace, scope, min_revision)
        result = tuple(
            decode_fact(b) for b in self.repository.facts(workspace, scope, fact=fact_id)
        )
        self.evidence.read_spans(
            workspace=workspace,
            scope=scope,
            spans=tuple(dict.fromkeys(span for fact in result for span in fact.evidence)),
        )
        return tuple(sorted(result, key=lambda f: (micros(f.knowledge.start), f.fact_version_id)))  # type: ignore[arg-type]

    def current_facts(
        self,
        *,
        workspace: str,
        scope: str,
        subject: str,
        predicate: str,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        min_revision: int | None = None,
    ) -> tuple[TemporalFact, ...]:
        self._scope(workspace, scope, min_revision)
        entity = self._entity(workspace, scope, subject)
        slot = digest([entity.entity_id, predicate, self.ontology_version])
        current_only = known_at is None
        valid_at, known_at = valid_at or datetime.now(UTC), known_at or datetime.now(UTC)
        utc(valid_at)
        utc(known_at)
        facts = tuple(
            decode_fact(b)
            for b in self.repository.facts(workspace, scope, slot=slot, current=current_only)
        )
        result = tuple(
            f
            for f in facts
            if f.state == "accepted"
            and f.validity.contains(valid_at)
            and f.knowledge.contains(known_at)
        )
        self.evidence.read_spans(
            workspace=workspace,
            scope=scope,
            spans=tuple(dict.fromkeys(span for fact in result for span in fact.evidence)),
        )
        return tuple(sorted(result, key=lambda f: f.fact_id))

    def doctor(self) -> dict[str, Any]:
        from capicove.knowledge.diagnostics import diagnose

        return diagnose(
            self.repository,
            extractor_configured=self.extractor is not None,
            dense_configured=self.dense is not None,
        )

    def rebuild_projections(self) -> None:
        self.repository.rebuild(
            lambda workspace, scope, span: self.evidence.read_text(
                workspace=workspace, scope=scope, span=span
            )
        )

    def recall(self, query: MemoryQuery) -> MemoryRecall:
        self._scope(query.workspace_ref, query.scope_ref, query.min_revision)
        return MemorySearchPlanner(self.repository, self.evidence, self.dense).recall(query)

    def record_memory(self, record: MemoryRecord, *, idempotency_key: str) -> str:
        """Persist an Agent policy decision only after rechecking accessible accepted Evidence."""
        workspace, scope = record.workspace_ref, record.scope_ref
        self._scope(workspace, scope)
        request = [
            "admit",
            scope,
            record.target_ref,
            record.kind,
            record.sensitivity,
            record.retention,
        ]
        for _ in range(8):
            prior = self.repository.replay_receipt(workspace, idempotency_key, request)
            if prior:
                return str(prior["memory_id"])
            revision, recorded = self.repository.head(workspace)
            if record.kind == "episodic":
                self.get_episode(workspace=workspace, scope=scope, episode_id=record.target_ref)
            else:
                history = self.get_fact_history(
                    workspace=workspace, scope=scope, fact_id=record.target_ref
                )
                if (
                    not history
                    or history[-1].state != "accepted"
                    or history[-1].knowledge.end is not None
                ):
                    raise MemoryError("memory_admission_rejected")
            body = {
                "id": record.memory_id,
                "target_ref": record.target_ref,
                "kind": record.kind,
                "sensitivity": record.sensitivity,
                "retention": record.retention,
                "retention_deadline": None
                if record.retention_deadline is None
                else micros(record.retention_deadline),
                "importance": record.importance,
                "admission_reason": record.admission_reason,
                "recorded": micros(recorded),
            }
            try:
                result = self._commit(
                    workspace=workspace,
                    scope=scope,
                    key=idempotency_key,
                    request=request,
                    expected_revision=revision,
                    recorded=recorded,
                    events=[("MemoryProposed", {"id": record.memory_id}), ("MemoryAdmitted", body)],
                    receipt={"memory_id": record.memory_id},
                )
                return str(result["memory_id"])
            except MemoryError as exc:
                if str(exc) != "concurrent_write":
                    raise
        raise MemoryError("write_conflict")

    def archive_memory(
        self, *, workspace: str, scope: str, memory_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        self._scope(workspace, scope)
        request = ["archive", scope, memory_id]
        for _ in range(8):
            prior = self.repository.replay_receipt(workspace, idempotency_key, request)
            if prior:
                return prior
            revision, recorded = self.repository.head(workspace)
            if not self.repository.records(workspace, scope, "memory", memory_id):
                raise MemoryError("memory_not_found")
            try:
                return self._commit(
                    workspace=workspace,
                    scope=scope,
                    key=idempotency_key,
                    request=request,
                    expected_revision=revision,
                    recorded=recorded,
                    events=[("MemoryArchived", {"id": memory_id, "recorded": micros(recorded)})],
                    receipt={"memory_id": memory_id, "state": "archived"},
                )
            except MemoryError as exc:
                if str(exc) != "concurrent_write":
                    raise
        raise MemoryError("write_conflict")

    def erase_scope(self, *, workspace: str, scope: str, idempotency_key: str) -> dict[str, Any]:
        """Deny source access durably before scrubbing every derived scope payload."""
        text(workspace)
        text(scope)
        text(idempotency_key)
        self.repository.head(workspace)
        self.evidence.erase_scope(workspace=workspace, scope=scope)
        request = ["erase", scope]
        for _ in range(8):
            prior = self.repository.replay_receipt(workspace, idempotency_key, request)
            if prior:
                self.repository.checkpoint()
                return prior
            revision, recorded = self.repository.head(workspace)
            try:
                result = self.repository.commit(
                    workspace=workspace,
                    scope=scope,
                    key=idempotency_key,
                    request=request,
                    expected_revision=revision,
                    recorded=recorded,
                    events=[("MemoryErased", {"scope": scope})],
                    receipt={"state": "erased"},
                )
                self.repository.checkpoint()
                return result
            except MemoryError as exc:
                if str(exc) != "concurrent_write":
                    raise
        raise MemoryError("write_conflict")

    def _plan_facts(
        self,
        workspace: str,
        scope: str,
        proposals: tuple[FactProposal, ...],
        recorded: datetime,
        *,
        new_entities: dict[str, Entity] | None = None,
        excluded: set[str] | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, str, str]], list[dict[str, Any]]]:
        return facts__plan_facts(
            self,
            workspace,
            scope,
            proposals,
            recorded,
            new_entities=new_entities,
            excluded=excluded,
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
        return extraction_retry_extraction(
            self,
            workspace=workspace,
            scope=scope,
            episode_id=episode_id,
            idempotency_key=idempotency_key,
            force=force,
        )

    def resolve_entity(
        self, *, workspace: str, scope: str, name: str, entity_type: str, idempotency_key: str
    ) -> Entity:
        return entity_resolution_resolve_entity(
            self,
            workspace=workspace,
            scope=scope,
            name=name,
            entity_type=entity_type,
            idempotency_key=idempotency_key,
        )

    def add_entity_alias(
        self, *, workspace: str, scope: str, entity_id: str, alias: str, idempotency_key: str
    ) -> dict[str, Any]:
        return entity_resolution_add_entity_alias(
            self,
            workspace=workspace,
            scope=scope,
            entity_id=entity_id,
            alias=alias,
            idempotency_key=idempotency_key,
        )

    def merge_entities(
        self, *, workspace: str, scope: str, source_id: str, target_id: str, idempotency_key: str
    ) -> dict[str, Any]:
        return entity_resolution_merge_entities(
            self,
            workspace=workspace,
            scope=scope,
            source_id=source_id,
            target_id=target_id,
            idempotency_key=idempotency_key,
        )

    def _commit(self, **arguments: Any) -> dict[str, Any]:
        with self.evidence.authorize_scope(
            workspace=arguments["workspace"], scope=arguments["scope"]
        ):
            return self.repository.commit(**arguments)

    def memory_policy_for(
        self, *, workspace: str, scope: str, target: str
    ) -> dict[str, Any] | None:
        self._scope(workspace, scope)
        return self.repository.memory_policy_for(workspace, scope, target)

    def due_extractions(self, *, workspace: str, scope: str, limit: int = 10) -> tuple[str, ...]:
        self._scope(workspace, scope)
        if self.extractor is None:
            raise MemoryError("extraction_unavailable")
        return self.repository.due_jobs(workspace, scope, now=datetime.now(UTC), limit=limit)

    def check_entity_resolution(
        self, *, workspace: str, scope: str, name: str, entity_type: str
    ) -> None:
        from capicove.knowledge.entity_resolution import check_resolution

        check_resolution(self, workspace=workspace, scope=scope, name=name, entity_type=entity_type)
