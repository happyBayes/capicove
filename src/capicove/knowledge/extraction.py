"""Extraction operations for temporal memory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from capicove.contracts.evidence import EvidenceSpan
from capicove.contracts.knowledge import (
    Authority,
    Entity,
    EntityRef,
    FactProposal,
    MemoryEpisode,
    digest,
    text,
)
from capicove.knowledge.entity_resolution import normalized_alias
from capicove.knowledge.errors import MemoryError
from capicove.knowledge.models import EntityMention, ExtractedFact
from capicove.source import EvidenceUnavailable


if TYPE_CHECKING:
    from capicove.knowledge.service import TemporalKnowledgeService


def retry_extraction(
    self: TemporalKnowledgeService,
    *,
    workspace: str,
    scope: str,
    episode_id: str,
    idempotency_key: str,
    force: bool = True,
) -> dict[str, Any]:
    self._scope(workspace, scope)
    text(idempotency_key)
    if self.extractor is None:
        return {
            "episode_id": episode_id,
            "state": "deferred",
            "warnings": ["extraction_deferred"],
        }
    request = [
        "extract",
        scope,
        episode_id,
        self.extractor.extractor_id,
        self.extractor.revision,
        self.ontology_version,
        "nfkc-casefold-v1",
    ]
    prior = self.repository.replay_receipt(workspace, idempotency_key, request)
    if prior:
        return prior
    episode = self.get_episode(workspace=workspace, scope=scope, episode_id=episode_id)
    fingerprint = digest(request)
    completed = self.repository.records(workspace, scope, "extraction", episode_id)
    if (
        completed
        and completed[0].get("fingerprint") == fingerprint
        and completed[0].get("state") in {"completed", "partially_completed"}
    ):
        return {
            "episode_id": episode_id,
            "state": completed[0]["state"],
            "facts": completed[0]["facts"],
            "committed_revision": self.repository.head(workspace)[0],
        }
    configuration_key = "extraction-configuration:" + fingerprint
    if self.repository.replay_receipt(workspace, configuration_key, request) is None:
        for _ in range(8):
            revision, recorded = self.repository.head(workspace)
            try:
                self._commit(
                    workspace=workspace,
                    scope=scope,
                    key=configuration_key,
                    request=request,
                    expected_revision=revision,
                    recorded=recorded,
                    events=[("ExtractionQueued", {"id": episode_id, "fingerprint": fingerprint})],
                    receipt={"episode_id": episode_id},
                )
                break
            except MemoryError as exc:
                if str(exc) != "concurrent_write":
                    raise
        else:
            raise MemoryError("write_conflict")
    job = self.repository.claim_job(
        workspace, scope, episode_id, now=datetime.now(UTC), lease_seconds=60, force=force
    )
    if job is None:
        raise MemoryError("extraction_unavailable")
    failure = False
    rejected_count = 0
    try:
        proposals = self.extractor.extract(
            episode,
            lambda start, end: self.evidence.read_text(
                workspace=workspace,
                scope=scope,
                span=EvidenceSpan(episode.evidence_refs[0], start, end),
            ),
        )
        if not isinstance(proposals, tuple) or len(proposals) > 100:
            raise ValueError("invalid extraction batch")
    except EvidenceUnavailable:
        raise
    except (ValueError, KeyError, RuntimeError, TimeoutError, ConnectionError):
        failure, proposals = True, ()
    for _ in range(8):
        self._scope(workspace, scope)
        revision, recorded = self.repository.head(workspace)
        events: list[tuple[str, dict[str, Any]]] = []
        indexes: list[tuple[str, str, str]] = []
        facts: list[dict[str, Any]] = []
        if not failure:
            try:
                normalized, entities, entity_events, rejected_count = prepare_candidates(
                    self, workspace, scope, episode, proposals
                )
                planned, indexes, facts = self._plan_facts(
                    workspace, scope, normalized, recorded, new_entities=entities
                )
                events = entity_events + planned
                failure = bool(rejected_count and not normalized)
            except (ValueError, MemoryError):
                failure = True
        state = (
            ("dead_letter" if job["attempts"] >= 3 else "retryable")
            if failure
            else ("partially_completed" if rejected_count else "completed")
        )
        events.append(
            (
                "ExtractionFailed"
                if failure
                else "ExtractionPartiallyCompleted"
                if rejected_count
                else "ExtractionCompleted",
                {
                    "id": episode_id,
                    "state": state,
                    "extractor_id": self.extractor.extractor_id,
                    "extractor_revision": self.extractor.revision,
                    "ontology_version": self.ontology_version,
                    "normalization_version": "nfkc-casefold-v1",
                    "fingerprint": fingerprint,
                    "rejected_count": rejected_count,
                    "facts": facts,
                    "attempts": job["attempts"],
                },
            )
        )
        try:
            return self._commit(
                workspace=workspace,
                scope=scope,
                key=idempotency_key,
                request=request,
                expected_revision=revision,
                recorded=recorded,
                events=events,
                indexes=indexes,
                receipt={"episode_id": episode_id, "state": state, "facts": facts},
                job=dict(job, state=state),
            )
        except MemoryError as exc:
            if str(exc) != "concurrent_write":
                raise
    raise MemoryError("write_conflict")


def prepare_candidates(
    service: TemporalKnowledgeService,
    workspace: str,
    scope: str,
    episode: MemoryEpisode,
    candidates: tuple[FactProposal | ExtractedFact, ...],
) -> tuple[tuple[FactProposal, ...], dict[str, Entity], list[tuple[str, dict[str, Any]]], int]:
    entities: dict[str, Entity] = {}
    events: list[tuple[str, dict[str, Any]]] = []
    accepted: list[FactProposal] = []
    rejected = 0

    def resolve(mention: EntityMention, pending: dict[str, Entity]) -> EntityRef:
        alias = normalized_alias(mention.name)
        existing = service.find_entity(workspace=workspace, scope=scope, name=mention.name)
        existing = existing or next(
            (
                e
                for e in (*entities.values(), *pending.values())
                if normalized_alias(e.display_name) == alias
            ),
            None,
        )
        if existing is not None:
            if existing.entity_type != mention.entity_type:
                raise ValueError("alias has incompatible type")
            return EntityRef(existing.entity_id)
        service.check_entity_resolution(
            workspace=workspace, scope=scope, name=mention.name, entity_type=mention.entity_type
        )
        entity = Entity(
            uuid4().hex,
            workspace,
            scope,
            mention.entity_type,
            mention.name,
            service.ontology_version,
        )
        pending[entity.entity_id] = entity
        return EntityRef(entity.entity_id)

    for candidate in candidates:
        pending: dict[str, Entity] = {}
        try:
            if isinstance(candidate, ExtractedFact):
                subject = resolve(candidate.subject, pending)
                obj = (
                    resolve(candidate.object, pending)
                    if isinstance(candidate.object, EntityMention)
                    else candidate.object
                )
                proposal = FactProposal(
                    subject,
                    candidate.predicate,
                    obj,
                    candidate.validity,
                    candidate.evidence,
                    episode.event_time,
                    Authority.MODEL,
                    candidate.confidence,
                )
            elif isinstance(candidate, FactProposal):
                proposal = candidate
            else:
                raise ValueError("invalid extraction candidate")
            if proposal.authority > Authority.MODEL or any(
                span.reference not in episode.evidence_refs for span in proposal.evidence
            ):
                raise ValueError("invalid extraction authority or evidence")
            service._plan_facts(
                workspace, scope, (proposal,), datetime.now(UTC), new_entities=entities | pending
            )
        except (ValueError, MemoryError):
            rejected += 1
            continue
        entities.update(pending)
        accepted.append(proposal)
    for entity in entities.values():
        events.extend(
            [
                (
                    "EntityResolved",
                    {
                        "id": entity.entity_id,
                        "name": entity.display_name,
                        "entity_type": entity.entity_type,
                        "ontology_version": entity.ontology_version,
                    },
                ),
                (
                    "EntityAliasAdded",
                    {
                        "id": normalized_alias(entity.display_name),
                        "entity_id": entity.entity_id,
                        "normalization_version": "nfkc-casefold-v1",
                    },
                ),
            ]
        )
    return tuple(accepted), entities, events, rejected
