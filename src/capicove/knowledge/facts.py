"""Facts operations for temporal memory."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from capicove.contracts.knowledge import (
    Authority,
    Entity,
    EntityRef,
    FactProposal,
    TimeInterval,
    TypedLiteral,
    digest,
    micros,
    text,
    utc,
)
from capicove.knowledge.codec import (
    decode_interval,
    decode_object,
    encode_object,
    encode_span,
    interval,
)
from capicove.knowledge.conflicts import Assertion, resolve_slot
from capicove.knowledge.entity_resolution import normalized_alias
from capicove.knowledge.errors import MemoryError


if TYPE_CHECKING:
    from capicove.knowledge.service import TemporalKnowledgeService


def assert_fact(
    self: TemporalKnowledgeService,
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
    self._scope(workspace, scope)
    text(idempotency_key)
    text(subject_name)
    utc(event_time)
    if not isinstance(authority, Authority) or not isinstance(validity, TimeInterval):
        raise ValueError("invalid authority/validity")
    definition = self.predicates.get(predicate)
    if definition is None or subject_type not in definition.subject_types:
        raise ValueError("invalid predicate domain")
    if definition.object_kind == "literal":
        if not isinstance(object, TypedLiteral) or object.kind not in definition.object_types:
            raise ValueError("invalid predicate range")
    elif (
        not isinstance(object, EntityRef)
        or self._entity(workspace, scope, object.entity_id).entity_type
        not in definition.object_types
    ):
        raise ValueError("invalid predicate range")
    if definition.temporal_kind == "static" and validity != TimeInterval():
        raise ValueError("static predicate cannot have validity bounds")
    text(evidence_text, 1024 * 1024)
    request = [
        "assert",
        scope,
        normalized_alias(subject_name),
        subject_type,
        predicate,
        encode_object(object),
        interval(validity),
        micros(event_time),
        hashlib.sha256(evidence_text.encode()).hexdigest(),
        int(authority),
        self.ontology_version,
        supersedes,
    ]
    for _ in range(8):
        prior = self.repository.replay_receipt(workspace, idempotency_key, request)
        if prior is not None:
            return prior
        revision, recorded = self.repository.head(workspace)
        entity = self.find_entity(workspace=workspace, scope=scope, name=subject_name)
        events: list[tuple[str, dict[str, Any]]] = []
        if entity is None:
            self.check_entity_resolution(
                workspace=workspace, scope=scope, name=subject_name, entity_type=subject_type
            )
            entity = Entity(
                uuid4().hex, workspace, scope, subject_type, subject_name, self.ontology_version
            )
            events.extend(
                [
                    (
                        "EntityResolved",
                        {
                            "id": entity.entity_id,
                            "name": subject_name,
                            "entity_type": subject_type,
                            "ontology_version": self.ontology_version,
                        },
                    ),
                    (
                        "EntityAliasAdded",
                        {
                            "id": normalized_alias(subject_name),
                            "entity_id": entity.entity_id,
                            "normalization_version": "nfkc-casefold-v1",
                        },
                    ),
                ]
            )
        elif entity.entity_type != subject_type:
            raise ValueError("alias has incompatible type")
        old = None
        if supersedes is not None:
            previous = self.repository.facts(workspace, scope, fact=supersedes, current=True)
            if (
                not previous
                or previous[0]["subject"] != entity.entity_id
                or previous[0]["predicate"] != predicate
                or previous[0].get("retracted")
            ):
                raise ValueError("invalid correction target")
            old = previous[0]
            if old["object"] == encode_object(object) and old["asserted_validity"] == interval(
                validity
            ):
                raise ValueError("correction must change the assertion")
        episode, span = self._episode(
            workspace, scope, evidence_text, "fact_assertion", event_time, recorded, "user"
        )
        events.extend([("EpisodeAccepted", episode), ("EpisodeIndexed", {"id": episode["id"]})])
        # Explicit assertions cite the complete source span; no substring inference.
        self.evidence.read_text(workspace=workspace, scope=scope, span=span)
        fact_events, fact_indexes, results = self._plan_facts(
            workspace,
            scope,
            (
                FactProposal(
                    EntityRef(entity.entity_id),
                    predicate,
                    object,
                    validity,
                    (span,),
                    event_time,
                    authority,
                ),
            ),
            recorded,
            new_entities={entity.entity_id: entity},
            excluded=set() if old is None else {old["id"]},
        )
        if old is not None:
            replacement = next(
                body["version"]
                for kind, body in fact_events
                if kind in {"FactAccepted", "FactDisputed", "FactRejected"}
                and body["id"] == results[0]["fact_id"]
            )
            retired = dict(
                old,
                version=uuid4().hex,
                parent_version_id=old["version"],
                state="rejected",
                recorded=micros(recorded),
                recorded_to=None,
                retracted=True,
                superseding_version_id=replacement,
            )
            events += [
                (
                    "FactRetracted",
                    {
                        "id": old["id"],
                        "version": old["version"],
                        "reason": "explicit_correction",
                    },
                ),
                ("FactRejected", retired),
            ]
        events.extend(fact_events)
        candidate = results[0]
        try:
            return self._commit(
                workspace=workspace,
                scope=scope,
                key=idempotency_key,
                request=request,
                expected_revision=revision,
                recorded=recorded,
                events=events,
                indexes=[
                    ("episode", episode["id"], evidence_text),
                    *fact_indexes,
                ],
                receipt={
                    "fact_id": candidate["fact_id"],
                    "episode_id": episode["id"],
                    "state": candidate["state"],
                },
            )
        except MemoryError as exc:
            if str(exc) != "concurrent_write":
                raise
    raise MemoryError("write_conflict")


def _plan_facts(
    self: TemporalKnowledgeService,
    workspace: str,
    scope: str,
    proposals: tuple[FactProposal, ...],
    recorded: datetime,
    *,
    new_entities: dict[str, Entity] | None = None,
    excluded: set[str] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, str, str]], list[dict[str, Any]]]:
    slots: dict[str, list[dict[str, Any]]] = {}
    dirty: set[str] = set()
    requested: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []
    indexes: list[tuple[str, str, str]] = []
    for proposal in proposals:
        entity = (new_entities or {}).get(proposal.subject_ref.entity_id) or self._entity(
            workspace, scope, proposal.subject_ref.entity_id
        )
        definition = self.predicates.get(proposal.predicate)
        if definition is None or entity.entity_type not in definition.subject_types:
            raise ValueError("invalid predicate domain")
        obj = proposal.object
        if isinstance(obj, EntityRef):
            obj = EntityRef(
                (
                    (new_entities or {}).get(obj.entity_id)
                    or self._entity(workspace, scope, obj.entity_id)
                ).entity_id
            )
        if definition.object_kind == "literal":
            if not isinstance(obj, TypedLiteral) or obj.kind not in definition.object_types:
                raise ValueError("invalid predicate range")
        elif (
            not isinstance(obj, EntityRef)
            or (
                (new_entities or {}).get(obj.entity_id)
                or self._entity(workspace, scope, obj.entity_id)
            ).entity_type
            not in definition.object_types
        ):
            raise ValueError("invalid predicate range")
        if definition.temporal_kind == "static" and proposal.validity != TimeInterval():
            raise ValueError("static predicate cannot have validity bounds")
        for span in proposal.evidence:
            self.evidence.read_text(workspace=workspace, scope=scope, span=span)
        slot = digest([entity.entity_id, proposal.predicate, self.ontology_version])
        assertion = digest(
            [
                entity.entity_id,
                proposal.predicate,
                encode_object(obj),
                interval(proposal.validity),
                self.ontology_version,
            ]
        )
        if slot not in slots:
            slots[slot] = [
                b
                for b in self.repository.facts(workspace, scope, slot=slot, current=True)
                if not b.get("retracted") and b["id"] not in (excluded or set())
            ]
        existing = next((b for b in slots[slot] if b["assertion_hash"] == assertion), None)
        if existing is None:
            earlier = self.repository.facts(workspace, scope, assertion=assertion, current=True)
            if earlier and earlier[0]["id"] not in (excluded or set()):
                existing = earlier[0]
                existing.pop("retracted", None)
                slots[slot].append(existing)
        if existing is None:
            existing = {
                "id": uuid4().hex,
                "subject": entity.entity_id,
                "predicate": proposal.predicate,
                "object": encode_object(obj),
                "asserted_validity": interval(proposal.validity),
                "assertion_hash": assertion,
                "conflict_key": slot,
                "ontology_version": self.ontology_version,
                "authority": int(proposal.authority),
                "evidence": [],
                "extraction_confidence": proposal.extraction_confidence,
            }
            slots[slot].append(existing)
        existing["authority"] = max(existing["authority"], int(proposal.authority))
        for span in proposal.evidence:
            support = encode_span(span)
            if support not in existing["evidence"]:
                existing["evidence"] = [*existing["evidence"], support]
        requested.append(existing["id"])
        dirty.add(existing["id"])
        events.append(
            (
                "FactProposed",
                {
                    "id": existing["id"],
                    "assertion_hash": assertion,
                    "event_time": micros(proposal.event_time),
                },
            )
        )
        indexes.append(
            (
                "fact",
                existing["id"],
                f"{entity.display_name} {proposal.predicate} {encode_object(obj)['value']}",
            )
        )
    states: dict[str, str] = {}
    for items in slots.values():
        definition = self.predicates[items[0]["predicate"]]
        resolutions = resolve_slot(
            tuple(
                Assertion(
                    b["id"],
                    decode_object(b["object"]),
                    decode_interval(b["asserted_validity"]),
                    Authority(b["authority"]),
                )
                for b in items
            ),
            cardinality=definition.cardinality,
            temporal_kind=definition.temporal_kind,
        )
        for body in items:
            resolution = resolutions[body["id"]]
            states[body["id"]] = resolution.state
            if (
                body["id"] not in dirty
                and body["validity"] == interval(resolution.validity)
                and body["state"] == resolution.state
            ):
                continue
            updated = dict(
                body,
                version=uuid4().hex,
                validity=interval(resolution.validity),
                state=resolution.state,
                recorded=micros(recorded),
                recorded_to=None,
                parent_version_id=body.get("version"),
                superseding_version_id=None,
            )
            if body.get("version"):
                events.append(
                    (
                        "FactSuperseded",
                        {
                            "id": body["id"],
                            "version": body["version"],
                            "superseding_version_id": updated["version"],
                        },
                    )
                )
            events.append(("Fact" + resolution.state.title(), updated))
    return (
        events,
        indexes,
        [{"fact_id": identity, "state": states[identity]} for identity in requested],
    )
