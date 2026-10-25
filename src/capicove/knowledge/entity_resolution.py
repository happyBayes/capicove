"""Entity resolution operations for temporal memory."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from capicove.contracts.knowledge import (
    Authority,
    Entity,
    EntityRef,
    FactProposal,
    digest,
    micros,
    text,
)
from capicove.knowledge.codec import (
    decode_interval,
    decode_object,
    decode_span,
)
from capicove.knowledge.errors import MemoryError


if TYPE_CHECKING:
    from capicove.knowledge.service import TemporalKnowledgeService


def find_entity(
    self: TemporalKnowledgeService, *, workspace: str, scope: str, name: str
) -> Entity | None:
    self._scope(workspace, scope)
    rows = self.repository.records(workspace, scope, "alias", normalized_alias(name))
    if not rows:
        return None
    return self._entity(workspace, scope, rows[0]["entity_id"])


def _entity(self: TemporalKnowledgeService, workspace: str, scope: str, identity: str) -> Entity:
    visited = set()
    while True:
        if identity in visited:
            raise MemoryError("invalid_entity_redirect")
        visited.add(identity)
        redirects = self.repository.records(workspace, scope, "redirect", identity)
        if not redirects:
            break
        identity = redirects[0]["target"]
    rows = self.repository.records(workspace, scope, "entity", identity)
    if not rows:
        raise MemoryError("entity_not_found")
    b = rows[0]
    return Entity(b["id"], workspace, scope, b["entity_type"], b["name"], b["ontology_version"])


def resolve_entity(
    self: TemporalKnowledgeService,
    *,
    workspace: str,
    scope: str,
    name: str,
    entity_type: str,
    idempotency_key: str,
) -> Entity:
    self._scope(workspace, scope)
    allowed = {kind for p in self.predicates.values() for kind in p.subject_types} | {
        kind
        for p in self.predicates.values()
        if p.object_kind == "entity"
        for kind in p.object_types
    }
    if entity_type not in allowed:
        raise ValueError("unknown entity type")
    alias = normalized_alias(name)
    request = ["entity", scope, alias, entity_type, self.ontology_version]
    for _ in range(8):
        prior = self.repository.replay_receipt(workspace, idempotency_key, request)
        if prior:
            return self._entity(workspace, scope, prior["entity_id"])
        revision, recorded = self.repository.head(workspace)
        existing = self.find_entity(workspace=workspace, scope=scope, name=name)
        if existing:
            if existing.entity_type != entity_type:
                raise ValueError("alias has incompatible type")
        identity = existing.entity_id if existing else uuid4().hex
        body = {
            "id": identity,
            "name": existing.display_name if existing else name,
            "entity_type": entity_type,
            "ontology_version": self.ontology_version,
        }
        try:
            self._commit(
                workspace=workspace,
                scope=scope,
                key=idempotency_key,
                request=request,
                expected_revision=revision,
                recorded=recorded,
                events=[
                    ("EntityResolved", body),
                    (
                        "EntityAliasAdded",
                        {
                            "id": alias,
                            "entity_id": identity,
                            "normalization_version": "nfkc-casefold-v1",
                        },
                    ),
                ],
                receipt={"entity_id": identity},
            )
            return self._entity(workspace, scope, identity)
        except MemoryError as exc:
            if str(exc) != "concurrent_write":
                raise
    raise MemoryError("write_conflict")


def add_entity_alias(
    self: TemporalKnowledgeService,
    *,
    workspace: str,
    scope: str,
    entity_id: str,
    alias: str,
    idempotency_key: str,
) -> dict[str, Any]:
    self._scope(workspace, scope)
    alias = normalized_alias(alias)
    request = ["alias", scope, entity_id, alias]
    for _ in range(8):
        prior = self.repository.replay_receipt(workspace, idempotency_key, request)
        if prior:
            return prior
        revision, recorded = self.repository.head(workspace)
        entity = self._entity(workspace, scope, entity_id)
        existing = self.find_entity(workspace=workspace, scope=scope, name=alias)
        if existing and existing.entity_id != entity.entity_id:
            raise MemoryError("ambiguous_entity_alias")
        try:
            return self._commit(
                workspace=workspace,
                scope=scope,
                key=idempotency_key,
                request=request,
                expected_revision=revision,
                recorded=recorded,
                events=[
                    (
                        "EntityAliasAdded",
                        {
                            "id": alias,
                            "entity_id": entity.entity_id,
                            "normalization_version": "nfkc-casefold-v1",
                        },
                    )
                ],
                receipt={"entity_id": entity.entity_id},
            )
        except MemoryError as exc:
            if str(exc) != "concurrent_write":
                raise
    raise MemoryError("write_conflict")


def merge_entities(
    self: TemporalKnowledgeService,
    *,
    workspace: str,
    scope: str,
    source_id: str,
    target_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    """Append a redirect and re-resolve affected typed facts in the same transaction."""
    self._scope(workspace, scope)
    request = ["merge", scope, source_id, target_id]
    for _ in range(8):
        prior = self.repository.replay_receipt(workspace, idempotency_key, request)
        if prior:
            return prior
        revision, recorded = self.repository.head(workspace)
        source, target = (
            self._entity(workspace, scope, source_id),
            self._entity(workspace, scope, target_id),
        )
        if source.entity_id == target.entity_id or source.entity_type != target.entity_type:
            raise ValueError("invalid entity merge")
        affected = self.repository.entity_assertions_for_merge(workspace, scope, source.entity_id)
        retired = {body["id"] for body in affected}
        proposals = []
        events: list[tuple[str, dict[str, Any]]] = []
        for body in affected:
            obj = decode_object(body["object"])
            if isinstance(obj, EntityRef) and obj.entity_id == source.entity_id:
                obj = EntityRef(target.entity_id)
            subject = target.entity_id if body["subject"] == source.entity_id else body["subject"]
            proposals.append(
                FactProposal(
                    EntityRef(subject),
                    body["predicate"],
                    obj,
                    decode_interval(body["asserted_validity"]),
                    tuple(decode_span(span) for span in body["evidence"]),
                    recorded,
                    Authority(body["authority"]),
                    body["extraction_confidence"],
                )
            )
            old = dict(
                body,
                version=uuid4().hex,
                parent_version_id=body["version"],
                state="rejected",
                recorded=micros(recorded),
                recorded_to=None,
                retracted=True,
            )
            events += [
                (
                    "FactRetracted",
                    {"id": body["id"], "version": body["version"], "reason": "entity_redirect"},
                ),
                ("FactRejected", old),
            ]
        planned, indexes, receipts = self._plan_facts(
            workspace, scope, tuple(proposals), recorded, excluded=retired
        )
        events += [
            ("EntityRedirected", {"id": source.entity_id, "target": target.entity_id}),
            *planned,
        ]
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
                receipt={"entity_id": target.entity_id, "facts": receipts},
            )
        except MemoryError as exc:
            if str(exc) != "concurrent_write":
                raise
    raise MemoryError("write_conflict")


def normalized_alias(name: str) -> str:
    return unicodedata.normalize("NFKC", text(name)).casefold().strip()


def check_resolution(
    service: TemporalKnowledgeService, *, workspace: str, scope: str, name: str, entity_type: str
) -> None:
    """Persist ambiguous near names; require an explicit alias or distinct identity decision."""
    alias = normalized_alias(name)
    candidates = service.repository.similar_entities(workspace, scope, alias, entity_type)
    ids = sorted(
        row["id"]
        for row in candidates
        if SequenceMatcher(None, alias, normalized_alias(row["name"])).ratio() >= 0.8
    )
    if not ids:
        return
    request = ["resolution_pending", scope, alias, entity_type, ids]
    key = "resolution:" + digest(request)
    if service.repository.replay_receipt(workspace, key, request) is None:
        revision, recorded = service.repository.head(workspace)
        service._commit(
            workspace=workspace,
            scope=scope,
            key=key,
            request=request,
            expected_revision=revision,
            recorded=recorded,
            events=[
                (
                    "EntityResolutionPending",
                    {
                        "id": digest([alias, entity_type]),
                        "name": name,
                        "entity_type": entity_type,
                        "candidates": ids,
                    },
                )
            ],
            receipt={"state": "resolution_pending"},
        )
    raise MemoryError("entity_resolution_required")
