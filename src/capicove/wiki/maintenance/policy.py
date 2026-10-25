"""Monotonic source withdrawal outside editorial rollback."""

import hashlib
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import MaintenanceRecord, SourcePolicyEvent
from capicove.wiki.maintenance.codec import decode_record, encode_record
from capicove.wiki.maintenance.store import (
    canonical_object,
    create_verified,
    read_authorized,
    record_path,
    write_authorized,
)
from capicove.wiki.maintenance.tasks import state


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


def withdraw(
    service: "MaintenanceService", source_id: str, reason: str, idempotency_key: str
) -> SourcePolicyEvent:
    """Record an immutable event and advance the live deny policy under lock."""
    revision, withdrawn = service._admit(allow_pending=True)
    workspace_id = state(service)["workspace_id"]
    identity = (
        "policy:"
        + hashlib.sha256(
            canonical_object(
                {
                    "workspace_id": workspace_id,
                    "idempotency_key": idempotency_key,
                }
            )
        ).hexdigest()
    )
    path = record_path(service.wiki, "policy-events", identity, "policy")
    event = SourcePolicyEvent(
        event_id=identity,
        workspace_id=workspace_id,
        source_id=source_id,
        policy_revision=revision + 1,
        reason=reason,
        idempotency_key=idempotency_key,
        action="withdraw",
    )
    if path.exists() or path.is_symlink():
        prior = decode_record(read_authorized(path, (service.wiki,), 1048576)).payload
        if (
            not isinstance(prior, SourcePolicyEvent)
            or prior.event_id != identity
            or prior.workspace_id != workspace_id
            or prior.source_id != source_id
            or prior.reason != reason
            or prior.idempotency_key != idempotency_key
        ):
            raise ValueError("idempotency_conflict")
        event = prior
        if source_id in withdrawn and event.policy_revision <= revision:
            return event
        if event.policy_revision != revision + 1:
            raise ValueError("policy_changed: reconcile incomplete policy event")
    digest = source_id.removeprefix("source:sha256:")
    sources = [service.wiki / "raw" / (digest + suffix) for suffix in (".md", ".pdf")]
    sources = [source for source in sources if source.exists() or source.is_symlink()]
    if not sources or any(
        hashlib.sha256(read_authorized(source, (service.wiki,), 256 * 1024 * 1024)).hexdigest()
        != digest
        for source in sources
    ):
        raise ValueError("evidence_unavailable: unknown or corrupt source")
    create_verified(
        path,
        encode_record(
            MaintenanceRecord(schema_version=1, record_type="source_policy_event", payload=event)
        ),
        service.wiki,
    )
    write_authorized(
        service.wiki / "maintenance/source-policy.json",
        canonical_object(
            {
                "schema_version": 1,
                "policy_revision": event.policy_revision,
                "withdrawn": sorted(set(withdrawn) | {source_id}),
            }
        ),
        service.wiki,
    )
    return event
