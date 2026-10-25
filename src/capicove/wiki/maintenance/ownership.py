"""Resolve managed ownership only through committed publication history."""

import hashlib
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import PageRevision, PublicationRecord
from capicove.wiki.maintenance.codec import decode_record
from capicove.wiki.maintenance.discovery import DiscoveryPage
from capicove.wiki.maintenance.store import (
    canonical_object,
    read_authorized,
    read_object,
    record_path,
)
from capicove.wiki.maintenance.tasks import state


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


def managed_revision(service: "MaintenanceService", page: DiscoveryPage) -> PageRevision:
    """Require a matching receipt reachable from the active publication pointer."""
    relative = "maintenance/managed/" + page.page_id[5:] + ".json"
    path = service.wiki / relative
    raw = read_authorized(path, (service.wiki,), 1048576)
    bundle = read_object(
        path, service.wiki, {"schema_version", "publication_id", "revision", "rendered_sha256"}
    )
    record = decode_record(canonical_object(bundle["revision"]))
    if (
        type(bundle["schema_version"]) is not int
        or bundle["schema_version"] != 1
        or not isinstance(record.payload, PageRevision)
        or record.payload.page_id != page.page_id
        or record.payload.page_path != page.path
        or bundle["rendered_sha256"] != page.revision_sha256
    ):
        raise ValueError("manual_edit_conflict: managed ownership mismatch")
    current = state(service)["publication_id"]
    seen: set[str] = set()
    while current is not None:
        if current in seen or len(seen) >= 10000:
            raise ValueError("invalid publication lineage")
        seen.add(current)
        receipt = decode_record(
            read_authorized(
                record_path(service.wiki, "publications", current, "publication"),
                (service.wiki,),
                1048576,
            )
        ).payload
        if (
            not isinstance(receipt, PublicationRecord)
            or receipt.publication_id != current
            or receipt.status == "aborted"
        ):
            raise ValueError("invalid committed publication")
        if current == bundle["publication_id"]:
            digest = hashlib.sha256(raw).hexdigest()
            if not any(
                change.relative_path == relative and change.after_sha256 == digest
                for change in receipt.changes
            ):
                raise ValueError("managed ownership not committed")
            return record.payload
        current = receipt.parent_id
    raise ValueError("managed ownership not committed")


def stale_pages(service: "MaintenanceService") -> frozenset[str]:
    """Resolve committed follow-ups until a later page revision supersedes them."""
    import json

    current = state(service)
    identity = current["publication_id"]
    expected_revision = current["wiki_revision"]
    seen: set[str] = set()
    revised: set[str] = set()
    stale: set[str] = set()
    while identity is not None:
        if identity in seen or len(seen) >= 10000:
            raise ValueError("invalid publication lineage")
        seen.add(identity)
        receipt = decode_record(
            read_authorized(
                record_path(service.wiki, "publications", identity, "publication"),
                (service.wiki,),
                1048576,
            )
        ).payload
        if (
            not isinstance(receipt, PublicationRecord)
            or receipt.publication_id != identity
            or receipt.status == "aborted"
            or receipt.wiki_revision != expected_revision
        ):
            raise ValueError("invalid publication lineage")
        for change in receipt.changes:
            if change.relative_path.startswith("maintenance/managed/"):
                revised.add("page:" + change.relative_path.rsplit("/", 1)[1].removesuffix(".json"))
            if change.relative_path.startswith("maintenance/follow-ups/"):
                raw = read_authorized(service.wiki / change.relative_path, (service.wiki,), 1048576)
                if hashlib.sha256(raw).hexdigest() != change.after_sha256:
                    raise ValueError("evidence_digest_mismatch: follow-up record")
                value = json.loads(raw)
                if (
                    not isinstance(value, dict)
                    or not isinstance(value.get("page_ids"), list)
                    or any(not isinstance(p, str) for p in value["page_ids"])
                    or len(value["page_ids"]) > 100
                ):
                    raise ValueError("invalid follow-up record")
                stale.update(set(value["page_ids"]) - revised)
        identity = receipt.parent_id
        expected_revision -= 1
    if expected_revision != 0:
        raise ValueError("invalid publication lineage")
    return frozenset(stale)
