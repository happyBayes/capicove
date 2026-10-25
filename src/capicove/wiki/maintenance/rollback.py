"""Reviewed compensating proposals from verified retained managed revisions."""

import hashlib
import json
import re
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import (
    ClaimDraft,
    CompilationSubmission,
    ImpactDecision,
    MaintenanceRecord,
    PageDraft,
    PageRevision,
    ProposalRecord,
    PublicationRecord,
    Relation,
    Section,
)
from capicove.wiki.maintenance.codec import decode_record, encode_record
from capicove.wiki.maintenance.impact import affected_pages
from capicove.wiki.maintenance.ownership import managed_revision
from capicove.wiki.maintenance.sources import SourceInterval
from capicove.wiki.maintenance.store import canonical_object, read_authorized, record_path
from capicove.wiki.maintenance.tasks import create_task, state
from capicove.wiki.maintenance.validation import submit


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


def historical_revision(
    service: "MaintenanceService", page_id: str, publication_id: str
) -> PageRevision:
    """Resolve a retained journal image through the active immutable lineage."""
    if re.fullmatch(r"page:[A-Z0-9]{10,64}", page_id) is None:
        raise ValueError("invalid page identity")
    record_path(service.wiki, "publications", publication_id, "publication")
    active = state(service)
    identity = active["publication_id"]
    revision = active["wiki_revision"]
    seen: set[str] = set()
    relative = "maintenance/managed/" + page_id[5:] + ".json"
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
            or receipt.wiki_revision != revision
            or receipt.status == "aborted"
        ):
            raise ValueError("invalid publication lineage")
        if identity == publication_id:
            change = next(
                (change for change in receipt.changes if change.relative_path == relative), None
            )
            if change is None or change.after_sha256 is None:
                raise ValueError("historical managed revision unavailable")
            raw = read_authorized(
                service.wiki
                / "_migration"
                / ("maintenance-" + identity[12:])
                / change.after_sha256,
                (service.wiki,),
                1048576,
            )
            if hashlib.sha256(raw).hexdigest() != change.after_sha256:
                raise ValueError("evidence_digest_mismatch: historical revision")
            value = decode_record(canonical_object(json.loads(raw)["revision"])).payload
            if not isinstance(value, PageRevision) or value.page_id != page_id:
                raise ValueError("invalid historical revision")
            return value
        identity = receipt.parent_id
        revision -= 1
    raise ValueError("historical publication is not committed")


def propose(
    service: "MaintenanceService", page_id: str, publication_id: str, idempotency_key: str
) -> ProposalRecord:
    """Restore earlier claim content only after current evidence and policy checks."""
    service._admit()
    historical = historical_revision(service, page_id, publication_id)
    inventory = service._inventory()
    page = next((page for page in inventory if page.page_id == page_id), None)
    if page is None:
        raise ValueError("current managed page unavailable")
    current = managed_revision(service, page)
    spans = tuple(dict.fromkeys(span for claim in historical.claims for span in claim.evidence))
    intervals: list[SourceInterval] = []
    for span in spans:
        if span.kind == "markdown_span":
            intervals.append((span.source_id, span.start_char, span.end_char))
        else:
            if span.page_index is None or span.block_id is None:
                raise ValueError("invalid historical PDF locator")
            intervals.append(
                (span.source_id, span.page_index, span.block_id, span.start_char, span.end_char)
            )
    task = create_task(
        service,
        tuple(sorted({span.source_id for span in spans})),
        historical.title,
        None,
        tuple(intervals),
        (page_id,),
    )
    aliases = {alias.span: alias.alias for alias in task.evidence}
    if not set(spans) <= aliases.keys():
        raise ValueError("evidence_unavailable: historical representation changed")
    drafts = {claim.claim_id: f"claim-{index}" for index, claim in enumerate(historical.claims)}
    claims = tuple(
        ClaimDraft(
            draft_id=drafts[claim.claim_id],
            text=claim.text,
            evidence_aliases=tuple(aliases[span] for span in claim.evidence),
            status=claim.status,
            knowledge=claim.knowledge,
            premise_ids=tuple(drafts[identity] for identity in claim.premise_ids),
            derivation=claim.derivation,
        )
        for claim in historical.claims
    )
    relations = tuple(
        Relation(
            kind=relation.kind,
            target_id=drafts.get(relation.target_id, relation.target_id),
            reason=relation.reason,
        )
        for relation in historical.relations
        if relation.target_id in drafts or relation.target_id.startswith("page:")
    )
    relations += tuple(
        Relation(
            kind="supersedes", target_id=claim.claim_id, reason="Reviewed compensating rollback."
        )
        for claim in current.claims
        if claim.claim_id not in drafts
    )
    impact = affected_pages(
        inventory,
        source_ids=tuple(source.source_id for source in task.sources),
        page_ids=tuple(candidate.page_id for candidate in task.candidates),
    )
    draft = PageDraft(
        draft_id="rollback",
        operation="revise",
        page_id=page_id,
        title=historical.title,
        kind=historical.kind,
        aliases=historical.aliases,
        claims=claims,
        sections=tuple(
            Section(
                heading=section.heading,
                claim_ids=tuple(drafts[identity] for identity in section.claim_ids),
            )
            for section in historical.sections
        ),
        relations=relations,
    )
    value = CompilationSubmission(
        task_id=task.task_id,
        task_version=task.task_version,
        idempotency_key=idempotency_key,
        dispositions=("update", "disputed")
        if any(claim.status == "disputed" for claim in claims)
        else ("update",),
        rationale="Compensating rollback to "
        + publication_id
        + "; current evidence and policy revalidated.",
        pages=(draft,),
        impact=tuple(
            ImpactDecision(
                page_id=identity,
                action="revise" if identity == page_id else "follow_up",
                reason="Compensating revision; related pages require follow-up.",
            )
            for identity in impact.page_ids
        ),
        provenance=None,
    )
    return submit(
        service,
        encode_record(
            MaintenanceRecord(schema_version=1, record_type="compilation_submission", payload=value)
        ),
    )
