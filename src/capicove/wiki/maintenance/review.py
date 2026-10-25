"""Host-bound semantic review; untrusted provenance never grants authority."""

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import MaintenanceRecord, ReviewDecision, ReviewRecord
from capicove.wiki.maintenance.codec import encode_record, record_digest
from capicove.wiki.maintenance.store import record_path, write_object
from capicove.wiki.maintenance.validation import preview


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


@dataclass(frozen=True, slots=True)
class ReviewAuthority:
    """Trusted invocation identity supplied by the host, never deserialized input."""

    reviewer_id: str
    permissions: frozenset[str]


def review(
    service: "MaintenanceService",
    proposal_id: str,
    decisions: tuple[ReviewDecision, ...],
    authority: ReviewAuthority,
    accept: bool,
) -> ReviewRecord:
    """Bind explicit judgments to the proposal, rendered output, base and policy."""
    if "wiki:review" not in authority.permissions or not authority.reviewer_id.strip():
        raise PermissionError("wiki:review permission required")
    value = preview(service, proposal_id)
    if len({d.subject_id for d in decisions}) != len(decisions) or {
        d.subject_id for d in decisions
    } != set(value.review_subjects):
        raise ValueError(
            "review incomplete: every claim, disposition and impact requires a decision"
        )
    if accept and any(d.verdict != "supported" for d in decisions):
        raise ValueError("review contains unsupported or uncertain decisions")
    proposal = value.proposal
    proposal_digest = record_digest(
        MaintenanceRecord(schema_version=1, record_type="proposal", payload=proposal)
    )
    result = ReviewRecord(
        review_id="review:" + "0" * 64,
        proposal_id=proposal_id,
        proposal_sha256=proposal_digest,
        rendered_sha256=proposal.rendered_sha256,
        wiki_revision=proposal.wiki_revision,
        policy_revision=proposal.policy_revision,
        reviewer_id=authority.reviewer_id,
        decisions=decisions,
        accepted=accept,
    )
    encoded = encode_record(
        MaintenanceRecord(schema_version=1, record_type="review", payload=result)
    )
    result = replace(result, review_id="review:" + hashlib.sha256(encoded).hexdigest())
    encoded = encode_record(
        MaintenanceRecord(schema_version=1, record_type="review", payload=result)
    )
    write_object(
        record_path(service.wiki, "reviews", result.review_id, "review"),
        json.loads(encoded),
        service.wiki,
    )
    return result
