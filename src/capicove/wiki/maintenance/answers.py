"""Ephemeral source-backed answer contexts with full submission revalidation."""

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import (
    AnswerResult,
    AnswerSubmission,
    AnswerTask,
    EvidenceAlias,
    MaintenanceRecord,
    ManagedClaim,
)
from capicove.wiki.maintenance.codec import decode_record, encode_record
from capicove.wiki.maintenance.discovery import discover, inventory_snapshot
from capicove.wiki.maintenance.ownership import managed_revision, stale_pages
from capicove.wiki.maintenance.store import canonical_object
from capicove.wiki.maintenance.tasks import state


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


def context(
    service: "MaintenanceService", question: str, include_history: bool
) -> tuple[AnswerTask, int]:
    """Select bounded eligible claims and expand only their original evidence."""
    policy, withdrawn = service._admit()
    current = state(service)
    pages = service._inventory()
    found = discover(service.wiki / "_derived/maintenance.sqlite", pages, question)
    stale = stale_pages(service)
    selected = {candidate.page_id for candidate in found.candidates}
    claims: list[ManagedClaim] = []
    evidence: list[EvidenceAlias] = []
    omitted = 0
    seen = set()
    text_bytes = 0
    for page in pages:
        if page.page_id not in selected:
            continue
        if page.page_id in stale:
            omitted += 1
            continue
        if not (service.wiki / "maintenance/managed" / (page.page_id[5:] + ".json")).exists():
            continue
        try:
            revision = managed_revision(service, page)
        except ValueError:
            # A dirty page is excluded, never repaired by asking a question.
            omitted += 1
            continue
        for claim in revision.claims:
            if (
                claim.claim_id in seen
                or claim.status == "withdrawn"
                or (claim.status == "superseded" and not include_history)
                or any(span.source_id in withdrawn for span in claim.evidence)
            ):
                omitted += 1
                continue
            additions: list[EvidenceAlias] = []
            try:
                for span in claim.evidence:
                    if any(alias.span == span for alias in (*evidence, *additions)):
                        continue
                    from capicove.wiki.maintenance.sources import load_source, resolve

                    text = resolve(
                        load_source(
                            service,
                            span.source_id,
                            withdrawn,
                            representation_sha256=span.representation_sha256
                            if include_history
                            else None,
                        ),
                        span,
                    )
                    additions.append(
                        EvidenceAlias(
                            alias=f"e{len(evidence) + len(additions) + 1}", span=span, text=text
                        )
                    )
            except ValueError:
                omitted += 1
                continue
            size = sum(len(alias.text.encode()) for alias in additions)
            if (
                len(claims) >= 128
                or len(evidence) + len(additions) > 128
                or text_bytes + size > 262144
            ):
                omitted += 1
                continue
            claims.append(claim)
            seen.add(claim.claim_id)
            evidence.extend(additions)
            text_bytes += size
    task = AnswerTask(
        task_id="task:" + "0" * 64,
        workspace_id=current["workspace_id"],
        wiki_revision=current["wiki_revision"],
        policy_revision=policy,
        question=question,
        evidence=tuple(evidence),
        claims=tuple(claims),
        include_history=include_history,
    )
    encoded = encode_record(
        MaintenanceRecord(schema_version=1, record_type="answer_task", payload=task)
    )
    digest = hashlib.sha256(
        encoded
        + canonical_object(
            {
                "inventory_sha256": hashlib.sha256(inventory_snapshot(pages)).hexdigest(),
                "omitted_claims": omitted,
            }
        )
    ).hexdigest()
    return replace(task, task_id="task:" + digest), omitted


def submit(service: "MaintenanceService", task: AnswerTask, raw: bytes) -> AnswerResult:
    """Recompute the entire host context; a client task never grants evidence."""
    if type(raw) is not bytes or len(raw) > 65536:
        raise ValueError("capacity_exceeded: answer output")
    fresh, omitted = context(service, task.question, task.include_history)
    if fresh != task:
        raise ValueError("answer_task_stale_or_forged")
    payload = decode_record(raw).payload
    if not isinstance(payload, AnswerSubmission) or payload.task_id != task.task_id:
        raise ValueError("invalid answer task")
    aliases = {alias.alias: alias for alias in fresh.evidence}
    for claim in payload.claims:
        spans = []
        for citation in claim.citations:
            alias = aliases.get(citation.alias)
            if alias is None or citation.quote not in alias.text:
                raise ValueError("citation_out_of_scope")
            spans.append(alias.span)
        statuses = {
            original.status
            for original in fresh.claims
            if any(span in original.evidence for span in spans)
        }
        if claim.status not in statuses:
            raise ValueError("citation_status_mismatch")
        if "disputed" in statuses and (
            claim.status != "disputed" or len({span.source_id for span in spans}) < 2
        ):
            raise ValueError("citation_dispute_pair_required")
        if "superseded" in statuses and claim.status == "current":
            raise ValueError("citation_historical_status_required")
    return AnswerResult(
        task_id=task.task_id,
        wiki_revision=task.wiki_revision,
        policy_revision=task.policy_revision,
        status="answered" if payload.claims else "insufficient_evidence",
        claims=payload.claims,
        evidence=tuple(
            alias
            for alias in fresh.evidence
            if any(
                citation.alias == alias.alias
                for claim in payload.claims
                for citation in claim.citations
            )
        ),
        semantic_status="not_verified",
        omitted_claims=omitted,
    )


def evidence_only(
    service: "MaintenanceService", question: str, include_history: bool
) -> tuple[AnswerTask, AnswerResult]:
    """Return original evidence without invoking a model or writing a record."""
    task, omitted = context(service, question, include_history)
    return task, AnswerResult(
        task_id=task.task_id,
        wiki_revision=task.wiki_revision,
        policy_revision=task.policy_revision,
        status="evidence_only" if task.evidence else "no_evidence",
        claims=(),
        evidence=task.evidence,
        semantic_status="not_verified",
        omitted_claims=omitted,
    )
