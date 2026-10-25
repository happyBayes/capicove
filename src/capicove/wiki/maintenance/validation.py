"""Proposal invariants and pure host-addressed concept compilation."""

import difflib
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import (
    ClaimDraft,
    CompilationSubmission,
    CompilationTask,
    EvidenceSpan,
    MaintenanceRecord,
    ManagedClaim,
    PageRevision,
    ProposalRecord,
    Section,
)
from capicove.wiki.maintenance.codec import decode_record, encode_record
from capicove.wiki.maintenance.impact import affected_pages
from capicove.wiki.maintenance.rendering import plain, render_page
from capicove.wiki.maintenance.store import (
    canonical_object,
    read_authorized,
    record_path,
    write_object,
)
from capicove.wiki.maintenance.tasks import load_task


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


def compile_pages(
    service: "MaintenanceService", task: CompilationTask, submission: CompilationSubmission
) -> tuple[PageRevision, ...]:
    """Resolve aliases, identity, impact and inference before accepting records."""
    if submission.task_id != task.task_id or submission.task_version != task.task_version:
        raise ValueError("task_superseded")
    if (
        any(page.operation == "create" for page in submission.pages)
        and "new" not in submission.dispositions
    ):
        raise ValueError("invalid_submission: create requires new disposition")
    if any(page.operation != "create" for page in submission.pages) and not {
        "update",
        "disputed",
    } & set(submission.dispositions):
        raise ValueError("invalid_submission: edit requires update or disputed disposition")
    if (
        any(claim.status == "disputed" for page in submission.pages for claim in page.claims)
        and "disputed" not in submission.dispositions
    ):
        raise ValueError("invalid_submission: disputed claims require disputed disposition")
    inventory = {page.page_id: page for page in service._inventory()}
    impact = affected_pages(
        tuple(inventory.values()),
        source_ids=tuple(s.source_id for s in task.sources),
        page_ids=tuple(c.page_id for c in task.candidates),
    )
    required = set(impact.page_ids)
    decisions = {d.page_id: d for d in submission.impact}
    if len(decisions) != len(submission.impact) or set(decisions) != required:
        raise ValueError("impact_incomplete")
    revised = {p.page_id for p in submission.pages if p.page_id}
    if {d.page_id for d in submission.impact if d.action == "revise"} != revised:
        raise ValueError("impact_incomplete: revise decisions disagree")
    if len({p.draft_id for p in submission.pages}) != len(submission.pages) or len(revised) != sum(
        p.page_id is not None for p in submission.pages
    ):
        raise ValueError("invalid_submission: duplicate page operation")
    # Rationale is mandatory in the closed contract. Its meaning is a V3
    # review decision, not an English keyword or source-hash heuristic.
    evidence = {alias.alias: alias.span for alias in task.evidence}
    pages = []
    for draft in submission.pages:
        plain(draft.title, heading=True)
        for alias in draft.aliases:
            plain(alias, heading=True)
        parent = None
        previous_revision: PageRevision | None = None
        if draft.page_id:
            if draft.page_id not in task.permitted_page_ids or draft.page_id not in inventory:
                raise ValueError("citation_out_of_scope: page")
            existing = inventory[draft.page_id]
            if existing.path.startswith("pages/books/"):
                raise ValueError("invalid_submission: Book ownership")
            # Until v3 publication installs an ownership receipt, only explicit
            # adoption may replace an existing unmanaged page.
            owned = service.wiki / "maintenance" / "managed" / (draft.page_id[5:] + ".json")
            if draft.operation == "revise" and not owned.is_file():
                raise ValueError("invalid_submission: unmanaged page requires adoption")
            if draft.operation == "adopt" and owned.exists():
                raise ValueError("invalid_submission: page already managed")
            if draft.operation == "revise":
                from capicove.wiki.maintenance.ownership import managed_revision

                previous_revision = managed_revision(service, existing)
            parent = existing.revision_sha256
            identity = draft.page_id
        else:
            identity = (
                "page:"
                + uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    task.workspace_id + task.task_id + submission.idempotency_key + draft.draft_id,
                ).hex.upper()
            )
        destination = "pages/concepts/concept-" + identity[5:].lower() + ".md"
        path = service.wiki / destination
        if path.exists() and (
            draft.page_id is None or inventory[draft.page_id].path != destination
        ):
            raise ValueError("invalid_submission: destination collision")
        by_draft = {claim.draft_id: claim for claim in draft.claims}
        resolved = _resolve_claims(draft.claims, evidence)
        claims = tuple(resolved[claim.draft_id] for claim in draft.claims)
        if len({claim.claim_id for claim in claims}) != len(claims):
            raise ValueError("invalid_submission: duplicate assertion")
        included = [identity for section in draft.sections for identity in section.claim_ids]
        if len(included) != len(set(included)) or set(included) != set(by_draft):
            raise ValueError("invalid_submission: sections must cover each claim once")
        sections = tuple(
            Section(
                heading=section.heading,
                claim_ids=tuple(resolved[name].claim_id for name in section.claim_ids),
            )
            for section in draft.sections
        )
        for section in sections:
            plain(section.heading, heading=True)
        relations = tuple(
            replace(relation, target_id=resolved[relation.target_id].claim_id)
            if relation.target_id in resolved
            else relation
            for relation in draft.relations
        )
        for relation in relations:
            plain(relation.reason)
            previous_claims = (
                {c.claim_id for c in previous_revision.claims} if previous_revision else set()
            )
            if (
                relation.target_id not in inventory
                and relation.target_id not in {c.claim_id for c in claims} | previous_claims
            ):
                raise ValueError("invalid_submission: unknown relation")
        if previous_revision:
            removed = {c.claim_id for c in previous_revision.claims} - {c.claim_id for c in claims}
            if not removed <= {r.target_id for r in relations if r.kind == "supersedes"}:
                raise ValueError(
                    "invalid_submission: changed claims require explicit supersedes lineage"
                )
        disputed = [c for c in claims if c.status == "disputed"]
        if disputed and (
            len(disputed) < 2
            or not any(r.kind == "disputed" for r in draft.relations)
            or len({span.source_id for c in disputed for span in c.evidence}) < 2
        ):
            raise ValueError(
                "invalid_submission: dispute requires both source-backed sides and reason"
            )
        if any(c.status == "superseded" for c in claims) and not any(
            r.kind == "supersedes" for r in draft.relations
        ):
            raise ValueError("invalid_submission: superseded claim lacks replacement")
        pages.append(
            PageRevision(
                page_id=identity,
                page_path=destination,
                parent_sha256=parent,
                title=draft.title,
                kind=draft.kind,
                aliases=draft.aliases,
                sections=sections,
                claims=claims,
                relations=relations,
                source_ids=tuple(sorted({s.source_id for c in claims for s in c.evidence})),
                rendering_version=1,
            )
        )
    return tuple(pages)


@dataclass(frozen=True, slots=True)
class DeferredTask:
    """Immutable follow-up intent activated against the post-publication view."""

    task_id: str
    proposal_id: str
    page_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    reason: str
    status: str = "awaiting_publication"


@dataclass(frozen=True, slots=True)
class ProposalPreview:
    """Exact rendered proposal, independent of any publication side effect."""

    proposal: ProposalRecord
    pages: tuple[PageRevision, ...]
    rendered: tuple[str, ...]
    diff: str
    review_subjects: tuple[str, ...]
    stale_page_ids: tuple[str, ...]
    follow_up_tasks: tuple[DeferredTask, ...]


def _proposal_identity(task: CompilationTask, submission: CompilationSubmission) -> str:
    key = {
        "workspace_id": task.workspace_id,
        "task_id": task.task_id,
        "task_version": task.task_version,
        "idempotency_key": submission.idempotency_key,
    }
    return "proposal:" + hashlib.sha256(canonical_object(key)).hexdigest()


def _rendered_digest(pages: tuple[PageRevision, ...], rendered: tuple[str, ...]) -> str:
    return hashlib.sha256(
        canonical_object({"pages": [asdict(page) for page in pages], "rendered": list(rendered)})
    ).hexdigest()


def submit(service: "MaintenanceService", raw: bytes) -> ProposalRecord:
    """Accept one fully validated proposal, or an exact idempotent retry."""
    record = decode_record(raw)
    if not isinstance(record.payload, CompilationSubmission):
        raise ValueError("invalid_submission: expected compilation submission")
    submission = record.payload
    task, bundle = load_task(service, submission.task_id)
    identity = _proposal_identity(task, submission)
    path = record_path(service.wiki, "proposals", identity, "proposal")
    if path.exists() or path.is_symlink():
        prior = decode_record(read_authorized(path, (service.wiki,), 1048576))
        if not isinstance(prior.payload, ProposalRecord) or prior.payload.submission != submission:
            raise ValueError("idempotency_conflict")
        preview(service, identity)
        return prior.payload
    pages = compile_pages(service, task, submission)
    rendered = _render_pages(service, pages, bundle["created_on"])
    proposal = ProposalRecord(
        proposal_id=identity,
        workspace_id=task.workspace_id,
        submission=submission,
        wiki_revision=task.wiki_revision,
        policy_revision=task.policy_revision,
        rendered_sha256=_rendered_digest(pages, rendered),
    )
    encoded = encode_record(
        MaintenanceRecord(schema_version=1, record_type="proposal", payload=proposal)
    )
    write_object(path, json.loads(encoded), service.wiki)
    return proposal


def preview(service: "MaintenanceService", proposal_id: str) -> ProposalPreview:
    """Revalidate and render the same proposal against its exact current base."""
    path = record_path(service.wiki, "proposals", proposal_id, "proposal")
    record = decode_record(read_authorized(path, (service.wiki,), 1048576))
    if not isinstance(record.payload, ProposalRecord):
        raise ValueError("invalid proposal record")
    proposal = record.payload
    task, bundle = load_task(service, proposal.submission.task_id)
    if (
        proposal.proposal_id != proposal_id
        or proposal_id != _proposal_identity(task, proposal.submission)
        or proposal.workspace_id != task.workspace_id
        or proposal.wiki_revision != task.wiki_revision
        or proposal.policy_revision != task.policy_revision
    ):
        raise ValueError("proposal_stale")
    pages = compile_pages(service, task, proposal.submission)
    rendered = _render_pages(service, pages, bundle["created_on"])
    if _rendered_digest(pages, rendered) != proposal.rendered_sha256:
        raise ValueError("review_stale: rendered content changed")
    inventory = {p.page_id: p for p in service._inventory()}
    diffs: list[str] = []
    for page, after in zip(pages, rendered, strict=True):
        before = ""
        old_path = "/dev/null"
        if page.parent_sha256:
            old_path = inventory[page.page_id].path
            before = read_authorized(
                service.wiki / old_path, (service.wiki,), 16 * 1024 * 1024
            ).decode()
        diffs.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=old_path,
                tofile=page.page_path,
            )
        )
    overflow = affected_pages(
        tuple(inventory.values()),
        source_ids=tuple(s.source_id for s in task.sources),
        page_ids=tuple(c.page_id for c in task.candidates),
    ).follow_up_page_ids
    stale = tuple(
        sorted(
            {d.page_id for d in proposal.submission.impact if d.action == "follow_up"}
            | set(overflow)
        )
    )
    overflow_subject = (
        "impact-overflow:"
        + hashlib.sha256(canonical_object({"page_ids": list(overflow)})).hexdigest()
    )

    subjects = tuple(
        dict.fromkeys(
            [c.claim_id for page in pages for c in page.claims]
            + ["disposition:" + value for value in proposal.submission.dispositions]
            + ["impact:" + d.page_id for d in proposal.submission.impact]
            + ["stale:" + identity for identity in stale if identity not in overflow]
            + ([overflow_subject] if overflow else [])
        )
    )
    follow_ups = tuple(
        DeferredTask(
            task_id="task:"
            + hashlib.sha256(
                canonical_object(
                    {
                        "proposal_id": proposal_id,
                        "page_id": decision.page_id,
                        "reason": decision.reason,
                    }
                )
            ).hexdigest(),
            proposal_id=proposal_id,
            page_ids=(decision.page_id,),
            source_ids=tuple(source.source_id for source in task.sources),
            reason=decision.reason,
        )
        for decision in proposal.submission.impact
        if decision.action == "follow_up"
    )
    for offset in range(0, len(overflow), 100):
        batch = overflow[offset : offset + 100]
        follow_ups += (
            DeferredTask(
                task_id="task:"
                + hashlib.sha256(
                    canonical_object({"proposal_id": proposal_id, "overflow": list(batch)})
                ).hexdigest(),
                proposal_id=proposal_id,
                page_ids=batch,
                source_ids=tuple(source.source_id for source in task.sources),
                reason="Mechanical impact exceeds the 100-page review window.",
            ),
        )
    return ProposalPreview(proposal, pages, rendered, "".join(diffs), subjects, stale, follow_ups)


def _resolve_claims(
    drafts: tuple[ClaimDraft, ...], evidence: dict[str, EvidenceSpan]
) -> dict[str, ManagedClaim]:
    """Resolve a local acyclic premise graph before assigning durable identities."""
    by_draft = {claim.draft_id: claim for claim in drafts}
    resolved: dict[str, ManagedClaim] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> ManagedClaim:
        if name in visiting:
            raise ValueError("invalid_submission: inference cycle")
        if name in resolved:
            return resolved[name]
        if name not in by_draft:
            raise ValueError("invalid_submission: unknown premise")
        visiting.add(name)
        value = by_draft[name]
        plain(value.text)
        if value.derivation:
            plain(value.derivation)
        if any(alias not in evidence for alias in value.evidence_aliases):
            raise ValueError("citation_out_of_scope")
        premises = tuple(resolve(premise) for premise in value.premise_ids)
        if any(premise.knowledge != "asserted" for premise in premises):
            raise ValueError("invalid_submission: inferred primary premise")
        spans = tuple(evidence[alias] for alias in value.evidence_aliases)
        contents = {
            "text": value.text,
            "evidence": [asdict(span) for span in spans],
            "knowledge": value.knowledge,
            "premise_ids": [p.claim_id for p in premises],
            "derivation": value.derivation,
        }
        claim = ManagedClaim(
            claim_id="claim:" + hashlib.sha256(canonical_object(contents)).hexdigest(),
            text=value.text,
            evidence=spans,
            status=value.status,
            knowledge=value.knowledge,
            premise_ids=tuple(p.claim_id for p in premises),
            derivation=value.derivation,
        )
        visiting.remove(name)
        resolved[name] = claim
        return claim

    for claim in drafts:
        resolve(claim.draft_id)
    return resolved


def _render_pages(
    service: "MaintenanceService", pages: tuple[PageRevision, ...], updated_on: str
) -> tuple[str, ...]:
    """Keep existing creation metadata and resolve navigation from host inventory."""
    from capicove.wiki.filesystem import _split_markdown_text

    inventory = {page.page_id: page for page in service._inventory()}
    paths = {identity: page.path for identity, page in inventory.items()}
    paths.update({page.page_id: page.page_path for page in pages})
    rendered = []
    for page in pages:
        created_on = updated_on
        if page.parent_sha256:
            path = service.wiki / inventory[page.page_id].path
            raw = read_authorized(path, (service.wiki,), 16 * 1024 * 1024)
            metadata, _ = _split_markdown_text(path, raw.decode(), require_frontmatter=True)
            created_on = str(metadata["created"])
        rendered.append(render_page(page, created_on, updated_on=updated_on, page_paths=paths))
    return tuple(rendered)
