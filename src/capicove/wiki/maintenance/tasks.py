"""Immutable bounded compilation tasks with exact workspace snapshot fences."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from capicove.contracts.wiki_maintenance import (
    Candidate,
    CompilationTask,
    EvidenceAlias,
    MaintenanceRecord,
    ManagedClaim,
    PublicationRecord,
)
from capicove.wiki.maintenance.codec import decode_record, encode_record
from capicove.wiki.maintenance.discovery import POLICY, discover, inventory_snapshot
from capicove.wiki.maintenance.sources import SourceInterval, load_source, resolve, slices
from capicove.wiki.maintenance.store import (
    canonical_object,
    read_authorized,
    read_object,
    record_path,
    write_object,
)


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService

TASK_FIELDS = {"schema_version", "task", "inventory_sha256", "created_on"}


def state(service: "MaintenanceService") -> dict[str, Any]:
    """Validate the current pointer before binding a task or proposal."""
    legacy_fields = {"schema_version", "workspace_id", "wiki_revision", "publication_id"}
    try:
        value = read_object(
            service.wiki / "maintenance/state.json",
            service.wiki,
            legacy_fields | {"policy_revision", "record_references"},
        )
    except ValueError:
        value = read_object(service.wiki / "maintenance/state.json", service.wiki, legacy_fields)
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["workspace_id"], str)
        or not value["workspace_id"].strip()
        or type(value["wiki_revision"]) is not int
        or value["wiki_revision"] < 0
        or (value["publication_id"] is not None and not isinstance(value["publication_id"], str))
    ):
        raise ValueError("invalid maintenance state")
    if "record_references" in value:
        refs = value["record_references"]
        if (
            type(value["policy_revision"]) is not int
            or value["policy_revision"] < 0
            or not isinstance(refs, dict)
            or set(refs) != {"proposal", "review", "parent_publication"}
        ):
            raise ValueError("invalid maintenance state references")
    if value["publication_id"] is not None:
        receipt = decode_record(
            read_authorized(
                record_path(service.wiki, "publications", value["publication_id"], "publication"),
                (service.wiki,),
                1048576,
            )
        ).payload
        if (
            not isinstance(receipt, PublicationRecord)
            or receipt.publication_id != value["publication_id"]
            or receipt.wiki_revision != value["wiki_revision"]
            or receipt.status == "aborted"
            or receipt.state_sha256 != hashlib.sha256(canonical_object(value)).hexdigest()
        ):
            raise ValueError("invalid maintenance state commitment")
        if "record_references" not in value or value["policy_revision"] != receipt.policy_revision:
            raise ValueError("invalid maintenance state references")
        for name, folder, identity, prefix in (
            ("proposal", "proposals", receipt.proposal_id, "proposal"),
            ("review", "reviews", receipt.review_id, "review"),
            ("parent_publication", "publications", receipt.parent_id, "publication"),
        ):
            expected = (
                hashlib.sha256(
                    read_authorized(
                        record_path(service.wiki, folder, identity, prefix),
                        (service.wiki,),
                        1048576,
                    )
                ).hexdigest()
                if identity is not None
                else None
            )
            if value["record_references"][name] != expected:
                raise ValueError("invalid maintenance state reference digest")
    elif value["wiki_revision"] != 0:
        raise ValueError("invalid initial maintenance state")
    return value


def advance_state(
    service: "MaintenanceService",
    current: dict[str, Any],
    publication_id: str,
    policy_revision: int,
    *,
    proposal_id: str | None = None,
    review_id: str | None = None,
) -> dict[str, Any]:
    """Bind bounded authoritative record references without a self-hash cycle."""
    references = {}
    for name, folder, identity, prefix in (
        ("proposal", "proposals", proposal_id, "proposal"),
        ("review", "reviews", review_id, "review"),
        ("parent_publication", "publications", current["publication_id"], "publication"),
    ):
        references[name] = (
            hashlib.sha256(
                read_authorized(
                    record_path(service.wiki, folder, identity, prefix), (service.wiki,), 1048576
                )
            ).hexdigest()
            if identity is not None
            else None
        )
    return {
        **current,
        "wiki_revision": current["wiki_revision"] + 1,
        "publication_id": publication_id,
        "policy_revision": policy_revision,
        "record_references": references,
    }


def _task_json(task: CompilationTask) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            encode_record(
                MaintenanceRecord(schema_version=1, record_type="compilation_task", payload=task)
            )
        ),
    )


def _identity(task: CompilationTask, fingerprint: str, created_on: str) -> str:
    value = {
        "task": _task_json(replace(task, task_id="task:" + "0" * 64)),
        "inventory_sha256": fingerprint,
        "created_on": created_on,
        "schema_version": 1,
    }
    return "task:" + hashlib.sha256(canonical_object(value)).hexdigest()


def load_task(
    service: "MaintenanceService", task_id: str
) -> tuple[CompilationTask, dict[str, Any]]:
    """Load only accepted, unsuperseded tasks whose full snapshot is current."""
    path = record_path(service.wiki, "tasks", task_id, "task")
    bundle = read_object(path, service.wiki, TASK_FIELDS)
    record = decode_record(canonical_object(bundle["task"]))
    if not isinstance(record.payload, CompilationTask):
        raise ValueError("invalid task record")
    task = record.payload
    if (
        task.task_id != task_id
        or type(bundle["schema_version"]) is not int
        or bundle["schema_version"] != 1
        or task_id != _identity(task, bundle["inventory_sha256"], bundle["created_on"])
    ):
        raise ValueError("task digest mismatch")
    superseded = record_path(service.wiki, "supersessions", task_id, "task")
    if superseded.exists() or superseded.is_symlink():
        raise ValueError("task_superseded")
    if task.previous_task_id:
        accepted = read_object(
            record_path(service.wiki, "supersessions", task.previous_task_id, "task"),
            service.wiki,
            {"schema_version", "next_task_id"},
        )
        if accepted != {"schema_version": 1, "next_task_id": task_id}:
            raise ValueError("task_superseded: context is not accepted")
    check_task(service, task, bundle)
    return task, bundle


def check_task(
    service: "MaintenanceService", task: CompilationTask, bundle: dict[str, Any]
) -> None:
    """Re-resolve policy, view and every exact source slice before acceptance."""
    policy, withdrawn = service._admit()
    current = state(service)
    if policy != task.policy_revision:
        raise ValueError("policy_changed")
    if (
        current["workspace_id"] != task.workspace_id
        or current["wiki_revision"] != task.wiki_revision
    ):
        raise ValueError("proposal_stale: workspace revision")
    fingerprint = hashlib.sha256(inventory_snapshot(service._inventory())).hexdigest()
    if fingerprint != bundle["inventory_sha256"]:
        raise ValueError("proposal_stale: page inventory changed")
    values = {}
    for source in task.sources:
        if source.source_id in withdrawn:
            raise ValueError("policy_changed")
        material = load_source(service, source.source_id, withdrawn)
        if material.reference != source:
            raise ValueError("evidence_digest_mismatch")
        values[source.source_id] = material
    for alias in task.evidence:
        selected = values.get(alias.span.source_id)
        if selected is None or resolve(selected, alias.span) != alias.text:
            raise ValueError("citation_out_of_scope")


def create_task(
    service: "MaintenanceService",
    source_ids: tuple[str, ...],
    query: str,
    previous_task_id: str | None,
    intervals: tuple[SourceInterval, ...] | None,
    page_ids: tuple[str, ...] = (),
) -> CompilationTask:
    """Capture a bounded complete context; refuse overflow rather than truncate."""
    if not 1 <= len(source_ids) <= 8 or len(set(source_ids)) != len(source_ids):
        raise ValueError("capacity_exceeded: selected sources")
    if intervals is not None and (
        not 1 <= len(intervals) <= 128
        or len(set(intervals)) != len(intervals)
        or {item[0] for item in intervals} != set(source_ids)
    ):
        raise ValueError("citation_out_of_scope: explicit intervals")
    policy, withdrawn = service._admit()
    current = state(service)
    previous = load_task(service, previous_task_id)[0] if previous_task_id else None
    pages = service._inventory()
    by_id = {page.page_id: page for page in pages}
    if (
        len(page_ids) > 20
        or len(set(page_ids)) != len(page_ids)
        or not set(page_ids) <= by_id.keys()
    ):
        raise ValueError("citation_out_of_scope: explicit candidate pages")
    result = discover(
        service.wiki / "_derived" / "maintenance.sqlite", pages, query, source_ids=source_ids
    )
    evidence: list[EvidenceAlias] = []
    sources = []
    for source_id in source_ids:
        material = load_source(service, source_id, withdrawn)
        sources.append(material.reference)
        for span, text in slices(material, intervals):
            evidence.append(EvidenceAlias(alias=f"e{len(evidence) + 1}", span=span, text=text))
            if len(evidence) > 128 or sum(len(item.text.encode()) for item in evidence) > 262144:
                raise ValueError("capacity_exceeded: task context")
    ranked = {
        item.page_id: Candidate(
            page_id=item.page_id,
            revision_sha256=item.revision_sha256,
            reason=",".join(f"{lane}:{rank}" for lane, rank in item.lane_ranks),
        )
        for item in result.candidates
    }
    selected_ids = page_ids or tuple(ranked)[:20]
    remaining = 262144 - sum(len(alias.text.encode()) for alias in evidence)
    candidate_list = []
    for position, identity in enumerate(selected_ids):
        page = by_id[identity]
        per_page = remaining // (len(selected_ids) - position)
        claim_context: tuple[ManagedClaim, ...] = ()
        if (service.wiki / "maintenance/managed" / (identity[5:] + ".json")).exists():
            from capicove.wiki.maintenance.ownership import managed_revision

            claim_context = managed_revision(service, page).claims
        original_claim_count = len(claim_context)
        title = page.title.encode()[:512].decode("utf-8", errors="ignore")
        candidate = Candidate(
            page_id=identity,
            revision_sha256=page.revision_sha256,
            reason=ranked[identity].reason if identity in ranked else "explicit_context",
            title=title,
            claim_context=claim_context,
        )

        def size(value: Candidate) -> int:
            return len(
                json.dumps(asdict(value), ensure_ascii=False, separators=(",", ":")).encode()
            )

        while size(candidate) > per_page and candidate.claim_context:
            candidate = replace(candidate, claim_context=candidate.claim_context[:-1])
        if size(candidate) > per_page:
            raise ValueError("capacity_exceeded: candidate context")
        available = min(16384, per_page - size(candidate))
        candidate = replace(
            candidate, body=page.body.encode()[:available].decode("utf-8", errors="ignore")
        )
        while size(candidate) > per_page:
            extra = max(1, size(candidate) - per_page)
            candidate = replace(
                candidate, body=candidate.body.encode()[:-extra].decode("utf-8", errors="ignore")
            )
        candidate = replace(
            candidate,
            context_complete=(
                candidate.body == page.body
                and title == page.title
                and len(candidate.claim_context) == original_claim_count
            ),
        )
        remaining -= size(candidate)
        candidate_list.append(candidate)
    candidates = tuple(candidate_list)
    from capicove.wiki.maintenance.impact import affected_pages

    impact = affected_pages(
        pages, source_ids=source_ids, page_ids=tuple(item.page_id for item in candidates)
    )
    task = CompilationTask(
        task_id="task:" + "0" * 64,
        task_version=previous.task_version + 1 if previous else 1,
        workspace_id=current["workspace_id"],
        wiki_revision=current["wiki_revision"],
        policy_revision=policy,
        sources=tuple(sources),
        evidence=tuple(evidence),
        candidates=candidates,
        permitted_page_ids=tuple(item.page_id for item in candidates),
        discovery_policy=POLICY,
        coverage="partial"
        if any(not item.context_complete for item in candidates)
        else result.coverage,
        max_output_bytes=1048576,
        previous_task_id=previous_task_id,
        affected_page_ids=impact.page_ids,
        deferred_page_count=len(impact.follow_up_page_ids),
        deferred_page_sha256=hashlib.sha256(
            canonical_object({"page_ids": list(impact.follow_up_page_ids)})
        ).hexdigest()
        if impact.follow_up_page_ids
        else None,
    )
    fingerprint = hashlib.sha256(inventory_snapshot(pages)).hexdigest()
    created_on = date.today().isoformat()
    task = replace(task, task_id=_identity(task, fingerprint, created_on))
    bundle = {
        "schema_version": 1,
        "task": _task_json(task),
        "inventory_sha256": fingerprint,
        "created_on": created_on,
    }
    write_object(record_path(service.wiki, "tasks", task.task_id, "task"), bundle, service.wiki)
    if previous_task_id:
        write_object(
            record_path(service.wiki, "supersessions", previous_task_id, "task"),
            {"schema_version": 1, "next_task_id": task.task_id},
            service.wiki,
        )
    return task
