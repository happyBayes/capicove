"""Apply exact host-reviewed concept proposals through the recovery journal."""

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from capicove.contracts.wiki_maintenance import (
    FileChange,
    MaintenanceRecord,
    PublicationRecord,
    ReviewRecord,
)
from capicove.wiki.maintenance.codec import decode_record, encode_record, record_digest
from capicove.wiki.maintenance.store import canonical_object, read_authorized, record_path
from capicove.wiki.maintenance.tasks import advance_state, state
from capicove.wiki.maintenance.transaction import transact_locked
from capicove.wiki.maintenance.validation import preview


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


def _record(kind: Any, payload: Any) -> bytes:
    return encode_record(MaintenanceRecord(schema_version=1, record_type=kind, payload=payload))


def _candidate_lint(service: "MaintenanceService", changes: dict[str, bytes | None]) -> None:
    """Reuse independent legacy lint on a private v2 compatibility projection.

    Typed Markdown representations are validated by task resolution, not passed
    to the Book-v2 parser. No public workspace is downgraded or enabled here.
    """
    from capicove.wiki.maintenance.evidence import resolve_markdown
    from capicove.wiki.maintenance.ownership import managed_revision
    from capicove.wiki.maintenance.representations import markdown_representation
    from capicove.wiki.validation import lint

    for page in service._inventory():
        ownership = service.wiki / "maintenance/managed" / (page.page_id[5:] + ".json")
        if ownership.exists() or ownership.is_symlink():
            revision = managed_revision(service, page)
            for claim in revision.claims:
                for span in claim.evidence:
                    if span.kind == "markdown_span":
                        raw = read_authorized(
                            service.wiki / "raw" / (span.source_sha256 + ".md"),
                            (service.wiki,),
                            16777216,
                        )
                        resolve_markdown(span, raw)
                    else:
                        from capicove.wiki.maintenance.sources import load_source, resolve

                        resolve(
                            load_source(
                                service,
                                span.source_id,
                                (),
                                representation_sha256=span.representation_sha256,
                            ),
                            span,
                        )

    with tempfile.TemporaryDirectory(prefix="capicove-wiki-candidate-") as directory:
        candidate = Path(directory).resolve() / "wiki"
        candidate.mkdir()
        total = 0
        for item in sorted(service.wiki.rglob("*")):
            relative = item.relative_to(service.wiki).as_posix()
            if relative.split("/", 1)[0] in {
                "maintenance",
                "_migration",
                "_derived",
                "representations",
            }:
                continue
            if item.is_symlink():
                raise ValueError("candidate_invalid: symlink")
            target = candidate / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                data = read_authorized(item, (service.wiki,), 256 * 1024 * 1024)
                total += len(data)
                if total > 512 * 1024 * 1024:
                    raise ValueError("capacity_exceeded: candidate validation")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for name in ("_migration", "_derived", "representations", "pages/books"):
            (candidate / name).mkdir(parents=True, exist_ok=True)
        for item in sorted((service.wiki / "representations").glob("*.json")):
            data = read_authorized(item, (service.wiki,), 16 * 1024 * 1024)
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError("evidence_digest_mismatch: invalid representation")
            if value.get("transform") == "wiki-markdown-v1":
                source_digest = value.get("source_sha256")
                if (
                    not isinstance(source_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
                ):
                    raise ValueError("evidence_digest_mismatch: invalid source identity")
                raw = read_authorized(
                    service.wiki / "raw" / (source_digest + ".md"), (service.wiki,), 16777216
                )
                expected = markdown_representation(raw)
                if data != expected.encoded or item.stem != expected.representation_sha256:
                    raise ValueError("evidence_digest_mismatch: representation")
            else:
                (candidate / "representations" / item.name).write_bytes(data)
        (candidate / "_migration/writer.lock").touch()
        citations = candidate / "graph/citations.jsonl"
        if not citations.exists():
            citations.write_bytes(b"")
        manifest = candidate / "wiki.toml"
        manifest.write_text(
            re.sub(
                r"(?m)^(\s*schema_version\s*=\s*)3(?=\s*(?:#.*)?$)", r"\g<1>2", manifest.read_text()
            )
        )
        for relative, content in changes.items():
            if not relative.startswith("pages/"):
                continue
            target = candidate / relative
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        result = lint(candidate)
        if not result.valid:
            raise ValueError(
                "candidate_invalid: "
                + "; ".join(f.code + ": " + f.message for f in result.findings)
            )


def apply(service: "MaintenanceService", proposal_id: str, review_id: str) -> PublicationRecord:
    """Validate review and complete candidate before any visible mutation."""
    service._admit()
    current = state(service)
    identity = (
        "publication:"
        + hashlib.sha256(
            canonical_object({"proposal_id": proposal_id, "review_id": review_id})
        ).hexdigest()
    )
    receipt_path = record_path(service.wiki, "publications", identity, "publication")
    if current["publication_id"] == identity:
        receipt = decode_record(read_authorized(receipt_path, (service.wiki,), 1048576)).payload
        if (
            not isinstance(receipt, PublicationRecord)
            or receipt.publication_id != identity
            or receipt.proposal_id != proposal_id
            or receipt.review_id != review_id
            or receipt.state_sha256 != hashlib.sha256(canonical_object(current)).hexdigest()
        ):
            raise ValueError("invalid publication receipt")
        for change in receipt.changes:
            path = service.wiki / change.relative_path
            actual = (
                hashlib.sha256(read_authorized(path, (service.wiki,), 16777216)).hexdigest()
                if path.exists()
                else None
            )
            if actual != change.after_sha256:
                raise ValueError("manual_edit_conflict")
        return receipt
    value = preview(service, proposal_id)
    record = decode_record(
        read_authorized(
            record_path(service.wiki, "reviews", review_id, "review"), (service.wiki,), 1048576
        )
    )
    review = record.payload
    if not isinstance(review, ReviewRecord):
        raise ValueError("invalid review")
    expected = (
        "review:"
        + hashlib.sha256(
            _record("review", replace(review, review_id="review:" + "0" * 64))
        ).hexdigest()
    )
    proposal = value.proposal
    if (
        review.review_id != review_id
        or expected != review_id
        or not review.accepted
        or review.proposal_id != proposal_id
        or review.proposal_sha256
        != record_digest(
            MaintenanceRecord(schema_version=1, record_type="proposal", payload=proposal)
        )
        or review.rendered_sha256 != proposal.rendered_sha256
        or review.wiki_revision != proposal.wiki_revision
        or review.policy_revision != proposal.policy_revision
        or {d.subject_id for d in review.decisions} != set(value.review_subjects)
        or len(review.decisions) != len(value.review_subjects)
        or any(d.verdict != "supported" for d in review.decisions)
    ):
        raise ValueError("review_stale")
    changes: dict[str, bytes | None] = {}
    inventory = {p.page_id: p for p in service._inventory()}
    for page, rendered in zip(value.pages, value.rendered, strict=True):
        previous = inventory.get(page.page_id)
        if previous and previous.path != page.page_path:
            changes[previous.path] = None
        changes[page.page_path] = rendered.encode()
        changes["maintenance/managed/" + page.page_id[5:] + ".json"] = canonical_object(
            {
                "schema_version": 1,
                "publication_id": identity,
                "revision": json.loads(_record("page_revision", page)),
                "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
    for follow_up in value.follow_up_tasks:
        changes["maintenance/follow-ups/" + follow_up.task_id[5:] + ".json"] = canonical_object(
            {
                **asdict(follow_up),
                "status": "ready",
                "wiki_revision": current["wiki_revision"] + 1,
                "publication_id": identity,
            }
        )
    _candidate_lint(service, changes)
    from capicove.wiki import WikiService
    from capicove.wiki.filesystem import append_log
    from capicove.wiki.maintenance.compatibility import projection
    from capicove.wiki.maintenance.discovery import build_discovery
    from capicove.wiki.maintenance.service import MaintenanceService

    with projection(service.wiki) as stage:
        for relative, content in changes.items():
            if not relative.startswith("pages/"):
                continue
            target = stage / relative
            if content is None:
                target.unlink()
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        WikiService(stage.parent).build(built_on=date.today())
        append_log(stage, f"## [{date.today().isoformat()}] maintenance | {identity}")
        build_discovery(
            stage / "_derived/maintenance.sqlite", MaintenanceService(stage.parent)._inventory()
        )
        for relative in (
            "index.md",
            "log.md",
            "_derived/search.sqlite3",
            "_derived/maintenance.sqlite",
        ):
            changes[relative] = read_authorized(stage / relative, (stage,), 16777216)
    next_state = advance_state(
        service,
        current,
        identity,
        proposal.policy_revision,
        proposal_id=proposal_id,
        review_id=review_id,
    )
    encoded_state = canonical_object(next_state)
    entries = []
    for relative, after in changes.items():
        path = service.wiki / relative
        before = read_authorized(path, (service.wiki,), 16777216) if path.exists() else None
        entries.append(
            FileChange(
                relative_path=relative,
                before_sha256=hashlib.sha256(before).hexdigest() if before is not None else None,
                after_sha256=hashlib.sha256(after).hexdigest() if after is not None else None,
            )
        )
    receipt = PublicationRecord(
        publication_id=identity,
        parent_id=current["publication_id"],
        proposal_id=proposal_id,
        review_id=review_id,
        wiki_revision=next_state["wiki_revision"],
        policy_revision=proposal.policy_revision,
        changes=tuple(entries),
        state_sha256=hashlib.sha256(encoded_state).hexdigest(),
        status="partial" if value.stale_page_ids else "complete",
    )
    changes[receipt_path.relative_to(service.wiki).as_posix()] = _record("publication", receipt)
    transact_locked(
        service.wiki,
        identity[12:],
        changes,
        state_bytes=encoded_state,
        policy_revision=proposal.policy_revision,
    )
    return receipt
