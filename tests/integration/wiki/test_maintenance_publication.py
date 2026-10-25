"""Reviewed publication binds exact content and preserves immutable evidence."""

from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import ProposalRecord, ReviewDecision, ReviewRecord
from capicove.wiki.maintenance.review import ReviewAuthority
from capicove.wiki.maintenance.service import MaintenanceService
from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.unit.wiki.maintenance.test_submissions import submission, wire
from tests.wiki_maintenance_support.oracle import snapshot


def prepared(root: Path) -> tuple[MaintenanceService, ProposalRecord, ReviewRecord]:
    service = experimental_workspace(root)
    source = root / "a.md"
    source.write_bytes(b"Plants use light.\r\n")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    proposal = service.submit(wire(submission(task)))
    view = service.preview(proposal.proposal_id)
    review = service.review(
        proposal.proposal_id,
        tuple(
            ReviewDecision(
                subject_id=s, verdict="supported", reason="Synthetic protocol check only."
            )
            for s in view.review_subjects
        ),
        authority=ReviewAuthority("test", frozenset({"wiki:review"})),
        accept=True,
    )
    return service, proposal, review


def test_apply_idempotent_and_commits_ownership(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.ownership import managed_revision

    service, proposal, review = prepared(tmp_path)
    before = snapshot(service.wiki / "raw")
    receipt = service.apply(proposal.proposal_id, review.review_id)
    assert receipt.wiki_revision == 1
    assert snapshot(service.wiki / "raw") == before
    pages = service._inventory()
    assert len(pages) == 1
    assert managed_revision(service, pages[0]).page_id == pages[0].page_id
    durable = snapshot(service.wiki)
    assert service.apply(proposal.proposal_id, review.review_id) == receipt
    assert snapshot(service.wiki) == durable
    with pytest.raises(ValueError, match="stale"):
        service.preview(proposal.proposal_id)


def test_changed_review_and_dirty_base_rejected_without_writes(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    path = service.wiki / "maintenance/reviews" / (review.review_id[7:] + ".json")
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"Synthetic protocol check only.", b"Altered review!"))
    before = snapshot(service.wiki)
    with pytest.raises(ValueError, match="review"):
        service.apply(proposal.proposal_id, review.review_id)
    assert snapshot(service.wiki) == before


def test_corrupt_active_state_is_rejected(tmp_path: Path) -> None:
    import json

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    path = service.wiki / "maintenance/state.json"
    value = json.loads(path.read_bytes())
    value["wiki_revision"] += 1
    path.write_text(json.dumps(value))
    before = snapshot(service.wiki)
    with pytest.raises(ValueError, match="state"):
        service.create_task(
            ("source:sha256:" + next((service.wiki / "raw").glob("*.md")).stem,), query="Plants"
        )
    assert snapshot(service.wiki) == before


def test_unused_corrupt_representation_blocks_candidate(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    other = tmp_path / "other.md"
    other.write_text("Unrelated evidence.")
    captured = service.capture(other)
    path = service.wiki / "representations" / (captured.representation_sha256 + ".json")
    path.write_bytes(path.read_bytes().replace(b"Unrelated evidence.", b"Corrupted evidence."))
    before = snapshot(service.wiki)
    with pytest.raises(ValueError, match="digest"):
        service.apply(proposal.proposal_id, review.review_id)
    assert snapshot(service.wiki) == before


def test_apply_publishes_search_projection_and_one_log_entry(tmp_path: Path) -> None:
    from capicove.wiki import WikiService

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    assert service.answer_task("Plants").claims
    assert WikiService(tmp_path).query("Plants").hits
    log = (service.wiki / "log.md").read_bytes()
    service.apply(proposal.proposal_id, review.review_id)
    assert (service.wiki / "log.md").read_bytes() == log


def test_rollback_is_a_new_reviewed_compensating_revision(tmp_path: Path) -> None:
    from dataclasses import replace

    from capicove.contracts.wiki_maintenance import ImpactDecision

    service, proposal, review = prepared(tmp_path)
    first = service.apply(proposal.proposal_id, review.review_id)
    page_id = service._inventory()[0].page_id
    captured_id = "source:sha256:" + next((service.wiki / "raw").glob("*.md")).stem
    task = service.create_task((captured_id,), query="Plants")
    updated = replace(
        submission(task),
        dispositions=("update",),
        idempotency_key="update",
        pages=(
            replace(
                submission(task).pages[0],
                operation="revise",
                page_id=page_id,
                title="Updated Plants",
            ),
        ),
        impact=(ImpactDecision(page_id=page_id, action="revise", reason="Update title."),),
    )
    changed = service.submit(wire(updated))

    def accept(identity: str) -> ReviewRecord:
        return service.review(
            identity,
            tuple(
                ReviewDecision(
                    subject_id=s, verdict="supported", reason="Synthetic protocol check."
                )
                for s in service.preview(identity).review_subjects
            ),
            authority=ReviewAuthority("test", frozenset({"wiki:review"})),
            accept=True,
        )

    second = service.apply(changed.proposal_id, accept(changed.proposal_id).review_id)
    before = snapshot(service.wiki / "pages")
    rollback = service.rollback_proposal(page_id, first.publication_id, idempotency_key="rollback")
    assert snapshot(service.wiki / "pages") == before
    assert service.preview(rollback.proposal_id).pages[0].title == "Plants"
    third = service.apply(rollback.proposal_id, accept(rollback.proposal_id).review_id)
    assert third.wiki_revision == 3 and third.parent_id == second.publication_id
    assert len(tuple((service.wiki / "maintenance/publications").glob("*.json"))) == 3


def test_state_binds_review_and_parent_record_bytes(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    path = service.wiki / "maintenance/reviews" / (review.review_id[7:] + ".json")
    path.write_bytes(
        path.read_bytes().replace(b"Synthetic protocol check only.", b"Changed after publication.")
    )
    with pytest.raises(ValueError, match="state"):
        service.status()
