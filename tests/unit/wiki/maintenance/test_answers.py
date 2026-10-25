"""Read-only answer contexts bind original evidence and live revisions."""

from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import (
    AnswerCitation,
    AnswerClaim,
    AnswerSubmission,
    MaintenanceRecord,
)
from capicove.wiki.maintenance.codec import encode_record
from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


def test_answer_is_read_only_and_checks_aliases_and_quotes(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    before = snapshot(service.wiki)
    task = service.answer_task("Plants")
    assert task.evidence[0].text == "Plants use light.\n"
    value = AnswerSubmission(
        task_id=task.task_id,
        claims=(
            AnswerClaim(
                text="Plants use light.",
                citations=(AnswerCitation(alias="e1", quote="Plants use light."),),
                status="current",
            ),
        ),
    )

    def wire(answer: AnswerSubmission) -> bytes:
        return encode_record(
            MaintenanceRecord(schema_version=1, record_type="answer_submission", payload=answer)
        )

    result = service.answer_submit(task, wire(value))
    assert result.status == "answered"
    assert result.evidence[0].span == task.evidence[0].span
    assert result.semantic_status == "not_verified"
    for citation in (
        AnswerCitation(alias="e999", quote="Plants"),
        AnswerCitation(alias="e1", quote="Invented"),
    ):
        bad = replace(value, claims=(replace(value.claims[0], citations=(citation,)),))
        with pytest.raises(ValueError, match="citation"):
            service.answer_submit(task, wire(bad))
    forged = replace(task, evidence=(replace(task.evidence[0], text="Invented"),))
    with pytest.raises(ValueError, match="task"):
        service.answer_submit(forged, wire(value))
    assert (
        service.answer_submit(task, wire(replace(value, claims=()))).status
        == "insufficient_evidence"
    )
    assert snapshot(service.wiki) == before


def test_withdrawn_and_dirty_claims_are_excluded(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    task = service.answer_task("Plants")
    page = next((service.wiki / "pages/concepts").glob("*.md"))
    original = page.read_bytes()
    page.write_bytes(original + b"Manual addition.\n")
    service.build_discovery()
    assert not service.answer_task("Plants").claims
    page.write_bytes(original)
    service.build_discovery()
    service.withdraw(task.evidence[0].span.source_id, reason="Unavailable", idempotency_key="a")
    assert not service.answer_task("Plants").claims
    empty = encode_record(
        MaintenanceRecord(
            schema_version=1,
            record_type="answer_submission",
            payload=AnswerSubmission(task_id=task.task_id, claims=()),
        )
    )
    with pytest.raises(ValueError, match=r"task|policy"):
        service.answer_submit(task, empty)


def test_deferred_managed_page_is_excluded_until_revised(tmp_path: Path) -> None:
    from capicove.contracts.wiki_maintenance import (
        CompilationSubmission,
        ImpactDecision,
        ReviewDecision,
    )
    from capicove.wiki.maintenance.review import ReviewAuthority
    from tests.unit.wiki.maintenance.test_submissions import wire

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    original = service.answer_task("Plants")
    page_id = service._inventory()[0].page_id
    task = service.create_task((original.evidence[0].span.source_id,), query="Plants")
    value = CompilationSubmission(
        task_id=task.task_id,
        task_version=task.task_version,
        idempotency_key="defer",
        dispositions=("no_material",),
        rationale="Needs follow-up review.",
        pages=(),
        impact=(
            ImpactDecision(page_id=page_id, action="follow_up", reason="Review applicability."),
        ),
        provenance=None,
    )
    proposed = service.submit(wire(value))
    view = service.preview(proposed.proposal_id)
    accepted = service.review(
        proposed.proposal_id,
        tuple(
            ReviewDecision(subject_id=s, verdict="supported", reason="Synthetic fixture.")
            for s in view.review_subjects
        ),
        authority=ReviewAuthority("test", frozenset({"wiki:review"})),
        accept=True,
    )
    service.apply(proposed.proposal_id, accepted.review_id)
    assert not service.answer_task("Plants").claims
