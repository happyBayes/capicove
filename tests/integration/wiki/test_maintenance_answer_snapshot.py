"""W4.01 answers stay read-only and are fenced to the live Wiki revision and policy."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import (
    AnswerCitation,
    AnswerClaim,
    AnswerSubmission,
    MaintenanceRecord,
)
from capicove.wiki import WikiService
from capicove.wiki.maintenance.codec import encode_record
from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


def _wire(answer: AnswerSubmission) -> bytes:
    return encode_record(
        MaintenanceRecord(schema_version=1, record_type="answer_submission", payload=answer)
    )


def _answer(task_id: str, status: str = "current") -> AnswerSubmission:
    return AnswerSubmission(
        task_id=task_id,
        claims=(
            AnswerClaim(
                text="Plants use light.",
                citations=(AnswerCitation(alias="e1", quote="Plants use light."),),
                status=status,  # type: ignore[arg-type]
            ),
        ),
    )


def test_answers_are_fenced_to_the_revision_and_claim_status(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    task = service.answer_task("Plants")
    before = snapshot(service.wiki)
    with pytest.raises(ValueError, match="status"):
        service.answer_submit(task, _wire(_answer(task.task_id, "disputed")))
    with pytest.raises(ValueError, match="capacity"):
        service.answer_submit(task, b"x" * 65537)
    with pytest.raises(ValueError):
        service.answer_submit(task, _wire(_answer("task:" + "1" * 64)))
    assert snapshot(service.wiki) == before
    assert service.answer_submit(task, _wire(_answer(task.task_id))).status == "answered"
    assert snapshot(service.wiki) == before
    source = tmp_path / "later.md"
    source.write_text("# Later\n\nUnrelated legacy evidence.\n")
    WikiService(tmp_path).ingest(source, ingested_on=date(2026, 10, 23))
    advanced = snapshot(service.wiki)
    with pytest.raises(ValueError, match="stale"):
        service.answer_submit(task, _wire(_answer(task.task_id)))
    assert snapshot(service.wiki) == advanced
    fresh = service.answer_task("Plants")
    assert fresh.wiki_revision == task.wiki_revision + 1
    assert service.answer_submit(fresh, _wire(_answer(fresh.task_id))).status == "answered"


def test_questions_without_material_return_no_evidence_without_writes(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    before = snapshot(service.wiki)
    task, result = service.evidence_only("quantum chromodynamics", include_history=False)
    assert result.status == "no_evidence" and task.evidence == () and result.claims == ()
    empty = replace(_answer(task.task_id), claims=())
    assert service.answer_submit(task, _wire(empty)).status == "insufficient_evidence"
    assert snapshot(service.wiki) == before
