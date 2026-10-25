"""Proposal acceptance validates before any durable effect."""

from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import (
    ClaimDraft,
    CompilationSubmission,
    CompilationTask,
    MaintenanceRecord,
    PageDraft,
    Section,
)
from capicove.wiki.maintenance.codec import encode_record
from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.wiki_maintenance_support.oracle import snapshot


def submission(task: CompilationTask) -> CompilationSubmission:
    claim = ClaimDraft(
        draft_id="claim-one",
        text="Plants use light.",
        evidence_aliases=("e1",),
        status="current",
        knowledge="asserted",
        premise_ids=(),
        derivation=None,
    )
    page = PageDraft(
        draft_id="plant",
        operation="create",
        page_id=None,
        title="Plants",
        kind="concept",
        aliases=(),
        sections=(Section(heading="Overview", claim_ids=("claim-one",)),),
        claims=(claim,),
        relations=(),
    )
    return CompilationSubmission(
        task_id=task.task_id,
        task_version=task.task_version,
        idempotency_key="request-one",
        dispositions=("new",),
        rationale="New source-backed concept.",
        pages=(page,),
        impact=(),
        provenance=None,
    )


def wire(value: CompilationSubmission) -> bytes:
    return encode_record(
        MaintenanceRecord(schema_version=1, record_type="compilation_submission", payload=value)
    )


def test_submit_retry_conflict_preview_and_page_identity(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    before_pages = snapshot(tmp_path / "wiki" / "pages")
    value = submission(task)
    proposal = service.submit(wire(value))
    accepted = snapshot(tmp_path / "wiki")
    assert service.submit(wire(value)) == proposal
    assert snapshot(tmp_path / "wiki") == accepted
    with pytest.raises(ValueError, match="idempotency_conflict"):
        service.submit(wire(replace(value, rationale="Changed payload")))
    assert snapshot(tmp_path / "wiki") == accepted
    preview = service.preview(proposal.proposal_id)
    assert preview.pages[0].page_id.startswith("page:")
    assert preview.pages[0].page_path.startswith("pages/concepts/")
    assert "Plants use light." in preview.diff
    assert preview.pages[0].claims[0].evidence[0] == task.evidence[0].span
    assert snapshot(tmp_path / "wiki" / "pages") == before_pages


def test_unknown_alias_and_malicious_text_have_zero_effect(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = submission(task)
    before = snapshot(tmp_path / "wiki")
    for claim in (
        replace(value.pages[0].claims[0], evidence_aliases=("e99",)),
        replace(value.pages[0].claims[0], text="<script>alert(1)</script>"),
    ):
        invalid = replace(value, pages=(replace(value.pages[0], claims=(claim,)),))
        with pytest.raises(ValueError):
            service.submit(wire(invalid))
        assert snapshot(tmp_path / "wiki") == before


def test_no_material_accepts_no_edits_and_stale_policy_rejects(tmp_path: Path) -> None:
    import json

    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = replace(
        submission(task),
        pages=(),
        dispositions=("no_material",),
        rationale="Irrelevant to this Wiki.",
    )
    proposal = service.submit(wire(value))
    assert service.preview(proposal.proposal_id).pages == ()
    policy = tmp_path / "wiki" / "maintenance" / "source-policy.json"
    policy.write_text(
        json.dumps({"schema_version": 1, "policy_revision": 1, "withdrawn": [captured.source_id]})
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match=r"policy|stale"):
        service.preview(proposal.proposal_id)
    assert snapshot(tmp_path / "wiki") == before


def test_no_material_rationale_is_language_independent(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("植物利用光能。")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="行星")
    value = replace(
        submission(task),
        pages=(),
        dispositions=("no_material",),
        rationale="这份植物资料与当前行星知识库主题无关。需要审查确认。",
    )
    proposal = service.submit(wire(value))
    preview = service.preview(proposal.proposal_id)
    assert preview.pages == ()
    assert "disposition:no_material" in preview.review_subjects
