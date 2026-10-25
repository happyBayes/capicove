"""Host authorization and content-bound semantic review."""

from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import ReviewDecision
from capicove.wiki.maintenance.review import ReviewAuthority
from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.unit.wiki.maintenance.test_submissions import submission, wire
from tests.wiki_maintenance_support.oracle import snapshot


def test_review_requires_host_permission_and_every_supported_decision(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    proposal = service.submit(wire(submission(task)))
    preview = service.preview(proposal.proposal_id)
    decisions = tuple(
        ReviewDecision(
            subject_id=item, verdict="supported", reason="Checked original evidence and full diff."
        )
        for item in preview.review_subjects
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(PermissionError):
        service.review(
            proposal.proposal_id,
            decisions,
            authority=ReviewAuthority("agent", frozenset({"wiki:write"})),
            accept=True,
        )
    assert snapshot(tmp_path / "wiki") == before
    authority = ReviewAuthority("operator:test", frozenset({"wiki:review"}))
    with pytest.raises(ValueError, match="incomplete"):
        service.review(proposal.proposal_id, decisions[:-1], authority=authority, accept=True)
    with pytest.raises(ValueError, match=r"unsupported|uncertain"):
        service.review(
            proposal.proposal_id,
            (replace(decisions[0], verdict="uncertain"), *decisions[1:]),
            authority=authority,
            accept=True,
        )
    review = service.review(proposal.proposal_id, decisions, authority=authority, accept=True)
    assert review.reviewer_id == "operator:test"
    assert review.rendered_sha256 == proposal.rendered_sha256
    assert review.accepted
    assert snapshot(tmp_path / "wiki" / "pages") == {}
