"""Live withdrawal is monotonic and independent of editorial transactions."""

from pathlib import Path

import pytest

from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


def test_withdrawal_invalidates_task_and_old_review(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    source_id = service.load_task(proposal.submission.task_id).sources[0].source_id
    pages = snapshot(service.wiki / "pages")
    event = service.withdraw(source_id, reason="Operator withdrawal", idempotency_key="withdraw-1")
    assert event.policy_revision == 1
    durable = snapshot(service.wiki)
    assert (
        service.withdraw(source_id, reason="Operator withdrawal", idempotency_key="withdraw-1")
        == event
    )
    assert snapshot(service.wiki) == durable
    with pytest.raises(ValueError, match="withdrawn"):
        service.read_markdown(source_id)
    with pytest.raises(ValueError, match="policy_changed"):
        service.apply(proposal.proposal_id, review.review_id)
    assert snapshot(service.wiki / "pages") == pages
    with pytest.raises(ValueError, match="idempotency_conflict"):
        service.withdraw(source_id, reason="Different", idempotency_key="withdraw-1")


def test_withdrawal_allowed_while_publication_pending(tmp_path: Path) -> None:
    service, proposal, _ = prepared(tmp_path)
    source_id = service.load_task(proposal.submission.task_id).sources[0].source_id
    marker = service.wiki / "_migration/maintenance.pending"
    marker.write_bytes(b"pending")
    event = service.withdraw(source_id, reason="Stop use now", idempotency_key="pending-1")
    assert event.policy_revision == 1
    assert marker.read_bytes() == b"pending"
