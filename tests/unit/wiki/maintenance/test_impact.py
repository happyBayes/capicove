"""Impact dispositions and stale-page review are explicit and byte-preserving."""

from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import ImpactDecision
from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.unit.wiki.maintenance.test_submissions import submission, wire
from tests.wiki_maintenance_support.oracle import snapshot


def test_every_known_affected_page_needs_a_disposition(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    note = tmp_path / "wiki" / "pages" / "manual.md"
    note.write_text(
        "---\nid: page:MANUAL0001\ntitle: Plants\nkind: concept\n"
        "created: 2026-10-22\nupdated: 2026-10-22\n"
        f"sources: [{captured.source_id}]\n---\nManual content.\n"
    )
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = submission(task)
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="impact_incomplete"):
        service.submit(wire(value))
    assert snapshot(tmp_path / "wiki") == before
    value = replace(
        value,
        impact=(
            ImpactDecision(
                page_id="page:MANUAL0001", action="follow_up", reason="Review manually."
            ),
        ),
    )
    proposal = service.submit(wire(value))
    preview = service.preview(proposal.proposal_id)
    assert preview.stale_page_ids == ("page:MANUAL0001",)
    assert "stale:page:MANUAL0001" in preview.review_subjects
    old = note.read_bytes()
    note.write_bytes(old + b"Manual edit.\n")
    with pytest.raises(ValueError, match="stale"):
        service.preview(proposal.proposal_id)
    assert note.read_bytes() == old + b"Manual edit.\n"


def test_follow_up_is_bound_to_an_explicit_deferred_task(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    note = tmp_path / "wiki" / "pages" / "manual.md"
    note.write_text(
        "---\nid: page:MANUAL0001\ntitle: Plants\nkind: concept\n"
        "created: 2026-10-22\nupdated: 2026-10-22\n"
        f"sources: [{captured.source_id}]\n---\nManual content.\n"
    )
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = replace(
        submission(task),
        impact=(
            ImpactDecision(
                page_id="page:MANUAL0001", action="follow_up", reason="Review manually."
            ),
        ),
    )
    proposal = service.submit(wire(value))
    preview = service.preview(proposal.proposal_id)
    assert preview.follow_up_tasks[0].page_ids == ("page:MANUAL0001",)
    assert preview.follow_up_tasks[0].proposal_id == proposal.proposal_id
    assert preview.follow_up_tasks[0].status == "awaiting_publication"
    assert service.preview(proposal.proposal_id).follow_up_tasks == preview.follow_up_tasks


def test_impact_overflow_is_explicitly_split(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    identities = tuple(f"page:MANUAL{i:04d}" for i in range(101))
    for index, identity in enumerate(identities):
        (service.wiki / "pages" / f"manual-{index}.md").write_text(
            f"---\nid: {identity}\ntitle: Plants {index}\nkind: concept\n"
            "created: 2026-10-22\nupdated: 2026-10-22\n"
            f"sources: [{captured.source_id}]\n---\nManual content.\n"
        )
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = replace(
        submission(task),
        impact=tuple(
            ImpactDecision(page_id=identity, action="unchanged", reason="Reviewed unchanged.")
            for identity in identities[:100]
        ),
    )
    proposal = service.submit(wire(value))
    view = service.preview(proposal.proposal_id)
    assert view.stale_page_ids == (identities[-1],)
    assert view.follow_up_tasks[0].page_ids == (identities[-1],)
    assert any(subject.startswith("impact-overflow:") for subject in view.review_subjects)


def test_published_follow_up_activates_against_new_revision(tmp_path: Path) -> None:
    from capicove.contracts.wiki_maintenance import ReviewDecision
    from capicove.wiki.maintenance.review import ReviewAuthority

    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    (service.wiki / "pages/manual.md").write_text(
        "---\nid: page:MANUAL0001\ntitle: Plants\nkind: concept\n"
        "created: 2026-10-22\nupdated: 2026-10-22\n"
        f"sources: [{captured.source_id}]\n---\nExisting explanation.\n"
    )
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = replace(
        submission(task),
        impact=(
            ImpactDecision(
                page_id="page:MANUAL0001", action="follow_up", reason="Review manually."
            ),
        ),
    )
    proposal = service.submit(wire(value))
    view = service.preview(proposal.proposal_id)
    review = service.review(
        proposal.proposal_id,
        tuple(
            ReviewDecision(subject_id=s, verdict="supported", reason="Fixture review.")
            for s in view.review_subjects
        ),
        authority=ReviewAuthority("test", frozenset({"wiki:review"})),
        accept=True,
    )
    with pytest.raises(ValueError):
        service.activate_follow_up(view.follow_up_tasks[0].task_id)
    service.apply(proposal.proposal_id, review.review_id)
    activated = service.activate_follow_up(view.follow_up_tasks[0].task_id)
    assert activated.task.wiki_revision == 1
    assert "page:MANUAL0001" in activated.task.permitted_page_ids
    assert activated.remaining_page_ids == ()
    assert service.load_task(activated.task.task_id) == activated.task
