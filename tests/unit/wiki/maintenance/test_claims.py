"""Inference, dispute and publication ownership adversarial cases."""

from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import ImpactDecision, Relation, Section
from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.unit.wiki.maintenance.test_submissions import submission, wire
from tests.wiki_maintenance_support.oracle import snapshot


def test_inference_cycles_and_uncited_sections_rejected(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = submission(task)
    first = replace(
        value.pages[0].claims[0],
        knowledge="inferred",
        premise_ids=("claim-two",),
        derivation="Because.",
    )
    second = replace(first, draft_id="claim-two", premise_ids=("claim-one",))
    invalid = replace(
        value.pages[0],
        claims=(first, second),
        sections=(Section(heading="Claims", claim_ids=("claim-one", "claim-two")),),
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="cycle"):
        service.submit(wire(replace(value, pages=(invalid,))))
    with pytest.raises(ValueError, match="sections"):
        service.submit(
            wire(
                replace(
                    value,
                    pages=(
                        replace(
                            value.pages[0], sections=(Section(heading="Claims", claim_ids=()),)
                        ),
                    ),
                )
            )
        )
    assert snapshot(tmp_path / "wiki") == before


def test_dispute_preserves_two_original_sources(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    ids = []
    for name, text in (("a", "Source A reports success."), ("b", "Source B reports failure.")):
        source = tmp_path / (name + ".md")
        source.write_text(text)
        ids.append(service.capture(source).source_id)
    service.build_discovery()
    task = service.create_task(tuple(ids), query="Results")
    value = submission(task)
    first = replace(value.pages[0].claims[0], text="Source A reports success.", status="disputed")
    second = replace(
        first, draft_id="claim-two", text="Source B reports failure.", evidence_aliases=("e2",)
    )
    page = replace(
        value.pages[0],
        claims=(first, second),
        sections=(Section(heading="Dispute", claim_ids=("claim-one", "claim-two")),),
        relations=(
            Relation(kind="disputed", target_id="claim-two", reason="Opposing source reports."),
        ),
    )
    proposal = service.submit(wire(replace(value, pages=(page,), dispositions=("new", "disputed"))))
    preview = service.preview(proposal.proposal_id)
    assert {c.evidence[0].source_id for c in preview.pages[0].claims} == set(ids)
    assert (
        "Source A reports success." in preview.diff and "Source B reports failure." in preview.diff
    )


def test_adoption_is_explicit_and_book_pages_are_protected(tmp_path: Path) -> None:
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
    page = replace(value.pages[0], operation="revise", page_id="page:MANUAL0001")
    value = replace(
        value,
        pages=(page,),
        dispositions=("update",),
        impact=(
            ImpactDecision(
                page_id="page:MANUAL0001", action="revise", reason="Replace after review."
            ),
        ),
    )
    before = note.read_bytes()
    with pytest.raises(ValueError, match="adoption"):
        service.submit(wire(value))
    proposal = service.submit(wire(replace(value, pages=(replace(page, operation="adopt"),))))
    assert "-Manual content." in service.preview(proposal.proposal_id).diff
    assert note.read_bytes() == before
    books = note.parent / "books"
    books.mkdir()
    note.rename(books / "manual.md")
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    with pytest.raises(ValueError, match="Book ownership"):
        service.submit(
            wire(replace(value, task_id=task.task_id, pages=(replace(page, operation="adopt"),)))
        )


def test_disposition_must_match_material_operations(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = replace(submission(task), dispositions=("update",))
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="disposition"):
        service.submit(wire(value))
    assert snapshot(tmp_path / "wiki") == before


def test_unverified_ownership_file_cannot_enable_revise(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    path = tmp_path / "wiki" / "pages" / "manual.md"
    path.write_text(
        "---\nid: page:MANUAL0001\ntitle: Plants\nkind: concept\n"
        "created: 2026-10-22\nupdated: 2026-10-22\n"
        f"sources: [{captured.source_id}]\n---\nManual content.\n"
    )
    owned = tmp_path / "wiki" / "maintenance" / "managed"
    owned.mkdir()
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    (owned / "MANUAL0001.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        service.create_task((captured.source_id,), query="Plants")
    value = submission(task)
    value = replace(
        value,
        pages=(replace(value.pages[0], operation="revise", page_id="page:MANUAL0001"),),
        dispositions=("update",),
        impact=(ImpactDecision(page_id="page:MANUAL0001", action="revise", reason="Review."),),
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError):
        service.submit(wire(value))
    assert snapshot(tmp_path / "wiki") == before
