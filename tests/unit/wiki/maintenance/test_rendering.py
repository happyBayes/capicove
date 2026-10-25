"""Independent renderer golden output and unsafe prose cases."""

from dataclasses import replace
from pathlib import Path

import pytest

from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.unit.wiki.maintenance.test_submissions import submission, wire


def test_rendered_body_has_only_cited_claims_and_safe_host_links(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    proposal = service.submit(wire(submission(task)))
    preview = service.preview(proposal.proposal_id)
    body = preview.rendered[0].split("---\n", 2)[2]
    expected = (
        "\n# Plants\n\n## Overview\n\n"
        "- Plants use light. **[current; asserted]** "
        f"[evidence](../../raw/{captured.source_sha256}.md) "
        "`chars 0:17; wiki-markdown-v1`\n"
    )
    assert body == expected
    assert preview == service.preview(proposal.proposal_id)


@pytest.mark.parametrize(
    "text", ["[open](file:///secret)", "<img src=x>", "```run", "![image](https://x)"]
)
def test_active_markup_is_rejected(tmp_path: Path, text: str) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = submission(task)
    invalid = replace(value, pages=(replace(value.pages[0], title=text),))
    with pytest.raises(ValueError):
        service.submit(wire(invalid))


def test_related_navigation_uses_host_path_and_adoption_retains_created_date(
    tmp_path: Path,
) -> None:
    from capicove.contracts.wiki_maintenance import ImpactDecision, Relation

    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    captured = service.capture(source)
    note = tmp_path / "wiki" / "pages" / "manual.md"
    note.write_text(
        "---\nid: page:MANUAL0001\ntitle: Plants\nkind: concept\n"
        "created: 2020-01-02\nupdated: 2026-10-22\n"
        f"sources: [{captured.source_id}]\n---\nManual content.\n"
    )
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Plants")
    value = submission(task)
    page = replace(
        value.pages[0],
        relations=(
            Relation(kind="related", target_id="page:MANUAL0001", reason="Related source."),
        ),
    )
    value = replace(
        value,
        pages=(page,),
        impact=(
            ImpactDecision(
                page_id="page:MANUAL0001", action="unchanged", reason="No edit requested."
            ),
        ),
    )
    proposal = service.submit(wire(value))
    assert "[page:MANUAL0001](../manual.md)" in service.preview(proposal.proposal_id).rendered[0]
    adoption = replace(
        value,
        idempotency_key="adopt",
        dispositions=("update",),
        pages=(replace(page, operation="adopt", page_id="page:MANUAL0001", relations=()),),
        impact=(
            ImpactDecision(page_id="page:MANUAL0001", action="revise", reason="Reviewed adoption."),
        ),
    )
    proposal = service.submit(wire(adoption))
    assert "created: 2020-01-02\n" in service.preview(proposal.proposal_id).rendered[0]
