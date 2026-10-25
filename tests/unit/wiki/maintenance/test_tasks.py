"""Immutable task snapshots and supersession behavior."""

from pathlib import Path

import pytest

from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.wiki_maintenance_support.oracle import snapshot


def test_task_pins_exact_evidence_and_preserves_pages(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_bytes("甲😀e\u0301\r\n乙".encode())
    captured = service.capture(source)
    service.build_discovery()
    pages = snapshot(tmp_path / "wiki" / "pages")
    task = service.create_task((captured.source_id,), query="植物")
    assert task.workspace_id == "workspace:test"
    assert task.wiki_revision == 0 and task.policy_revision == 0
    assert task.evidence[0].text == "甲😀e\u0301\n乙"
    assert task.evidence[0].span.start_char == 0
    assert task.evidence[0].span.end_char == 6
    assert task.sources[0].representation_sha256 == captured.representation_sha256
    assert service.load_task(task.task_id) == task
    assert snapshot(tmp_path / "wiki" / "pages") == pages
    next_task = service.create_task(
        (captured.source_id,), query="乙", previous_task_id=task.task_id
    )
    assert next_task.task_version == 2
    assert next_task.previous_task_id == task.task_id
    with pytest.raises(ValueError, match="superseded"):
        service.load_task(task.task_id)


def test_task_overflow_rejects_before_persisting(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("x" * 262145)
    captured = service.capture(source)
    service.build_discovery()
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match=r"capacity|context"):
        service.create_task((captured.source_id,), query="x")
    assert snapshot(tmp_path / "wiki") == before


def test_explicit_context_expansion_preserves_codepoint_scope(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("abcdef")
    captured = service.capture(source)
    service.build_discovery()
    first = service.create_task(
        (captured.source_id,), query="x", intervals=((captured.source_id, 1, 3),)
    )
    assert first.evidence[0].text == "bc"
    second = service.create_task(
        (captured.source_id,),
        query="x",
        intervals=((captured.source_id, 0, 6),),
        previous_task_id=first.task_id,
    )
    assert second.evidence[0].text == "abcdef"
    with pytest.raises(ValueError, match="superseded"):
        service.load_task(first.task_id)
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="scope"):
        service.create_task(
            (captured.source_id,), query="x", intervals=(("source:sha256:" + "0" * 64, 0, 2),)
        )
    assert snapshot(tmp_path / "wiki") == before


@pytest.mark.parametrize("change", ["task", "source", "representation", "state"])
def test_changed_pinned_dependencies_fail_closed(tmp_path: Path, change: str) -> None:
    import json

    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("Evidence")
    captured = service.capture(source)
    service.build_discovery()
    task = service.create_task((captured.source_id,), query="Evidence")
    wiki = tmp_path / "wiki"
    if change == "task":
        path = wiki / "maintenance" / "tasks" / (task.task_id[5:] + ".json")
        value = json.loads(path.read_bytes())
        value["task"]["payload"]["evidence"][0]["text"] = "Spoofed"
        path.write_text(json.dumps(value))
    elif change == "source":
        (wiki / "raw" / (captured.source_sha256 + ".md")).write_bytes(b"Replaced")
    elif change == "representation":
        (wiki / "representations" / (captured.representation_sha256 + ".json")).write_bytes(b"bad")
    else:
        path = wiki / "maintenance" / "state.json"
        value = json.loads(path.read_bytes())
        value["wiki_revision"] = 1
        path.write_text(json.dumps(value))
    before = snapshot(wiki)
    with pytest.raises(ValueError):
        service.load_task(task.task_id)
    assert snapshot(wiki) == before


def test_candidate_context_contains_bounded_page_content(tmp_path: Path) -> None:
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
    assert task.candidates[0].title == "Plants"
    assert "Existing explanation." in task.candidates[0].body
    assert task.candidates[0].context_complete


def test_managed_task_exposes_claim_identity_and_impact_scope(tmp_path: Path) -> None:
    from tests.integration.wiki.test_maintenance_publication import prepared

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    source_id = "source:sha256:" + next((service.wiki / "raw").glob("*.md")).stem
    task = service.create_task((source_id,), query="Plants")
    assert task.candidates[0].claim_context[0].claim_id.startswith("claim:")
    assert task.affected_page_ids == (task.candidates[0].page_id,)
    assert task.deferred_page_count == 0
