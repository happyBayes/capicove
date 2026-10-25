"""W3.03 writer matrix: legacy operations on v1, v2 and v3, and pending-fence refusal."""

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from capicove.wiki import WikiService
from tests.wiki_maintenance_support.oracle import snapshot


DAY = date(2026, 10, 23)


def _workspace(tmp_path: Path, version: int) -> tuple[WikiService, Path]:
    from capicove.wiki.filesystem import migrate_to_v2
    from capicove.wiki.maintenance.migration import migrate_v3

    wiki = WikiService(tmp_path)
    wiki.initialize(created_on=DAY)
    if version >= 2:
        migrate_to_v2(wiki.wiki_path, preview=False)
    if version == 3:
        migrate_v3(wiki.wiki_path, preview=False)
    source = tmp_path / "first.md"
    source.write_text("# First\n\nmatrixtokenalpha evidence.\n")
    return wiki, source


@pytest.mark.parametrize("version", [1, 2, 3])
def test_every_legacy_operation_keeps_working_on_each_schema(tmp_path: Path, version: int) -> None:
    from capicove.wiki.maintenance.service import MaintenanceService

    wiki, source = _workspace(tmp_path, version)
    assert wiki.scan().schema_version == version
    raw_before = source.read_bytes()
    wiki.ingest(source, ingested_on=DAY)
    assert wiki.lint().valid
    wiki.build(built_on=DAY)
    assert wiki.query("matrixtokenalpha").hits
    assert wiki.doctor() is not None
    assert source.read_bytes() == raw_before
    if version == 3:
        status = MaintenanceService(tmp_path).status()
        # Migration is revision 1; ingest and build each publish one system revision.
        assert status["wiki_revision"] >= 2
        assert status["publication_pending"] is False
    else:
        assert not (wiki.wiki_path / "maintenance").exists()


def _operations(wiki: WikiService, source: Path) -> dict[str, Callable[[], Any]]:
    return {
        "ingest": lambda: wiki.ingest(source, ingested_on=DAY),
        "build": lambda: wiki.build(built_on=DAY),
        "lint": wiki.lint,
        "query": lambda: wiki.query("matrixtokenalpha"),
        "doctor": wiki.doctor,
    }


@pytest.mark.parametrize("operation", ["ingest", "build", "lint", "query", "doctor"])
def test_pending_v3_publication_fences_every_legacy_operation(
    tmp_path: Path, operation: str
) -> None:
    from capicove.wiki.maintenance.migration import migrate_v3

    wiki, source = _workspace(tmp_path, 3)
    wiki.ingest(source, ingested_on=DAY)
    wiki.build(built_on=DAY)
    marker = wiki.wiki_path / "_migration" / "maintenance.pending"
    marker.parent.mkdir(exist_ok=True)
    marker.write_bytes(b"{}")
    second = tmp_path / "second.md"
    second.write_text("# Second\n\nmore evidence.\n")
    before = snapshot(wiki.wiki_path)
    with pytest.raises(Exception, match="pending"):
        _operations(wiki, second)[operation]()
    assert snapshot(wiki.wiki_path) == before
    with pytest.raises(ValueError, match="pending"):
        migrate_v3(wiki.wiki_path, preview=False)
    assert snapshot(wiki.wiki_path) == before


def test_legacy_writer_cannot_revise_a_managed_page(tmp_path: Path) -> None:
    from tests.integration.wiki.test_maintenance_publication import prepared

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    managed = next((service.wiki / "pages" / "concepts").glob("*.md"))
    wiki = WikiService(tmp_path)
    before = snapshot(service.wiki)
    from capicove.wiki.maintenance.compatibility import execute

    def rewrite(stage: Path) -> None:
        target = stage / managed.relative_to(service.wiki)
        target.write_bytes(target.read_bytes() + b"\nlegacy edit\n")

    with pytest.raises(ValueError, match="manual_edit_conflict"):
        execute(wiki.wiki_path, "ingest", rewrite)
    assert snapshot(service.wiki) == before


def test_a_legacy_publication_stales_tasks_pinned_to_the_old_revision(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.service import MaintenanceService
    from tests.unit.wiki.maintenance.test_submissions import submission, wire

    wiki, source = _workspace(tmp_path, 3)
    service = MaintenanceService(tmp_path, source_roots=(tmp_path,))
    evidence = tmp_path / "evidence.md"
    evidence.write_text("Plants use light.")
    captured = service.capture(evidence)
    task = service.create_task((captured.source_id,), query="Plants")
    revision = service.status()["wiki_revision"]
    wiki.ingest(source, ingested_on=DAY)
    assert service.status()["wiki_revision"] == revision + 1
    before = snapshot(wiki.wiki_path)
    with pytest.raises(ValueError, match=r"stale|superseded|revision"):
        service.submit(wire(submission(task)))
    assert snapshot(wiki.wiki_path) == before
