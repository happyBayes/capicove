"""Explicit migration preserves original evidence and old Wiki operations."""

from datetime import date
from pathlib import Path

from capicove.wiki import WikiService
from tests.wiki_maintenance_support.oracle import snapshot


def test_explicit_migration_preserves_pages_and_sources(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.migration import migrate_v3
    from capicove.wiki.maintenance.service import MaintenanceService

    wiki = WikiService(tmp_path)
    wiki.initialize(created_on=date(2026, 10, 23))
    source = tmp_path / "original.md"
    source.write_bytes(b"# Original\r\n\r\nSource evidence.\r\n")
    wiki.ingest(source, ingested_on=date(2026, 10, 23))
    before = snapshot(wiki.wiki_path)
    plan = migrate_v3(wiki.wiki_path, preview=True)
    assert plan.to_version == 3
    assert snapshot(wiki.wiki_path) == before
    migrate_v3(wiki.wiki_path, preview=False)
    after = snapshot(wiki.wiki_path)
    for path, value in before.items():
        if path.startswith(("raw/", "pages/")):
            assert after[path] == value
    assert wiki.scan().schema_version == 3
    assert wiki.lint().valid
    wiki.build(built_on=date(2026, 10, 23))
    assert wiki.query("evidence").hits
    revision = MaintenanceService(tmp_path).status()["wiki_revision"]
    source.write_text("# Added\n\nAdditional evidence.")
    wiki.ingest(source, ingested_on=date(2026, 10, 23))
    assert MaintenanceService(tmp_path).status()["wiki_revision"] == revision + 1


def test_migration_accepts_compact_toml_and_ingest_rebuilds_search(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.migration import migrate_v3

    wiki = WikiService(tmp_path)
    wiki.initialize(created_on=date(2026, 10, 23))
    manifest = wiki.wiki_path / "wiki.toml"
    manifest.write_text(manifest.read_text().replace("schema_version = 1", "schema_version=1"))
    migrate_v3(wiki.wiki_path, preview=False)
    source = tmp_path / "new.md"
    source.write_text("# Rebuilt\n\nuniquesearchtoken")
    wiki.ingest(source, ingested_on=date(2026, 10, 23))
    assert wiki.query("uniquesearchtoken").hits


def test_migrated_wiki_is_ready_for_tasks_without_private_rebuilds(tmp_path: Path) -> None:
    from capicove.wiki.maintenance.migration import migrate_v3
    from capicove.wiki.maintenance.service import MaintenanceService

    wiki = WikiService(tmp_path)
    wiki.initialize(created_on=date(2026, 10, 23))
    plan = migrate_v3(wiki.wiki_path, preview=True)
    assert "_derived/maintenance.sqlite" in plan.created_paths
    migrate_v3(wiki.wiki_path, preview=False)
    source = tmp_path / "a.md"
    source.write_text("Plants use light.")
    service = MaintenanceService(tmp_path, source_roots=(tmp_path,))
    captured = service.capture(source)
    task = service.create_task((captured.source_id,), query="Plants")
    assert task.evidence[0].text == "Plants use light."
