"""Discovery uses current bytes and never mutates published pages."""

from pathlib import Path

import pytest

from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.wiki_maintenance_support.oracle import snapshot


def test_discovery_service_detects_manual_edits_and_denies_unknown_sources(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text("植物利用光能")
    captured = service.capture(source)
    path = tmp_path / "wiki" / "pages" / "plant.md"
    path.write_text(
        "---\nid: page:PLANT0001\ntitle: 植物\nkind: concept\n"
        "created: 2026-10-22\nupdated: 2026-10-22\n"
        f"sources: [{captured.source_id}]\naliases: [光合作用]\n---\n植物利用光能。\n"
    )
    before = snapshot(tmp_path / "wiki" / "pages")
    service.build_discovery()
    assert service.discover("光能").candidates[0].page_id == "page:PLANT0001"
    assert snapshot(tmp_path / "wiki" / "pages") == before
    with pytest.raises(ValueError, match="scope"):
        service.discover("光能", source_ids=("source:sha256:" + "0" * 64,))
    path.write_text(path.read_text().replace("光能。", "光能与水。"))
    with pytest.raises(ValueError, match="stale"):
        service.discover("光能")
