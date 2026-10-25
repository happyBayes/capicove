from datetime import date
from pathlib import Path

import pytest

import capicove.wiki.service as wiki_service
from capicove.wiki import WikiError, WikiService


def test_scan_reports_page_source_fact_and_identity_counts(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    wiki = tmp_path / "wiki"
    (wiki / "pages" / "alpha.md").write_text(
        "---\nid: page:ALPHA000001\ntitle: Alpha\nkind: concept\n"
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\nAlpha body.\n",
        encoding="utf-8",
    )
    (wiki / "raw" / "source.txt").write_text("source", encoding="utf-8")
    (wiki / "graph" / "facts.jsonl").write_text('{"factId":"fact:ONE"}\n', encoding="utf-8")
    (wiki / "graph" / "identity-events.jsonl").write_text(
        '{"eventId":"identity:ONE"}\n', encoding="utf-8"
    )

    inventory = service.scan()

    assert inventory.pages == 1
    assert inventory.sources == 1
    assert inventory.facts == 1
    assert inventory.identity_events == 1


def test_build_translates_output_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))

    def fail_index_write(_path: Path, _content: str | bytes) -> None:
        raise OSError("injected index write failure")

    monkeypatch.setattr(wiki_service, "atomic_write", fail_index_write)

    with pytest.raises(WikiError, match="Wiki build failed"):
        service.build(built_on=date(2026, 9, 1))


def test_query_rejects_non_utf8_encodable_text(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))

    with pytest.raises(ValueError, match="valid Unicode"):
        service.query("\ud800")
