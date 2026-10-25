import argparse
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

import capicove.wiki.service as wiki_service
from capicove.cli.commands.wiki import run as run_wiki
from capicove.wiki import (
    WikiError,
    WikiFormatError,
    WikiIndexUnavailableError,
    WikiLintReport,
    WikiService,
    WikiValidationError,
)
from capicove.wiki._adapters import sqlite_index
from capicove.wiki.filesystem import atomic_write, wiki_write_lock
from capicove.wiki.indexing import WikiDocument, render_index


def _source(path: Path, title: str, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def test_build_writes_a_sorted_index_and_searches_cited_pages(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    beta = service.ingest(
        _source(tmp_path / "beta.md", "Beta", "Oranges grow in a sheltered grove."),
        ingested_on=date(2026, 9, 1),
    )
    alpha = service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Apples grow in a local orchard."),
        ingested_on=date(2026, 9, 1),
    )

    result = service.build(built_on=date(2026, 9, 2))
    search = service.query("local orchard", top_k=5)

    assert result.pages == 2
    assert result.index_path == "index.md"
    assert result.search_index_path == "_derived/search.sqlite3"
    assert (tmp_path / "wiki" / "_derived" / "search.sqlite3").is_file()
    assert (tmp_path / "wiki" / "index.md").read_text(encoding="utf-8") == (
        "# Wiki Index\n\n"
        "> Last built: 2026-09-02 | Total pages: 2\n\n"
        f"- [Alpha](pages/alpha.md) — concept — `{alpha.page_id}`\n"
        f"- [Beta](pages/beta.md) — concept — `{beta.page_id}`\n"
    )
    assert search.query == "local orchard"
    assert len(search.hits) == 1
    hit = search.hits[0]
    assert hit.page_id == alpha.page_id
    assert hit.page_path == "pages/alpha.md"
    assert hit.title == "Alpha"
    assert hit.kind == "concept"
    assert "local orchard" in hit.snippet
    assert hit.source_ids == (alpha.source_id,)
    assert hit.score == 1.0


def test_chunk_publication_pending_marker_blocks_query_until_cleared(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Apples grow in an orchard."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 2))
    pending = service.wiki_path / "_migration" / "chunk-publication.pending"
    atomic_write(pending, "Unfinished chunk publication.\n")

    with pytest.raises(WikiIndexUnavailableError, match="chunk publication is incomplete"):
        service.query("orchard")

    pending.unlink()
    assert len(service.query("orchard").hits) == 1


def test_build_url_encodes_nested_page_paths_in_the_human_index(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    result = service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Nested page content."),
        ingested_on=date(2026, 9, 1),
    )
    raw_page_path = "pages/research #1/(draft)/literal%2F/研究/alpha.md"
    nested_page = service.wiki_path / raw_page_path
    nested_page.parent.mkdir(parents=True)
    (service.wiki_path / result.page_path).rename(nested_page)

    assert service.lint().valid is True
    service.build(built_on=date(2026, 9, 2))

    expected_path = "pages/research%20%231/%28draft%29/literal%252F/%E7%A0%94%E7%A9%B6/alpha.md"
    assert f"- [Alpha]({expected_path}) — concept — `{result.page_id}`" in (
        service.wiki_path / "index.md"
    ).read_text(encoding="utf-8")
    assert service.query("Nested").hits[0].page_path == raw_page_path


def test_render_index_url_encodes_question_marks_without_using_a_real_path() -> None:
    document = WikiDocument(
        page_id="page:ALPHA000001",
        page_path="pages/question?/alpha.md",
        title="Alpha",
        kind="concept",
        body="",
        source_ids=(),
    )

    rendered = render_index((document,), built_on=date(2026, 9, 2))

    assert "- [Alpha](pages/question%3F/alpha.md)" in rendered


@pytest.mark.parametrize("operation", ("build", "query"))
def test_locked_operations_surface_invalid_manifest_as_a_format_error(
    tmp_path: Path, operation: str
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    (service.wiki_path / "wiki.toml").write_text(
        'schema_version = 3\nauthority = "standalone"\n',
        encoding="utf-8",
    )

    with pytest.raises(WikiFormatError, match="unsupported Wiki schema version"):
        if operation == "build":
            service.build(built_on=date(2026, 9, 2))
        else:
            service.query("anything")


def test_build_refuses_invalid_pages_without_creating_an_index(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    (tmp_path / "wiki" / "pages" / "broken.md").write_text(
        "---\nid: [\n---\nBroken.\n", encoding="utf-8"
    )

    with pytest.raises(WikiValidationError) as raised:
        service.build(built_on=date(2026, 9, 2))

    assert raised.value.report.valid is False
    assert not (tmp_path / "wiki" / "_derived" / "search.sqlite3").exists()


def test_edited_surrogate_title_blocks_build_and_reports_unhealthy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    result = service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Searchable content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    page = service.wiki_path / result.page_path
    page.write_text(
        page.read_text(encoding="utf-8").replace('title: "Alpha"', 'title: "\\uD800"'),
        encoding="utf-8",
    )
    before = {
        path.relative_to(service.wiki_path): path.read_bytes()
        for path in service.wiki_path.rglob("*")
        if path.is_file()
    }

    code = run_wiki(argparse.Namespace(workspace=tmp_path, wiki_command="doctor"))
    report = json.loads(capsys.readouterr().out)

    assert code == 1
    assert report["healthy"] is False
    assert report["index_healthy"] is False
    assert report["lint"]["findings"][0]["code"] == "page_title_invalid"
    with pytest.raises(WikiValidationError):
        service.build(built_on=date(2026, 9, 2))
    after = {
        path.relative_to(service.wiki_path): path.read_bytes()
        for path in service.wiki_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_index_encoding_failure_preserves_index_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    database = tmp_path / "search.sqlite3"
    sqlite_index.rebuild_index(database, ())
    original = database.read_bytes()
    document = WikiDocument("page:INVALIDTITLE01", "pages/invalid.md", "\ud800", "concept", "", ())

    with pytest.raises(WikiIndexUnavailableError, match="build failed"):
        sqlite_index.rebuild_index(database, (document,))

    assert database.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["search.sqlite3"]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, RuntimeError])
def test_index_interruption_preserves_index_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    database = tmp_path / "search.sqlite3"
    sqlite_index.rebuild_index(database, ())
    original = database.read_bytes()

    def interrupt_fingerprint(documents: tuple[WikiDocument, ...]) -> str:
        raise error_type("injected build interruption")

    monkeypatch.setattr(sqlite_index, "_content_fingerprint", interrupt_fingerprint)

    with pytest.raises(error_type, match="injected build interruption"):
        sqlite_index.rebuild_index(database, ())

    assert database.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["search.sqlite3"]


def test_rebuild_replaces_the_search_projection_with_current_pages(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "first.md", "First", "Mercury is visible."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "second.md", "Second", "Venus is visible."),
        ingested_on=date(2026, 9, 2),
    )

    service.build(built_on=date(2026, 9, 2))

    assert len(service.query("Mercury", top_k=10).hits) == 1
    assert len(service.query("Venus", top_k=10).hits) == 1


@pytest.mark.parametrize("failed_output", ["index.md", "log"])
def test_interrupted_build_fails_closed_until_successful_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failed_output: str,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Alpha content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    old_catalog = (service.wiki_path / "index.md").read_bytes()
    service.ingest(
        _source(tmp_path / "beta.md", "Beta", "Beta content."),
        ingested_on=date(2026, 9, 2),
    )

    def fail_catalog(path: Path, content: str | bytes) -> None:
        if path.name == "index.md":
            raise OSError("injected catalog publication failure")
        atomic_write(path, content)

    def fail_log(wiki: Path, entry: str) -> None:
        raise OSError("injected build log failure")

    with monkeypatch.context() as scoped_patch:
        if failed_output == "index.md":
            scoped_patch.setattr(wiki_service, "atomic_write", fail_catalog)
        else:
            scoped_patch.setattr(wiki_service, "append_log", fail_log)
        with pytest.raises(WikiError, match="Wiki build failed"):
            service.build(built_on=date(2026, 9, 2))

    if failed_output == "index.md":
        assert (service.wiki_path / "index.md").read_bytes() == old_catalog
    code = run_wiki(argparse.Namespace(workspace=tmp_path, wiki_command="doctor"))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["healthy"] is False
    assert report["index_healthy"] is False
    with pytest.raises(WikiIndexUnavailableError, match="incomplete"):
        service.query("Beta")

    service.build(built_on=date(2026, 9, 2))

    assert not (service.wiki_path / "_migration" / "build.pending").exists()
    assert service.index_is_healthy() is True
    assert len(service.query("Beta").hits) == 1
    assert "[Beta]" in (service.wiki_path / "index.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("reader", ["query", "health"])
def test_index_reader_holds_writer_lock_through_validation_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Alpha content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    beta = _source(tmp_path / "beta.md", "Beta", "Beta content.")
    request_write = threading.Event()
    writer_ready = threading.Event()
    writer_entered = threading.Event()
    original_check = sqlite_index.check_index
    original_search = sqlite_index.search_index

    def writer() -> None:
        writer_ready.set()
        assert request_write.wait(timeout=5)
        with wiki_write_lock(service.wiki_path):
            writer_entered.set()
        service.ingest(beta, ingested_on=date(2026, 9, 2))
        service.build(built_on=date(2026, 9, 2))

    def checked(database: Path, documents: tuple[WikiDocument, ...]) -> None:
        original_check(database, documents)
        request_write.set()
        assert not writer_entered.wait(timeout=0.1)

    def searched(
        database: Path,
        query: str,
        *,
        tokens: tuple[str, ...],
        top_k: int,
    ) -> tuple[WikiDocument, ...]:
        assert writer_entered.is_set() is False
        return original_search(database, query, tokens=tokens, top_k=top_k)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(writer)
        assert writer_ready.wait(timeout=5)
        try:
            with monkeypatch.context() as scoped_patch:
                scoped_patch.setattr(sqlite_index, "check_index", checked)
                scoped_patch.setattr(sqlite_index, "search_index", searched)
                if reader == "query":
                    assert service.query("Beta").hits == ()
                else:
                    assert service.index_is_healthy() is True
        finally:
            request_write.set()
        future.result(timeout=5)

    assert writer_entered.is_set()
    assert len(service.query("Beta").hits) == 1


def test_doctor_holds_writer_lock_through_the_complete_health_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Alpha content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    request_write = threading.Event()
    writer_ready = threading.Event()
    writer_entered = threading.Event()
    original_check = sqlite_index.check_index
    original_lint = service.lint
    original_regular_file_check = wiki_service._is_regular_file_without_following
    lint_calls = 0

    def writer() -> None:
        writer_ready.set()
        assert request_write.wait(timeout=5)
        with wiki_write_lock(service.wiki_path):
            writer_entered.set()

    def observed_lint() -> WikiLintReport:
        nonlocal lint_calls
        report = original_lint()
        lint_calls += 1
        if lint_calls == 2:
            request_write.set()
            assert not writer_entered.wait(timeout=0.1)
        return report

    def observed_regular_file_check(path: Path) -> bool:
        assert request_write.is_set()
        assert not writer_entered.wait(timeout=0.1)
        return original_regular_file_check(path)

    def checked(database: Path, documents: tuple[WikiDocument, ...]) -> None:
        original_check(database, documents)
        assert request_write.is_set()
        assert not writer_entered.wait(timeout=0.1)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(writer)
        assert writer_ready.wait(timeout=5)
        try:
            with monkeypatch.context() as scoped_patch:
                scoped_patch.setattr(service, "lint", observed_lint)
                scoped_patch.setattr(
                    wiki_service,
                    "_is_regular_file_without_following",
                    observed_regular_file_check,
                )
                scoped_patch.setattr(sqlite_index, "check_index", checked)
                report = service.doctor()
                assert report.healthy is True
                assert report.index_built is True
                assert report.index_healthy is True
                assert report.lint.valid is True
                assert lint_calls == 2
        finally:
            request_write.set()
        future.result(timeout=5)

    assert writer_entered.is_set()


def test_doctor_reports_a_symlinked_index_as_not_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Alpha content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    database = service.wiki_path / "_derived" / "search.sqlite3"
    actual = tmp_path / "actual-search.sqlite3"
    database.replace(actual)
    database.symlink_to(actual)

    code = run_wiki(argparse.Namespace(workspace=tmp_path, wiki_command="doctor"))
    report = json.loads(capsys.readouterr().out)

    assert code == 1
    assert report["healthy"] is False
    assert report["index_built"] is False
    assert report["index_healthy"] is False
    assert report["lint"]["valid"] is False


def test_doctor_reports_an_unavailable_writer_lock_as_unhealthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "alpha.md", "Alpha", "Alpha content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))

    def unavailable(_wiki: Path) -> None:
        raise WikiError("Wiki writer lock is unavailable")

    monkeypatch.setattr(wiki_service, "wiki_write_lock", unavailable)

    report = service.doctor()

    assert report.healthy is False
    assert report.index_built is True
    assert report.index_healthy is False
    assert report.lint.valid is True


def test_query_requires_a_built_index_and_valid_bounds(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))

    with pytest.raises(WikiIndexUnavailableError, match="build"):
        service.query("anything")
    with pytest.raises(ValueError, match="nonblank"):
        service.query("   ")
    with pytest.raises(ValueError, match="between 1 and 100"):
        service.query("anything", top_k=0)


def test_query_snippet_is_plain_and_bounded(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    long_body = "prefix " * 80 + "needle phrase " + "suffix " * 80
    service.ingest(
        _source(tmp_path / "long.md", "Long", long_body),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))

    hit = service.query("needle phrase", top_k=1).hits[0]

    assert len(hit.snippet) <= 240
    assert "needle phrase" in hit.snippet
    assert "<b>" not in hit.snippet


def test_query_uses_unicode61_separator_boundaries(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    for name, body in (
        ("underscore", "Adjacent foo_bar terms."),
        ("space", "Adjacent foo bar terms."),
        ("hyphen", "Adjacent foo-bar terms."),
        ("nonadjacent", "Separated foo and later bar terms."),
    ):
        service.ingest(
            _source(tmp_path / f"{name}.md", name.title(), body),
            ingested_on=date(2026, 9, 1),
        )
    service.build(built_on=date(2026, 9, 1))

    result = service.query("foo_bar", top_k=10)

    assert {hit.title for hit in result.hits} == {"Underscore", "Space", "Hyphen"}


def test_query_accepts_a_unicode61_token_outside_python_word_characters(
    tmp_path: Path,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "private-use.md", "Private Use", "Marker \ue000 token."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))

    result = service.query("\ue000")

    assert [hit.title for hit in result.hits] == ["Private Use"]


def test_query_keeps_a_mixed_unicode61_token_intact(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "mixed-token.md", "Mixed Token", "Marker a\ue000b token."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))

    result = service.query("a\ue000b")

    assert [hit.title for hit in result.hits] == ["Mixed Token"]


@pytest.mark.parametrize(
    ("query", "message"),
    (
        ("!!!", "searchable text"),
        ("_", "searchable text"),
        ("x" * 2049, "at most 2048"),
        (" ".join(["x"] * 129), "at most 128 searchable tokens"),
        ("_".join(["x"] * 129), "at most 128 searchable tokens"),
    ),
)
def test_query_rejects_invalid_text_before_opening_the_index(
    tmp_path: Path, query: str, message: str
) -> None:
    service = WikiService(tmp_path)

    with pytest.raises(ValueError, match=message):
        service.query(query)


def test_query_translates_corrupt_index_rows(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "source.md", "Source", "Searchable content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    database = tmp_path / "wiki" / "_derived" / "search.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE pages_fts SET source_ids = 'not-json'")

    with pytest.raises(WikiIndexUnavailableError, match="unreadable"):
        service.query("Searchable")


def test_query_and_health_reject_tampered_index_rows(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "source.md", "Source", "Original searchable content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    database = tmp_path / "wiki" / "_derived" / "search.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE pages_fts SET body = 'tampered searchable content'")

    assert service.index_is_healthy() is False
    with pytest.raises(WikiIndexUnavailableError, match="stale"):
        service.query("searchable")


def test_query_translates_invalid_user_edited_pages(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    result = service.ingest(
        _source(tmp_path / "source.md", "Source", "Original searchable content."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))
    page = tmp_path / "wiki" / result.page_path
    page.write_text(
        page.read_text(encoding="utf-8").replace('title: "Source"\n', ""),
        encoding="utf-8",
    )

    with pytest.raises(WikiValidationError):
        service.query("searchable")


def test_query_supports_uri_metacharacters_in_workspace_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace?local"
    workspace.mkdir()
    service = WikiService(workspace)
    service.initialize(created_on=date(2026, 9, 1))
    service.ingest(
        _source(tmp_path / "uri.md", "URI", "Question-safe search."),
        ingested_on=date(2026, 9, 1),
    )
    service.build(built_on=date(2026, 9, 1))

    assert len(service.query("Question-safe").hits) == 1


def test_rendered_index_normalizes_multiline_markdown_titles(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    page = tmp_path / "wiki" / "pages" / "multiline.md"
    page.write_text(
        "---\nid: page:MULTILINE01\ntitle: |\n  First [line]\n  second line\n"
        "kind: concept\ncreated: 2026-09-01\nupdated: 2026-09-01\nsources: []\n"
        "---\n\nBody.\n",
        encoding="utf-8",
    )

    service.build(built_on=date(2026, 9, 1))

    assert "- [First \\[line\\] second line](pages/multiline.md)" in (
        tmp_path / "wiki" / "index.md"
    ).read_text(encoding="utf-8")
