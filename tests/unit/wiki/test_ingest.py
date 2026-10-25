import builtins
import errno
import hashlib
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from threading import Barrier

import pytest

import capicove.wiki.filesystem as wiki_filesystem
from capicove.wiki import WikiDependencyError, WikiError, WikiFormatError, WikiService


_UNSUPPORTED_HARD_LINK_ERRNOS = tuple(
    dict.fromkeys((errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV))
)
_PERMISSION_HARD_LINK_ERRNOS = tuple(dict.fromkeys((errno.EACCES, errno.EPERM)))


def test_ingest_preserves_raw_bytes_and_creates_a_cited_page(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "Alpha Note.md"
    source_bytes = "# Alpha Note\n\nA durable café observation.\n".encode()
    source.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()

    result = service.ingest(source, ingested_on=date(2026, 9, 1))

    assert result.created is True
    assert result.source_id == f"source:sha256:{digest}"
    assert result.page_id == f"page:{digest[:26].upper()}"
    assert result.page_path == "pages/alpha-note.md"
    assert (tmp_path / "wiki" / "raw" / f"{digest}.md").read_bytes() == source_bytes
    rendered = (tmp_path / "wiki" / result.page_path).read_text(encoding="utf-8")
    assert 'title: "Alpha Note"' in rendered
    assert f'sources: ["source:sha256:{digest}"]' in rendered
    assert rendered.endswith("# Alpha Note\n\nA durable café observation.\n")


@pytest.mark.parametrize(
    ("manifest", "message"),
    (
        ('schema_version = 3\nauthority = "standalone"\n', "unsupported Wiki schema version"),
        ('schema_version = 1\nauthority = "future"\n', "wiki.toml is invalid"),
    ),
)
def test_ingest_rejects_an_unsupported_manifest_without_changing_the_wiki(
    tmp_path: Path, manifest: str, message: str
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nMust not be ingested.\n", encoding="utf-8")
    wiki = service.wiki_path
    (wiki / "wiki.toml").write_text(manifest, encoding="utf-8")
    (wiki / "_migration" / ".create-stale").write_bytes(b"existing operational state")
    before = {
        path.relative_to(wiki): path.read_bytes() for path in wiki.rglob("*") if path.is_file()
    }

    with pytest.raises(WikiFormatError, match=message):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    after = {
        path.relative_to(wiki): path.read_bytes() for path in wiki.rglob("*") if path.is_file()
    }
    assert after == before


def test_ingest_accepts_unknown_manifest_extension_fields(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    manifest = service.wiki_path / "wiki.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + 'future_hint = "preserve"\n',
        encoding="utf-8",
    )
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nCompatible extension field.\n", encoding="utf-8")

    result = service.ingest(source, ingested_on=date(2026, 9, 1))

    assert result.created is True
    assert 'future_hint = "preserve"' in manifest.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("source_bytes", "case"),
    (
        (
            b'---\r\ntitle: "Configured title"\r\nsourceOnly: must-not-leak\r\n---\r\n\r\n'
            b"# Body heading\r\n\r\nBody.\r\n",
            "crlf",
        ),
        (
            b'---\rtitle: "Configured title"\rsourceOnly: must-not-leak\r---\r\r'
            b"# Body heading\r\rBody.\r",
            "cr",
        ),
        (
            b'\xef\xbb\xbf---\ntitle: "Configured title"\nsourceOnly: must-not-leak\n'
            b"---\n\n# Body heading\n\nBody.\n",
            "bom",
        ),
        (
            b'\xef\xbb\xbf---\r\ntitle: "Configured title"\r\n'
            b"sourceOnly: must-not-leak\r\n---\r\n\r\n"
            b"# Body heading\r\n\r\nBody.\r\n",
            "bom-crlf",
        ),
    ),
)
def test_ingest_normalizes_source_markdown_without_changing_raw_bytes(
    tmp_path: Path, source_bytes: bytes, case: str
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / f"{case}.md"
    source.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()

    result = service.ingest(source, ingested_on=date(2026, 9, 1))

    assert result.source_id == f"source:sha256:{digest}"
    assert result.page_id == f"page:{digest[:26].upper()}"
    assert result.page_path == "pages/configured-title.md"
    assert (tmp_path / "wiki" / "raw" / f"{digest}.md").read_bytes() == source_bytes
    page_bytes = (tmp_path / "wiki" / result.page_path).read_bytes()
    assert b"\r" not in page_bytes
    rendered = page_bytes.decode("utf-8")
    assert 'title: "Configured title"' in rendered
    assert "sourceOnly" not in rendered
    assert rendered.count("---\n") == 2
    assert rendered.endswith("# Body heading\n\nBody.\n")


@pytest.mark.parametrize(
    ("name", "source_bytes", "expected_title", "expected_body"),
    (
        ("empty-body", b"---\n---\n", "empty-body", ""),
        ("empty-at-eof", b"---\n---", "empty-at-eof", ""),
        ("empty-frontmatter", b"---\n---\n\n# Note\nHello\n", "Note", "# Note\nHello\n"),
        ("body-at-eof", b"---\n---\n# Note\nHello", "Note", "# Note\nHello"),
        ("closing-at-eof", b'---\ntitle: "EOF title"\n---', "EOF title", ""),
        (
            "bom-crlf-empty",
            b"\xef\xbb\xbf---\r\n---\r\n\r\n# Note\r\nBody\r\n",
            "Note",
            "# Note\nBody\n",
        ),
    ),
)
def test_ingest_accepts_empty_frontmatter_and_a_closing_delimiter_at_eof(
    tmp_path: Path,
    name: str,
    source_bytes: bytes,
    expected_title: str,
    expected_body: str,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / f"{name}.md"
    source.write_bytes(source_bytes)

    result = service.ingest(source, ingested_on=date(2026, 9, 1))

    digest = hashlib.sha256(source_bytes).hexdigest()
    assert (service.wiki_path / "raw" / f"{digest}.md").read_bytes() == source_bytes
    rendered = (service.wiki_path / result.page_path).read_text(encoding="utf-8")
    assert f'title: "{expected_title}"' in rendered
    assert rendered.endswith(f"---\n\n{expected_body}")


def test_frontmatter_closing_fix_preserves_block_scalar_chomping() -> None:
    metadata, body = wiki_filesystem._split_markdown_text(
        Path("note.md"),
        "---\ntitle: |\n  Block title\n---\nBody\n",
    )

    assert metadata == {"title": "Block title"}
    assert body == "Body\n"


def test_frontmatter_closing_delimiter_at_eof_returns_an_empty_body() -> None:
    metadata, body = wiki_filesystem._split_markdown_text(
        Path("note.md"),
        '---\ntitle: "EOF title"\n---',
    )

    assert metadata == {"title": "EOF title"}
    assert body == ""


def test_frontmatter_scanner_preserves_later_delimiters_in_the_body() -> None:
    metadata, body = wiki_filesystem._split_markdown_text(
        Path("note.md"),
        "---\ntitle: Note\n---\nBody\n---\nmore\n",
    )

    assert metadata == {"title": "Note"}
    assert body == "Body\n---\nmore\n"


@pytest.mark.parametrize("closing_line", ("----", "--- "))
def test_frontmatter_requires_an_exact_closing_delimiter(closing_line: str) -> None:
    with pytest.raises(WikiFormatError, match="frontmatter is unclosed"):
        wiki_filesystem._split_markdown_text(
            Path("note.md"),
            f"---\ntitle: Note\n{closing_line}\nBody\n",
        )


def test_ingest_strips_a_bom_before_selecting_a_heading_title(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "fallback.md"
    source_bytes = b"\xef\xbb\xbf# Heading title\r\n\r\nBody.\r\n"
    source.write_bytes(source_bytes)

    first = service.ingest(source, ingested_on=date(2026, 9, 1))
    second = service.ingest(source, ingested_on=date(2026, 9, 2))

    assert first.page_path == "pages/heading-title.md"
    assert second.created is False
    assert second == first.__class__(
        source_id=first.source_id,
        page_id=first.page_id,
        page_path=first.page_path,
        created=False,
    )
    digest = first.source_id.removeprefix("source:sha256:")
    assert (tmp_path / "wiki" / "raw" / f"{digest}.md").read_bytes() == source_bytes
    assert service.scan().pages == 1
    assert service.scan().sources == 1


def test_reingesting_identical_bytes_is_idempotent(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nSame bytes.\n", encoding="utf-8")

    first = service.ingest(source, ingested_on=date(2026, 9, 1))
    second = service.ingest(source, ingested_on=date(2026, 9, 2))

    assert first.created is True
    assert second.created is False
    assert second.source_id == first.source_id
    assert second.page_id == first.page_id
    assert second.page_path == first.page_path
    assert service.scan().pages == 1
    assert service.scan().sources == 1


def test_reingest_accepts_an_existing_page_with_additional_provenance(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First\n\nPrimary source.\n", encoding="utf-8")
    second.write_text("# Second\n\nAdditional source.\n", encoding="utf-8")
    first_result = service.ingest(first, ingested_on=date(2026, 9, 1))
    second_result = service.ingest(second, ingested_on=date(2026, 9, 1))
    page = tmp_path / "wiki" / first_result.page_path
    rendered = page.read_text(encoding="utf-8")
    rendered = rendered.replace(
        f'sources: ["{first_result.source_id}"]',
        f'sources: ["{first_result.source_id}","{second_result.source_id}"]',
    )
    page.write_text(rendered, encoding="utf-8")

    repeated = service.ingest(first, ingested_on=date(2026, 9, 2))

    assert repeated.created is False
    assert repeated.page_path == first_result.page_path


def test_reingest_restores_a_missing_raw_capture_without_duplicate_pages(
    tmp_path: Path,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source_bytes = b"# Note\n\nRecover the exact source bytes.\n"
    source.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    first = service.ingest(source, ingested_on=date(2026, 9, 1))
    raw_path = tmp_path / "wiki" / "raw" / f"{digest}.md"
    raw_path.unlink()

    recovered = service.ingest(source, ingested_on=date(2026, 9, 2))

    assert recovered.created is False
    assert recovered.page_path == first.page_path
    assert raw_path.read_bytes() == source_bytes
    assert service.scan().pages == 1
    assert service.scan().sources == 1


def test_ingest_cleans_a_stale_private_publication_file(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    stale = tmp_path / "wiki" / "_migration" / ".create-stale"
    stale.write_text("sensitive interrupted content", encoding="utf-8")
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nFresh content.\n", encoding="utf-8")

    service.ingest(source, ingested_on=date(2026, 9, 1))

    assert not stale.exists()


@pytest.mark.parametrize("unsupported_errno", _UNSUPPORTED_HARD_LINK_ERRNOS)
def test_ingest_fails_closed_when_hard_links_are_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsupported_errno: int,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source_bytes = b"# Note\n\nExact source bytes.\n"
    source.write_bytes(source_bytes)
    wiki = tmp_path / "wiki"
    log_path = wiki / "log.md"
    log_before = log_path.read_bytes()

    def reject_hard_link(_source: str, _destination: Path) -> None:
        raise OSError(unsupported_errno, os.strerror(unsupported_errno))

    monkeypatch.setattr(os, "link", reject_hard_link)

    with pytest.raises(
        WikiError,
        match="local Wiki filesystem does not support safe create-only publication with hard links",
    ):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert source.read_bytes() == source_bytes
    assert not tuple((wiki / "raw").iterdir())
    assert not tuple((wiki / "pages").iterdir())
    assert log_path.read_bytes() == log_before
    assert not tuple((wiki / "_migration").glob(".create-*"))


@pytest.mark.parametrize("permission_errno", _PERMISSION_HARD_LINK_ERRNOS)
def test_ingest_does_not_misreport_hard_link_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    permission_errno: int,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source.write_bytes(b"# Note\n\nMust remain unpublished.\n")
    wiki = tmp_path / "wiki"

    def reject_hard_link(_source: str, _destination: Path) -> None:
        raise OSError(permission_errno, os.strerror(permission_errno))

    monkeypatch.setattr(os, "link", reject_hard_link)

    with pytest.raises(WikiError, match="Wiki ingest failed") as captured:
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert "does not support safe create-only publication" not in str(captured.value)
    cause = captured.value.__cause__
    assert isinstance(cause, OSError)
    assert cause.errno == permission_errno
    assert not tuple((wiki / "raw").iterdir())
    assert not tuple((wiki / "pages").iterdir())
    assert not tuple((wiki / "_migration").glob(".create-*"))


def test_ingest_uses_a_stable_suffix_when_page_names_collide(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "note.md"
    second = second_dir / "note.md"
    first.write_text("# Shared title\n\nFirst.\n", encoding="utf-8")
    second.write_text("# Shared title\n\nSecond.\n", encoding="utf-8")

    first_result = service.ingest(first, ingested_on=date(2026, 9, 1))
    second_result = service.ingest(second, ingested_on=date(2026, 9, 1))

    second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
    assert first_result.page_path == "pages/shared-title.md"
    assert second_result.page_path == f"pages/shared-title-{second_digest[:8]}.md"


def test_ingest_rejects_a_symbolic_link_source(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    log_path = tmp_path / "wiki" / "log.md"
    log_before = log_path.read_bytes()
    target = tmp_path / "target.md"
    target.write_text("# Target\n\nMust not be followed through a link.\n", encoding="utf-8")
    source = tmp_path / "source.md"
    source.symlink_to(target)

    with pytest.raises(WikiFormatError, match="regular file"):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert service.scan().pages == 0
    assert service.scan().sources == 0
    assert log_path.read_bytes() == log_before


def test_ingest_rejects_source_replaced_by_symlink_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source.write_text("# Original\n\nAuthorized bytes.\n", encoding="utf-8")
    replacement = tmp_path / "replacement.md"
    replacement.write_text("# Replacement\n\nMust not be followed.\n", encoding="utf-8")
    log_path = tmp_path / "wiki" / "log.md"
    log_before = log_path.read_bytes()
    original_open = os.open
    replaced = False

    def replace_before_open(
        path: Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        nonlocal replaced
        if path == source and not replaced:
            source.unlink()
            source.symlink_to(replacement)
            replaced = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_open)

    with pytest.raises(WikiFormatError, match="regular file"):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert replaced is True
    assert service.scan().pages == 0
    assert service.scan().sources == 0
    assert log_path.read_bytes() == log_before


def test_ingest_rejects_source_beneath_a_symlinked_directory(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    actual = tmp_path / "actual"
    actual.mkdir()
    source = actual / "source.md"
    source.write_text("# Linked parent\n\nMust not be followed.\n", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    log_path = tmp_path / "wiki" / "log.md"
    log_before = log_path.read_bytes()

    with pytest.raises(WikiFormatError, match="symbolic link"):
        service.ingest(linked / "source.md", ingested_on=date(2026, 9, 1))

    assert service.scan().pages == 0
    assert service.scan().sources == 0
    assert log_path.read_bytes() == log_before


def test_ingest_rejects_parent_traversal_after_a_symlinked_directory(
    tmp_path: Path,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    outside = tmp_path / "outside"
    child = outside / "child"
    child.mkdir(parents=True)
    secret = outside / "secret.md"
    secret.write_text("# Secret\n\nMust not be captured through a link.\n", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.symlink_to(child, target_is_directory=True)
    crafted = linked / ".." / "secret.md"
    log_path = tmp_path / "wiki" / "log.md"
    log_before = log_path.read_bytes()

    with pytest.raises(WikiFormatError, match="parent traversal"):
        service.ingest(crafted, ingested_on=date(2026, 9, 1))

    assert service.scan().pages == 0
    assert service.scan().sources == 0
    assert log_path.read_bytes() == log_before


def test_ingest_rejects_a_non_utf8_encodable_title_before_writing(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source.write_text('---\ntitle: "\\uD800"\n---\n\nBody.\n', encoding="utf-8")
    log_path = tmp_path / "wiki" / "log.md"
    log_before = log_path.read_bytes()

    with pytest.raises(WikiFormatError, match="Unicode"):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert service.scan().pages == 0
    assert service.scan().sources == 0
    assert log_path.read_bytes() == log_before


def test_ingest_rejects_an_unreadable_log_before_publishing(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nMust remain unpublished.\n", encoding="utf-8")
    log_path = tmp_path / "wiki" / "log.md"
    log_path.write_bytes(b"\xff")

    with pytest.raises(WikiFormatError, match="Wiki log is unreadable"):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert service.scan().pages == 0
    assert service.scan().sources == 0
    assert log_path.read_bytes() == b"\xff"


def test_ingest_refuses_to_replace_a_mismatched_raw_capture(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source_bytes = b"# Intended\n\nImmutable source bytes.\n"
    source.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    raw_path = tmp_path / "wiki" / "raw" / f"{digest}.md"
    raw_path.write_bytes(b"mismatched existing bytes")
    log_path = tmp_path / "wiki" / "log.md"
    log_before = log_path.read_bytes()

    with pytest.raises(WikiFormatError, match="content-addressed source bytes disagree"):
        service.ingest(source, ingested_on=date(2026, 9, 1))

    assert raw_path.read_bytes() == b"mismatched existing bytes"
    assert service.scan().pages == 0
    assert log_path.read_bytes() == log_before


def test_recovery_does_not_replace_raw_created_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source_bytes = b"# Intended\n\nImmutable source bytes.\n"
    source.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    first = service.ingest(source, ingested_on=date(2026, 9, 1))
    raw_path = tmp_path / "wiki" / "raw" / f"{digest}.md"
    raw_path.unlink()
    original_exists = Path.exists
    injected = False

    def create_mismatch_after_missing_check(path: Path) -> bool:
        nonlocal injected
        result = original_exists(path)
        if path == raw_path and not result and not injected:
            raw_path.write_bytes(b"concurrent mismatched bytes")
            injected = True
        return result

    monkeypatch.setattr(Path, "exists", create_mismatch_after_missing_check)

    with pytest.raises(WikiFormatError, match="content-addressed source bytes disagree"):
        service.ingest(source, ingested_on=date(2026, 9, 2))

    assert injected is True
    assert raw_path.read_bytes() == b"concurrent mismatched bytes"
    assert service.scan().pages == 1
    assert (tmp_path / "wiki" / first.page_path).is_file()


def test_concurrent_ingests_with_same_title_preserve_both_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "note.md"
    second = second_dir / "note.md"
    first.write_text("# Shared title\n\nFirst body.\n", encoding="utf-8")
    second.write_text("# Shared title\n\nSecond body.\n", encoding="utf-8")
    barrier = Barrier(2)
    original_render_page = wiki_filesystem._render_page

    def render_together(
        *,
        page_id: str,
        title: str,
        source_id: str,
        body: str,
        page_date: date,
    ) -> str:
        rendered = original_render_page(
            page_id=page_id,
            title=title,
            source_id=source_id,
            body=body,
            page_date=page_date,
        )
        barrier.wait(timeout=5)
        return rendered

    monkeypatch.setattr(wiki_filesystem, "_render_page", render_together)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda source: service.ingest(source, ingested_on=date(2026, 9, 1)),
                (first, second),
            )
        )

    assert all(result.created for result in results)
    assert len({result.page_path for result in results}) == 2
    assert service.scan().pages == 2
    assert service.scan().sources == 2
    for result in results:
        rendered = (tmp_path / "wiki" / result.page_path).read_text(encoding="utf-8")
        assert result.source_id in rendered
    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    for result in results:
        assert log.count(f"ingest | {result.page_path}") == 1


def test_ingest_bounds_a_long_title_without_losing_the_full_title(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    title = "a" * 300
    source = tmp_path / "source.md"
    source.write_text(f"# {title}\n\nLong titles remain valid content.\n", encoding="utf-8")

    result = service.ingest(source, ingested_on=date(2026, 9, 1))
    repeated = service.ingest(source, ingested_on=date(2026, 9, 2))
    second = tmp_path / "second.md"
    second.write_text(f"# {title}\n\nDifferent content.\n", encoding="utf-8")
    second_result = service.ingest(second, ingested_on=date(2026, 9, 2))

    page_name = Path(result.page_path).name
    assert len(page_name.encode("utf-8")) <= 255
    assert len(Path(second_result.page_path).name.encode("utf-8")) <= 255
    assert second_result.page_path != result.page_path
    assert result.page_path == repeated.page_path
    assert repeated.created is False
    rendered = (tmp_path / "wiki" / result.page_path).read_text(encoding="utf-8")
    assert f'title: "{title}"' in rendered
    assert service.lint().valid is True


def test_collision_suffix_keeps_page_stem_within_200_characters(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    title = "a" * 200
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text(f"# {title}\n\nFirst content.\n", encoding="utf-8")
    second.write_text(f"# {title}\n\nSecond content.\n", encoding="utf-8")

    first_result = service.ingest(first, ingested_on=date(2026, 9, 1))
    second_result = service.ingest(second, ingested_on=date(2026, 9, 1))

    assert first_result.page_path != second_result.page_path
    assert len(Path(first_result.page_path).stem) <= 200
    assert len(Path(second_result.page_path).stem) <= 200


def test_ingest_extends_the_digest_suffix_without_overwriting_incumbents(
    tmp_path: Path,
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source.write_text("# Shared title\n\nNew content.\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    pages = tmp_path / "wiki" / "pages"
    base = pages / "shared-title.md"
    short = pages / f"shared-title-{digest[:8]}.md"
    base.write_text("incumbent base", encoding="utf-8")
    short.write_text("incumbent short digest", encoding="utf-8")

    result = service.ingest(source, ingested_on=date(2026, 9, 1))

    assert result.page_path == f"pages/shared-title-{digest[:12]}.md"
    assert base.read_text(encoding="utf-8") == "incumbent base"
    assert short.read_text(encoding="utf-8") == "incumbent short digest"


def test_ingest_retry_repairs_log_after_post_page_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nRecover the audit entry.\n", encoding="utf-8")

    def fail_log(_wiki: Path, _entry: str) -> None:
        raise OSError("injected log failure")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(wiki_filesystem, "_append_log", fail_log)
        with pytest.raises(WikiError, match="Wiki ingest failed"):
            service.ingest(source, ingested_on=date(2026, 9, 1))

    recovered = service.ingest(source, ingested_on=date(2026, 9, 2))

    assert recovered.created is False
    assert service.scan().pages == 1
    assert service.scan().sources == 1
    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    assert log.count(f"ingest | {recovered.page_path}") == 1


def test_ingest_retry_does_not_duplicate_an_ambiguously_committed_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nKeep one audit entry.\n", encoding="utf-8")
    original_append_log = wiki_filesystem._append_log

    def append_then_fail(wiki: Path, entry: str) -> None:
        original_append_log(wiki, entry)
        raise OSError("ambiguous log result")

    with monkeypatch.context() as scoped_patch:
        scoped_patch.setattr(wiki_filesystem, "_append_log", append_then_fail)
        with pytest.raises(WikiError, match="Wiki ingest failed"):
            service.ingest(source, ingested_on=date(2026, 9, 1))

    recovered = service.ingest(source, ingested_on=date(2026, 9, 2))

    assert recovered.created is False
    log = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    assert log.count(f"ingest | {recovered.page_path}") == 1
    assert log.count(recovered.source_id) == 1


@pytest.mark.parametrize("title", ("CON", "prn", "AUX", "nul", "COM1", "LPT9"))
def test_ingest_avoids_windows_reserved_page_names(tmp_path: Path, title: str) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "source.md"
    source.write_text(f"# {title}\n\nPortable page name.\n", encoding="utf-8")

    result = service.ingest(source, ingested_on=date(2026, 9, 1))

    assert Path(result.page_path).stem.lower() not in {
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "lpt9",
    }


def test_reingest_propagates_missing_yaml_support_without_creating_a_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    source = tmp_path / "note.md"
    source.write_text("# Note\n\nSame bytes.\n", encoding="utf-8")
    first = service.ingest(source, ingested_on=date(2026, 9, 1))
    original_import = builtins.__import__

    def import_without_yaml(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "yaml":
            raise ImportError("simulated missing PyYAML")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)

    with pytest.raises(WikiDependencyError, match=r"capicove\[wiki\]"):
        service.ingest(source, ingested_on=date(2026, 9, 2))

    assert service.scan().pages == 1
    assert service.scan().sources == 1
    assert (tmp_path / "wiki" / first.page_path).is_file()


def test_build_propagates_missing_yaml_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    original_import = builtins.__import__

    def import_without_yaml(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "yaml":
            raise ImportError("simulated missing PyYAML")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_yaml)

    with pytest.raises(WikiDependencyError, match=r"capicove\[wiki\]"):
        service.build(built_on=date(2026, 9, 1))
