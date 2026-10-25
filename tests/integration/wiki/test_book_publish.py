from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from capicove.books import (
    AtomicClaim,
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
    SubmissionVersion,
    SummarySubmission,
    create_summary_tasks,
)
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)
from capicove.wiki import WikiService, migrate_to_v2, publish_book_wiki


def test_direct_publication_rejects_failed_pages_before_writing(tmp_path: Path) -> None:
    import hashlib

    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 3))
    migrate_to_v2(service.wiki_path, preview=False)
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-failed")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    receipt = DocumentParserReceipt("plugin", "1", "profile", 1, "c" * 64, "d" * 64, "e" * 64)
    page = DocumentPage(0, None, 100, 200, DocumentPageStatus.FAILED, (), "OCR failed")
    result = DocumentParseResult(source_digest, "b" * 64, (page,), receipt)
    structure = BookStructure(source_digest, result.representation_sha256, 1, (), ())

    with pytest.raises(ValueError, match="failed pages"):
        publish_book_wiki(
            service.wiki_path,
            book_id="book:" + "1" * 24,
            source_pdf=source,
            result=result,
            structure=structure,
            tasks=(),
            submissions={},
            published_on=date(2026, 9, 3),
        )
    assert not (service.wiki_path / "raw" / f"{source_digest}.pdf").exists()
    assert not (service.wiki_path / "representations" / f"{'b' * 64}.json").exists()


def test_v2_migration_preview_has_no_writes_and_apply_is_explicit(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 2))

    preview = migrate_to_v2(service.wiki_path, preview=True)

    assert preview.applied is False
    assert "representations" in preview.created_paths
    assert not (service.wiki_path / "representations").exists()
    assert "schema_version = 1" in (service.wiki_path / "wiki.toml").read_text()

    applied = migrate_to_v2(service.wiki_path, preview=False)

    assert applied.applied is True
    assert (service.wiki_path / "representations").is_dir()
    assert (service.wiki_path / "graph" / "citations.jsonl").is_file()
    assert "schema_version = 2" in (service.wiki_path / "wiki.toml").read_text()


def test_publish_renders_printed_label_and_zero_based_pdf_target(tmp_path: Path) -> None:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 2))
    migrate_to_v2(service.wiki_path, preview=False)
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nbook")
    import hashlib

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    block = DocumentBlock(
        "p0002:b000001",
        2,
        DocumentBlockType.TEXT,
        "Exact page evidence.",
        "Exact page evidence.",
        NormalizedBox(0.1, 0.1, 0.9, 0.2),
        0,
        0.99,
    )
    page = DocumentPage(2, "17", 100, 200, DocumentPageStatus.SUCCEEDED, (block,))
    receipt = DocumentParserReceipt("plugin", "1", "profile", 1, "c" * 64, "d" * 64, "e" * 64)
    result = DocumentParseResult(source_digest, "b" * 64, (page,), receipt)
    section = BookSection(
        "section:one",
        "Chapter One",
        "Chapter One",
        1,
        None,
        (),
        2,
        3,
        "17",
        "17",
        (block.block_id,),
        StructureMethod.OUTLINE,
        StructureConfidence.HIGH,
        ("embedded_outline",),
        (),
    )
    structure = BookStructure(
        source_digest, result.representation_sha256, 1, (section,), (section.section_id,)
    )
    task = create_summary_tasks(structure, result)[0]
    submission = SummarySubmission(
        task.task_id,
        task.idempotency_key,
        "Section abstract.",
        ("Key point",),
        (AtomicClaim("claim:one", "A grounded fact.", (block.block_id,)),),
    )

    published = publish_book_wiki(
        service.wiki_path,
        book_id="book:" + "1" * 24,
        source_pdf=source,
        result=result,
        structure=structure,
        tasks=(task,),
        submissions={task.task_id: SubmissionVersion(task.task_id, 1, None, submission)},
        published_on=date(2026, 9, 2),
    )

    page_text = (service.wiki_path / published.page_paths[0]).read_text()
    assert "Printed page 17 · PDF page index 2" in page_text
    assert "#page=3" in page_text
    assert "PDF page index 2" in page_text
    assert published.citations == 1
    assert service.lint().valid is True

    ledger = service.wiki_path / "graph" / "citations.jsonl"
    record = json.loads(ledger.read_text())
    record["evidence_text_sha256"] = "f" * 64
    ledger.write_text(json.dumps(record) + "\n")
    assert service.lint().valid is False
