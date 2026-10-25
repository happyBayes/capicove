from __future__ import annotations

from datetime import date
from pathlib import Path

from capicove.books import (
    AtomicClaim,
    BookService,
    LocalBookWorkspace,
    SummarySubmission,
)
from capicove.contracts import (
    DocumentBlock,
    DocumentBlockType,
    DocumentPage,
    DocumentPageStatus,
    DocumentParseRequest,
    DocumentParseResult,
    DocumentParserReceipt,
    NormalizedBox,
)
from capicove.wiki import WikiService


class _DeterministicBookParser:
    parser_id = "e2e-parser"

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        pages = []
        for page_index, title in enumerate(("Chapter 1 Evidence", "Chapter 2 Workflows")):
            blocks = (
                DocumentBlock(
                    f"p{page_index:04d}:b000000",
                    page_index,
                    DocumentBlockType.DOCUMENT_TITLE,
                    title,
                    title,
                    NormalizedBox(0.1, 0.1, 0.9, 0.2),
                    0,
                    0.99,
                ),
                DocumentBlock(
                    f"p{page_index:04d}:b000001",
                    page_index,
                    DocumentBlockType.TEXT,
                    f"Grounded content for {title}.",
                    f"Grounded content for {title}.",
                    NormalizedBox(0.1, 0.3, 0.9, 0.6),
                    1,
                    0.99,
                ),
            )
            pages.append(
                DocumentPage(
                    page_index,
                    str(page_index + 1),
                    1000,
                    1400,
                    DocumentPageStatus.SUCCEEDED,
                    blocks,
                )
            )
        receipt = DocumentParserReceipt(
            "capicove.processor.e2e",
            "1",
            request.profile_id,
            request.profile_version,
            "c" * 64,
            "d" * 64,
            "e" * 64,
        )
        return DocumentParseResult(
            request.source_sha256,
            "b" * 64,
            tuple(pages),
            receipt,
        )


def test_pdf_to_agent_tasks_to_page_aware_searchable_wiki(tmp_path: Path) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nend-to-end book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    service = BookService(workspace)

    prepared = service.prepare(source, _DeterministicBookParser())
    while (task := service.next_task(prepared.book_id)) is not None:
        citation_id = task.allowed_citation_ids[-1]
        service.submit(
            prepared.book_id,
            SummarySubmission(
                task.task_id,
                task.idempotency_key,
                f"Fast-reading abstract for {task.section_id}.",
                ("Grounded key point",),
                (
                    AtomicClaim(
                        f"claim:{task.task_id.removeprefix('summary:')}",
                        "This section contains grounded content.",
                        (citation_id,),
                    ),
                ),
                "caller-agent",
                "caller-model",
            ),
        )

    published = service.publish(prepared.book_id, published_on=date(2026, 9, 2))
    wiki = WikiService(workspace.root)
    build = wiki.build(built_on=date(2026, 9, 2))
    search = wiki.query("grounded content")

    assert published.citations == 2
    assert len(published.page_paths) == 2
    assert build.pages == 2
    assert search.hits
    assert search.hits[0].source_ids == (f"source:sha256:{prepared.source_sha256}",)
