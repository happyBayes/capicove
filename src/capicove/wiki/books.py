"""Validated page-aware Wiki projection for completed book summaries."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from capicove.books import (
    AtomicClaim,
    BookStructure,
    SubmissionVersion,
    SummaryTask,
)
from capicove.contracts import DocumentBlock, DocumentPageStatus, DocumentParseResult
from capicove.documents import document_parse_result_to_json
from capicove.wiki.citations import validate_book_citations
from capicove.wiki.filesystem import atomic_write, jsonl_records, scan
from capicove.wiki.models import BookWikiPublishResult, ClaimCitation


def publish_book_wiki(
    wiki: Path,
    *,
    book_id: str,
    source_pdf: Path,
    result: DocumentParseResult,
    structure: BookStructure,
    tasks: tuple[SummaryTask, ...],
    submissions: dict[str, SubmissionVersion],
    published_on: date,
) -> BookWikiPublishResult:
    """Validate all claims, then atomically project one complete book into Wiki v2."""
    from capicove.wiki.maintenance.compatibility import execute, is_v3

    if is_v3(wiki):
        return execute(
            wiki,
            "book_section",
            lambda stage: publish_book_wiki(
                stage,
                book_id=book_id,
                source_pdf=source_pdf,
                result=result,
                structure=structure,
                tasks=tasks,
                submissions=submissions,
                published_on=published_on,
            ),
        )

    if any(page.status is not DocumentPageStatus.SUCCEEDED for page in result.pages):
        raise ValueError("failed pages prevent strict book publication")
    if scan(wiki).schema_version != 2:
        raise ValueError("Wiki v2 migration is required before book publication")
    if source_pdf.is_symlink() or not source_pdf.is_file():
        raise ValueError("book source PDF must be a regular file")
    source_bytes = source_pdf.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != result.source_sha256:
        raise ValueError("book source PDF digest does not match parsed evidence")
    tasks_by_section = {task.section_id: task for task in tasks}
    blocks = {block.block_id: block for page in result.pages for block in page.blocks}
    labels = {page.page_index: page.printed_page_label for page in result.pages}
    page_ids = {
        section.section_id: _page_id(book_id, section.section_id) for section in structure.sections
    }
    citations: list[ClaimCitation] = []
    claims: list[AtomicClaim] = []
    claim_ids: set[str] = set()
    for section in structure.sections:
        task = tasks_by_section.get(section.section_id)
        if task is None or task.task_id not in submissions:
            raise ValueError("every book section requires a completed summary submission")
        version = submissions[task.task_id]
        for claim in version.submission.claims:
            if claim.claim_id in claim_ids:
                raise ValueError("claim IDs must be unique across a published book")
            claim_ids.add(claim.claim_id)
            claims.append(claim)
            grouped: dict[int, list[DocumentBlock]] = defaultdict(list)
            for block_id in claim.citation_ids:
                block = blocks.get(block_id)
                if block is None:
                    raise ValueError("submission citation does not resolve to parsed evidence")
                grouped[block.page_index].append(block)
            for group_position, (page_index, cited_blocks) in enumerate(sorted(grouped.items())):
                block_ids = tuple(block.block_id for block in cited_blocks)
                exact = "\n".join(block.exact_text for block in cited_blocks)
                citation_identity = "|".join(
                    (claim.claim_id, str(version.version), str(page_index), *block_ids)
                )
                citations.append(
                    ClaimCitation(
                        f"citation:{hashlib.sha256(citation_identity.encode()).hexdigest()[:24]}",
                        claim.claim_id,
                        page_ids[section.section_id],
                        book_id,
                        result.source_sha256,
                        result.representation_sha256,
                        page_index,
                        labels[page_index],
                        block_ids,
                        hashlib.sha256(exact.encode("utf-8")).hexdigest(),
                        claim.quotation if group_position == 0 else None,
                        version.version,
                    )
                )
    report = validate_book_citations(tuple(citations), tuple(claims), result)
    if not report.valid:
        codes = ", ".join(item.code for item in report.findings)
        raise ValueError(f"book citation validation failed: {codes}")

    source_relative = f"raw/{result.source_sha256}.pdf"
    representation_relative = f"representations/{result.representation_sha256}.json"
    _write_content_addressed(wiki / source_relative, source_bytes)
    representation = (
        json.dumps(
            document_parse_result_to_json(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    _write_content_addressed(wiki / representation_relative, representation.encode("utf-8"))
    citations_by_claim: dict[str, list[ClaimCitation]] = defaultdict(list)
    for citation in citations:
        citations_by_claim[citation.claim_id].append(citation)
    page_paths: list[str] = []
    for section in structure.sections:
        task = tasks_by_section[section.section_id]
        version = submissions[task.task_id]
        relative = (
            f"pages/books/{book_id.replace(':', '-')}/{section.section_id.replace(':', '-')}.md"
        )
        atomic_write(
            wiki / relative,
            _render_page(
                page_id=page_ids[section.section_id],
                title=section.exact_title,
                source_sha256=result.source_sha256,
                submission=version,
                citations_by_claim=citations_by_claim,
                published_on=published_on,
            ),
        )
        page_paths.append(relative)
    _replace_book_citations(wiki, book_id, tuple(citations))
    return BookWikiPublishResult(
        book_id,
        tuple(page_paths),
        len(citations),
        representation_relative,
        source_relative,
    )


def _render_page(
    *,
    page_id: str,
    title: str,
    source_sha256: str,
    submission: SubmissionVersion,
    citations_by_claim: dict[str, list[ClaimCitation]],
    published_on: date,
) -> str:
    value = submission.submission
    lines = [
        "---",
        f"id: {page_id}",
        f"title: {json.dumps(title, ensure_ascii=False)}",
        "kind: summary",
        f"created: {published_on.isoformat()}",
        f"updated: {published_on.isoformat()}",
        f'sources: ["source:sha256:{source_sha256}"]',
        "---",
        "",
        f"# {title}",
        "",
        value.abstract,
        "",
        "## Key points",
        "",
        *(f"- {point}" for point in value.key_points),
        "",
        "## Grounded claims",
        "",
    ]
    for claim in value.claims:
        links = " ".join(
            _citation_link(citation) for citation in citations_by_claim[claim.claim_id]
        )
        lines.append(f"- {claim.text} {links}")
    return "\n".join(lines) + "\n"


def _citation_link(citation: ClaimCitation) -> str:
    visible = (
        f"Printed page {citation.printed_page_label} · PDF page index {citation.page_index}"
        if citation.printed_page_label is not None
        else f"PDF page index {citation.page_index}"
    )
    return f"[{visible}](../../../raw/{citation.source_sha256}.pdf#page={citation.page_index + 1})"


def _replace_book_citations(
    wiki: Path,
    book_id: str,
    citations: tuple[ClaimCitation, ...],
) -> None:
    path = wiki / "graph" / "citations.jsonl"
    retained = [record for record in jsonl_records(path) if record.get("book_id") != book_id]
    records = retained + [_citation_json(item) for item in citations]
    records.sort(key=lambda item: str(item.get("citation_id", "")))
    content = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for item in records
    )
    atomic_write(path, content)


def _citation_json(value: ClaimCitation) -> dict[str, object]:
    return {
        "citation_id": value.citation_id,
        "claim_id": value.claim_id,
        "page_id": value.page_id,
        "book_id": value.book_id,
        "source_sha256": value.source_sha256,
        "representation_sha256": value.representation_sha256,
        "page_index": value.page_index,
        "printed_page_label": value.printed_page_label,
        "block_ids": list(value.block_ids),
        "evidence_text_sha256": value.evidence_text_sha256,
        "quotation": value.quotation,
        "submission_version": value.submission_version,
    }


def _write_content_addressed(path: Path, content: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError("content-addressed Wiki artifact disagrees with durable bytes")
        return
    atomic_write(path, content)


def _page_id(book_id: str, section_id: str) -> str:
    digest = hashlib.sha256(f"{book_id}|{section_id}".encode()).hexdigest().upper()
    return f"page:{digest[:26]}"


__all__ = ["publish_book_wiki"]
