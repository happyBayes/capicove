"""Book-v3 operations preserve v2 packet rejection and pin source snapshots."""

from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from capicove.books import BookChunkResultService, BookService, LocalBookWorkspace
from capicove.books.chunk_config import resolve_chunk_config
from capicove.books.chunk_service import BookChunkService
from capicove.contracts.wiki_maintenance import ImpactDecision
from capicove.wiki import (
    ChunkPublicationPacket,
    ChunkPublicationPage,
    MaintenanceService,
    WikiService,
)
from capicove.wiki.maintenance.migration import migrate_v3
from tests.integration.books.test_chunk_result_publication import _submit_claim
from tests.integration.books.test_service import _Parser
from tests.wiki_maintenance_support.oracle import snapshot


def _accepted_book(tmp_path: Path) -> tuple[Any, ...]:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-authorized-result-test")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book_id = BookService(workspace).prepare(source, _Parser()).book_id
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, parsed, _ = chunks._load(book_id, run_id)
    by_chunk = {chunk.chunk_id: chunk for chunk in run.chunks}
    accepted: dict[str, dict[str, Any]] = {}
    citations: dict[str, dict[str, Any]] = {}
    for number, task in enumerate(tasks):
        dependencies = task["dependency_task_ids"]
        citation = (
            asdict(by_chunk[task["chunk_ids"][0]].core_spans[0])
            if task["task_type"] == "chunk_summary"
            else citations[dependencies[0]]
        )
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"publish-{number}",
            [accepted[identifier]["result_id"] for identifier in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 10, 23))
    migrate_v3(wiki.wiki_path, preview=False)
    return source, workspace, book_id, run_id, parsed, service, wiki


def test_chunk_v3_uses_public_adapter_and_records_verified_pdf(tmp_path: Path) -> None:
    source, workspace, book_id, run_id, parsed, service, wiki = _accepted_book(tmp_path)
    manifest = service.publication_manifest(book_id, run_id)
    bare_packet = ChunkPublicationPacket(
        manifest["publication_id"],
        book_id,
        run_id,
        tuple(manifest["result_ids"]),
        parsed.source_sha256,
        parsed.representation_sha256,
        2,
        1,
        source,
        parsed,
        (ChunkPublicationPage("book", "Overview", ()),),
    )
    before = snapshot(wiki.wiki_path)
    with pytest.raises(Exception, match="unsupported Wiki schema version"):
        wiki.publish_chunk_results(bare_packet, published_on=date(2026, 10, 23))
    assert snapshot(wiki.wiki_path) == before
    published = service.publish(book_id, run_id, wiki, published_on=date(2026, 10, 23))
    assert published["wiki_schema_version"] == 3
    assert published["publication_id"].startswith("publication:")
    repeated = service.publish(book_id, run_id, wiki, published_on=date(2026, 10, 23))
    assert repeated["publication_id"] == published["publication_id"]
    assert wiki.query("Cited fact").hits
    assert MaintenanceService(workspace.root).status()["wiki_revision"] == 2
    assert (
        wiki.wiki_path / "maintenance/book-snapshots" / (parsed.source_sha256 + ".json")
    ).is_file()
    from capicove.wiki.maintenance.compatibility import evidence_records

    commitments = evidence_records(wiki.wiki_path)
    assert (
        "maintenance/book-snapshots/"
        + parsed.source_sha256
        + "-"
        + parsed.representation_sha256
        + ".json"
        in commitments
    )
    maintenance = MaintenanceService(workspace.root)
    task = maintenance.create_task(("source:sha256:" + parsed.source_sha256,), query="Cited fact")
    assert task.evidence and all(alias.span.kind == "pdf_block_span" for alias in task.evidence)
    assert task.evidence[0].text == parsed.pages[0].blocks[0].exact_text
    registry = wiki.wiki_path / "maintenance/book-snapshots" / (parsed.source_sha256 + ".json")
    registry.write_bytes(
        registry.read_bytes().replace(b'"schema_version":1', b'"schema_version":2')
    )
    with pytest.raises(ValueError, match=r"digest|committed"):
        maintenance.load_task(task.task_id)


def test_pdf_task_proposal_preview_and_review_cite_the_committed_snapshot(
    tmp_path: Path,
) -> None:
    from capicove.contracts.wiki_maintenance import ReviewDecision
    from capicove.wiki.maintenance.review import ReviewAuthority
    from tests.unit.wiki.maintenance.test_submissions import submission, wire

    _, workspace, book_id, run_id, parsed, service, wiki = _accepted_book(tmp_path)
    service.publish(book_id, run_id, wiki, published_on=date(2026, 10, 23))
    maintenance = MaintenanceService(workspace.root)
    task = maintenance.create_task(("source:sha256:" + parsed.source_sha256,), query="Cited fact")
    pages_before = snapshot(wiki.wiki_path / "pages")
    raw_before = (wiki.wiki_path / "raw" / (parsed.source_sha256 + ".pdf")).read_bytes()
    value = submission(task)
    claim = replace(
        value.pages[0].claims[0],
        text=task.evidence[0].text,
        evidence_aliases=(task.evidence[0].alias,),
    )
    value = replace(
        value,
        pages=(replace(value.pages[0], title="Cited fact", claims=(claim,)),),
        impact=tuple(
            ImpactDecision(page_id=page_id, action="unchanged", reason="Book page kept.")
            for page_id in task.affected_page_ids
        ),
    )
    proposal = maintenance.submit(wire(value))
    preview = maintenance.preview(proposal.proposal_id)
    cited = preview.pages[0].claims[0].evidence[0]
    assert cited.kind == "pdf_block_span" and cited == task.evidence[0].span
    assert task.evidence[0].text in preview.diff
    decisions = tuple(
        ReviewDecision(subject_id=item, verdict="supported", reason="Checked PDF block.")
        for item in preview.review_subjects
    )
    review = maintenance.review(
        proposal.proposal_id,
        decisions,
        authority=ReviewAuthority("operator:test", frozenset({"wiki:review"})),
        accept=True,
    )
    assert review.accepted and review.rendered_sha256 == proposal.rendered_sha256
    assert snapshot(wiki.wiki_path / "pages") == pages_before
    assert (wiki.wiki_path / "raw" / (parsed.source_sha256 + ".pdf")).read_bytes() == raw_before
