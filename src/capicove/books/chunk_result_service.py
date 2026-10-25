"""Result-task projection for immutable Book chunk runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
from typing import Any

from capicove.books.chunk_config import integer_limit
from capicove.books.chunk_models import BookChunk, ChunkRun, SourceSpan, digest
from capicove.books.chunk_result_ledger import ChunkResultLedger
from capicove.books.chunk_result_submission import canonical_json, validate_submission
from capicove.books.chunk_service import BookChunkService
from capicove.books.chunk_tasks import reference_tasks
from capicove.books.workspace import LocalBookWorkspace
from capicove.contracts import DocumentBlock, DocumentPageStatus, DocumentParseResult


def _task_states(
    tasks: tuple[dict[str, Any], ...], current: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Project current readiness in the reference DAG's bottom-up order."""
    states: dict[str, str] = {}
    for task in tasks:
        task_id = task["task_id"]
        dependencies = task["dependency_task_ids"]
        record = current.get(task_id)
        dependencies_complete = all(states[identifier] == "complete" for identifier in dependencies)
        if record is None:
            state = "ready" if dependencies_complete else "blocked"
        else:
            actual = record["submission"]["dependency_result_ids"]
            expected = [
                current[identifier]["result_id"]
                for identifier in dependencies
                if identifier in current
            ]
            state = "complete" if dependencies_complete and actual == expected else "stale"
        states[task_id] = state
    return states


def _validate_citations(
    submission: dict[str, Any],
    task: dict[str, Any],
    dependencies: tuple[dict[str, Any], ...],
    chunks: Mapping[str, BookChunk],
    blocks: Mapping[str, DocumentBlock],
) -> None:
    """Verify each claimed source span against the task's exact citation scope."""
    if task["task_type"] == "chunk_summary":
        core_spans = chunks[task["chunk_ids"][0]].core_spans
        allowed_citations = None
    else:
        core_spans = ()
        allowed_citations = {
            canonical_json(citation)
            for record in dependencies
            for claim in record["submission"]["claims"]
            for citation in claim["citations"]
        }
    for claim in submission["claims"]:
        for raw_citation in claim["citations"]:
            citation = SourceSpan(**raw_citation)
            if allowed_citations is None:
                allowed = any(
                    citation.block_id == core.block_id
                    and citation.page_index == core.page_index
                    and core.start_char <= citation.start_char
                    and citation.end_char <= core.end_char
                    for core in core_spans
                )
            else:
                allowed = canonical_json(raw_citation) in allowed_citations
            if not allowed:
                raise ValueError("citation_invalid")
            block = blocks.get(citation.block_id)
            if block is None:
                raise ValueError("citation_invalid")
            try:
                citation.extract(block)
            except ValueError as error:
                raise ValueError("citation_invalid") from error


def _validate_complete_citations(
    tasks: tuple[dict[str, Any], ...],
    current: dict[str, dict[str, Any]],
    run: ChunkRun,
    parsed: DocumentParseResult,
) -> None:
    """Recheck every accepted citation against the current source and DAG."""
    chunks = {chunk.chunk_id: chunk for chunk in run.chunks}
    blocks = {block.block_id: block for page in parsed.pages for block in page.blocks}
    for task in tasks:
        dependencies = tuple(current[identifier] for identifier in task["dependency_task_ids"])
        _validate_citations(
            current[task["task_id"]]["submission"], task, dependencies, chunks, blocks
        )


def _validate_against_run(
    submission: dict[str, Any],
    *,
    book_id: str,
    run_id: str,
    task_rows: tuple[dict[str, Any], ...],
    run: ChunkRun,
    parsed: DocumentParseResult,
    current: dict[str, dict[str, Any]],
) -> None:
    """Check task readiness and exact source citations before accepting a result."""
    if submission["book_id"] != book_id or submission["run_id"] != run_id:
        raise ValueError("invalid_submission")
    tasks = {row["task_id"]: row for row in task_rows}
    task = tasks.get(submission["task_id"])
    if task is None:
        raise ValueError("invalid_submission")
    dependencies = [current.get(identifier) for identifier in task["dependency_task_ids"]]
    if any(record is None for record in dependencies):
        raise ValueError("dependency_incomplete")
    states = _task_states(task_rows, current)
    if any(states[identifier] != "complete" for identifier in task["dependency_task_ids"]):
        raise ValueError("dependency_stale")
    verified_dependencies = tuple(record for record in dependencies if record is not None)
    current_ids = [record["result_id"] for record in verified_dependencies]
    if submission["dependency_result_ids"] != current_ids:
        raise ValueError("dependency_stale")
    chunks = {chunk.chunk_id: chunk for chunk in run.chunks}
    blocks = {block.block_id: block for page in parsed.pages for block in page.blocks}
    ancestors = set(task["dependency_task_ids"])
    for prior in reversed(task_rows):
        if prior["task_id"] in ancestors:
            ancestors.update(prior["dependency_task_ids"])
    for prior in task_rows:
        if prior["task_id"] in ancestors:
            _validate_citations(
                current[prior["task_id"]]["submission"],
                prior,
                tuple(current[identifier] for identifier in prior["dependency_task_ids"]),
                chunks,
                blocks,
            )
    _validate_citations(submission, task, verified_dependencies, chunks, blocks)


class BookChunkResultService:
    """Read accepted result tasks without changing legacy reference tasks."""

    def __init__(self, workspace: LocalBookWorkspace) -> None:
        self._chunks = BookChunkService(workspace)

    def publication_manifest(self, book_id: str, run_id: str) -> dict[str, Any]:
        """Pin one complete current result DAG to immutable source evidence."""
        run, result, structure = self._chunks._load(book_id, run_id)
        tasks = reference_tasks(run, structure)
        if not tasks or any(
            page.status is not DocumentPageStatus.SUCCEEDED for page in result.pages
        ):
            raise ValueError("publication_incomplete")
        current = ChunkResultLedger(
            self._chunks._directory(book_id, run_id), run_id, book_id=book_id
        ).current_results()
        states = _task_states(tasks, current)
        if any(state != "complete" for state in states.values()):
            raise ValueError("publication_incomplete")
        _validate_complete_citations(tasks, current, run, result)
        manifest = {
            "book_id": book_id,
            "run_id": run_id,
            "result_ids": [current[task["task_id"]]["result_id"] for task in tasks],
            "source_sha256": result.source_sha256,
            "representation_sha256": result.representation_sha256,
            "wiki_schema_version": 2,
            "render_policy_version": 1,
        }
        return {"publication_id": "chunk-publication:" + digest(manifest)[:24], **manifest}

    def publish(
        self, book_id: str, run_id: str, wiki_service: Any, *, published_on: date
    ) -> dict[str, Any]:
        """Delegate one complete, verified result set to the public Wiki publisher."""
        from capicove.wiki import (
            ChunkPublicationCitation,
            ChunkPublicationClaim,
            ChunkPublicationPacket,
            ChunkPublicationPage,
            WikiService,
        )

        if not isinstance(wiki_service, WikiService):
            raise TypeError("wiki_service must be a WikiService")
        manifest = self.publication_manifest(book_id, run_id)
        run, parsed, structure = self._chunks._load(book_id, run_id)
        tasks = reference_tasks(run, structure)
        current = ChunkResultLedger(
            self._chunks._directory(book_id, run_id), run_id, book_id=book_id
        ).current_results()
        if [current[task["task_id"]]["result_id"] for task in tasks] != manifest["result_ids"]:
            raise ValueError("publication_incomplete")
        sections = {section.section_id: section for section in structure.sections}
        pages = []
        for task in tasks:
            if task["task_type"] not in ("section_summary", "book_summary"):
                continue
            record = current[task["task_id"]]
            title = (
                sections[task["parent_id"]].exact_title
                if task["task_type"] == "section_summary"
                else f"Book overview {book_id}"
            )
            claims = tuple(
                ChunkPublicationClaim(
                    claim["text"],
                    tuple(ChunkPublicationCitation(**citation) for citation in claim["citations"]),
                    record["result_id"],
                    record["version"],
                )
                for claim in record["submission"]["claims"]
            )
            pages.append(ChunkPublicationPage(task["parent_id"], title, claims))
        packet = ChunkPublicationPacket(
            manifest["publication_id"],
            book_id,
            run_id,
            tuple(manifest["result_ids"]),
            manifest["source_sha256"],
            manifest["representation_sha256"],
            manifest["wiki_schema_version"],
            manifest["render_policy_version"],
            self._chunks.workspace.root / "raw" / f"{parsed.source_sha256}.pdf",
            parsed,
            tuple(pages),
        )
        if wiki_service.scan().schema_version == 3:
            from capicove.wiki import ChunkPublicationPacketV3, MaintenanceService

            current_wiki = MaintenanceService(wiki_service.wiki_path.parent).status()
            return wiki_service.publish_chunk_results_v3(
                ChunkPublicationPacketV3(
                    current_wiki["wiki_revision"], current_wiki["policy_revision"], packet
                ),
                published_on=published_on,
            )
        return wiki_service.publish_chunk_results(packet, published_on=published_on)

    def submit(self, value: object) -> dict[str, Any]:
        """Accept one cited result after validating immutable source evidence."""
        submission = validate_submission(value)
        book_id = submission["book_id"]
        run_id = submission["run_id"]
        run, result, structure = self._chunks._load(book_id, run_id)
        task_rows = reference_tasks(run, structure)
        directory = self._chunks._directory(book_id, run_id)
        ledger = ChunkResultLedger(directory, run_id, book_id=book_id)

        def validate_candidate(
            candidate: dict[str, Any], accepted: dict[str, dict[str, Any]]
        ) -> None:
            _validate_against_run(
                candidate,
                book_id=book_id,
                run_id=run_id,
                task_rows=task_rows,
                run=run,
                parsed=result,
                current=accepted,
            )

        ledger.recover_pending(validate_orphan=validate_candidate)
        prior = ledger.find_idempotent(submission)
        if prior is not None:
            return prior
        current = ledger.current_results()
        validate_candidate(submission, current)
        return ledger.append(submission, validate_orphan=validate_candidate)

    def status(
        self, book_id: str, run_id: str, *, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Report bounded current readiness without changing accepted history."""
        integer_limit("offset", offset, minimum=0, maximum=1000000)
        integer_limit("limit", limit, minimum=1, maximum=50)
        run, parsed, structure = self._chunks._load(book_id, run_id)
        tasks = reference_tasks(run, structure)
        current = ChunkResultLedger(
            self._chunks._directory(book_id, run_id), run_id, book_id=book_id
        ).current_results()
        states = _task_states(tasks, current)
        publishable = (
            bool(tasks)
            and all(state == "complete" for state in states.values())
            and all(page.status is DocumentPageStatus.SUCCEEDED for page in parsed.pages)
        )
        if publishable:
            _validate_complete_citations(tasks, current, run, parsed)
        items: list[dict[str, Any]] = []
        for task in tasks:
            task_id = task["task_id"]
            record = current.get(task_id)
            items.append(
                {
                    "task_id": task_id,
                    "state": states[task_id],
                    "current_result_id": record["result_id"] if record is not None else None,
                    "version": record["version"] if record is not None else None,
                }
            )
        page: list[dict[str, Any]] = []
        used = 0
        for item in items[offset : offset + limit]:
            size = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
            if used + size > 1024 * 1024:
                if not page:
                    raise ValueError("one result status exceeds the response byte limit")
                break
            page.append(item)
            used += size
        next_offset = offset + len(page)
        return {
            "book_id": book_id,
            "run_id": run_id,
            "items": page,
            "total": len(items),
            "next_offset": next_offset if next_offset < len(items) else None,
            "publishable": publishable,
        }

    def list_tasks(
        self, book_id: str, run_id: str, *, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """Return a bounded task page scoped to one verified chunk run."""
        integer_limit("offset", offset, minimum=0, maximum=1000000)
        integer_limit("limit", limit, minimum=1, maximum=50)
        run, _, structure = self._chunks._load(book_id, run_id)
        chunks = {chunk.chunk_id: chunk for chunk in run.chunks}
        tasks = reference_tasks(run, structure)
        current = ChunkResultLedger(
            self._chunks._directory(book_id, run_id), run_id, book_id=book_id
        ).current_results()
        rows: list[dict[str, Any]] = []
        used = 0
        for index in range(offset, min(offset + limit, len(tasks))):
            task = tasks[index]
            leaf = task["task_type"] == "chunk_summary"
            if leaf:
                citation_scope = [asdict(span) for span in chunks[task["chunk_ids"][0]].core_spans]
            else:
                citation_scope = [
                    citation
                    for identifier in task["dependency_task_ids"]
                    if identifier in current
                    for claim in current[identifier]["submission"]["claims"]
                    for citation in claim["citations"]
                ]
            row = {
                "schema_version": 1,
                "book_id": book_id,
                "run_id": run_id,
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "parent_id": task["parent_id"],
                "dependency_task_ids": task["dependency_task_ids"],
                "chunk_ids": task["chunk_ids"],
                "citation_policy": "chunk_core" if leaf else "dependency_claims",
                "citation_scope_sha256": digest(citation_scope),
                "max_result_bytes": 1024 * 1024,
                "submission_contract": "chunk-result-v1",
            }
            size = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
            if used + size > 1024 * 1024:
                if not rows:
                    raise ValueError("one result task exceeds the response byte limit")
                break
            rows.append(row)
            used += size
        next_offset = offset + len(rows)
        return {
            "book_id": book_id,
            "run_id": run_id,
            "items": rows,
            "total": len(tasks),
            "next_offset": next_offset if next_offset < len(tasks) else None,
        }
