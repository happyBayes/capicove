"""Application service for the complete local book workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.books.models import StructureConfidence, StructureReviewTask
from capicove.books.review import StructureReviewSubmission, apply_structure_reviews
from capicove.books.structure import (
    build_book_structure,
)
from capicove.books.structure import (
    structure_review_tasks as create_structure_review_tasks,
)
from capicove.books.tasks import (
    SummarySubmission,
    SummaryTask,
    create_summary_tasks,
    validate_summary_submission,
)
from capicove.books.workspace import (
    LocalBookWorkspace,
    StructureReviewVersion,
    SubmissionVersion,
)
from capicove.contracts import (
    DocumentPageRange,
    DocumentPageStatus,
    DocumentParseRequest,
    DocumentParseResult,
    LocalPathSource,
)
from capicove.documents import DocumentParserBackend


if TYPE_CHECKING:
    from capicove.wiki.models import BookWikiPublishResult


@dataclass(frozen=True, slots=True)
class BookPreparationResult:
    """Describe one idempotent prepared book workflow."""

    book_id: str
    source_sha256: str
    representation_sha256: str
    sections: int
    summary_tasks: int
    structure_review_tasks: int
    created: bool


@dataclass(frozen=True, slots=True)
class BookStatus:
    """Summarize durable task progress and publication readiness."""

    book_id: str
    total_tasks: int
    completed_tasks: int
    pending_tasks: int
    structure_review_tasks: int
    publishable: bool


def _validate_parser_result(
    result: DocumentParseResult,
    *,
    source_sha256: str,
    profile_id: str,
    profile_version: int,
    page_range: DocumentPageRange | None,
) -> None:
    """Reject fresh or cached evidence with a different request identity."""
    if result.source_sha256 != source_sha256:
        raise ValueError("parser result source identity does not match the preserved PDF")
    if result.receipt.profile_id != profile_id or result.receipt.profile_version != profile_version:
        raise ValueError("parser result profile does not match the requested profile")
    if page_range is not None and tuple(page.page_index for page in result.pages) != tuple(
        range(page_range.start_page_index, page_range.end_page_index)
    ):
        raise ValueError("parser result does not cover the requested page range")


class BookService:
    """Coordinate local parsing, task exchange, and durable progress."""

    def __init__(self, workspace: LocalBookWorkspace) -> None:
        self._workspace = workspace

    def prepare(
        self,
        source: Path,
        parser: DocumentParserBackend,
        *,
        profile_id: str = "book-balanced",
        profile_version: int = 2,
        language_hints: tuple[str, ...] = (),
        page_range: DocumentPageRange | None = None,
        evidence_only: bool = False,
    ) -> BookPreparationResult:
        """Preserve, parse, structure, and task one authorized local PDF."""
        if source.is_symlink() or not source.is_file():
            raise ValueError("book source must be a regular PDF")
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("profile_id must be a nonblank string")
        if (
            isinstance(profile_version, bool)
            or not isinstance(profile_version, int)
            or profile_version < 1
        ):
            raise ValueError("profile_version must be a positive integer")
        if not isinstance(evidence_only, bool):
            raise ValueError("evidence_only must be boolean")
        if (
            not isinstance(language_hints, tuple)
            or len(language_hints) > 16
            or any(
                not isinstance(hint, str) or not hint.strip() or len(hint) > 64
                for hint in language_hints
            )
            or len(set(language_hints)) != len(language_hints)
        ):
            raise ValueError("language_hints must contain at most 16 unique bounded values")
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        parser_id = parser.parser_id
        identity = f"{source_sha256}|{parser_id}|{profile_id}|{profile_version}"
        if page_range is not None:
            identity += f"|pages:{page_range.start_page_index}:{page_range.end_page_index}"
        if language_hints:
            identity += "|language-hints:" + json.dumps(
                language_hints, ensure_ascii=True, separators=(",", ":")
            )
        if evidence_only:
            identity += "|evidence-only:v1"
        book_id = f"book:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        preserved = self._workspace.preserve_source(source, expected_sha256=source_sha256)
        request = DocumentParseRequest(
            LocalPathSource(str(preserved)),
            source_sha256,
            "application/pdf",
            profile_id,
            profile_version,
            language_hints=language_hints,
            page_range=page_range,
        )
        if self._workspace.has_book(book_id):
            manifest = self._workspace.read_manifest(book_id)
            cached_evidence_only = self._workspace.is_evidence_only(book_id)
            if cached_evidence_only is not evidence_only:
                raise ValueError("cached book preparation policy does not match the request")
            result = self._workspace.read_result(book_id)
            _validate_parser_result(
                result,
                source_sha256=source_sha256,
                profile_id=profile_id,
                profile_version=profile_version,
                page_range=page_range,
            )
            if (
                manifest["source_sha256"] != result.source_sha256
                or manifest["representation_sha256"] != result.representation_sha256
            ):
                raise ValueError("cached book manifest does not match its parser result")
            if self._workspace.has_unverified_reviewed_projection(book_id):
                review_tasks = self._workspace.read_structure_tasks(book_id)
                decisions = tuple(
                    version.submission
                    for version in self._workspace.latest_structure_submissions(book_id).values()
                )
                reviewed = apply_structure_reviews(
                    self._workspace.read_structure(book_id, original=True),
                    result,
                    review_tasks,
                    decisions,
                    require_complete=True,
                )
                reviewed_tasks = () if evidence_only else create_summary_tasks(reviewed, result)
                self._workspace.save_reviewed(book_id, reviewed, reviewed_tasks)
            structure = self._workspace.read_structure(book_id)
            if (
                structure.profile_version != profile_version
                or structure.source_sha256 != result.source_sha256
                or structure.representation_sha256 != result.representation_sha256
            ):
                raise ValueError("cached book structure does not match its parser result")
            tasks = self._workspace.read_tasks(book_id)
            return BookPreparationResult(
                book_id,
                source_sha256,
                str(manifest["representation_sha256"]),
                len(structure.sections),
                len(tasks),
                len(self.structure_tasks(book_id)),
                False,
            )
        result = parser.parse(request)
        _validate_parser_result(
            result,
            source_sha256=source_sha256,
            profile_id=profile_id,
            profile_version=profile_version,
            page_range=page_range,
        )
        structure = build_book_structure(result, profile_version=profile_version)
        reviews = create_structure_review_tasks(structure, result)
        summary_tasks = (
            ()
            if evidence_only
            or any(
                section.confidence is not StructureConfidence.HIGH for section in structure.sections
            )
            else create_summary_tasks(structure, result)
        )
        created = self._workspace.save_prepared(
            book_id,
            result,
            structure,
            summary_tasks,
            reviews,
            evidence_only=evidence_only,
        )
        return BookPreparationResult(
            book_id,
            source_sha256,
            result.representation_sha256,
            len(structure.sections),
            len(summary_tasks),
            len(reviews),
            created,
        )

    def tasks(self, book_id: str) -> tuple[SummaryTask, ...]:
        """Return immutable summary tasks in bottom-up dependency order."""
        return self._workspace.read_tasks(book_id)

    def structure_tasks(self, book_id: str) -> tuple[StructureReviewTask, ...]:
        """Return bounded low-confidence structure review tasks."""
        if self._workspace.has_reviewed(book_id):
            return ()
        completed = frozenset(self._workspace.latest_structure_submissions(book_id))
        tasks = self._workspace.read_structure_tasks(book_id)
        pending = tuple(task for task in tasks if task.task_id not in completed)
        # An interrupted finalization (or a legacy invalid decision) must be retryable.
        return pending or tasks

    def next_task(self, book_id: str) -> SummaryTask | None:
        """Return the first dependency-ready incomplete task."""
        completed = frozenset(self._workspace.latest_submissions(book_id))
        for task in self.tasks(book_id):
            if task.task_id in completed:
                continue
            if all(item in completed for item in task.dependency_task_ids):
                return task
        return None

    def submit(self, book_id: str, submission: SummarySubmission) -> SubmissionVersion:
        """Validate then append one caller-owned summary version."""
        tasks = {task.task_id: task for task in self.tasks(book_id)}
        task = tasks.get(submission.task_id)
        if task is None:
            raise ValueError("submission references an unknown summary task")
        completed = frozenset(self._workspace.latest_submissions(book_id))
        validate_summary_submission(task, submission, completed_task_ids=completed)
        return self._workspace.append_submission(book_id, submission)

    def submit_structure(
        self,
        book_id: str,
        submission: StructureReviewSubmission,
    ) -> StructureReviewVersion:
        """Append one local correction and finalize structure when all are complete."""
        if self.tasks(book_id) or self._workspace.has_reviewed(book_id):
            raise ValueError("structure review is already finalized")
        all_tasks = self._workspace.read_structure_tasks(book_id)
        if submission.task_id not in {task.task_id for task in all_tasks}:
            raise ValueError("submission references an unknown structure review task")
        previous = self._workspace.latest_structure_submissions(book_id)
        decisions = {key: value.submission for key, value in previous.items()}
        decisions[submission.task_id] = submission
        complete = len(decisions) == len(all_tasks)
        result = self._workspace.read_result(book_id)
        reviewed = apply_structure_reviews(
            self._workspace.read_structure(book_id, original=True),
            result,
            all_tasks,
            tuple(decisions.values()),
            require_complete=complete,
        )
        tasks = (
            create_summary_tasks(reviewed, result)
            if complete and not self._workspace.is_evidence_only(book_id)
            else ()
        )
        existing = previous.get(submission.task_id)
        version = (
            existing
            if existing is not None and existing.submission == submission
            else self._workspace.append_structure_submission(book_id, submission)
        )
        if complete:
            self._workspace.save_reviewed(book_id, reviewed, tasks)
        return version

    def status(self, book_id: str) -> BookStatus:
        """Return durable task counts and strict publication readiness."""
        tasks = self.tasks(book_id)
        completed = frozenset(self._workspace.latest_submissions(book_id))
        reviews = len(self.structure_tasks(book_id))
        completed_count = sum(task.task_id in completed for task in tasks)
        pages_succeeded = all(
            page.status is DocumentPageStatus.SUCCEEDED
            for page in self._workspace.read_result(book_id).pages
        )
        return BookStatus(
            book_id,
            len(tasks),
            completed_count,
            len(tasks) - completed_count,
            reviews,
            bool(tasks) and completed_count == len(tasks) and reviews == 0 and pages_succeeded,
        )

    def publish(self, book_id: str, *, published_on: date) -> BookWikiPublishResult:
        """Create a strict validated Wiki-v2 projection without invoking a model."""
        status = self.status(book_id)
        if not status.publishable:
            raise ValueError("book is not publishable until all tasks and reviews are complete")
        from capicove.wiki import WikiService, migrate_to_v2, publish_book_wiki

        wiki_service = WikiService(self._workspace.root)
        if not wiki_service.wiki_path.exists():
            wiki_service.initialize(created_on=published_on)
            migrate_to_v2(wiki_service.wiki_path, preview=False)
        elif wiki_service.scan().schema_version not in {2, 3}:
            raise ValueError("existing Wiki requires an explicit schema-v2 migration")
        result = self._workspace.read_result(book_id)
        source = self._workspace.root / "raw" / f"{result.source_sha256}.pdf"
        return publish_book_wiki(
            wiki_service.wiki_path,
            book_id=book_id,
            source_pdf=source,
            result=result,
            structure=self._workspace.read_structure(book_id),
            tasks=self.tasks(book_id),
            submissions=self._workspace.latest_submissions(book_id),
            published_on=published_on,
        )


__all__ = ["BookPreparationResult", "BookService", "BookStatus"]


def _manifest_count(manifest: dict[str, object], name: str) -> int:
    value = manifest[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"book manifest {name} is invalid")
    return value
