from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from capicove.books import (
    AtomicClaim,
    StructureReviewSubmission,
    StructureReviewTask,
    SummaryEvidence,
    SummarySubmission,
    SummaryTask,
)
from capicove.books.codec import (
    structure_review_submission_to_json,
    summary_submission_to_json,
)
from capicove.contracts import InvocationContext, PrivacyClass, ToolStatus
from capicove.tools.book import BOOK_TOOL_SPECS, BookToolHandler


class _Service:
    def __init__(self) -> None:
        evidence = SummaryEvidence("p0000:b000001", 0, "1", "text", "Evidence", "a" * 64)
        self.summary_task = SummaryTask(
            "summary:" + "1" * 24,
            "section:one",
            "Summarize",
            (evidence,),
            (evidence.citation_id,),
            (),
            1,
            "b" * 64,
        )
        self.review_task = StructureReviewTask(
            "structure-review:" + "2" * 24,
            "section:one",
            ("p0000:b000000",),
            ("p0000:b000000",),
            (),
            ("accept",),
        )
        self.return_structure = False
        self.return_no_summary = False
        self.prepared_profile_version: int | None = None
        self.prepared_language_hints: tuple[str, ...] | None = None

    def structure_tasks(self, book_id: str) -> tuple[StructureReviewTask, ...]:
        del book_id
        return (self.review_task,) if self.return_structure else ()

    def next_task(self, book_id: str) -> SummaryTask | None:
        del book_id
        if self.return_no_summary:
            return None
        return self.summary_task

    def status(self, book_id: str) -> object:
        return SimpleNamespace(
            book_id=book_id,
            completed_tasks=1,
            pending_tasks=0,
            publishable=True,
            structure_review_tasks=0,
            total_tasks=1,
        )

    def submit(self, book_id: str, submission: SummarySubmission) -> object:
        del book_id
        return SimpleNamespace(task_id=submission.task_id, version=1, previous_version=None)

    def submit_structure(self, book_id: str, submission: StructureReviewSubmission) -> object:
        del book_id
        return SimpleNamespace(task_id=submission.task_id, version=2, previous_version=1)

    def publish(self, book_id: str, *, published_on: object) -> object:
        del published_on
        return SimpleNamespace(
            book_id=book_id,
            citations=1,
            page_paths=("pages/book.md",),
            representation_path="representations/one.json",
            source_path="raw/one.pdf",
        )

    def prepare(
        self,
        source: Path,
        parser: object,
        *,
        profile_id: str,
        profile_version: int,
        language_hints: tuple[str, ...],
    ) -> object:
        del source, parser, profile_id
        self.prepared_profile_version = profile_version
        self.prepared_language_hints = language_hints
        return SimpleNamespace(
            book_id="book:" + "1" * 24,
            created=True,
            representation_sha256="b" * 64,
            sections=1,
            source_sha256="a" * 64,
            structure_review_tasks=0,
            summary_tasks=1,
        )


def _context(operation: str) -> InvocationContext:
    spec = BOOK_TOOL_SPECS[operation]
    return InvocationContext(
        "request",
        "trace",
        "workspace",
        "agent",
        "local",
        datetime.now(UTC) + timedelta(minutes=1),
        spec.required_permissions,
        PrivacyClass.CONFIDENTIAL,
    )


def test_next_submit_and_publish_tools_share_canonical_service() -> None:
    service = _Service()

    def factory(workspace: Path) -> _Service:
        del workspace
        return service

    book_id = "book:" + "1" * 24

    next_result = BookToolHandler("next-task", service_factory=factory)(
        _context("next-task"), {"workspace": "/books", "book_id": book_id}
    )
    assert next_result.status is ToolStatus.SUCCEEDED
    assert next_result.result is not None
    task_value = next_result.result["task"]
    assert isinstance(task_value, Mapping)
    assert task_value["task_type"] == "summary"

    submission = SummarySubmission(
        service.summary_task.task_id,
        service.summary_task.idempotency_key,
        "Abstract",
        ("Point",),
        (AtomicClaim("claim:one", "Fact", ("p0000:b000001",)),),
    )
    submit_result = BookToolHandler("submit", service_factory=factory)(
        _context("submit"),
        {
            "workspace": "/books",
            "book_id": book_id,
            "submission": summary_submission_to_json(submission),
        },
    )
    assert submit_result.status is ToolStatus.SUCCEEDED
    assert submit_result.result is not None
    assert submit_result.result["version"] == 1

    publish_result = BookToolHandler("publish", service_factory=factory)(
        _context("publish"),
        {"workspace": "/books", "book_id": book_id, "date": "2026-09-02"},
    )
    assert publish_result.status is ToolStatus.SUCCEEDED
    assert publish_result.result is not None
    assert publish_result.result["citations"] == 1


def test_next_task_returns_structure_review_before_summary() -> None:
    service = _Service()
    service.return_structure = True

    result = BookToolHandler("next-task", service_factory=lambda workspace: service)(
        _context("next-task"),
        {"workspace": "/books", "book_id": "book:" + "1" * 24},
    )

    assert result.result is not None
    task_value = result.result["task"]
    assert isinstance(task_value, Mapping)
    assert task_value["task_type"] == "structure_review"


def test_prepare_tool_uses_explicit_local_parser_configuration(
    monkeypatch: MonkeyPatch,
) -> None:
    service = _Service()

    class _Plugin:
        def availability(self) -> object:
            return SimpleNamespace(available=True, detail=None)

        def create(self, configuration: Mapping[str, object]) -> object:
            assert configuration["model_root"] == "/models"
            return object()

    monkeypatch.setattr("capicove.tools.book.load_document_parser_plugin", lambda name: _Plugin())
    result = BookToolHandler("prepare", service_factory=lambda workspace: service)(
        _context("prepare"),
        {
            "workspace": "/books",
            "source": "/books/book.pdf",
            "model_root": "/models",
            "profile_version": 7,
            "language_hints": ["ru", "en"],
        },
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["summary_tasks"] == 1
    assert service.prepared_profile_version == 7
    assert service.prepared_language_hints == ("ru", "en")

    invalid = BookToolHandler("prepare", service_factory=lambda workspace: service)(
        _context("prepare"),
        {
            "workspace": "/books",
            "source": "/books/book.pdf",
            "model_root": "/models",
            "profile_version": True,
        },
    )
    assert invalid.status is ToolStatus.FAILED
    assert invalid.error is not None
    assert invalid.error.message == "profile_version must be a positive integer"


def test_book_tool_rejects_missing_permissions_and_invalid_input() -> None:
    handler = BookToolHandler("status", service_factory=lambda workspace: _Service())
    denied = handler(
        InvocationContext(
            "request",
            "trace",
            "workspace",
            "agent",
            "local",
            datetime.now(UTC) + timedelta(minutes=1),
            frozenset(),
            PrivacyClass.CONFIDENTIAL,
        ),
        {"workspace": "/books", "book_id": "book:" + "1" * 24},
    )
    invalid = handler(_context("status"), {"workspace": "", "book_id": "bad"})

    assert denied.status is ToolStatus.FAILED
    assert invalid.status is ToolStatus.FAILED


def test_status_empty_queue_and_structure_submission_cover_workflow_edges() -> None:
    service = _Service()
    service.return_no_summary = True

    def factory(workspace: Path) -> _Service:
        del workspace
        return service

    book_id = "book:" + "1" * 24

    status = BookToolHandler("status", service_factory=factory)(
        _context("status"), {"workspace": "/books", "book_id": book_id}
    )
    next_result = BookToolHandler("next-task", service_factory=factory)(
        _context("next-task"), {"workspace": "/books", "book_id": book_id}
    )
    review = StructureReviewSubmission(service.review_task.task_id, "accept")
    submitted = BookToolHandler("submit", service_factory=factory)(
        _context("submit"),
        {
            "workspace": "/books",
            "book_id": book_id,
            "submission": structure_review_submission_to_json(review),
        },
    )
    published = BookToolHandler("publish", service_factory=factory)(
        _context("publish"), {"workspace": "/books", "book_id": book_id}
    )

    assert status.result is not None and status.result["publishable"] is True
    assert next_result.result == {"task": None}
    assert submitted.result is not None and submitted.result["previous_version"] == 1
    assert published.status is ToolStatus.SUCCEEDED


def test_unknown_book_tool_operation_is_rejected() -> None:
    try:
        BookToolHandler("unknown")
    except ValueError as error:
        assert str(error) == "unknown book Tool operation"
    else:
        raise AssertionError("unknown operation should be rejected")
