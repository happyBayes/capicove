from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import CaptureFixture, MonkeyPatch

import capicove.cli.commands.book as book_command
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
from capicove.cli import main
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
from capicove.plugins import PluginAvailability, PluginMetadata


class _Parser:
    parser_id = "fake"

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        pages = []
        for index, title in enumerate(("Chapter 1", "Chapter 2")):
            block = DocumentBlock(
                f"p{index:04d}:b000000",
                index,
                DocumentBlockType.DOCUMENT_TITLE,
                title,
                title,
                NormalizedBox(0.1, 0.1, 0.9, 0.2),
                0,
                0.99,
            )
            pages.append(
                DocumentPage(
                    index,
                    str(index + 1),
                    100,
                    200,
                    DocumentPageStatus.SUCCEEDED,
                    (block,),
                )
            )
        receipt = DocumentParserReceipt(
            "fake.plugin",
            "1",
            request.profile_id,
            request.profile_version,
            "c" * 64,
            "d" * 64,
            "e" * 64,
        )
        return DocumentParseResult(request.source_sha256, "b" * 64, tuple(pages), receipt)


class _Plugin:
    metadata = PluginMetadata(
        "capicove.processor.fake", "1", "document_parser", 1, 1, (), (), False
    )

    def availability(self) -> PluginAvailability:
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> _Parser:
        del configuration
        return _Parser()


@pytest.mark.parametrize(
    "failure_stage", ("load", "availability", "unavailable", "create", "parse")
)
def test_book_prepare_does_not_expose_parser_errors(
    failure_stage: str,
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    secret = "private-parser-token"  # pragma: allowlist secret

    def fail() -> None:
        raise RuntimeError(secret)

    if failure_stage == "load":
        monkeypatch.setattr(book_command, "load_document_parser_plugin", lambda name: fail())
    else:
        plugin = _Plugin()
        if failure_stage == "availability":
            monkeypatch.setattr(plugin, "availability", fail)
        elif failure_stage == "unavailable":
            monkeypatch.setattr(plugin, "availability", lambda: PluginAvailability(False, secret))
        elif failure_stage == "create":
            monkeypatch.setattr(plugin, "create", lambda configuration: fail())
        else:
            parser = _Parser()
            monkeypatch.setattr(parser, "parse", lambda request: fail())
            monkeypatch.setattr(plugin, "create", lambda configuration: parser)
        monkeypatch.setattr(book_command, "load_document_parser_plugin", lambda name: plugin)

    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nbook")

    assert main(["book", "prepare", str(tmp_path / "workspace"), str(source)]) == 2
    output = capsys.readouterr().out
    assert secret not in output
    assert json.loads(output)["error"]["code"] == "book_operation_failed"
    if failure_stage != "parse":
        assert not (tmp_path / "workspace").exists()


def test_book_cli_prepares_and_emits_agent_neutral_jsonl_tasks(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.7\nbook")
    monkeypatch.setattr(book_command, "load_document_parser_plugin", lambda name: _Plugin())

    assert main(["book", "prepare", str(tmp_path / "workspace"), str(source)]) == 0
    prepared = json.loads(capsys.readouterr().out)

    assert prepared["summary_tasks"] == 2
    assert (
        main(
            ["book", "tasks", str(tmp_path / "workspace"), prepared["book_id"], "--format", "jsonl"]
        )
        == 0
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["task_type"] for line in lines] == ["summary", "summary"]
    assert all("evidence" in line for line in lines)


def test_book_command_handles_status_publish_and_both_submission_types(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    evidence = SummaryEvidence("p0000:b000001", 0, "1", "text", "Evidence", "a" * 64)
    summary_task = SummaryTask(
        "summary:" + "1" * 24,
        "section:one",
        "Summarize",
        (evidence,),
        (evidence.citation_id,),
        (),
        1,
        "b" * 64,
    )
    review_task = StructureReviewTask(
        "structure-review:" + "2" * 24,
        "section:one",
        ("p0000:b000000",),
        ("p0000:b000000",),
        (),
        ("accept",),
    )

    class _Service:
        def structure_tasks(self, book_id: str) -> tuple[StructureReviewTask, ...]:
            del book_id
            return (review_task,)

        def tasks(self, book_id: str) -> tuple[SummaryTask, ...]:
            del book_id
            return (summary_task,)

        def submit(self, book_id: str, submission: SummarySubmission) -> object:
            del book_id
            return SimpleNamespace(task_id=submission.task_id, version=1, previous_version=None)

        def submit_structure(self, book_id: str, submission: StructureReviewSubmission) -> object:
            del book_id
            return SimpleNamespace(task_id=submission.task_id, version=2, previous_version=1)

        def status(self, book_id: str) -> object:
            return SimpleNamespace(
                book_id=book_id,
                completed_tasks=1,
                pending_tasks=0,
                publishable=True,
                structure_review_tasks=0,
                total_tasks=1,
            )

        def publish(self, book_id: str, *, published_on: date) -> object:
            assert published_on == date(2026, 9, 2)
            return SimpleNamespace(
                book_id=book_id,
                citations=1,
                page_paths=("pages/book.md",),
                representation_path="representations/book.json",
                source_path="raw/book.pdf",
            )

    service = _Service()
    monkeypatch.setattr(book_command, "_workspace", lambda path, create: object())
    monkeypatch.setattr(book_command, "BookService", lambda workspace: service)
    common = {"workspace": tmp_path, "book_id": "book:" + "1" * 24}

    assert book_command.run(Namespace(book_command="tasks", format="json", **common)) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert [item["task_type"] for item in tasks] == ["structure_review", "summary"]

    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary_submission_to_json(
                SummarySubmission(
                    summary_task.task_id,
                    summary_task.idempotency_key,
                    "Abstract",
                    ("Point",),
                    (AtomicClaim("claim:one", "Fact", (evidence.citation_id,)),),
                )
            )
        ),
        encoding="utf-8",
    )
    assert book_command.run(Namespace(book_command="submit", input=summary_path, **common)) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 1

    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(
            structure_review_submission_to_json(
                StructureReviewSubmission(review_task.task_id, "accept")
            )
        ),
        encoding="utf-8",
    )
    assert book_command.run(Namespace(book_command="submit", input=review_path, **common)) == 0
    assert json.loads(capsys.readouterr().out)["previous_version"] == 1

    assert book_command.run(Namespace(book_command="status", **common)) == 0
    assert json.loads(capsys.readouterr().out)["publishable"] is True
    assert book_command.run(Namespace(book_command="publish", date=date(2026, 9, 2), **common)) == 0
    assert json.loads(capsys.readouterr().out)["citations"] == 1


def test_book_command_reports_invalid_submission_input(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    result = book_command.run(
        Namespace(
            book_command="submit",
            workspace=tmp_path,
            book_id="book:" + "1" * 24,
            input=tmp_path / "missing.json",
        )
    )

    assert result == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "book_operation_failed"


def test_prepare_page_range_has_distinct_identity_and_keeps_original_indices(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    requests: list[DocumentParseRequest] = []

    class RangeParser(_Parser):
        def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
            requests.append(request)
            result = super().parse(request)
            if request.page_range is not None:
                return replace(result, pages=result.pages[1:2])
            return result

    plugin = _Plugin()
    monkeypatch.setattr(plugin, "create", lambda config: RangeParser())
    monkeypatch.setattr(book_command, "load_document_parser_plugin", lambda name: plugin)
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-test")
    common = ["book", "prepare", str(tmp_path / "books"), str(source)]
    assert main(common) == 0
    full = json.loads(capsys.readouterr().out)

    assert main([*common, "--start-page-index", "1", "--end-page-index", "2"]) == 0
    sample = json.loads(capsys.readouterr().out)

    assert sample["book_id"] != full["book_id"]
    assert sample["sections"] == 1
    assert requests[-1].page_range is not None
    assert requests[-1].page_range.start_page_index == 1
    assert requests[-1].page_range.end_page_index == 2

    assert main([*common, "--language-hint", "ru", "--language-hint", "en"]) == 0
    hinted = json.loads(capsys.readouterr().out)
    assert hinted["book_id"] != full["book_id"]
    assert requests[-1].language_hints == ("ru", "en")
