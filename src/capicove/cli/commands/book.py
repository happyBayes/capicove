"""CLI translation for the complete local book workflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, redirect_stdout
from datetime import date
from pathlib import Path

from capicove.books import (
    BookService,
    LocalBookWorkspace,
    StructureReviewVersion,
    SubmissionVersion,
)
from capicove.books.codec import (
    structure_review_submission_from_json,
    structure_review_task_to_json,
    summary_submission_from_json,
    summary_task_to_json,
)
from capicove.cli.commands import book_chunk_results, book_chunks
from capicove.contracts import DocumentPageRange
from capicove.plugins import load_document_parser_plugin


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove book` commands."""
    book = parent.add_parser("book", help="Parse and rapidly read structured PDF books.")
    commands = book.add_subparsers(dest="book_command", required=True)
    prepare = commands.add_parser("prepare", help="Parse a PDF and create bounded Agent tasks.")
    prepare.add_argument("workspace", type=Path)
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--parser", default="paddleocr")
    prepare.add_argument("--profile", default="book-balanced")
    prepare.add_argument("--profile-version", type=int, default=2)
    prepare.add_argument("--model-root", type=Path)
    prepare.add_argument("--device", default="cpu")
    prepare.add_argument(
        "--model-config", type=Path, help="Offline model factory JSON/TOML configuration."
    )
    prepare.add_argument(
        "--model-binding", default="book.parser", help="Configured document parser binding."
    )
    prepare.add_argument(
        "--language-hint",
        action="append",
        help="BCP 47 language hint; repeat for multiple languages.",
    )
    prepare.add_argument("--start-page-index", type=int, help="First zero-based PDF page index.")
    prepare.add_argument("--end-page-index", type=int, help="Exclusive zero-based PDF page end.")
    prepare.add_argument(
        "--evidence-only",
        action="store_true",
        help="Prepare OCR and structure without legacy summary tasks.",
    )

    tasks = commands.add_parser("tasks", help="Export bounded Agent-neutral tasks.")
    tasks.add_argument("workspace", type=Path)
    tasks.add_argument("book_id")
    tasks.add_argument("--format", choices=("json", "jsonl"), default="jsonl")

    submit = commands.add_parser("submit", help="Validate and append a summary submission.")
    submit.add_argument("workspace", type=Path)
    submit.add_argument("book_id")
    submit.add_argument("--input", type=Path, required=True)

    status = commands.add_parser("status", help="Report durable workflow progress.")
    status.add_argument("workspace", type=Path)
    status.add_argument("book_id")

    publish = commands.add_parser("publish", help="Publish a validated page-aware local Wiki.")
    publish.add_argument("workspace", type=Path)
    publish.add_argument("book_id")
    publish.add_argument("--date", type=date.fromisoformat, default=date.today())
    book_chunks.add_parsers(commands)
    book_chunk_results.add_parsers(commands)


def run(arguments: argparse.Namespace) -> int:
    """Execute one book workflow command and emit deterministic JSON."""
    try:
        if arguments.book_command in book_chunks.COMMANDS:
            return book_chunks.run(arguments)
        if arguments.book_command in book_chunk_results.COMMANDS:
            return book_chunk_results.run(arguments)
        if arguments.book_command == "prepare":
            page_range = None
            start = arguments.start_page_index
            end = arguments.end_page_index
            if start is not None or end is not None:
                if start is None or end is None:
                    raise ValueError("supply both --start-page-index and --end-page-index")
                page_range = DocumentPageRange(start, end)
            try:
                with ExitStack() as stack:
                    if getattr(arguments, "model_config", None) is not None:
                        from capicove.model import ModelFactory, load_configuration

                        if (
                            arguments.model_root is not None
                            or arguments.device != "cpu"
                            or arguments.parser != "paddleocr"
                        ):
                            raise ValueError("model configuration owns parser, root and device")
                        factory = stack.enter_context(
                            ModelFactory(load_configuration(arguments.model_config))
                        )
                        parser = stack.enter_context(
                            factory.document_parser(arguments.model_binding)
                        )
                    else:
                        plugin = load_document_parser_plugin(arguments.parser)
                        availability = plugin.availability()
                        if not availability.available:
                            raise RuntimeError("document parser is unavailable")
                        model_root = arguments.model_root or _default_model_root(
                            arguments.parser, arguments.profile
                        )
                        parser = plugin.create(
                            {"model_root": str(model_root), "device": arguments.device}
                        )
                    workspace = _workspace(arguments.workspace, create=True)
                    stack.enter_context(_parser_progress())
                    result = BookService(workspace).prepare(
                        arguments.source,
                        parser,
                        profile_id=arguments.profile,
                        profile_version=arguments.profile_version,
                        language_hints=tuple(arguments.language_hint or ()),
                        page_range=page_range,
                        evidence_only=arguments.evidence_only,
                    )
            except Exception:
                raise RuntimeError(
                    "book preparation failed; check local source and parser assets"
                ) from None
            _emit(
                {
                    "book_id": result.book_id,
                    "created": result.created,
                    "representation_sha256": result.representation_sha256,
                    "sections": result.sections,
                    "source_sha256": result.source_sha256,
                    "structure_review_tasks": result.structure_review_tasks,
                    "summary_tasks": result.summary_tasks,
                }
            )
            return 0
        service = BookService(_workspace(arguments.workspace, create=False))
        if arguments.book_command == "tasks":
            values = [
                {"task_type": "structure_review", **structure_review_task_to_json(task)}
                for task in service.structure_tasks(arguments.book_id)
            ]
            values.extend(
                {"task_type": "summary", **summary_task_to_json(task)}
                for task in service.tasks(arguments.book_id)
            )
            if arguments.format == "jsonl":
                for value in values:
                    _emit(value)
            else:
                print(json.dumps(values, ensure_ascii=False, sort_keys=True))
            return 0
        if arguments.book_command == "submit":
            if arguments.input.is_symlink() or not arguments.input.is_file():
                raise ValueError("submission input must be a regular JSON file")
            raw_submission = json.loads(arguments.input.read_text(encoding="utf-8"))
            version: StructureReviewVersion | SubmissionVersion
            if isinstance(raw_submission, dict) and "operation" in raw_submission:
                version = service.submit_structure(
                    arguments.book_id,
                    structure_review_submission_from_json(raw_submission),
                )
            else:
                version = service.submit(
                    arguments.book_id,
                    summary_submission_from_json(raw_submission),
                )
            _emit(
                {
                    "previous_version": version.previous_version,
                    "task_id": version.task_id,
                    "version": version.version,
                }
            )
            return 0
        if arguments.book_command == "status":
            status = service.status(arguments.book_id)
            _emit(
                {
                    "book_id": status.book_id,
                    "completed_tasks": status.completed_tasks,
                    "pending_tasks": status.pending_tasks,
                    "publishable": status.publishable,
                    "structure_review_tasks": status.structure_review_tasks,
                    "total_tasks": status.total_tasks,
                }
            )
            return 0
        if arguments.book_command == "publish":
            published = service.publish(arguments.book_id, published_on=arguments.date)
            _emit(
                {
                    "book_id": published.book_id,
                    "citations": published.citations,
                    "page_paths": list(published.page_paths),
                    "representation_path": published.representation_path,
                    "source_path": published.source_path,
                }
            )
            return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        _emit({"error": {"code": "book_operation_failed", "message": str(error)}})
        return 2
    _emit({"error": {"code": "book_invalid_command", "message": "unknown book command"}})
    return 2


def _workspace(path: Path, *, create: bool) -> LocalBookWorkspace:
    marker = path.expanduser().resolve() / "workspace.json"
    if create and not marker.exists():
        return LocalBookWorkspace.create(path)
    return LocalBookWorkspace.open(path)


@contextmanager
def _parser_progress() -> Iterator[None]:
    """Keep count-only parser progress on stderr and JSON on stdout."""
    logger = logging.getLogger("capicove_processor_paddleocr.backend")
    handler = logging.StreamHandler(sys.stderr)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with redirect_stdout(sys.stderr):
            yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _default_model_root(plugin: str, profile: str) -> Path:
    configured = os.environ.get("CAPICOVE_MODEL_HOME")
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".local/share/capicove/models"
    )
    return (root / plugin / profile).resolve()


def _emit(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


__all__ = ["add_parser", "run"]
