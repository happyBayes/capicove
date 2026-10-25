"""CLI translation for standalone local Wiki operations."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from capicove.wiki import (
    WikiDependencyError,
    WikiError,
    WikiExistsError,
    WikiFormatError,
    WikiIndexUnavailableError,
    WikiNotFoundError,
    WikiService,
    WikiUnsafePathError,
    WikiValidationError,
    migrate_to_v2,
)
from capicove.wiki.models import WikiInventory, WikiLintReport, WikiSearchResult


def add_parser(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register `capicove wiki` commands."""
    wiki = parent.add_parser("wiki", help="Build and query a standalone local LLM Wiki.")
    commands = wiki.add_subparsers(dest="wiki_command", required=True)
    from capicove.cli.commands import wiki_maintenance

    wiki_maintenance.add_parser(commands)
    for name, help_text in (
        ("init", "Initialize a new standalone Wiki."),
        ("scan", "Inventory durable Wiki records."),
        ("lint", "Validate Wiki structure and content."),
        ("build", "Build the Markdown catalog and local FTS index."),
        ("doctor", "Report Wiki validation and index health."),
    ):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("workspace", type=Path)
        if name in {"init", "build"}:
            parser.add_argument("--date", type=_date, default=date.today())
    ingest = commands.add_parser("ingest", help="Ingest one UTF-8 Markdown source.")
    ingest.add_argument("workspace", type=Path)
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--date", type=_date, default=date.today())
    query = commands.add_parser("query", help="Search the built local Wiki index.")
    query.add_argument("workspace", type=Path)
    query.add_argument("query")
    query.add_argument("--top-k", type=int, default=20)
    migrate = commands.add_parser("migrate-v2", help="Preview or apply the Wiki v2 migration.")
    migrate.add_argument("workspace", type=Path)
    migrate.add_argument("--apply", action="store_true")


def run(arguments: argparse.Namespace) -> int:
    """Execute one Wiki command and emit deterministic JSON."""
    if arguments.wiki_command in {"maintain", "ask"}:
        from capicove.cli.commands import wiki_maintenance

        return wiki_maintenance.run(arguments)
    try:
        service = WikiService(arguments.workspace)
        if arguments.wiki_command == "init":
            inventory = service.initialize(created_on=arguments.date)
            _emit({**_inventory(inventory), "wiki": str(service.wiki_path)})
            return 0
        if arguments.wiki_command == "scan":
            _emit(_inventory(service.scan()))
            return 0
        if arguments.wiki_command == "ingest":
            ingest_result = service.ingest(arguments.source, ingested_on=arguments.date)
            _emit(
                {
                    "created": ingest_result.created,
                    "page_id": ingest_result.page_id,
                    "page_path": ingest_result.page_path,
                    "source_id": ingest_result.source_id,
                }
            )
            return 0
        if arguments.wiki_command == "lint":
            report = service.lint()
            _emit(_lint(report))
            return 0 if report.valid else 1
        if arguments.wiki_command == "build":
            build_result = service.build(built_on=arguments.date)
            _emit(
                {
                    "index_path": build_result.index_path,
                    "pages": build_result.pages,
                    "search_index_path": build_result.search_index_path,
                }
            )
            return 0
        if arguments.wiki_command == "query":
            search_result = service.query(arguments.query, top_k=arguments.top_k)
            _emit(_search(search_result))
            return 0
        if arguments.wiki_command == "doctor":
            result = service.doctor()
            _emit(
                {
                    "healthy": result.healthy,
                    "index_built": result.index_built,
                    "index_healthy": result.index_healthy,
                    "lint": _lint(result.lint),
                }
            )
            return 0 if result.healthy else 1
        if arguments.wiki_command == "migrate-v2":
            plan = migrate_to_v2(service.wiki_path, preview=not arguments.apply)
            _emit(
                {
                    "applied": plan.applied,
                    "created_paths": list(plan.created_paths),
                    "from_version": plan.from_version,
                    "to_version": plan.to_version,
                    "updated_paths": list(plan.updated_paths),
                }
            )
            return 0
    except (WikiError, ValueError) as error:
        _emit({"error": {"code": _error_code(error), "message": str(error)}})
        return 2
    _emit({"error": {"code": "wiki_invalid_command", "message": "unknown Wiki command"}})
    return 2


def _inventory(value: WikiInventory) -> dict[str, object]:
    return {
        "authority": value.authority.value,
        "facts": value.facts,
        "identity_events": value.identity_events,
        "pages": value.pages,
        "schema_version": value.schema_version,
        "sources": value.sources,
    }


def _lint(report: WikiLintReport) -> dict[str, object]:
    return {
        "findings": [
            {
                "code": item.code,
                "message": item.message,
                "path": item.path,
                "severity": item.severity.value,
            }
            for item in report.findings
        ],
        "valid": report.valid,
    }


def _search(result: WikiSearchResult) -> dict[str, object]:
    return {
        "hits": [
            {
                "kind": hit.kind,
                "page_id": hit.page_id,
                "page_path": hit.page_path,
                "score": hit.score,
                "snippet": hit.snippet,
                "source_ids": list(hit.source_ids),
                "title": hit.title,
            }
            for hit in result.hits
        ],
        "query": result.query,
    }


def _error_code(error: Exception) -> str:
    if isinstance(error, WikiExistsError):
        return "wiki_exists"
    if isinstance(error, WikiNotFoundError):
        return "wiki_not_found"
    if isinstance(error, WikiUnsafePathError):
        return "wiki_unsafe_path"
    if isinstance(error, WikiDependencyError):
        return "wiki_dependency_unavailable"
    if isinstance(error, WikiFormatError):
        return "wiki_format_invalid"
    if isinstance(error, WikiValidationError):
        return "wiki_validation_failed"
    if isinstance(error, WikiIndexUnavailableError):
        return "wiki_index_unavailable"
    if isinstance(error, ValueError):
        return "wiki_invalid_request"
    return "wiki_operation_failed"


def _date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD")
    return parsed


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


__all__ = ["add_parser", "run"]
