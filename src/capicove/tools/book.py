"""Canonical Tools for the local provider-neutral book workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

from capicove.books import BookService, LocalBookWorkspace
from capicove.books.codec import (
    structure_review_submission_from_json,
    structure_review_task_to_json,
    summary_submission_from_json,
    summary_task_to_json,
)
from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.plugins import load_document_parser_plugin
from capicove.tools.registry import ToolRegistry


_WORKSPACE = {"type": "string", "minLength": 1}
_BOOK_ID = {"type": "string", "pattern": "^book:[0-9a-f]{24}$"}
_COMMON_OUTPUT: dict[str, object] = {"type": "object"}


def _input(required: list[str], properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


BOOK_TOOL_SPECS = {
    "prepare": ToolSpec(
        "capicove.book.prepare",
        2,
        "Parse an authorized local PDF and create bounded model-neutral tasks.",
        _input(
            ["workspace", "source"],
            {
                "workspace": _WORKSPACE,
                "source": _WORKSPACE,
                "parser": {"type": "string", "default": "paddleocr"},
                "profile": {"type": "string", "default": "book-balanced"},
                "profile_version": {"type": "integer", "minimum": 1, "default": 2},
                "model_root": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "language_hints": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 64},
                    "maxItems": 16,
                    "uniqueItems": True,
                    "default": [],
                },
            },
        ),
        _COMMON_OUTPUT,
        ("document", "text"),
        frozenset({"book:read", "book:write", "filesystem:read", "filesystem:write"}),
        PrivacyClass.CONFIDENTIAL,
        SideEffect.WRITE,
        False,
        True,
        True,
        1800.0,
        ("workspace", "source", "model_root"),
        ToolResources(filesystem_read=True, filesystem_write=True),
    ),
    "next-task": ToolSpec(
        "capicove.book.next-task",
        1,
        "Return the next bounded dependency-ready book task.",
        _input(["workspace", "book_id"], {"workspace": _WORKSPACE, "book_id": _BOOK_ID}),
        _COMMON_OUTPUT,
        ("text",),
        frozenset({"book:read", "filesystem:read"}),
        PrivacyClass.CONFIDENTIAL,
        SideEffect.READ,
        False,
        True,
        True,
        30.0,
        ("workspace",),
        ToolResources(filesystem_read=True),
    ),
    "submit": ToolSpec(
        "capicove.book.submit",
        1,
        "Validate citations and append one Agent-generated summary version.",
        _input(
            ["workspace", "book_id", "submission"],
            {"workspace": _WORKSPACE, "book_id": _BOOK_ID, "submission": {"type": "object"}},
        ),
        _COMMON_OUTPUT,
        ("text",),
        frozenset({"book:read", "book:write", "filesystem:read", "filesystem:write"}),
        PrivacyClass.CONFIDENTIAL,
        SideEffect.WRITE,
        False,
        True,
        False,
        30.0,
        ("workspace",),
        ToolResources(filesystem_read=True, filesystem_write=True),
    ),
    "status": ToolSpec(
        "capicove.book.status",
        1,
        "Report durable book workflow progress.",
        _input(["workspace", "book_id"], {"workspace": _WORKSPACE, "book_id": _BOOK_ID}),
        _COMMON_OUTPUT,
        ("text",),
        frozenset({"book:read", "filesystem:read"}),
        PrivacyClass.CONFIDENTIAL,
        SideEffect.READ,
        False,
        True,
        True,
        30.0,
        ("workspace",),
        ToolResources(filesystem_read=True),
    ),
    "publish": ToolSpec(
        "capicove.book.publish",
        1,
        "Publish completed summaries into a validated page-aware local Wiki.",
        _input(
            ["workspace", "book_id"],
            {"workspace": _WORKSPACE, "book_id": _BOOK_ID, "date": {"type": "string"}},
        ),
        _COMMON_OUTPUT,
        ("text",),
        frozenset({"book:read", "book:write", "filesystem:read", "filesystem:write"}),
        PrivacyClass.CONFIDENTIAL,
        SideEffect.WRITE,
        False,
        True,
        True,
        60.0,
        ("workspace",),
        ToolResources(filesystem_read=True, filesystem_write=True),
    ),
}


class BookToolHandler:
    """Execute one canonical book operation against an authorized local workspace."""

    def __init__(
        self,
        operation: str,
        *,
        service_factory: Callable[[Path], object] | None = None,
    ) -> None:
        if operation not in BOOK_TOOL_SPECS:
            raise ValueError("unknown book Tool operation")
        self._operation = operation
        self._service_factory = service_factory or _open_service

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        """Validate permissions and return one canonical Tool result."""
        started = datetime.now(UTC)
        spec = BOOK_TOOL_SPECS[self._operation]
        missing = spec.required_permissions.difference(context.permissions)
        if missing:
            return _failure(
                context,
                spec,
                started,
                ToolErrorCode.PERMISSION_DENIED,
                "book Tool permissions are incomplete",
            )
        try:
            workspace = Path(_string(arguments, "workspace"))
            service = self._service_factory(workspace)
            payload = self._execute(service, arguments)
            return ToolResult(
                context.request_id,
                context.trace_id,
                spec.tool_id,
                spec.contract_version,
                ToolStatus.SUCCEEDED,
                payload,
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return _failure(
                context,
                spec,
                started,
                ToolErrorCode.INVALID_INPUT,
                str(error)[:500],
            )

    def _execute(self, service: object, arguments: Mapping[str, object]) -> dict[str, object]:
        if self._operation == "status":
            status = service.status(_string(arguments, "book_id"))  # type: ignore[attr-defined]
            return {
                "book_id": status.book_id,
                "completed_tasks": status.completed_tasks,
                "pending_tasks": status.pending_tasks,
                "publishable": status.publishable,
                "structure_review_tasks": status.structure_review_tasks,
                "total_tasks": status.total_tasks,
            }
        if self._operation == "next-task":
            book_id = _string(arguments, "book_id")
            structure_tasks = service.structure_tasks(book_id)  # type: ignore[attr-defined]
            if structure_tasks:
                return {
                    "task": {
                        "task_type": "structure_review",
                        **structure_review_task_to_json(structure_tasks[0]),
                    }
                }
            task = service.next_task(book_id)  # type: ignore[attr-defined]
            return {
                "task": None
                if task is None
                else {"task_type": "summary", **summary_task_to_json(task)}
            }
        if self._operation == "submit":
            raw_submission = arguments.get("submission")
            if isinstance(raw_submission, Mapping) and "operation" in raw_submission:
                version = service.submit_structure(  # type: ignore[attr-defined]
                    _string(arguments, "book_id"),
                    structure_review_submission_from_json(raw_submission),
                )
            else:
                version = service.submit(  # type: ignore[attr-defined]
                    _string(arguments, "book_id"),
                    summary_submission_from_json(raw_submission),
                )
            return {
                "task_id": version.task_id,
                "version": version.version,
                "previous_version": version.previous_version,
            }
        if self._operation == "publish":
            raw_date = arguments.get("date")
            published_on = date.today() if raw_date is None else date.fromisoformat(str(raw_date))
            value = service.publish(  # type: ignore[attr-defined]
                _string(arguments, "book_id"), published_on=published_on
            )
            return {
                "book_id": value.book_id,
                "citations": value.citations,
                "page_paths": list(value.page_paths),
                "representation_path": value.representation_path,
                "source_path": value.source_path,
            }
        if self._operation == "prepare":
            parser_name = str(arguments.get("parser", "paddleocr"))
            profile = str(arguments.get("profile", "book-balanced"))
            profile_version = _positive_integer(arguments, "profile_version", default=2)
            plugin = load_document_parser_plugin(parser_name)
            availability = plugin.availability()
            if not availability.available:
                raise RuntimeError(availability.detail or "document parser is unavailable")
            model_root = arguments.get("model_root")
            if not isinstance(model_root, str) or not model_root.strip():
                raise ValueError("model_root is required for book preparation")
            parser = plugin.create(
                {"model_root": model_root, "device": str(arguments.get("device", "cpu"))}
            )
            value = service.prepare(  # type: ignore[attr-defined]
                Path(_string(arguments, "source")),
                parser,
                profile_id=profile,
                profile_version=profile_version,
                language_hints=_string_tuple(arguments, "language_hints"),
            )
            return {
                "book_id": value.book_id,
                "created": value.created,
                "representation_sha256": value.representation_sha256,
                "sections": value.sections,
                "source_sha256": value.source_sha256,
                "structure_review_tasks": value.structure_review_tasks,
                "summary_tasks": value.summary_tasks,
            }
        raise ValueError("unsupported book Tool operation")


def register_book_tools(registry: ToolRegistry) -> None:
    """Register all five book Tools."""
    for operation, spec in BOOK_TOOL_SPECS.items():
        registry.register(spec, BookToolHandler(operation))


def _open_service(workspace: Path) -> BookService:
    return BookService(LocalBookWorkspace.open(workspace))


def _string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _positive_integer(arguments: Mapping[str, object], name: str, *, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _string_tuple(arguments: Mapping[str, object], name: str) -> tuple[str, ...]:
    value = arguments.get(name, ())
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a string array")
    result = tuple(value)
    if (
        len(result) > 16
        or any(not isinstance(item, str) or not item.strip() or len(item) > 64 for item in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} must contain at most 16 unique bounded strings")
    return result


def _failure(
    context: InvocationContext,
    spec: ToolSpec,
    started: datetime,
    code: ToolErrorCode,
    message: str,
) -> ToolResult:
    return ToolResult(
        context.request_id,
        context.trace_id,
        spec.tool_id,
        spec.contract_version,
        ToolStatus.FAILED,
        error=ToolError(code, message, False),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


__all__ = ["BOOK_TOOL_SPECS", "BookToolHandler", "register_book_tools"]
