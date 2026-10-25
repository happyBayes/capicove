"""Workspace-bound canonical Tools for accepted Book chunk results."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

from capicove.books.chunk_config import integer_limit
from capicove.books.chunk_result_service import BookChunkResultService
from capicove.books.workspace import LocalBookWorkspace
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


if TYPE_CHECKING:
    from capicove.wiki import WikiService


def _read_spec(operation: str) -> ToolSpec:
    """Describe one bounded result read without caller-selected authority."""
    return ToolSpec(
        tool_id=f"capicove.book.{operation}",
        contract_version=1,
        description={
            "chunk-tasks": "Read a bounded page of saved chunk-result tasks.",
            "chunk-status": "Read bounded chunk-result readiness and publishability.",
        }[operation],
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["book_id", "run_id"],
            "properties": {
                "book_id": {"type": "string", "pattern": "^book:[0-9a-f]{24}$"},
                "run_id": {"type": "string", "pattern": "^chunk-run:[0-9a-f]{24}$"},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
        },
        output_schema={"type": "object"},
        modalities=("text",),
        required_permissions=frozenset({"book:read"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        side_effect=SideEffect.READ,
        resources=ToolResources(filesystem_read=True),
    )


BOOK_RESULT_TOOL_SPECS = {
    operation: _read_spec(operation) for operation in ("chunk-tasks", "chunk-status")
}

BOOK_RESULT_TOOL_SPECS["chunk-submit"] = ToolSpec(
    tool_id="capicove.book.chunk-submit",
    contract_version=1,
    description="Append one cited chunk-result version to the bound Book workspace.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "book_id",
            "run_id",
            "task_id",
            "idempotency_key",
            "dependency_result_ids",
            "abstract",
            "key_points",
            "claims",
        ],
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "book_id": {"type": "string", "pattern": "^book:[0-9a-f]{24}$"},
            "run_id": {"type": "string", "pattern": "^chunk-run:[0-9a-f]{24}$"},
            "task_id": {"type": "string", "pattern": "^chunk-task:[0-9a-f]{24}$"},
            "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
            "dependency_result_ids": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string", "pattern": "^chunk-result:[0-9a-f]{24}$"},
            },
            "abstract": {"type": "string", "minLength": 1, "maxLength": 16384},
            "key_points": {
                "type": "array",
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1, "maxLength": 2048},
            },
            "claims": {
                "type": "array",
                "maxItems": 64,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "citations"],
                    "properties": {
                        "text": {"type": "string", "minLength": 1, "maxLength": 2048},
                        "citations": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "block_id",
                                    "page_index",
                                    "start_char",
                                    "end_char",
                                    "text_sha256",
                                ],
                                "properties": {
                                    "block_id": {"type": "string"},
                                    "page_index": {"type": "integer", "minimum": 0},
                                    "start_char": {"type": "integer", "minimum": 0},
                                    "end_char": {"type": "integer", "minimum": 1},
                                    "text_sha256": {
                                        "type": "string",
                                        "pattern": "^[0-9a-f]{64}$",
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "provenance": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "provider_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "model_revision": {"type": "string", "minLength": 1, "maxLength": 256},
                    "caller_run_id": {"type": "string", "minLength": 1, "maxLength": 256},
                },
            },
        },
    },
    output_schema={"type": "object"},
    modalities=("text",),
    required_permissions=frozenset({"book:write"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    resources=ToolResources(filesystem_read=True, filesystem_write=True),
)

BOOK_RESULT_TOOL_SPECS["chunk-publish"] = ToolSpec(
    tool_id="capicove.book.chunk-publish",
    contract_version=1,
    description="Publish one verified result DAG to the startup-bound Wiki.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["book_id", "run_id"],
        "properties": {
            "book_id": {"type": "string", "pattern": "^book:[0-9a-f]{24}$"},
            "run_id": {"type": "string", "pattern": "^chunk-run:[0-9a-f]{24}$"},
        },
    },
    output_schema={"type": "object"},
    modalities=("text",),
    required_permissions=frozenset({"book:write", "wiki:write"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    deterministic=False,
    resources=ToolResources(filesystem_read=True, filesystem_write=True),
)

_SAFE_ERRORS = frozenset(
    {
        "permission_denied",
        "wiki_not_configured",
        "invalid_submission",
        "citation_invalid",
        "source_changed",
        "dependency_incomplete",
        "dependency_stale",
        "idempotency_conflict",
        "publication_incomplete",
        "publication_conflict",
    }
)


class BookChunkResultToolHandler:
    """Execute an accepted Book result read within one bound workspace."""

    def __init__(
        self,
        operation: str,
        *,
        workspace: LocalBookWorkspace,
        wiki_service: WikiService | None = None,
    ) -> None:
        if operation not in BOOK_RESULT_TOOL_SPECS:
            raise ValueError("unknown Book result Tool operation")
        self.operation = operation
        self._service = BookChunkResultService(workspace)
        self._wiki_service = wiki_service

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)
        spec = BOOK_RESULT_TOOL_SPECS[self.operation]
        code = ToolErrorCode.INVALID_INPUT
        try:
            if not spec.required_permissions <= context.permissions:
                code = ToolErrorCode.PERMISSION_DENIED
                raise ValueError("permission_denied")
            if not isinstance(arguments, Mapping):
                raise ValueError("invalid_input")
            allowed = cast(Mapping[str, object], spec.input_schema["properties"])
            if set(arguments) - set(allowed):
                raise ValueError("invalid_input")
            if self.operation == "chunk-submit":
                payload = self._service.submit(dict(arguments))
            elif self.operation == "chunk-publish":
                if self._wiki_service is None:
                    code = ToolErrorCode.UNAVAILABLE
                    raise ValueError("wiki_not_configured")
                payload = self._service.publish(
                    _identifier(arguments, "book_id", "book:"),
                    _identifier(arguments, "run_id", "chunk-run:"),
                    self._wiki_service,
                    published_on=datetime.now(UTC).date(),
                )
            else:
                book_id = _identifier(arguments, "book_id", "book:")
                run_id = _identifier(arguments, "run_id", "chunk-run:")
                offset = integer_limit(
                    "offset", arguments.get("offset", 0), minimum=0, maximum=1000000
                )
                limit = integer_limit("limit", arguments.get("limit", 20), minimum=1, maximum=50)
                operation = (
                    self._service.list_tasks
                    if self.operation == "chunk-tasks"
                    else self._service.status
                )
                payload = operation(book_id, run_id, offset=offset, limit=limit)
            return ToolResult(
                context.request_id,
                context.trace_id,
                spec.tool_id,
                1,
                ToolStatus.SUCCEEDED,
                payload,
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            message = str(error)
            if message not in _SAFE_ERRORS:
                message = "invalid_input"
            return ToolResult(
                context.request_id,
                context.trace_id,
                spec.tool_id,
                1,
                ToolStatus.FAILED,
                error=ToolError(code, message, False),
                started_at=started,
                completed_at=datetime.now(UTC),
            )


def _identifier(arguments: Mapping[str, object], name: str, prefix: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("invalid_input")
    return value
