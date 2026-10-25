"""Canonical, offline Tools for independently versioned book chunks."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from capicove.books.chunk_config import CONTENT_TYPES, PRESETS, integer_limit, resolve_chunk_config
from capicove.books.chunk_service import BookChunkService
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
from capicove.tools.registry import ToolRegistry


_CHUNK_PROPERTIES: dict[str, object] = {
    "preset": {"enum": list(PRESETS)},
    "boundary_level": {"enum": ["deepest", "section", "chapter"]},
    "oversize_policy": {"enum": ["split", "error"]},
    "page_boundary": {"enum": ["strict", "allow_continuations"]},
    "front_matter_policy": {"enum": ["navigation-v1", "preserve"]},
    "tokenizer": {"type": "string", "maxLength": 4096},
    "protected_types": {
        "type": "array",
        "uniqueItems": True,
        "items": {"enum": list(CONTENT_TYPES)},
    },
    "merge_small_chunks": {"type": "boolean"},
}
for _name in (
    "target_core_tokens",
    "max_chunk_tokens",
    "context_before_tokens",
    "context_after_tokens",
    "max_context_read_tokens",
):
    _CHUNK_PROPERTIES[_name] = {
        "type": "integer",
        "minimum": 0 if _name.startswith("context_") else 1,
        "maximum": 32768,
    }


def _spec(operation: str) -> ToolSpec:
    write = operation == "chunk"
    required = ["workspace", "book_id"]
    properties: dict[str, object] = {
        "workspace": {"type": "string", "minLength": 1},
        "book_id": {"type": "string", "pattern": "^book:[0-9a-f]{24}$"},
    }
    if operation in {"chunk", "chunk-preview"}:
        properties.update(
            {
                "chunking": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": _CHUNK_PROPERTIES,
                },
                "compare_run": {"type": "string"},
            }
        )
    else:
        required.append("run_id")
        properties["run_id"] = {"type": "string", "pattern": "^chunk-run:[0-9a-f]{24}$"}
        if operation == "chunks":
            properties.update(
                {
                    "offset": {"type": "integer", "minimum": 0, "maximum": 1000000},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "tasks": {"type": "boolean"},
                }
            )
        else:
            required.append("chunk_id")
            properties.update(
                {
                    "chunk_id": {"type": "string"},
                    "max_context_read_tokens": {"type": "integer", "minimum": 1, "maximum": 32768},
                    "read_policy_id": {"type": "string"},
                }
            )
    permissions = {"book:read", "filesystem:read"}
    if write:
        permissions.update({"book:write", "filesystem:write"})
    return ToolSpec(
        "capicove.book." + operation,
        1,
        {
            "chunk": "Create immutable offline chunks without changing legacy summaries.",
            "chunk-preview": "Preview chunk quality and configuration without writes or OCR.",
            "chunks": "Read bounded chunk pages or reference-only caller task graphs.",
            "context": "Read original Parent evidence within a saved read policy.",
        }[operation],
        {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        },
        {"type": "object"},
        ("text",),
        frozenset(permissions),
        PrivacyClass.CONFIDENTIAL,
        SideEffect.WRITE if write else SideEffect.READ,
        False,
        True,
        True,
        120.0,
        ("workspace",),
        ToolResources(filesystem_read=True, filesystem_write=write),
    )


CHUNK_TOOL_SPECS = {name: _spec(name) for name in ("chunk", "chunk-preview", "chunks", "context")}


class BookChunkToolHandler:
    """Validate a canonical request before accessing authorized evidence."""

    def __init__(self, operation: str, *, workspace: Path | None = None) -> None:
        if operation not in CHUNK_TOOL_SPECS:
            raise ValueError("unknown chunk Tool operation")
        self.operation = operation
        self.workspace = workspace.resolve() if workspace is not None else None

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)
        spec = CHUNK_TOOL_SPECS[self.operation]
        code = ToolErrorCode.INVALID_INPUT
        try:
            if not spec.required_permissions <= context.permissions:
                code = ToolErrorCode.PERMISSION_DENIED
                raise ValueError("book chunk Tool permissions are incomplete")
            allowed = cast(Mapping[str, object], spec.input_schema["properties"])
            if set(arguments) - set(allowed):
                raise ValueError("unknown book chunk Tool argument")
            workspace = Path(_string(arguments, "workspace")).expanduser().resolve()
            if self.workspace is not None and workspace != self.workspace:
                raise ValueError("workspace is outside the authorized MCP root")
            book_id = _string(arguments, "book_id")
            config = None
            if self.operation in {"chunk", "chunk-preview"}:
                raw_config = arguments.get("chunking", {})
                if not isinstance(raw_config, Mapping):
                    raise ValueError("chunking must be an object")
                config = resolve_chunk_config(overrides=raw_config)
                self._check_tokenizer(config.tokenizer)
            service = BookChunkService(
                LocalBookWorkspace.open(workspace), tokenizer_root=self.workspace
            )
            if config is not None:
                compare = _string(arguments, "compare_run") if "compare_run" in arguments else None
                payload = service.create(
                    book_id, config, dry_run=self.operation == "chunk-preview", compare_run=compare
                )
            elif self.operation == "chunks":
                tasks = arguments.get("tasks", False)
                if not isinstance(tasks, bool):
                    raise ValueError("tasks must be boolean")
                payload = service.read_chunks(
                    book_id,
                    _string(arguments, "run_id"),
                    offset=integer_limit(
                        "offset", arguments.get("offset", 0), minimum=0, maximum=1000000
                    ),
                    limit=integer_limit("limit", arguments.get("limit", 20), minimum=1, maximum=50),
                    tasks=tasks,
                )
            else:
                payload = service.read_context(
                    book_id,
                    _string(arguments, "run_id"),
                    _string(arguments, "chunk_id"),
                    max_tokens=integer_limit(
                        "max_context_read_tokens",
                        arguments["max_context_read_tokens"],
                        minimum=1,
                        maximum=32768,
                    )
                    if "max_context_read_tokens" in arguments
                    else None,
                    read_policy_id=_string(arguments, "read_policy_id")
                    if "read_policy_id" in arguments
                    else None,
                )
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
            return ToolResult(
                context.request_id,
                context.trace_id,
                spec.tool_id,
                1,
                ToolStatus.FAILED,
                error=ToolError(code, str(error)[:500], False),
                started_at=started,
                completed_at=datetime.now(UTC),
            )

    def _check_tokenizer(self, tokenizer: str) -> None:
        if (
            self.workspace is not None
            and tokenizer.startswith("local:")
            and not Path(tokenizer[6:]).resolve().is_relative_to(self.workspace)
        ):
            raise ValueError("MCP local tokenizer must be inside the authorized workspace")


def _string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def register_book_chunk_tools(registry: ToolRegistry) -> None:
    """Register four canonical Tools without changing the legacy book contract."""
    for operation, spec in CHUNK_TOOL_SPECS.items():
        registry.register(spec, BookChunkToolHandler(operation))
