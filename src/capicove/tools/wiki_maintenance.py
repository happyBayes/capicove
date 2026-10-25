"""Closed host-bound Wiki maintenance Tools, with mutations disabled by default."""

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from capicove.contracts.wiki_maintenance import AnswerTask, ReviewDecision
from capicove.wiki.maintenance.codec import decode_record
from capicove.wiki.maintenance.review import ReviewAuthority
from capicove.wiki.maintenance.service import MaintenanceService


_TEXT = {"type": "string", "minLength": 1, "maxLength": 8192}
_ID = {"type": "string", "minLength": 1, "maxLength": 128}
_WIRE = {"type": "string", "minLength": 1, "maxLength": 1048576}
_BOOL = {"type": "boolean"}
_PAGES = {"type": "array", "minItems": 0, "maxItems": 20, "items": _ID}
_INTERVAL = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source_id", "start", "end"],
    "properties": {
        "source_id": _ID,
        "start": {"type": "integer", "minimum": 0, "maximum": 16777216},
        "end": {"type": "integer", "minimum": 1, "maximum": 16777216},
        "page_index": {"type": "integer", "minimum": 0, "maximum": 100000},
        "block_id": _ID,
    },
}
_OPTIONAL = {"task": {"page_ids", "intervals"}, "context": {"page_ids", "intervals"}}

_SOURCES = {"type": "array", "minItems": 1, "maxItems": 8, "items": _ID}
_DECISIONS = {
    "type": "array",
    "minItems": 1,
    "maxItems": 256,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["subject_id", "verdict", "reason"],
        "properties": {
            "subject_id": _ID,
            "verdict": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]},
            "reason": _TEXT,
        },
    },
}
_OPERATIONS: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "capture": (("wiki:write",), {"source": _TEXT}),
    "task": (("wiki:read", "wiki:write"), {"source_ids": _SOURCES, "query": _TEXT}),
    "context": (
        ("wiki:read", "wiki:write"),
        {"task_id": _ID, "source_ids": _SOURCES, "query": _TEXT},
    ),
    "submit": (("wiki:write",), {"submission": _WIRE}),
    "preview": (("wiki:read",), {"proposal_id": _ID}),
    "review": (("wiki:review",), {"proposal_id": _ID, "decisions": _DECISIONS, "accept": _BOOL}),
    "apply": (("wiki:write", "wiki:publish"), {"proposal_id": _ID, "review_id": _ID}),
    "follow-up": (("wiki:read", "wiki:write"), {"intent_id": _ID, "page_ids": _PAGES}),
    "rollback": (
        ("wiki:read", "wiki:write"),
        {"page_id": _ID, "publication_id": _ID, "idempotency_key": _ID},
    ),
    "status": (("wiki:read",), {}),
    "recover": (("wiki:publish",), {"abort": _BOOL}),
    "withdraw": (("wiki:publish",), {"source_id": _ID, "reason": _TEXT, "idempotency_key": _ID}),
    "answer-task": (("wiki:read",), {"question": _TEXT, "include_history": _BOOL}),
    "answer-submit": (("wiki:read",), {"task": _WIRE, "submission": _WIRE}),
    "ask": (("wiki:read",), {"question": _TEXT, "include_history": _BOOL}),
}
_READ_ONLY = frozenset({"preview", "status", "answer-task", "answer-submit", "ask"})


def _spec(operation: str) -> ToolSpec:
    permissions, fields = _OPERATIONS[operation]
    properties = {"workspace_id": _ID, **fields}
    if operation in {"task", "context"}:
        properties.update(
            {
                "page_ids": _PAGES,
                "intervals": {"type": "array", "minItems": 1, "maxItems": 128, "items": _INTERVAL},
            }
        )
    suffix = (
        operation
        if operation in {"answer-task", "answer-submit", "ask"}
        else "maintenance-" + operation
    )
    return ToolSpec(
        "capicove.wiki." + suffix,
        1,
        "Execute host-bound Wiki " + operation + ".",
        {
            "type": "object",
            "additionalProperties": False,
            "required": [key for key in properties if key not in _OPTIONAL.get(operation, set())],
            "properties": properties,
        },
        {"type": "object"},
        required_permissions=frozenset(permissions),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        side_effect=SideEffect.READ if operation in _READ_ONLY else SideEffect.WRITE,
        redacted_fields=("source", "submission", "task", "question"),
        resources=ToolResources(filesystem_read=True, filesystem_write=operation not in _READ_ONLY),
    )


def maintenance_specs(*, enable_mutations: bool = False) -> tuple[ToolSpec, ...]:
    """Enumerate read-only operations unless the host explicitly enables writes."""
    return tuple(_spec(name) for name in _OPERATIONS if enable_mutations or name in _READ_ONLY)


def _validate(value: Any, schema: Mapping[str, Any]) -> None:
    kind = schema["type"]
    if kind == "object":
        if (
            not isinstance(value, Mapping)
            or not set(schema["required"]) <= set(value)
            or not set(value) <= set(schema["properties"])
        ):
            raise ValueError("request fields must exactly match schema")
        for name, child in schema["properties"].items():
            if name in value:
                _validate(value[name], child)
    elif kind == "array":
        if (
            not isinstance(value, (list, tuple))
            or not schema["minItems"] <= len(value) <= schema["maxItems"]
        ):
            raise ValueError("invalid bounded array")
        for item in value:
            _validate(item, schema["items"])
    elif kind == "integer":
        if type(value) is not int or not schema["minimum"] <= value <= schema["maximum"]:
            raise ValueError("invalid bounded integer")
    elif kind == "boolean":
        if type(value) is not bool:
            raise ValueError("expected boolean")
    elif kind == "string":
        if not isinstance(value, str) or not schema.get("minLength", 1) <= len(
            value.encode()
        ) <= schema.get("maxLength", 8192):
            raise ValueError("invalid bounded text")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError("invalid enum value")


class MaintenanceToolHandler:
    """Use only host-registered services and authenticated invocation identity."""

    def __init__(
        self,
        operation: str,
        workspaces: Mapping[str, MaintenanceService],
        *,
        enable_mutations: bool = False,
    ) -> None:
        if operation not in _OPERATIONS:
            raise ValueError("unknown maintenance operation")
        self.operation = operation
        self.workspaces = dict(workspaces)
        self.enable_mutations = enable_mutations

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)
        spec = _spec(self.operation)
        try:
            if not spec.required_permissions <= context.permissions or (
                self.operation not in _READ_ONLY and not self.enable_mutations
            ):
                raise PermissionError("maintenance operation is not authorized")
            if context.deadline <= started:
                raise TimeoutError("maintenance deadline exceeded")
            _validate(arguments, spec.input_schema)
            identity = arguments["workspace_id"]
            if identity != context.workspace_id or identity not in self.workspaces:
                raise PermissionError("workspace is not authorized")
            service = self.workspaces[str(identity)]
            payload = self._execute(service, context, dict(arguments))
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
        except (ValueError, OSError, TypeError, TimeoutError) as error:
            code = (
                ToolErrorCode.PERMISSION_DENIED
                if isinstance(error, PermissionError)
                else ToolErrorCode.INVALID_INPUT
            )
            # Do not expose filesystem paths or provider configuration in envelopes.
            message = (
                "maintenance request denied"
                if code == ToolErrorCode.PERMISSION_DENIED
                else "maintenance request failed validation"
            )
            return ToolResult(
                context.request_id,
                context.trace_id,
                spec.tool_id,
                1,
                ToolStatus.REJECTED,
                error=ToolError(code, message, False),
                started_at=started,
                completed_at=datetime.now(UTC),
            )

    def _execute(
        self, service: MaintenanceService, context: InvocationContext, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        operation = self.operation
        value: Any
        if operation == "capture":
            value = service.capture(Path(arguments["source"]))
        elif operation in {"task", "context"}:
            intervals = []
            for item in arguments.get("intervals", []):
                if ("page_index" in item) != ("block_id" in item):
                    raise ValueError("PDF interval needs page and block")
                prefix: tuple[Any, ...] = (item["source_id"],)
                if "page_index" in item:
                    prefix += (item["page_index"], item["block_id"])
                intervals.append((*prefix, item["start"], item["end"]))
            value = service.create_task(
                tuple(arguments["source_ids"]),
                query=arguments["query"],
                previous_task_id=arguments.get("task_id"),
                intervals=tuple(intervals) if intervals else None,
                page_ids=tuple(arguments.get("page_ids", ())),
            )
        elif operation == "follow-up":
            value = service.activate_follow_up(
                arguments["intent_id"], page_ids=tuple(arguments["page_ids"])
            )
        elif operation == "rollback":
            value = service.rollback_proposal(
                arguments["page_id"],
                arguments["publication_id"],
                idempotency_key=arguments["idempotency_key"],
            )
        elif operation == "submit":
            value = service.submit(arguments["submission"].encode())
        elif operation == "preview":
            value = service.preview(arguments["proposal_id"])
        elif operation == "review":
            value = service.review(
                arguments["proposal_id"],
                tuple(ReviewDecision(**decision) for decision in arguments["decisions"]),
                authority=ReviewAuthority(context.actor_id, context.permissions),
                accept=arguments["accept"],
            )
        elif operation == "apply":
            value = service.apply(arguments["proposal_id"], arguments["review_id"])
        elif operation == "recover":
            service.recover(abort=arguments["abort"])
            return {"recovered": True}
        elif operation == "withdraw":
            value = service.withdraw(
                arguments["source_id"],
                reason=arguments["reason"],
                idempotency_key=arguments["idempotency_key"],
            )
        elif operation == "answer-task":
            value = service.answer_task(
                arguments["question"], include_history=arguments["include_history"]
            )
        elif operation == "answer-submit":
            task = decode_record(arguments["task"].encode()).payload
            if not isinstance(task, AnswerTask):
                raise ValueError("invalid answer task")
            value = service.answer_submit(task, arguments["submission"].encode())
        elif operation == "ask":
            task, result = service.evidence_only(
                arguments["question"], include_history=arguments["include_history"]
            )
            return {"task": asdict(task), "answer": asdict(result)}
        else:
            return service.status()
        return dict(json.loads(json.dumps(asdict(value))))


def maintenance_operations() -> tuple[str, ...]:
    """Return supported operation names for explicit host adapter registration."""
    return tuple(_OPERATIONS)


def encode_result(result: ToolResult) -> dict[str, Any]:
    """Encode the same canonical envelope for CLI and local MCP consumers."""

    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": plain(result.result),
        "evidence_refs": list(result.evidence_refs),
        "error": None
        if result.error is None
        else {
            "code": result.error.code.value,
            "message": result.error.message,
            "retryable": result.error.retryable,
        },
    }
