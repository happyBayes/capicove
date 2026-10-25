"""Closed temporal-memory operations with host-owned workspace and scope authority."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from capicove.agent import AgentMemoryService
from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.contracts.evidence import EvidenceRef
from capicove.contracts.knowledge import EntityRef, TimeInterval, TypedLiteral, utc
from capicove.contracts.memory import MemoryQuery
from capicove.traces import TraceRecorder


_FIELDS = {
    "remember_episode": {
        "scope",
        "content",
        "event_time",
        "idempotency_key",
        "episode_type",
        "sensitivity",
        "pinned",
        "extraction",
    },
    "assert_fact": {
        "scope",
        "supersedes",
        "subject_name",
        "subject_type",
        "predicate",
        "object",
        "valid_from",
        "valid_to",
        "event_time",
        "evidence_text",
        "idempotency_key",
        "sensitivity",
        "pinned",
    },
    "recall": {
        "scope",
        "text",
        "valid_at",
        "known_at",
        "types",
        "states",
        "center_entities",
        "top_k",
        "retrieval_profile_ref",
        "data_policy_ref",
        "min_revision",
        "max_characters",
        "max_tokens",
    },
    "get_episode": {"scope", "episode_id", "min_revision"},
    "get_fact_history": {"scope", "fact_id", "min_revision"},
    "retry_extraction": {"scope", "episode_id", "idempotency_key"},
    "archive_memory": {"scope", "memory_id", "idempotency_key"},
    "erase_scope": {"scope", "idempotency_key"},
    "doctor": {"scope"},
    "work_extractions": {"scope", "limit"},
    "resolve_entity": {"scope", "name", "entity_type", "idempotency_key"},
}
_READS = {"recall", "get_episode", "get_fact_history", "doctor"}


_REQUIRED = {
    "remember_episode": ("scope", "content", "event_time", "idempotency_key"),
    "assert_fact": (
        "scope",
        "subject_name",
        "subject_type",
        "predicate",
        "object",
        "event_time",
        "evidence_text",
        "idempotency_key",
    ),
    "recall": ("scope",),
    "get_episode": ("scope", "episode_id"),
    "get_fact_history": ("scope", "fact_id"),
    "retry_extraction": ("scope", "episode_id", "idempotency_key"),
    "archive_memory": ("scope", "memory_id", "idempotency_key"),
    "erase_scope": ("scope", "idempotency_key"),
    "doctor": ("scope",),
    "work_extractions": ("scope",),
    "resolve_entity": ("scope", "name", "entity_type", "idempotency_key"),
}


def _property_schema(name: str) -> dict[str, object]:
    if name in {"pinned", "extraction"}:
        return {"type": "boolean"}
    if name in {"top_k", "min_revision", "max_characters", "max_tokens", "limit"}:
        return {
            "type": "integer",
            "minimum": 0 if name == "min_revision" else 1,
            "maximum": 100
            if name in {"top_k", "limit"}
            else 1000000
            if name in {"max_characters", "max_tokens"}
            else 2**63 - 1,
        }
    if name in {"valid_from", "valid_to", "valid_at", "known_at", "event_time"}:
        return {
            "type": ["string", "null"] if name != "event_time" else "string",
            "format": "date-time",
        }
    if name in {"content", "evidence_text", "text"}:
        return {
            "type": "string",
            "minLength": 0 if name == "text" else 1,
            "maxLength": 8192 if name == "text" else 1048576,
        }
    if name in {"types", "states", "center_entities"}:
        items = (
            {"enum": ["episodic", "semantic"]}
            if name == "types"
            else {"enum": ["accepted", "disputed", "rejected"]}
            if name == "states"
            else {"type": "string", "minLength": 1, "maxLength": 512}
        )
        return {
            "type": "array",
            "items": items,
            "uniqueItems": True,
            "maxItems": 100,
            "minItems": 0 if name == "center_entities" else 1,
        }
    if name == "object":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "value"],
            "properties": {
                "kind": {
                    "enum": ["entity", "string", "boolean", "int64", "decimal", "date", "datetime"]
                },
                "value": {"type": ["string", "boolean", "integer"]},
            },
        }
    enums = {
        "episode_type": ["message", "document", "json", "fact_assertion"],
        "sensitivity": ["public", "internal", "confidential", "restricted"],
        "retrieval_profile_ref": ["rrf-v1"],
        "data_policy_ref": ["default"],
    }
    return (
        {"enum": enums[name]}
        if name in enums
        else {"type": "string", "minLength": 1, "maxLength": 512}
    )


def memory_spec(operation: str) -> ToolSpec:
    fields = _FIELDS[operation]
    return ToolSpec(
        tool_id="capicove.memory." + operation,
        contract_version=1,
        description="Run the scoped " + operation + " memory operation.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": list(_REQUIRED[operation]),
            "properties": {name: _property_schema(name) for name in sorted(fields)},
        },
        output_schema={"type": "object"},
        required_permissions=frozenset(
            {
                "memory:erase"
                if operation == "erase_scope"
                else "memory:read"
                if operation in _READS
                else "memory:write"
            }
        ),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        side_effect=SideEffect.READ if operation in _READS else SideEffect.WRITE,
        resources=ToolResources(filesystem_read=True, filesystem_write=operation not in _READS),
        redacted_fields=tuple(sorted(fields)),
    )


MEMORY_SPECS = {name: memory_spec(name) for name in _FIELDS}


def _json_value(value: Any) -> Any:
    if isinstance(value, EvidenceRef):
        return {"evidence_id": value.evidence_id, "content_sha256": value.content_sha256}
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


class MemoryToolHandler:
    def __init__(
        self,
        operation: str,
        service: AgentMemoryService,
        *,
        workspace: str,
        authorize_scope: Callable[[InvocationContext, str], bool],
    ) -> None:
        self.spec = MEMORY_SPECS[operation]
        self.operation, self.service, self.workspace, self.authorize_scope = (
            operation,
            service,
            workspace,
            authorize_scope,
        )

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)

        def failure(code: ToolErrorCode, message: str) -> ToolResult:
            return ToolResult(
                context.request_id,
                context.trace_id,
                self.spec.tool_id,
                1,
                ToolStatus.FAILED,
                error=ToolError(
                    code, message, code in {ToolErrorCode.RETRYABLE, ToolErrorCode.UNAVAILABLE}
                ),
                started_at=started,
                completed_at=datetime.now(UTC),
            )

        scope = arguments.get("scope")
        if (
            context.workspace_id != self.workspace
            or self.spec.required_permissions - context.permissions
            or not isinstance(scope, str)
        ):
            return failure(ToolErrorCode.PERMISSION_DENIED, "memory_permission_denied")
        try:
            authorized = self.authorize_scope(context, scope) is True
        except Exception:
            authorized = False
        if not authorized:
            return failure(ToolErrorCode.PERMISSION_DENIED, "memory_permission_denied")
        if context.deadline <= started:
            return failure(ToolErrorCode.TIMEOUT, "memory_deadline_exceeded")
        if set(arguments) - _FIELDS[self.operation]:
            return failure(ToolErrorCode.INVALID_INPUT, "invalid_request")
        try:
            args: dict[str, Any] = dict(arguments, workspace=self.workspace)
            for key in ("event_time", "valid_at", "known_at", "valid_from", "valid_to"):
                if key in args and args[key] is not None:
                    if not isinstance(args[key], str):
                        raise ValueError("invalid time")
                    args[key] = utc(datetime.fromisoformat(args[key]))
            operation = self.operation
            if operation == "assert_fact":
                obj = args.get("object")
                if not isinstance(obj, Mapping) or set(obj) != {"kind", "value"}:
                    raise ValueError("invalid object")
                args["object"] = (
                    EntityRef(obj["value"])
                    if obj["kind"] == "entity"
                    else TypedLiteral(obj["kind"], obj["value"])
                )
                args["validity"] = TimeInterval(
                    args.pop("valid_from", None), args.pop("valid_to", None)
                )
                result: Any = self.service.assert_fact(**args)
            elif operation == "remember_episode":
                result = self.service.remember_episode(**args)
            elif operation == "recall":
                max_chars, max_tokens = (
                    args.pop("max_characters", 12000),
                    args.pop("max_tokens", 4000),
                )
                args["workspace_ref"] = args.pop("workspace")
                args["scope_ref"] = args.pop("scope")
                for field in ("types", "states", "center_entities"):
                    if field in args:
                        if not isinstance(args[field], (tuple, list)):
                            raise ValueError("invalid filter")
                        args[field] = tuple(args[field])
                result = self.service.recall(
                    MemoryQuery(**args), max_characters=max_chars, max_tokens=max_tokens
                )
            elif operation == "work_extractions":
                result = self.service.work_extractions(**args)
            elif operation == "resolve_entity":
                result = self.service.resolve_entity(**args)
            elif operation == "get_episode":
                result = self.service.get_episode(**args)
            elif operation == "get_fact_history":
                result = {"facts": self.service.get_fact_history(**args)}
            elif operation == "retry_extraction":
                result = self.service.retry_extraction(**args)
            elif operation == "archive_memory":
                result = self.service.archive_memory(**args)
            elif operation == "erase_scope":
                result = self.service.erase_scope(**args)
            else:
                result = self.service.doctor()
            payload = _json_value(result)
            json.dumps(payload, allow_nan=False)
        except (ValueError, TypeError, KeyError) as exc:
            from capicove.agent.memory_policy import request_error

            return failure(ToolErrorCode.INVALID_INPUT, request_error(exc))
        except RuntimeError as exc:
            stable = {
                "permission_denied",
                "scope_erased",
                "evidence_unavailable",
                "privacy_authority_unavailable",
                "corrupt_event_log",
                "write_conflict",
                "consistency_timeout",
                "lease_lost",
                "extraction_unavailable",
                "episode_not_found",
                "entity_not_found",
                "memory_not_found",
                "erasure_checkpoint_pending",
                "memory_backend_unavailable",
                "entity_resolution_required",
            }
            code = str(exc) if str(exc) in stable else "memory_unavailable"
            return failure(
                ToolErrorCode.PERMISSION_DENIED
                if code == "permission_denied"
                else ToolErrorCode.UNAVAILABLE,
                code,
            )
        if datetime.now(UTC) >= context.deadline:
            return failure(ToolErrorCode.TIMEOUT, "memory_deadline_exceeded_retry_same_key")
        return ToolResult(
            context.request_id,
            context.trace_id,
            self.spec.tool_id,
            1,
            ToolStatus.SUCCEEDED,
            evidence_refs=tuple(sorted(_evidence_ids(payload))),
            result=payload,
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def encode_memory_tool_result(result: ToolResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "tool_id": result.tool_id,
        "tool_version": result.tool_version,
        "status": result.status.value,
        "result": _plain(result.result),
        "evidence_refs": list(result.evidence_refs),
        "error": None
        if result.error is None
        else {
            "code": result.error.code.value,
            "message": result.error.message,
            "retryable": result.error.retryable,
        },
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def invoke_memory_tool(
    *,
    recorder: TraceRecorder | None,
    spec: ToolSpec,
    handler: ToolHandler,
    context: InvocationContext,
    arguments: Mapping[str, object],
) -> ToolResult:
    """Record content-free memory outcomes; return authorized content only to the caller."""
    if recorder is None:
        return handler(context, arguments)
    from capicove.tools.tracing import invoke_one_step_tool

    captured: list[ToolResult] = []

    def metadata_handler(ctx: InvocationContext, args: Mapping[str, object], /) -> ToolResult:
        original = handler(ctx, args)
        captured.append(original)
        metadata = {"content_redacted": True}
        return dataclasses.replace(
            original, result=metadata if original.result is not None else None
        )

    invoke_one_step_tool(
        recorder=recorder, spec=spec, handler=metadata_handler, context=context, arguments=arguments
    )
    return captured[0]


def _evidence_ids(value: Any) -> set[str]:
    if isinstance(value, dict):
        own = {value["evidence_id"]} if isinstance(value.get("evidence_id"), str) else set()
        return own.union(*(_evidence_ids(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_evidence_ids(item) for item in value))
    return set()
