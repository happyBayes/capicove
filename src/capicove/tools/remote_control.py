"""Remote artifact and operation Tool contracts bound to one server-owned authority.

These are the RFC-0028 control-plane Tools. Each handler set is created per grant
with the complete ``AuthorityKey`` captured from server policy; no argument can
name an owner, root, path or backend. Artifact writes and operation submission
carry their own durable idempotency records, which is why the remote dispatcher
may admit them although they write.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.contracts.remote import AuthorityKey


_ID = {"type": "string", "minLength": 1, "maxLength": 64}
_SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_KEY = {"type": "string", "pattern": "^[A-Za-z0-9._:-]{1,128}$"}
_COUNT = {"type": "integer", "minimum": 0, "maximum": 2**40}
_CHUNK_TEXT = {"type": "string", "minLength": 1, "maxLength": 349_528}


def _spec(
    tool_id: str,
    description: str,
    properties: Mapping[str, object],
    permission: str,
    effect: SideEffect,
) -> ToolSpec:
    return ToolSpec(
        tool_id=tool_id,
        contract_version=1,
        description=description,
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(properties),
            "properties": dict(properties),
        },
        output_schema={"type": "object"},
        modalities=("text",),
        required_permissions=frozenset({permission}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        side_effect=effect,
        network_access=False,
        idempotent=True,
        timeout_seconds=30,
    )


ARTIFACT_SPECS: dict[str, ToolSpec] = {
    "begin-upload": _spec(
        "capicove.artifact.begin-upload",
        "Reserve quota for one upload; repeating the same key returns the same upload.",
        {
            "size": {"type": "integer", "minimum": 1, "maximum": 2**40},
            "sha256": _SHA,
            "media_type": {"type": "string", "minLength": 3, "maxLength": 192},
            "idempotency_key": _KEY,
        },
        "artifact:write",
        SideEffect.WRITE,
    ),
    "write-chunk": _spec(
        "capicove.artifact.write-chunk",
        "Store one numbered base64 chunk of at most 256 KiB decoded bytes.",
        {"upload_id": _ID, "index": _COUNT, "data_base64": _CHUNK_TEXT},
        "artifact:write",
        SideEffect.WRITE,
    ),
    "upload-status": _spec(
        "capicove.artifact.upload-status",
        "Report received chunk indices and digests of one owned upload.",
        {"upload_id": _ID},
        "artifact:write",
        SideEffect.READ,
    ),
    "abort-upload": _spec(
        "capicove.artifact.abort-upload",
        "Release an incomplete owned upload exactly once.",
        {"upload_id": _ID},
        "artifact:write",
        SideEffect.WRITE,
    ),
    "commit-upload": _spec(
        "capicove.artifact.commit-upload",
        "Verify size, SHA-256 and media type, then publish one immutable artifact.",
        {"upload_id": _ID},
        "artifact:write",
        SideEffect.WRITE,
    ),
    "stat": _spec(
        "capicove.artifact.stat",
        "Return the reference of one owned live artifact.",
        {"artifact_id": _ID},
        "artifact:read",
        SideEffect.READ,
    ),
    "read-chunk": _spec(
        "capicove.artifact.read-chunk",
        "Read at most 256 KiB of one owned artifact as base64.",
        {
            "artifact_id": _ID,
            "offset": _COUNT,
            "length": {"type": "integer", "minimum": 1, "maximum": 262_144},
        },
        "artifact:read",
        SideEffect.READ,
    ),
    "delete": _spec(
        "capicove.artifact.delete",
        "Tombstone one owned artifact and release its stored quota.",
        {"artifact_id": _ID},
        "artifact:delete",
        SideEffect.WRITE,
    ),
}
OPERATION_SPECS: dict[str, ToolSpec] = {
    "status": _spec(
        "capicove.operation.status",
        "Return the current state of one owned operation.",
        {"operation_id": _ID},
        "operation:submit",
        SideEffect.READ,
    ),
    "result": _spec(
        "capicove.operation.result",
        "Return the terminal output of one owned operation.",
        {"operation_id": _ID},
        "operation:submit",
        SideEffect.READ,
    ),
    "cancel": _spec(
        "capicove.operation.cancel",
        "Cancel queued work or request cancellation of running work.",
        {"operation_id": _ID},
        "operation:submit",
        SideEffect.WRITE,
    ),
}
_FAILURES = {
    "permission_denied": (ToolErrorCode.PERMISSION_DENIED, False),
    "approval_required": (ToolErrorCode.PERMISSION_DENIED, False),
    "quota_exceeded": (ToolErrorCode.UNAVAILABLE, True),
    "draining": (ToolErrorCode.UNAVAILABLE, True),
    "audit_unavailable": (ToolErrorCode.UNAVAILABLE, True),
}


def _result(
    spec: ToolSpec,
    context: InvocationContext,
    started: datetime,
    payload: dict[str, Any] | None,
    code: str | None = None,
) -> ToolResult:
    error = None
    if code is not None:
        kind, retryable = _FAILURES.get(code, (ToolErrorCode.INVALID_INPUT, False))
        error = ToolError(
            code=kind, message=f"Remote control request failed: {code}.", retryable=retryable
        )
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=spec.tool_id,
        tool_version=spec.contract_version,
        status=ToolStatus.SUCCEEDED if code is None else ToolStatus.REJECTED,
        result=payload,
        error=error,
        started_at=started,
        completed_at=datetime.now(UTC),
    )


Handler = Callable[[InvocationContext, Mapping[str, Any]], ToolResult]


def artifact_handlers(service: Any, authority: AuthorityKey) -> dict[str, Handler]:
    """Bind artifact Tools to one artifact service and one server-owned authority."""
    from capicove.runtime.remote_artifacts import ArtifactError

    def call(operation: str, run: Callable[[Mapping[str, Any]], Any]) -> Handler:
        spec = ARTIFACT_SPECS[operation]

        def handler(context: InvocationContext, arguments: Mapping[str, Any]) -> ToolResult:
            started = datetime.now(UTC)
            try:
                value = run(arguments)
            except ArtifactError as error:
                return _result(spec, context, started, None, error.code)
            return _result(spec, context, started, value)

        return handler

    def write(arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            data = base64.b64decode(arguments["data_base64"], validate=True)
        except (binascii.Error, ValueError):
            raise ArtifactError("invalid_input", "chunk is not base64") from None
        return asdict(
            service.write_chunk(authority, arguments["upload_id"], arguments["index"], data)
        )

    def read(arguments: Mapping[str, Any]) -> dict[str, Any]:
        data = service.read_chunk(
            authority, arguments["artifact_id"], arguments["offset"], arguments["length"]
        )
        return {"data_base64": base64.b64encode(data).decode("ascii"), "length": len(data)}

    def delete(arguments: Mapping[str, Any]) -> dict[str, Any]:
        service.delete(authority, arguments["artifact_id"])
        return {"deleted": True}

    def abort(arguments: Mapping[str, Any]) -> dict[str, Any]:
        service.abort(authority, arguments["upload_id"])
        return {"aborted": True}

    return {
        ARTIFACT_SPECS["begin-upload"].tool_id: call(
            "begin-upload",
            lambda arguments: asdict(
                service.begin(
                    authority,
                    size=arguments["size"],
                    sha256=arguments["sha256"],
                    media_type=arguments["media_type"],
                    idempotency_key=arguments["idempotency_key"],
                )
            ),
        ),
        ARTIFACT_SPECS["write-chunk"].tool_id: call("write-chunk", write),
        ARTIFACT_SPECS["upload-status"].tool_id: call(
            "upload-status",
            lambda arguments: asdict(service.status(authority, arguments["upload_id"])),
        ),
        ARTIFACT_SPECS["abort-upload"].tool_id: call("abort-upload", abort),
        ARTIFACT_SPECS["commit-upload"].tool_id: call(
            "commit-upload",
            lambda arguments: asdict(service.commit(authority, arguments["upload_id"])),
        ),
        ARTIFACT_SPECS["stat"].tool_id: call(
            "stat", lambda arguments: asdict(service.stat(authority, arguments["artifact_id"]))
        ),
        ARTIFACT_SPECS["read-chunk"].tool_id: call("read-chunk", read),
        ARTIFACT_SPECS["delete"].tool_id: call("delete", delete),
    }


def operation_handlers(service: Any, authority: AuthorityKey) -> dict[str, Handler]:
    """Bind operation Tools to one operation service and one server-owned authority."""
    from capicove.runtime.remote_operations import OperationError

    def call(operation: str, run: Callable[[InvocationContext, Mapping[str, Any]], Any]) -> Handler:
        spec = OPERATION_SPECS[operation]

        def handler(context: InvocationContext, arguments: Mapping[str, Any]) -> ToolResult:
            started = datetime.now(UTC)
            try:
                value = run(context, arguments)
            except OperationError as error:
                return _result(spec, context, started, None, error.code)
            return _result(spec, context, started, value)

        return handler

    return {
        OPERATION_SPECS["status"].tool_id: call(
            "status",
            lambda context, arguments: asdict(
                service.status(authority, context.permissions, arguments["operation_id"])
            ),
        ),
        OPERATION_SPECS["result"].tool_id: call(
            "result",
            lambda context, arguments: service.result(
                authority, context.permissions, arguments["operation_id"]
            ),
        ),
        OPERATION_SPECS["cancel"].tool_id: call(
            "cancel",
            lambda context, arguments: asdict(
                service.cancel(authority, context.permissions, arguments["operation_id"])
            ),
        ),
    }


__all__ = ["ARTIFACT_SPECS", "OPERATION_SPECS", "artifact_handlers", "operation_handlers"]
