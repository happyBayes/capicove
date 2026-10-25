"""Shared fail-closed tracing boundary for transport-neutral Tool handlers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import cast

from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolSpec, ToolStatus
from capicove.traces import EpisodeStatus, TraceRecorder
from capicove.traces._json import freeze_json, thaw_json
from capicove.traces.codec import canonical_json_bytes
from capicove.traces.recorder import TraceSession
from capicove.traces.sanitization import TraceSanitizer


class TraceProtocolError(RuntimeError):
    """Report a Tool result that violates invocation identity invariants."""


class TracedToolHandler:
    """Record one sanitized call and canonical result around a real Tool handler."""

    def __init__(
        self,
        spec: ToolSpec,
        handler: ToolHandler,
        session: TraceSession,
        sanitizer: TraceSanitizer,
        *,
        attempt: int = 1,
    ) -> None:
        if not isinstance(spec, ToolSpec):
            raise ValueError("spec must be a ToolSpec")
        if not callable(handler):
            raise ValueError("handler must be callable")
        if not isinstance(session, TraceSession):
            raise ValueError("session must be a TraceSession")
        if not isinstance(sanitizer, TraceSanitizer):
            raise ValueError("sanitizer must be a TraceSanitizer")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        self._spec = spec
        self._handler = handler
        self._session = session
        self._sanitizer = sanitizer
        self._attempt = attempt

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        """Invoke the wrapped Tool with fixed fail-closed recording order."""
        sanitized = self._sanitizer.sanitize(self._spec, arguments)
        self._session.record_tool_call(
            context=context,
            spec=self._spec,
            arguments=sanitized.arguments,
            media=sanitized.media,
            attempt=self._attempt,
        )
        try:
            result = self._handler(context, arguments)
        except Exception as primary_error:
            try:
                self._session.record_tool_exception(
                    context=context,
                    spec=self._spec,
                    attempt=self._attempt,
                )
            except Exception:
                _add_trace_failure_note(primary_error, "Tool exception result")
            raise
        try:
            _validate_tool_result_identity(context, self._spec, result)
        except TraceProtocolError as primary_error:
            try:
                self._session.record_tool_exception(
                    context=context,
                    spec=self._spec,
                    attempt=self._attempt,
                )
            except Exception:
                _add_trace_failure_note(primary_error, "Tool protocol failure")
            raise
        self._session.record_tool_result(
            result=result,
            spec=self._spec,
            attempt=self._attempt,
        )
        return result


def invoke_one_step_tool(
    *,
    recorder: TraceRecorder | None,
    spec: ToolSpec,
    handler: ToolHandler,
    context: InvocationContext,
    arguments: Mapping[str, object],
    attempt: int = 1,
    episode_id: str | None = None,
) -> ToolResult:
    """Invoke one Tool directly or as one complete opt-in trace Episode."""
    if recorder is None:
        return handler(context, arguments)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("attempt must be a positive integer")
    session = recorder.start_session(
        episode_id=episode_id or context.trace_id,
        trace_id=context.trace_id,
        goal=f"Invoke {spec.tool_id}",
        workspace_id=context.workspace_id,
        actor_id=context.actor_id,
        environment_id=context.environment_id,
        environment_version=1,
        tool_catalog_sha256=tool_catalog_sha256((spec,)),
        active_asset_versions={},
        privacy_class=context.privacy_class,
    )
    traced = TracedToolHandler(
        spec,
        handler,
        session,
        recorder.sanitizer,
        attempt=attempt,
    )
    try:
        result = traced(context, arguments)
    except Exception as primary_error:
        try:
            session.complete(status=EpisodeStatus.FAILED)
        except Exception:
            _add_trace_failure_note(primary_error, "Episode completion")
        raise
    session.complete(status=_episode_status(result.status))
    return result


def tool_catalog_sha256(specs: Iterable[ToolSpec], /) -> str:
    """Hash complete public Tool contracts independently of iterable order."""
    encoded: list[dict[str, object]] = []
    for spec in specs:
        if not isinstance(spec, ToolSpec):
            raise ValueError("Tool catalog must contain only ToolSpec values")
        encoded.append(
            {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "description": spec.description,
                "input_schema": _plain_json_object(spec.input_schema),
                "output_schema": _plain_json_object(spec.output_schema),
                "modalities": list(spec.modalities),
                "required_permissions": sorted(spec.required_permissions),
                "privacy_class": spec.privacy_class.value,
                "side_effect": spec.side_effect.value,
                "network_access": spec.network_access,
                "deterministic": spec.deterministic,
                "idempotent": spec.idempotent,
                "timeout_seconds": spec.timeout_seconds,
                "redacted_fields": list(spec.redacted_fields),
                "resources": {
                    "filesystem_read": spec.resources.filesystem_read,
                    "filesystem_write": spec.resources.filesystem_write,
                    "subprocesses": list(spec.resources.subprocesses),
                },
            }
        )
    encoded.sort(key=lambda item: (cast(str, item["tool_id"]), cast(int, item["contract_version"])))
    return hashlib.sha256(canonical_json_bytes({"tools": encoded})).hexdigest()


def _episode_status(status: ToolStatus) -> EpisodeStatus:
    return {
        ToolStatus.SUCCEEDED: EpisodeStatus.SUCCEEDED,
        ToolStatus.FAILED: EpisodeStatus.FAILED,
        ToolStatus.REJECTED: EpisodeStatus.REJECTED,
    }[status]


def _validate_tool_result_identity(
    context: InvocationContext,
    spec: ToolSpec,
    result: object,
) -> None:
    if not isinstance(result, ToolResult):
        raise TraceProtocolError("Tool result identity does not match the invocation")
    if (
        result.request_id != context.request_id
        or result.trace_id != context.trace_id
        or (result.tool_id, result.tool_version) != spec.registry_key
    ):
        raise TraceProtocolError("Tool result identity does not match the invocation")


def _plain_json_object(value: Mapping[str, object]) -> dict[str, object]:
    frozen = freeze_json(value, field_name="Tool schema")
    plain = thaw_json(frozen)
    if not isinstance(plain, dict):
        raise ValueError("Tool schema must be a JSON object")
    return cast(dict[str, object], plain)


def _add_trace_failure_note(error: BaseException, phase: str) -> None:
    error.add_note(f"Capicove could not persist the {phase} trace record.")
