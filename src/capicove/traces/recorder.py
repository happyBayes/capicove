"""Explicit fail-closed recording of observable multi-step Agent Episodes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime

from capicove.contracts import InvocationContext, PrivacyClass, ToolResult, ToolSpec, ToolStatus
from capicove.traces._json import freeze_json, thaw_json
from capicove.traces.codec import canonical_json_bytes, encode_trace_draft
from capicove.traces.content import (
    AgentLineage,
    MediaContent,
    MediaReference,
    MessageRole,
    TextContent,
    TraceError,
    TraceLineage,
    TraceModality,
)
from capicove.traces.events import (
    ArtifactProduced,
    BatchItemObserved,
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    EpisodeStatus,
    MessageObserved,
    ToolCallObserved,
    ToolResultObserved,
    TracePayload,
)
from capicove.traces.models import TraceRecord, TraceRecordDraft
from capicove.traces.ports import (
    TraceConflictError,
    TraceLifecycleError,
    TraceStore,
)
from capicove.traces.sanitization import (
    TracePrivacyError,
    TraceSanitizer,
    validate_trace_privacy,
)


_UNEXPECTED_TOOL_ERROR = "Tool execution raised an unexpected exception."
_PRIVACY_ORDER = {
    PrivacyClass.PUBLIC: 0,
    PrivacyClass.INTERNAL: 1,
    PrivacyClass.CONFIDENTIAL: 2,
    PrivacyClass.RESTRICTED: 3,
}


class TraceRecordingError(RuntimeError):
    """Report a fail-closed recording or session protocol failure."""


class TraceRecorder:
    """Start explicit trace sessions against one configured local store."""

    def __init__(
        self,
        store: TraceStore,
        *,
        sanitizer: TraceSanitizer,
        clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(sanitizer, TraceSanitizer):
            raise ValueError("sanitizer must be a TraceSanitizer")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._store = store
        self._sanitizer = sanitizer
        self._clock = clock

    @property
    def sanitizer(self) -> TraceSanitizer:
        """Return the recorder-owned privacy policy implementation."""
        return self._sanitizer

    def start_session(
        self,
        *,
        episode_id: str,
        trace_id: str,
        goal: str,
        workspace_id: str,
        actor_id: str,
        environment_id: str,
        environment_version: int,
        tool_catalog_sha256: str,
        active_asset_versions: Mapping[str, int],
        agent_lineage: AgentLineage | None = None,
        privacy_class: PrivacyClass = PrivacyClass.INTERNAL,
    ) -> TraceSession:
        """Persist one Episode start before returning an active session."""
        draft = TraceRecordDraft(
            episode_id=episode_id,
            trace_id=trace_id,
            occurred_at=self._now(),
            payload=EpisodeStarted(
                goal=goal,
                workspace_id=workspace_id,
                actor_id=actor_id,
                environment_id=environment_id,
                environment_version=environment_version,
                tool_catalog_sha256=tool_catalog_sha256,
                active_asset_versions=active_asset_versions,
                agent_lineage=agent_lineage,
            ),
            modalities=(),
            privacy_class=privacy_class,
        )
        start_record = self._append(draft, idempotency_key="episode.started")
        return TraceSession(
            recorder=self,
            episode_id=episode_id,
            trace_id=trace_id,
            privacy_class=privacy_class,
            start_record=start_record,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise TraceRecordingError("trace clock must return a timezone-aware datetime")
        return value

    def _append(self, draft: TraceRecordDraft, *, idempotency_key: str) -> TraceRecord:
        try:
            validate_trace_privacy(encode_trace_draft(draft))
        except TracePrivacyError as error:
            raise TraceRecordingError("trace privacy validation failed") from error
        try:
            return self._store.append(draft, idempotency_key=idempotency_key)
        except (TraceConflictError, TraceLifecycleError):
            raise
        except Exception as error:
            raise TraceRecordingError("trace persistence failed") from error


class TraceSession:
    """Record one explicit observable Agent Episode in append order."""

    def __init__(
        self,
        *,
        recorder: TraceRecorder,
        episode_id: str,
        trace_id: str,
        privacy_class: PrivacyClass,
        start_record: TraceRecord,
    ) -> None:
        self._recorder = recorder
        self._episode_id = episode_id
        self._trace_id = trace_id
        self._privacy_class = privacy_class
        self._start_record = start_record
        self._completed = False
        self._open_calls: dict[str, tuple[ToolSpec, int, TraceRecord, PrivacyClass]] = {}

    @property
    def episode_id(self) -> str:
        """Return the stable Episode identity."""
        return self._episode_id

    @property
    def trace_id(self) -> str:
        """Return the distributed trace identity."""
        return self._trace_id

    def record_message(
        self,
        *,
        message_id: str,
        role: MessageRole,
        content: tuple[TextContent | MediaContent, ...],
        media: tuple[MediaReference, ...] = (),
    ) -> TraceRecord:
        """Append one visible, ordered, sanitized message."""
        self._require_active()
        _require_non_blank("message_id", message_id)
        sanitized_content = self._recorder.sanitizer.sanitize_content(content)
        content_media_ids = tuple(
            block.media_id for block in sanitized_content if isinstance(block, MediaContent)
        )
        if len(set(content_media_ids)) != len(content_media_ids):
            raise TraceRecordingError("message content contains duplicate media IDs")
        reference_ids = tuple(reference.media_id for reference in media)
        if len(set(reference_ids)) != len(reference_ids):
            raise TraceRecordingError("message contains duplicate media references")
        if set(content_media_ids) != set(reference_ids):
            raise TraceRecordingError("message media blocks must match attached media references")
        return self._append(
            MessageObserved(role, sanitized_content),
            idempotency_key=f"message:{message_id}",
            modalities=_modalities(media),
            media=media,
        )

    def record_tool_call(
        self,
        *,
        context: InvocationContext,
        spec: ToolSpec,
        arguments: Mapping[str, object],
        media: tuple[MediaReference, ...],
        attempt: int,
    ) -> TraceRecord:
        """Append one sanitized Tool call before execution."""
        self._require_active()
        self._validate_context(context)
        _require_positive_int("attempt", attempt)
        if context.request_id in self._open_calls:
            raise TraceRecordingError("request_id already has an open Tool call")
        privacy = _most_restrictive(
            self._privacy_class,
            context.privacy_class,
            spec.privacy_class,
        )
        record = self._append(
            ToolCallObserved(
                tool_id=spec.tool_id,
                tool_version=spec.contract_version,
                arguments=arguments,
                arguments_schema_sha256=_schema_sha256(spec.input_schema),
                attempt=attempt,
            ),
            idempotency_key=f"tool.call:{context.request_id}:{attempt}",
            request_id=context.request_id,
            modalities=_tool_modalities(spec, media),
            media=media,
            privacy_class=privacy,
        )
        self._open_calls[context.request_id] = (spec, attempt, record, privacy)
        return record

    def record_tool_result(
        self,
        *,
        result: ToolResult,
        spec: ToolSpec,
        attempt: int,
    ) -> TraceRecord:
        """Append one canonical Tool outcome after identity validation."""
        self._require_active()
        call_record, privacy = self._validate_result_identity(result, spec, attempt)
        sanitized_result = (
            None
            if result.result is None
            else self._recorder.sanitizer.sanitize_json(
                result.result,
                field_name="Tool result",
            )
        )
        trace_error = None
        if result.error is not None:
            trace_error = TraceError(
                result.error.code.value,
                self._recorder.sanitizer.sanitize_text(result.error.message),
                result.error.retryable,
            )
        lineage = _result_lineage(sanitized_result)
        record = self._append(
            ToolResultObserved(
                tool_id=result.tool_id,
                tool_version=result.tool_version,
                status=result.status,
                result_schema_sha256=_schema_sha256(spec.output_schema),
                result=sanitized_result,
                evidence_refs=result.evidence_refs,
                metrics=result.metrics,
            ),
            idempotency_key=f"tool.result:{result.request_id}:{attempt}",
            request_id=result.request_id,
            parent_record_id=call_record.record_id,
            modalities=_tool_modalities(spec, ()),
            lineage=lineage,
            error=trace_error,
            privacy_class=privacy,
        )
        self._open_calls.pop(result.request_id, None)
        return record

    def record_tool_exception(
        self,
        *,
        context: InvocationContext,
        spec: ToolSpec,
        attempt: int,
    ) -> TraceRecord:
        """Append a fixed public internal failure without exception text."""
        self._require_active()
        self._validate_context(context)
        call_record, privacy = self._validate_open_call(context.request_id, spec, attempt)
        record = self._append(
            ToolResultObserved(
                tool_id=spec.tool_id,
                tool_version=spec.contract_version,
                status=ToolStatus.FAILED,
                result_schema_sha256=_schema_sha256(spec.output_schema),
            ),
            idempotency_key=f"tool.result:{context.request_id}:{attempt}",
            request_id=context.request_id,
            parent_record_id=call_record.record_id,
            modalities=_tool_modalities(spec, ()),
            error=TraceError("internal", _UNEXPECTED_TOOL_ERROR, False),
            privacy_class=privacy,
        )
        self._open_calls.pop(context.request_id, None)
        return record

    def record_artifact(
        self,
        *,
        event_id: str,
        artifact_ref: str,
        media: MediaReference,
        derived_from: tuple[str, ...] = (),
    ) -> TraceRecord:
        """Append a produced artifact and its immutable media identity."""
        self._require_active()
        _require_non_blank("event_id", event_id)
        if derived_from != media.derived_from:
            raise TraceRecordingError("artifact derivation must match its attached media reference")
        return self._append(
            ArtifactProduced(artifact_ref, media.media_id, derived_from),
            idempotency_key=f"artifact:{event_id}",
            modalities=(media.modality,),
            media=(media,),
        )

    def record_batch_item(
        self,
        *,
        event_id: str,
        payload: BatchItemObserved,
    ) -> TraceRecord:
        """Append one terminal batch item fact."""
        self._require_active()
        _require_non_blank("event_id", event_id)
        return self._append(
            payload,
            idempotency_key=event_id,
            modalities=(payload.modality,),
        )

    def record_feedback(
        self,
        *,
        feedback_id: str,
        payload: EpisodeFeedback,
    ) -> TraceRecord:
        """Append linked feedback after Episode completion."""
        if not self._completed:
            raise TraceRecordingError("feedback requires a completed session")
        _require_non_blank("feedback_id", feedback_id)
        sanitized = EpisodeFeedback(
            payload.kind,
            self._recorder.sanitizer.sanitize_json(
                payload.value,
                field_name="feedback value",
            ),
            payload.provenance,
            payload.target_record_id,
        )
        return self._append(sanitized, idempotency_key=f"feedback:{feedback_id}")

    def complete(
        self,
        *,
        status: EpisodeStatus,
        outcome: str | None = None,
    ) -> TraceRecord:
        """Append the one terminal Episode record."""
        self._require_active()
        if self._open_calls:
            raise TraceRecordingError("cannot complete a session with an open Tool call")
        record = self._append(
            EpisodeCompleted(status, outcome),
            idempotency_key="episode.completed",
        )
        self._completed = True
        return record

    def _append(
        self,
        payload: TracePayload,
        *,
        idempotency_key: str,
        modalities: tuple[TraceModality, ...] = (),
        request_id: str | None = None,
        parent_record_id: str | None = None,
        media: tuple[MediaReference, ...] = (),
        lineage: TraceLineage | None = None,
        error: TraceError | None = None,
        privacy_class: PrivacyClass | None = None,
    ) -> TraceRecord:
        draft = TraceRecordDraft(
            episode_id=self._episode_id,
            trace_id=self._trace_id,
            occurred_at=self._recorder._now(),
            payload=payload,
            modalities=modalities,
            request_id=request_id,
            parent_record_id=parent_record_id,
            media=media,
            lineage=lineage,
            error=error,
            privacy_class=privacy_class or self._privacy_class,
        )
        return self._recorder._append(draft, idempotency_key=idempotency_key)

    def _require_active(self) -> None:
        if self._completed:
            raise TraceRecordingError("trace session is already completed")

    def _validate_context(self, context: InvocationContext) -> None:
        if not isinstance(context, InvocationContext):
            raise TraceRecordingError("context must be an InvocationContext")
        if context.trace_id != self._trace_id:
            raise TraceRecordingError("context trace_id must match the session")

    def _validate_open_call(
        self,
        request_id: str,
        spec: ToolSpec,
        attempt: int,
    ) -> tuple[TraceRecord, PrivacyClass]:
        open_call = self._open_calls.get(request_id)
        if open_call is None:
            raise TraceRecordingError("request_id has no open Tool call")
        expected_spec, expected_attempt, call_record, privacy = open_call
        if expected_spec.registry_key != spec.registry_key or expected_attempt != attempt:
            raise TraceRecordingError("Tool result does not match its call identity")
        return call_record, privacy

    def _validate_result_identity(
        self,
        result: ToolResult,
        spec: ToolSpec,
        attempt: int,
    ) -> tuple[TraceRecord, PrivacyClass]:
        if result.trace_id != self._trace_id:
            raise TraceRecordingError("Tool result trace_id must match the session")
        call_record, privacy = self._validate_open_call(result.request_id, spec, attempt)
        if (result.tool_id, result.tool_version) != spec.registry_key:
            raise TraceRecordingError("Tool result identity must match the Tool spec")
        return call_record, privacy


def _schema_sha256(schema: Mapping[str, object]) -> str:
    frozen = freeze_json(schema, field_name="schema")
    thawed = thaw_json(frozen)
    if not isinstance(thawed, dict):
        raise TraceRecordingError("Tool schema must be a JSON object")
    return hashlib.sha256(canonical_json_bytes(thawed)).hexdigest()


def _result_lineage(result: Mapping[str, object] | None) -> TraceLineage | None:
    if result is None:
        return None
    lineage = result.get("lineage")
    if not isinstance(lineage, Mapping):
        return None
    return TraceLineage({"result": lineage})


def _modalities(media: tuple[MediaReference, ...]) -> tuple[TraceModality, ...]:
    return tuple(dict.fromkeys(reference.modality for reference in media))


def _tool_modalities(
    spec: ToolSpec,
    media: tuple[MediaReference, ...],
) -> tuple[TraceModality, ...]:
    modalities = list(_modalities(media))
    for value in spec.modalities:
        try:
            modality = TraceModality(value)
        except ValueError:
            continue
        if modality not in modalities:
            modalities.append(modality)
    return tuple(modalities)


def _most_restrictive(*values: PrivacyClass) -> PrivacyClass:
    return max(values, key=_PRIVACY_ORDER.__getitem__)


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TraceRecordingError(f"{field_name} must not be blank")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TraceRecordingError(f"{field_name} must be a positive integer")
