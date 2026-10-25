"""Canonical immutable audio import Tool (RFC-0022)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

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
from capicove.processing.audio import AudioImportError
from capicove.processing.audio.ports import AudioPathPolicyError
from capicove.processing.audio.processor import AudioProcessor, evidence_exists
from capicove.tools.audio_transcribe import AudioPathAuthorizer, audio_failure_result


AUDIO_IMPORT_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source"],
    "properties": {
        "source": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "path"],
            "properties": {
                "type": {"const": "path"},
                "path": {"type": "string", "minLength": 1, "maxLength": 4096},
            },
        },
        "audio_stream_index": {"type": "integer", "minimum": 0},
    },
}

AUDIO_IMPORT_SPEC = ToolSpec(
    tool_id="capicove.audio.import",
    contract_version=1,
    description="Preserve one local audio file as immutable content-addressed evidence.",
    input_schema=AUDIO_IMPORT_INPUT_SCHEMA,
    output_schema={"type": "object"},
    modalities=("audio",),
    required_permissions=frozenset(
        {"audio:read", "audio:write", "artifact:write", "subprocess:execute"}
    ),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    network_access=False,
    deterministic=True,
    idempotent=True,
    timeout_seconds=300.0,
    resources=ToolResources(
        filesystem_read=True, filesystem_write=True, subprocesses=("ffmpeg", "ffprobe")
    ),
)


class AudioImportToolHandler:
    """Authorize, preserve, and describe one audio file without transcribing it."""

    def __init__(
        self,
        processor: AudioProcessor,
        path_authorizer: AudioPathAuthorizer,
        *,
        execution_policy_ref: str,
        store_root: Path,
    ) -> None:
        if not store_root.is_absolute():
            raise ValueError("store_root must be absolute")
        self._processor = processor
        self._authorizer = path_authorizer
        self._policy = execution_policy_ref
        self._store_root = store_root

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        """Import once; identical bytes reuse the existing evidence."""
        started = datetime.now(UTC)
        if AUDIO_IMPORT_SPEC.required_permissions.difference(context.permissions):
            return _failure(context, started, ToolErrorCode.PERMISSION_DENIED)
        try:
            path, index = _decode(arguments)
            source = self._authorizer.authorize_read(self._policy, path)
            digest = _sha256(source)
            existed = evidence_exists(self._store_root, digest)
            options: dict[str, object] = {"deadline": context.deadline}
            if index is not None:
                options["audio_stream_index"] = index
            imported = self._processor.import_file_with_selection(
                source,
                self._store_root,
                **options,  # type: ignore[arg-type]
            )
        except AudioPathPolicyError:
            return _failure(context, started, ToolErrorCode.PERMISSION_DENIED)
        except AudioImportError as error:
            return audio_failure_result(
                AUDIO_IMPORT_SPEC,
                context,
                started,
                error.code,
                str(error),
                preflight_audio_stream_count=error.preflight_audio_stream_count,
            )
        except (OSError, TypeError, ValueError):
            return _failure(context, started, ToolErrorCode.INVALID_INPUT)
        evidence, selection = imported.evidence, imported.stream_selection
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=AUDIO_IMPORT_SPEC.tool_id,
            tool_version=AUDIO_IMPORT_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED,
            result={
                "evidence": {
                    "evidence_id": evidence.evidence_id,
                    "sha256": evidence.sha256,
                    "size_bytes": evidence.size_bytes,
                    "media_type": evidence.media_type,
                    "source_name": evidence.source_name,
                },
                "audio_stream_selection": {
                    "schema_version": selection.schema_version,
                    "selected_audio_stream_index": selection.selected_audio_stream_index,
                    "index_semantics": selection.index_semantics,
                    "selection_method": selection.selection_method,
                    "preflight_audio_stream_count": selection.preflight_audio_stream_count,
                    "validation_scope": selection.validation_scope,
                },
                "reused": existed and evidence.sha256 == digest,
            },
            evidence_refs=(evidence.evidence_id,),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def _decode(arguments: Mapping[str, object]) -> tuple[Path, int | None]:
    if set(arguments) - {"source", "audio_stream_index"} or "source" not in arguments:
        raise ValueError("arguments contain unknown or missing fields")
    source = arguments["source"]
    if not isinstance(source, Mapping) or set(source) != {"type", "path"}:
        raise ValueError("source must be a path")
    path = source.get("path")
    if source.get("type") != "path" or not isinstance(path, str) or not path.strip():
        raise ValueError("source must be a path")
    index = arguments.get("audio_stream_index")
    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
        raise ValueError("audio_stream_index must be a non-negative integer")
    return Path(path), index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _failure(context: InvocationContext, started: datetime, code: ToolErrorCode) -> ToolResult:
    messages = {
        ToolErrorCode.PERMISSION_DENIED: "Audio access is not permitted.",
        ToolErrorCode.INVALID_INPUT: "Audio import arguments are invalid.",
    }
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=AUDIO_IMPORT_SPEC.tool_id,
        tool_version=AUDIO_IMPORT_SPEC.contract_version,
        status=ToolStatus.FAILED,
        error=ToolError(code, messages[code], retryable=False),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


__all__ = ["AUDIO_IMPORT_SPEC", "AudioImportToolHandler"]
