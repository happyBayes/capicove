"""Canonical, path-redacted local audio clipping Tool (RFC-0005)."""

from __future__ import annotations

import stat
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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
from capicove.processing.audio import AudioClippingError, AudioClippingService, AudioClipRequest
from capicove.processing.audio.models import AudioFailureCode
from capicove.tools.registry import ToolRegistry


def _closed(properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


_NUMBER = {"type": "number"}
_INTEGER = {"type": "integer"}
_STRING = {"type": "string"}
_HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_PATH = _closed(
    {"type": {"const": "path"}, "path": {"type": "string", "minLength": 1, "maxLength": 4096}}
)
_METADATA: dict[str, object] = {
    "duration_seconds": _NUMBER,
    "sample_rate_hz": _INTEGER,
    "channels": _INTEGER,
    "codec": _STRING,
    "container": _STRING,
    "bit_depth": {"type": ["integer", "null"]},
    "bit_rate_bps": {"type": ["integer", "null"]},
    "audio_stream_index": _INTEGER,
    "container_stream_index": _INTEGER,
}
_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source", "destination", "start_seconds", "end_seconds"],
    "properties": {
        "source": _PATH,
        "destination": _PATH,
        "start_seconds": {"type": "number", "minimum": 0},
        "end_seconds": {"type": "number", "exclusiveMinimum": 0},
        "profile": {"const": "pcm16-wav"},
        "sample_rate_hz": {"type": "integer", "minimum": 8000, "maximum": 96000},
        "channels": {"type": "integer", "enum": [1, 2]},
    },
}

AUDIO_CLIP_SPEC = ToolSpec(
    tool_id="capicove.audio.clip",
    contract_version=1,
    description="Extract one sample-accurate time range from a local audio file to a new PCM16 WAV. Requires exactly one audio stream; never overwrites a destination.",
    input_schema=_INPUT_SCHEMA,
    output_schema=_closed(
        {
            "requested_range": _closed({"start_seconds": _NUMBER, "end_seconds": _NUMBER}),
            "actual_range": _closed(
                {
                    "start_seconds": _NUMBER,
                    "end_seconds": _NUMBER,
                    "start_frame": _INTEGER,
                    "end_frame": _INTEGER,
                }
            ),
            "input": _closed(_METADATA),
            "output": _closed({**_METADATA, "frame_count": _INTEGER}),
            "evidence": _closed(
                {
                    "evidence_id": _STRING,
                    "sha256": _HASH,
                    "size_bytes": _INTEGER,
                    "media_type": _STRING,
                    "source_name": _STRING,
                }
            ),
            "artifact": _closed(
                {
                    "artifact_id": _STRING,
                    "sha256": _HASH,
                    "size_bytes": _INTEGER,
                    "media_type": {"const": "audio/wav"},
                    "destination_name": _STRING,
                }
            ),
            "lineage": _closed(
                {
                    "backend": {"const": "ffmpeg"},
                    "backend_version": _STRING,
                    "profile": {"const": "pcm16-wav"},
                    "configuration_sha256": _HASH,
                    "source_audio_stream_index": _INTEGER,
                }
            ),
        }
    ),
    modalities=("audio",),
    required_permissions=frozenset({"audio:read", "artifact:write", "subprocess:execute"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    network_access=False,
    deterministic=True,
    idempotent=False,
    timeout_seconds=300,
    redacted_fields=("source.path", "destination.path"),
    resources=ToolResources(
        filesystem_read=True, filesystem_write=True, subprocesses=("ffmpeg", "ffprobe")
    ),
)


def decode_clip_request(arguments: Mapping[str, object]) -> AudioClipRequest:
    """Decode a closed request without coercing null, booleans, or numeric strings."""
    required = {"source", "destination", "start_seconds", "end_seconds"}
    optional = {"profile", "sample_rate_hz", "channels"}
    if not required <= set(arguments) <= required | optional:
        raise ValueError("unknown or missing clipping fields")

    def path_ref(value: object) -> Path:
        if (
            not isinstance(value, Mapping)
            or set(value) != {"type", "path"}
            or value["type"] != "path"
        ):
            raise ValueError("a local path reference is required")
        path = value["path"]
        if not isinstance(path, str) or not path.strip() or len(path) > 4096 or "\0" in path:
            raise ValueError("invalid path reference")
        parsed = Path(path)
        if not parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError("paths must be absolute and normalized")
        return parsed

    if any(arguments.get(field) is None for field in optional if field in arguments):
        raise ValueError("optional fields must be omitted rather than null")
    return AudioClipRequest(
        source=path_ref(arguments["source"]),
        destination=path_ref(arguments["destination"]),
        start_seconds=cast(float, arguments["start_seconds"]),
        end_seconds=cast(float, arguments["end_seconds"]),
        profile=cast(str, arguments.get("profile", "pcm16-wav")),
        sample_rate_hz=cast(int | None, arguments.get("sample_rate_hz")),
        channels=cast(int | None, arguments.get("channels")),
    )


class AudioClipToolHandler:
    """Authorize local reads and writes before invoking the clipping service."""

    def __init__(
        self,
        service: AudioClippingService,
        *,
        read_roots: tuple[Path, ...],
        output_root: Path,
        store_root: Path,
    ) -> None:
        if not read_roots or any(
            not path.is_absolute() for path in (*read_roots, output_root, store_root)
        ):
            raise ValueError("audio authority requires absolute roots")
        self._service = service
        self._read_roots = read_roots
        self._output_root = output_root
        self._store_root = store_root

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        """Return a single canonical result; every failure forbids automatic replay."""
        started = datetime.now(UTC)

        def failure(code: ToolErrorCode, message: str, *, rejected: bool = False) -> ToolResult:
            return ToolResult(
                context.request_id,
                context.trace_id,
                AUDIO_CLIP_SPEC.tool_id,
                1,
                ToolStatus.REJECTED if rejected else ToolStatus.FAILED,
                error=ToolError(code, message, False),
                started_at=started,
                completed_at=datetime.now(UTC),
            )

        if AUDIO_CLIP_SPEC.required_permissions.difference(context.permissions):
            return failure(
                ToolErrorCode.PERMISSION_DENIED,
                "Audio clipping permissions are missing.",
                rejected=True,
            )
        try:
            request = decode_clip_request(arguments)
        except (ValueError, TypeError, OverflowError):
            return failure(ToolErrorCode.INVALID_INPUT, "Invalid audio clipping request.")
        if (
            not any(
                request.source != root and request.source.is_relative_to(root)
                for root in self._read_roots
            )
            or request.destination == self._output_root
            or not request.destination.is_relative_to(self._output_root)
        ):
            return failure(
                ToolErrorCode.PERMISSION_DENIED,
                "Audio paths are outside the configured authority.",
                rejected=True,
            )
        try:
            if not stat.S_ISREG(request.source.lstat().st_mode):
                return failure(
                    ToolErrorCode.PERMISSION_DENIED,
                    "Audio source must be a regular local file.",
                    rejected=True,
                )
            result = self._service.clip_file(request, self._store_root, deadline=context.deadline)
        except AudioClippingError as error:
            if error.code in (
                AudioFailureCode.BACKEND_UNAVAILABLE,
                AudioFailureCode.DECODER_UNAVAILABLE,
            ):
                return failure(
                    ToolErrorCode.UNAVAILABLE,
                    "Secure audio clipping is unavailable in this runtime.",
                )
            if error.code is AudioFailureCode.TIMEOUT:
                return failure(ToolErrorCode.TIMEOUT, "Audio clipping timed out.")
            code = (
                ToolErrorCode.INTERNAL
                if error.code
                in (
                    AudioFailureCode.OUTPUT_FAILED,
                    AudioFailureCode.OUTPUT_CORRUPTED,
                    AudioFailureCode.STORE_FAILED,
                    AudioFailureCode.STORE_CORRUPTED,
                )
                else ToolErrorCode.INVALID_INPUT
            )
            return failure(code, str(error))
        except OSError:
            return failure(ToolErrorCode.INVALID_INPUT, "Audio source could not be read.")
        evidence = cast(dict[str, object], result["evidence"])
        artifact = cast(dict[str, object], result["artifact"])
        return ToolResult(
            context.request_id,
            context.trace_id,
            AUDIO_CLIP_SPEC.tool_id,
            1,
            ToolStatus.SUCCEEDED,
            result=result,
            evidence_refs=(str(evidence["evidence_id"]), str(artifact["artifact_id"])),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def register_audio_clip_tool(registry: ToolRegistry, handler: AudioClipToolHandler) -> None:
    """Register the canonical clipping contract exactly once."""
    registry.register(AUDIO_CLIP_SPEC, handler)
