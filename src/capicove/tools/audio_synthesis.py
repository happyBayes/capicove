"""Two speech Tools sharing one application operation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

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
from capicove.processing.audio.synthesis import SpeechRequest, SynthesisError


def speech_spec(clone: bool) -> ToolSpec:
    """Build a closed operation-specific schema, independent of providers."""
    properties: dict[str, object] = {
        "text": {"type": "string", "minLength": 1, "maxLength": 200},
        "language": {"type": "string", "default": "zh"},
        "voice": {"type": "string"},
    }
    required = ["text"]
    if clone:
        properties.update(
            {
                "reference": {"type": "string", "minLength": 1},
                "authorization": {"type": "string", "minLength": 1},
                "reference_text": {"type": "string"},
            }
        )
        required += ["reference", "authorization"]
    return ToolSpec(
        tool_id="capicove.audio.clone" if clone else "capicove.audio.synthesize",
        contract_version=1,
        description="Generate a local speech artifact using an explicitly configured model.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        },
        output_schema={
            "type": "object",
            "required": [
                "path",
                "provenance_path",
                "duration_seconds",
                "sample_rate",
                "channels",
                "warnings",
                "lineage",
            ],
        },
        modalities=("text", "audio"),
        required_permissions=frozenset({"audio:read", "artifact:write", "subprocess:execute"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        side_effect=SideEffect.WRITE,
        network_access=False,
        deterministic=False,
        idempotent=False,
        timeout_seconds=240,
        redacted_fields=("text", "reference", "authorization", "reference_text"),
        resources=ToolResources(
            filesystem_read=True, filesystem_write=True, subprocesses=("llama-tts",)
        ),
    )


AUDIO_SYNTHESIZE_SPEC = speech_spec(False)
AUDIO_CLONE_SPEC = speech_spec(True)


class SpeechToolHandler:
    """Apply invocation authority and fail-fast single-request admission."""

    def __init__(
        self,
        service: Callable[[SpeechRequest, datetime], Mapping[str, object]],
        *,
        clone: bool,
        admission: Lock | None = None,
    ) -> None:
        self.service = service
        self.spec = AUDIO_CLONE_SPEC if clone else AUDIO_SYNTHESIZE_SPEC
        self.clone = clone
        self.admission = admission if admission is not None else Lock()

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)

        def failure(code: ToolErrorCode, message: str, rejected: bool = False) -> ToolResult:
            return ToolResult(
                context.request_id,
                context.trace_id,
                self.spec.tool_id,
                1,
                ToolStatus.REJECTED if rejected else ToolStatus.FAILED,
                error=ToolError(code, message, False),
                started_at=started,
                completed_at=datetime.now(UTC),
            )

        if self.spec.required_permissions - context.permissions:
            return failure(ToolErrorCode.PERMISSION_DENIED, "speech permissions missing", True)
        if context.deadline <= started:
            return failure(ToolErrorCode.TIMEOUT, "timeout")
        allowed = {"text", "language", "voice"} | (
            {"reference", "authorization", "reference_text"} if self.clone else set()
        )
        if (
            set(arguments) - allowed
            or not isinstance(arguments.get("text"), str)
            or any(not isinstance(value, str) for value in arguments.values())
            or (self.clone and not {"reference", "authorization"} <= set(arguments))
        ):
            return failure(ToolErrorCode.INVALID_INPUT, "invalid_input")
        if not self.admission.acquire(blocking=False):
            return failure(ToolErrorCode.RETRYABLE, "server_busy")
        try:
            request = SpeechRequest(
                str(arguments["text"]),
                str(arguments.get("language", "zh")),
                Path(str(arguments["reference"])) if "reference" in arguments else None,
                str(arguments["authorization"]) if "authorization" in arguments else None,
                str(arguments["reference_text"]) if "reference_text" in arguments else None,
                str(arguments["voice"]) if "voice" in arguments else None,
            )
            result = self.service(request, context.deadline)
            return ToolResult(
                context.request_id,
                context.trace_id,
                self.spec.tool_id,
                1,
                ToolStatus.SUCCEEDED,
                result=result,
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        except SynthesisError as error:
            code = str(error)
            outer = (
                ToolErrorCode.TIMEOUT
                if code == "timeout"
                else ToolErrorCode.POLICY_REJECTED
                if code in {"unsafe_path", "voice_unauthorized"}
                else ToolErrorCode.INVALID_INPUT
                if code in {"invalid_input", "unsupported_feature", "invalid_reference"}
                else ToolErrorCode.UNAVAILABLE
            )
            return failure(outer, code, outer == ToolErrorCode.POLICY_REJECTED)
        except (OSError, ValueError):
            return failure(ToolErrorCode.UNAVAILABLE, "speech_unavailable")
        finally:
            self.admission.release()
