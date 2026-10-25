"""Permission-checked, bounded evidence expansion through one Tool contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

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


SECTIONS = ("metadata", "preview", "transcript", "ocr", "scores", "lineage", "provider_diagnostics")
CAPICOVE_EVIDENCE_SPEC = ToolSpec(
    tool_id="capicove.get_evidence",
    contract_version=1,
    description="Read authorized sections of immutable evidence.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_id"],
        "properties": {
            "evidence_id": {"type": "string", "minLength": 1},
            "sections": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"enum": list(SECTIONS)},
            },
        },
    },
    output_schema={"type": "object"},
    modalities=("text", "image"),
    required_permissions=frozenset({"evidence:read"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    timeout_seconds=30,
    redacted_fields=("evidence_id",),
)


class EvidenceReader(Protocol):
    """Authorize actor/workspace/resource on every read, including erased previews."""

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]: ...


def decode_evidence_arguments(arguments: Mapping[str, object]) -> tuple[str, tuple[str, ...]]:
    """Decode only closed bounded section requests."""
    if set(arguments) - {"evidence_id", "sections"}:
        raise ValueError("unsupported evidence argument")
    evidence_id = arguments.get("evidence_id")
    sections = arguments.get("sections", ["metadata"])
    if not isinstance(evidence_id, str) or not evidence_id.strip() or len(evidence_id) > 512:
        raise ValueError("invalid evidence id")
    if not isinstance(sections, (list, tuple)) or not sections or len(sections) > len(SECTIONS):
        raise ValueError("invalid sections")
    if any(not isinstance(section, str) or section not in SECTIONS for section in sections):
        raise ValueError("unsupported section")
    values = tuple(str(section) for section in sections)
    if len(set(values)) != len(values):
        raise ValueError("duplicate section")
    return evidence_id, values


class EvidenceToolHandler:
    """Apply Tool permission and result limits around a resource-authorizing reader."""

    def __init__(self, reader: EvidenceReader, *, spec: ToolSpec = CAPICOVE_EVIDENCE_SPEC) -> None:
        self._reader = reader
        self._spec = spec

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        started = datetime.now(UTC)
        code: ToolErrorCode | None = None
        payload: dict[str, object] | None = None
        try:
            if "evidence:read" not in context.permissions:
                raise PermissionError
            if started >= context.deadline:
                raise TimeoutError
            evidence_id, sections = decode_evidence_arguments(arguments)
            if (
                "provider_diagnostics" in sections
                and "evidence:diagnostics" not in context.permissions
            ):
                raise PermissionError
        except (TypeError, ValueError):
            code = ToolErrorCode.INVALID_INPUT
        except PermissionError:
            code = ToolErrorCode.PERMISSION_DENIED
        except TimeoutError:
            code = ToolErrorCode.TIMEOUT
        if code is None:
            try:
                read = dict(
                    self._reader.read(context=context, evidence_id=evidence_id, sections=sections)
                )
                if set(read) - set(sections):
                    raise ValueError("unrequested section")
                for section, value in read.items():
                    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 4096:
                        raise ValueError("section exceeds limit")
                    if isinstance(value, (list, tuple)) and len(value) > 32:
                        raise ValueError("section exceeds item limit")
                    if section == "preview":
                        if not isinstance(value, Mapping) or set(value) != {
                            "resource_ref",
                            "media_type",
                        }:
                            raise ValueError("invalid preview")
                        ref = value["resource_ref"]
                        if (
                            not isinstance(ref, str)
                            or not ref.startswith("resource:preview:")
                            or any(c in ref for c in ("/", "\\", "\n"))
                        ):
                            raise ValueError("invalid preview ref")
                        if value["media_type"] not in {"image/jpeg", "image/png", "image/webp"}:
                            raise ValueError("invalid preview media type")
                payload = {"evidence_id": evidence_id, **read}
                if datetime.now(UTC) >= context.deadline:
                    raise TimeoutError
            except PermissionError:
                code = ToolErrorCode.PERMISSION_DENIED
            except LookupError:
                code = ToolErrorCode.UNAVAILABLE
            except TimeoutError:
                code = ToolErrorCode.TIMEOUT
            except Exception:
                code = ToolErrorCode.INTERNAL
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=self._spec.tool_id,
            tool_version=self._spec.contract_version,
            status=ToolStatus.SUCCEEDED if code is None else ToolStatus.FAILED,
            result=payload if code is None else None,
            error=None
            if code is None
            else ToolError(code=code, message="Evidence request failed.", retryable=False),
            started_at=started,
            completed_at=datetime.now(UTC),
        )
