"""Canonical read-only exact-file video deduplication Tool (RFC-0021)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

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


MAX_PATHS = 1000
MAX_FAILURES = 1000

VIDEO_DEDUPLICATE_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["paths"],
    "properties": {
        "paths": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_PATHS,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^/", "maxLength": 4096},
        }
    },
}

VIDEO_DEDUPLICATE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["file_count", "groups", "failures", "lineage"],
    "properties": {
        "file_count": {"type": "integer", "minimum": 0},
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sha256", "xxh3_128", "size_bytes", "paths"],
                "properties": {
                    "sha256": {"type": "string"},
                    "xxh3_128": {"type": "string"},
                    "size_bytes": {"type": "integer", "minimum": 0},
                    "paths": {"type": "array", "minItems": 2, "items": {"type": "string"}},
                },
            },
        },
        "failures": {
            "type": "array",
            "maxItems": MAX_FAILURES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "code"],
                "properties": {
                    "path": {"type": "string"},
                    "code": {"enum": ["unreadable", "changed", "too_large"]},
                },
            },
        },
        "lineage": {"type": "object"},
    },
}

VIDEO_DEDUPLICATE_SPEC = ToolSpec(
    tool_id="capicove.video.deduplicate",
    contract_version=1,
    description="Group byte-identical local video files without modifying them.",
    input_schema=VIDEO_DEDUPLICATE_INPUT_SCHEMA,
    output_schema=VIDEO_DEDUPLICATE_OUTPUT_SCHEMA,
    modalities=("video",),
    required_permissions=frozenset({"video:read"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    network_access=False,
    deterministic=True,
    idempotent=True,
    timeout_seconds=600.0,
    resources=ToolResources(filesystem_read=True),
)


class DigestUnavailable(Exception):
    """One file could not be hashed; ``code`` is a stable failure code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DuplicateDigestPort(Protocol):
    """Authorize and hash files for exact grouping."""

    hash_version: str

    def authorize(self, path: Path) -> None:
        """Raise PermissionError for a path outside the configured authority."""

    def digest(self, path: Path) -> tuple[int, str, str]:
        """Return size, xxHash3-128, and SHA-256, or raise DigestUnavailable."""


class VideoDeduplicateToolHandler:
    """Apply permissions, authority, and deadlines around exact grouping."""

    def __init__(self, port: DuplicateDigestPort) -> None:
        self._port = port

    def __call__(
        self, context: InvocationContext, arguments: Mapping[str, object], /
    ) -> ToolResult:
        """Group exact duplicates; nothing is moved, deleted, or written."""
        started = datetime.now(UTC)
        if VIDEO_DEDUPLICATE_SPEC.required_permissions.difference(context.permissions):
            return _failure(context, started, ToolErrorCode.PERMISSION_DENIED)
        try:
            paths = _decode(arguments)
        except (TypeError, ValueError):
            return _failure(context, started, ToolErrorCode.INVALID_INPUT)
        try:
            for path in paths:
                self._port.authorize(path)
        except PermissionError:
            return _failure(context, started, ToolErrorCode.PERMISSION_DENIED)
        buckets: dict[tuple[int, str, str], list[str]] = defaultdict(list)
        failures: list[dict[str, str]] = []
        for path in paths:
            if datetime.now(UTC) >= context.deadline:
                return _failure(context, started, ToolErrorCode.TIMEOUT)
            try:
                size, fast, strong = self._port.digest(path)
            except DigestUnavailable as error:
                failures.append({"path": str(path), "code": error.code})
                continue
            buckets[(size, fast, strong)].append(str(path))
        groups = sorted(
            (
                {"sha256": strong, "xxh3_128": fast, "size_bytes": size, "paths": sorted(group)}
                for (size, fast, strong), group in buckets.items()
                if len(group) > 1
            ),
            key=lambda item: item["paths"][0],  # type: ignore[index]
        )
        payload: dict[str, object] = {
            "file_count": len(paths),
            "groups": groups,
            "failures": sorted(failures, key=lambda item: item["path"])[:MAX_FAILURES],
            "lineage": {
                "fast_hash": "xxh3_128",
                "xxhash_version": self._port.hash_version,
                "confirmation": "sha256",
            },
        }
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=VIDEO_DEDUPLICATE_SPEC.tool_id,
            tool_version=VIDEO_DEDUPLICATE_SPEC.contract_version,
            status=ToolStatus.SUCCEEDED,
            result=payload,
            evidence_refs=tuple(f"sha256:{group['sha256']}" for group in groups),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def _decode(arguments: Mapping[str, object]) -> tuple[Path, ...]:
    if set(arguments) != {"paths"}:
        raise ValueError("arguments must contain only paths")
    values = arguments["paths"]
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_PATHS:
        raise ValueError("paths must contain 1 to 1,000 entries")
    if any(not isinstance(value, str) or not value.startswith("/") for value in values):
        raise ValueError("paths must be absolute strings")
    if len(set(values)) != len(values):
        raise ValueError("paths must be unique")
    return tuple(Path(value) for value in values)


def _failure(context: InvocationContext, started: datetime, code: ToolErrorCode) -> ToolResult:
    messages = {
        ToolErrorCode.PERMISSION_DENIED: "Video access is not permitted.",
        ToolErrorCode.INVALID_INPUT: "Deduplication arguments are invalid.",
        ToolErrorCode.TIMEOUT: "Deduplication deadline exceeded.",
    }
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=VIDEO_DEDUPLICATE_SPEC.tool_id,
        tool_version=VIDEO_DEDUPLICATE_SPEC.contract_version,
        status=ToolStatus.FAILED,
        error=ToolError(code, messages.get(code, "Deduplication failed."), retryable=False),
        started_at=started,
        completed_at=datetime.now(UTC),
    )


__all__ = [
    "MAX_PATHS",
    "VIDEO_DEDUPLICATE_SPEC",
    "DigestUnavailable",
    "DuplicateDigestPort",
    "VideoDeduplicateToolHandler",
]
