"""Canonical operator trace Tool contracts (RFC-0016, amendment 1).

Requests select configured store and export-root aliases only. Results are
counts, validation codes, and opaque export IDs; they never include trace
content, record IDs, or local paths.
"""

from __future__ import annotations

from capicove.contracts import PrivacyClass, SideEffect, ToolResources, ToolSpec


MAX_FINDINGS = 100
_ALIAS: dict[str, object] = {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$"}
_EXPORT_ID: dict[str, object] = {"type": "string", "pattern": "^export:[0-9a-f]{32}$"}
_TIMESTAMP: dict[str, object] = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$",
}


def _set_of(item: dict[str, object], limit: int) -> dict[str, object]:
    return {"type": "array", "minItems": 1, "maxItems": limit, "uniqueItems": True, "items": item}


_BOUNDED_ID: dict[str, object] = {"type": "string", "minLength": 1, "maxLength": 256}
TRACE_INSPECT_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"store": _ALIAS, "export_id": _EXPORT_ID},
    "oneOf": [{"required": ["store"]}, {"required": ["export_id"]}],
}
TRACE_VALIDATE_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["export_id"],
    "properties": {"export_id": _EXPORT_ID, "verify_media": {"type": "boolean"}},
}
TRACE_EXPORT_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["store", "output_root", "idempotency_key"],
    "properties": {
        "store": _ALIAS,
        "output_root": _ALIAS,
        "idempotency_key": {"type": "string", "pattern": "^[A-Za-z0-9._:-]{1,128}$"},
        "media_mode": {"enum": ["reference", "copy"]},
        "include_incomplete": {"type": "boolean"},
        "include_restricted": {"type": "boolean"},
        "episode_ids": _set_of(_BOUNDED_ID, 1000),
        "tool_ids": _set_of(_BOUNDED_ID, 100),
        "modalities": _set_of({"enum": ["audio", "image", "video"]}, 3),
        "tool_statuses": _set_of({"enum": ["succeeded", "failed", "rejected"]}, 3),
        "episode_statuses": _set_of({"enum": ["succeeded", "failed", "rejected", "incomplete"]}, 4),
        "started_after": _TIMESTAMP,
        "completed_before": _TIMESTAMP,
    },
}

_COUNT: dict[str, object] = {"type": "integer", "minimum": 0}
TRACE_STORE_RESULT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "kind",
        "store",
        "episode_count",
        "incomplete_episode_count",
        "omitted_restricted_episode_count",
    ],
    "properties": {
        "kind": {"const": "store"},
        "store": _ALIAS,
        "episode_count": _COUNT,
        "incomplete_episode_count": _COUNT,
        "omitted_restricted_episode_count": _COUNT,
    },
}
TRACE_EXPORT_RESULT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "kind",
        "export_id",
        "valid",
        "records_checked",
        "episodes_checked",
        "media_checked",
        "findings",
        "findings_truncated",
    ],
    "properties": {
        "kind": {"const": "export"},
        "export_id": _EXPORT_ID,
        "valid": {"type": "boolean"},
        "records_checked": _COUNT,
        "episodes_checked": _COUNT,
        "media_checked": _COUNT,
        "findings": {
            "type": "array",
            "maxItems": MAX_FINDINGS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["severity", "code"],
                "properties": {
                    "severity": {"type": "string", "maxLength": 16},
                    "code": {"type": "string", "pattern": "^[A-Z][A-Z0-9_]{0,63}$"},
                },
            },
        },
        "findings_truncated": {"type": "boolean"},
    },
}
TRACE_INSPECT_OUTPUT_SCHEMA: dict[str, object] = {
    "anyOf": [TRACE_STORE_RESULT, TRACE_EXPORT_RESULT]
}
TRACE_PACKAGE_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "export_id",
        "output_root",
        "included_episode_count",
        "incomplete_episode_count",
        "omitted_restricted_episode_count",
        "record_count",
        "shard_count",
        "media_count",
        "missing_media_count",
        "media_mode",
    ],
    "properties": {
        "export_id": _EXPORT_ID,
        "output_root": _ALIAS,
        "included_episode_count": _COUNT,
        "incomplete_episode_count": _COUNT,
        "omitted_restricted_episode_count": _COUNT,
        "record_count": _COUNT,
        "shard_count": _COUNT,
        "media_count": _COUNT,
        "missing_media_count": _COUNT,
        "media_mode": {"enum": ["reference", "copy"]},
    },
}

TRACE_INSPECT_SPEC = ToolSpec(
    tool_id="capicove.trace.inspect",
    contract_version=1,
    description="Count Episodes in a configured trace store or inspect a configured export.",
    input_schema=TRACE_INSPECT_INPUT_SCHEMA,
    output_schema=TRACE_INSPECT_OUTPUT_SCHEMA,
    required_permissions=frozenset({"trace:read"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.READ,
    timeout_seconds=120.0,
    resources=ToolResources(filesystem_read=True),
)
TRACE_VALIDATE_SPEC = ToolSpec(
    tool_id="capicove.trace.validate",
    contract_version=1,
    description="Validate one configured trace export and return content-free findings.",
    input_schema=TRACE_VALIDATE_INPUT_SCHEMA,
    output_schema=TRACE_EXPORT_RESULT,
    required_permissions=frozenset({"trace:read"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.READ,
    timeout_seconds=600.0,
    resources=ToolResources(filesystem_read=True),
)
TRACE_EXPORT_SPEC = ToolSpec(
    tool_id="capicove.trace.export",
    contract_version=1,
    description="Publish an idempotent trace package into a configured private export root.",
    input_schema=TRACE_EXPORT_INPUT_SCHEMA,
    output_schema=TRACE_PACKAGE_OUTPUT_SCHEMA,
    required_permissions=frozenset({"trace:export"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    timeout_seconds=1800.0,
    resources=ToolResources(filesystem_read=True, filesystem_write=True),
)
TRACE_SPECS = {
    spec.tool_id: spec for spec in (TRACE_INSPECT_SPEC, TRACE_VALIDATE_SPEC, TRACE_EXPORT_SPEC)
}

__all__ = ["TRACE_EXPORT_SPEC", "TRACE_INSPECT_SPEC", "TRACE_SPECS", "TRACE_VALIDATE_SPEC"]
