"""Canonical operator batch-job Tool contracts (RFC-0016, amendment 1).

These specs describe the closed request and result shapes only. A host binds
them to startup-configured roots, operations, and a durable job registry; no
argument can name a path, provider, executable, or another workspace.
"""

from __future__ import annotations

from capicove.batch.compose import OPERATIONS
from capicove.contracts import PrivacyClass, SideEffect, ToolResources, ToolSpec


MAX_STATUS_ITEMS = 100
MAX_PLAN_PREVIEW_ITEMS = 100
JOB_STATES = (
    "queued",
    "running",
    "cancelling",
    "interrupted",
    "succeeded",
    "failed",
    "cancelled",
)
ITEM_STATES = ("pending", "running", "succeeded", "failed", "skipped", "source_changed")

_ALIAS: dict[str, object] = {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$"}
_SHA256: dict[str, object] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_PLAN_ID: dict[str, object] = {"type": "string", "pattern": "^plan:[0-9a-f]{64}$"}
_JOB_ID: dict[str, object] = {"type": "string", "pattern": "^job:[0-9a-f]{24}$"}
_TIMESTAMP: dict[str, object] = {"type": "string", "maxLength": 64}
_NULLABLE_TIMESTAMP: dict[str, object] = {"anyOf": [_TIMESTAMP, {"type": "null"}]}
_SELECTORS: dict[str, object] = {
    "input_root": _ALIAS,
    "operation": {"enum": list(OPERATIONS)},
    "parameters": {"type": "object", "maxProperties": 32},
    "provider_fingerprint": _SHA256,
}

BATCH_PLAN_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["input_root", "operation", "parameters", "provider_fingerprint"],
    "properties": {**_SELECTORS, "output_root": _ALIAS},
}
BATCH_ENQUEUE_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "input_root",
        "operation",
        "parameters",
        "provider_fingerprint",
        "output_root",
        "plan_id",
        "idempotency_key",
    ],
    "properties": {
        **_SELECTORS,
        "output_root": _ALIAS,
        "plan_id": _PLAN_ID,
        "idempotency_key": {"type": "string", "pattern": "^[A-Za-z0-9._:-]{1,128}$"},
    },
}
BATCH_STATUS_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job_id"],
    "properties": {
        "job_id": _JOB_ID,
        "offset": {"type": "integer", "minimum": 0, "maximum": 1_000_000},
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_STATUS_ITEMS},
    },
}
BATCH_CANCEL_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job_id", "expected_revision"],
    "properties": {
        "job_id": _JOB_ID,
        "expected_revision": {"type": "integer", "minimum": 1},
    },
}

_PLAN_ITEM: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["item_id", "relative_path", "media_type", "size_bytes", "sha256"],
    "properties": {
        "item_id": {"type": "string"},
        "relative_path": {"type": "string"},
        "media_type": {"type": "string"},
        "size_bytes": {"type": "integer", "minimum": 0},
        "sha256": _SHA256,
    },
}
BATCH_PLAN_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "plan_id",
        "input_root",
        "output_root",
        "operation",
        "operation_fingerprint",
        "item_count",
        "total_bytes",
        "items",
        "truncated",
    ],
    "properties": {
        "plan_id": _PLAN_ID,
        "input_root": _ALIAS,
        "output_root": {"anyOf": [_ALIAS, {"type": "null"}]},
        "operation": {"enum": list(OPERATIONS)},
        "operation_fingerprint": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "operation_id",
                "operation_version",
                "parameters_sha256",
                "provider_fingerprint",
            ],
            "properties": {
                "operation_id": {"type": "string"},
                "operation_version": {"type": "string"},
                "parameters_sha256": _SHA256,
                "provider_fingerprint": _SHA256,
            },
        },
        "item_count": {"type": "integer", "minimum": 0},
        "total_bytes": {"type": "integer", "minimum": 0},
        "items": {"type": "array", "maxItems": MAX_PLAN_PREVIEW_ITEMS, "items": _PLAN_ITEM},
        "truncated": {"type": "boolean"},
    },
}
JOB_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "job_id",
        "plan_id",
        "operation",
        "input_root",
        "output_root",
        "state",
        "revision",
        "total_items",
        "completed_items",
        "created_at",
        "updated_at",
        "lease_until",
        "terminal_manifest_sha256",
        "error_code",
        "diagnostic",
    ],
    "properties": {
        "job_id": _JOB_ID,
        "plan_id": _PLAN_ID,
        "operation": {"anyOf": [{"enum": list(OPERATIONS)}, {"type": "null"}]},
        "input_root": _ALIAS,
        "output_root": _ALIAS,
        "state": {"enum": list(JOB_STATES)},
        "revision": {"type": "integer", "minimum": 1},
        "total_items": {"type": "integer", "minimum": 0},
        "completed_items": {"type": "integer", "minimum": 0},
        "created_at": _TIMESTAMP,
        "updated_at": _TIMESTAMP,
        "lease_until": _NULLABLE_TIMESTAMP,
        "terminal_manifest_sha256": {"anyOf": [_SHA256, {"type": "null"}]},
        "error_code": {"anyOf": [{"type": "string", "maxLength": 64}, {"type": "null"}]},
        "diagnostic": {"anyOf": [{"type": "string", "maxLength": 512}, {"type": "null"}]},
    },
}
BATCH_JOB_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job"],
    "properties": {"job": JOB_SCHEMA},
}
BATCH_STATUS_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["job", "items", "offset", "limit", "next_offset"],
    "properties": {
        "job": JOB_SCHEMA,
        "items": {
            "type": "array",
            "maxItems": MAX_STATUS_ITEMS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "item_id",
                    "relative_path",
                    "state",
                    "attempts",
                    "failure_code",
                    "result",
                ],
                "properties": {
                    "item_id": {"type": "string"},
                    "relative_path": {"type": "string"},
                    "state": {"enum": list(ITEM_STATES)},
                    "attempts": {"type": "integer", "minimum": 0},
                    "failure_code": {
                        "anyOf": [{"type": "string", "maxLength": 64}, {"type": "null"}]
                    },
                    "result": {"anyOf": [{"type": "object"}, {"type": "null"}]},
                },
            },
        },
        "offset": {"type": "integer", "minimum": 0},
        "limit": {"type": "integer", "minimum": 1, "maximum": MAX_STATUS_ITEMS},
        "next_offset": {"anyOf": [{"type": "integer", "minimum": 0}, {"type": "null"}]},
    },
}

BATCH_PLAN_SPEC = ToolSpec(
    tool_id="capicove.batch.plan",
    contract_version=1,
    description="Plan one allowlisted operation over a configured input root without side effects.",
    input_schema=BATCH_PLAN_INPUT_SCHEMA,
    output_schema=BATCH_PLAN_OUTPUT_SCHEMA,
    modalities=("image", "video"),
    required_permissions=frozenset({"batch:read"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    timeout_seconds=600.0,
    resources=ToolResources(filesystem_read=True),
)
BATCH_ENQUEUE_SPEC = ToolSpec(
    tool_id="capicove.batch.enqueue",
    contract_version=1,
    description="Durably queue a recomputed plan for the operator-started local worker.",
    input_schema=BATCH_ENQUEUE_INPUT_SCHEMA,
    output_schema=BATCH_JOB_OUTPUT_SCHEMA,
    modalities=("image", "video"),
    required_permissions=frozenset({"batch:write"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.WRITE,
    timeout_seconds=600.0,
    resources=ToolResources(filesystem_read=True, filesystem_write=True),
)
BATCH_STATUS_SPEC = ToolSpec(
    tool_id="capicove.batch.status",
    contract_version=1,
    description="Read one durable job and a bounded page of its item checkpoints.",
    input_schema=BATCH_STATUS_INPUT_SCHEMA,
    output_schema=BATCH_STATUS_OUTPUT_SCHEMA,
    required_permissions=frozenset({"batch:read"}),
    privacy_class=PrivacyClass.CONFIDENTIAL,
    side_effect=SideEffect.READ,
    timeout_seconds=60.0,
    resources=ToolResources(filesystem_read=True),
)
BATCH_CANCEL_SPEC = ToolSpec(
    tool_id="capicove.batch.cancel",
    contract_version=1,
    description="Request revision-checked cooperative cancellation of one durable job.",
    input_schema=BATCH_CANCEL_INPUT_SCHEMA,
    output_schema=BATCH_JOB_OUTPUT_SCHEMA,
    required_permissions=frozenset({"batch:write"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.WRITE,
    timeout_seconds=60.0,
    resources=ToolResources(filesystem_write=True),
)
BATCH_SPECS = {
    spec.tool_id: spec
    for spec in (BATCH_PLAN_SPEC, BATCH_ENQUEUE_SPEC, BATCH_STATUS_SPEC, BATCH_CANCEL_SPEC)
}

__all__ = [
    "BATCH_CANCEL_SPEC",
    "BATCH_ENQUEUE_SPEC",
    "BATCH_PLAN_SPEC",
    "BATCH_SPECS",
    "BATCH_STATUS_SPEC",
    "MAX_STATUS_ITEMS",
]
