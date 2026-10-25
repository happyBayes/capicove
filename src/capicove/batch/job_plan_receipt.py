"""Canonical private plan receipts for durable operator jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from capicove.batch.models import (
    BatchCacheScope,
    BatchItem,
    BatchModality,
    BatchPlan,
    OperationFingerprint,
    SourceSnapshot,
)
from capicove.batch.planner import BatchPlanner


MAX_PLAN_RECEIPT_BYTES = 8 * 1024 * 1024


def encode_plan_receipt(plan: BatchPlan, input_root: Path) -> tuple[str, str, str]:
    """Encode a verified plan without absolute paths or source payloads."""
    _require_safe_root(input_root)
    if len(plan.items) > 1_000_000:
        raise ValueError("plan item count exceeds the registry limit")
    if BatchPlanner().plan((entry.item for entry in plan.items), plan.operation) != plan:
        raise ValueError("plan identity or cache keys are inconsistent")
    operation = _operation_body(plan.operation)
    items: list[dict[str, object]] = []
    for entry in plan.items:
        item = entry.item
        if item.source_path != input_root.joinpath(*item.relative_path.parts):
            raise ValueError("plan source must match its configured relative path")
        items.append(
            {
                "item_id": item.item_id,
                "modality": item.modality.value,
                "relative_path": item.relative_path.as_posix(),
                "media_type": item.media_type,
                "sha256": item.sha256,
                "cache_key": entry.cache_key,
                "snapshot": {
                    "device": item.snapshot.device,
                    "inode": item.snapshot.inode,
                    "size_bytes": item.snapshot.size_bytes,
                    "modified_ns": item.snapshot.modified_ns,
                    "changed_ns": item.snapshot.changed_ns,
                },
            }
        )
    body = {"schema_version": 1, "plan_id": plan.plan_id, "operation": operation, "items": items}
    encoded = _canonical_json(body)
    if len(encoded.encode("utf-8")) > MAX_PLAN_RECEIPT_BYTES:
        raise ValueError("plan receipt exceeds the registry size limit")
    return encoded, _sha256(encoded), operation_sha256(plan.operation)


def operation_sha256(operation: OperationFingerprint) -> str:
    """Return the registry identity of one operation fingerprint."""
    return hashlib.sha256(_canonical_json(_operation_body(operation)).encode("utf-8")).hexdigest()


def _operation_body(operation: OperationFingerprint) -> dict[str, object]:
    return {
        "operation_id": operation.operation_id,
        "operation_version": operation.operation_version,
        "modalities": sorted(modality.value for modality in operation.modalities),
        "parameters_sha256": operation.parameters_sha256,
        "provider_fingerprint": operation.provider_fingerprint,
        "cache_scope": operation.cache_scope.value,
    }


def decode_plan_receipt(encoded: str, plan_id: str, input_root: Path) -> BatchPlan:
    """Reconstruct a plan under the configured root and verify its identity."""
    _require_safe_root(input_root)
    try:
        encoded_size = len(encoded.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("invalid plan receipt") from error
    if encoded_size > MAX_PLAN_RECEIPT_BYTES:
        raise ValueError("invalid plan receipt size")
    try:
        body: Any = json.loads(encoded)
        if not isinstance(body, dict) or set(body) != {
            "schema_version",
            "plan_id",
            "operation",
            "items",
        }:
            raise ValueError("invalid plan receipt fields")
        if body["schema_version"] != 1 or body["plan_id"] != plan_id:
            raise ValueError("invalid plan receipt identity")
        operation_body = body["operation"]
        items_body = body["items"]
        if not isinstance(operation_body, dict) or not isinstance(items_body, list):
            raise ValueError("invalid plan receipt values")
        if set(operation_body) != {
            "operation_id",
            "operation_version",
            "modalities",
            "parameters_sha256",
            "provider_fingerprint",
            "cache_scope",
        }:
            raise ValueError("invalid plan operation fields")
        if len(items_body) > 1_000_000:
            raise ValueError("plan receipt item count exceeds the registry limit")
        operation = OperationFingerprint(
            operation_id=operation_body["operation_id"],
            operation_version=operation_body["operation_version"],
            modalities=frozenset(BatchModality(value) for value in operation_body["modalities"]),
            parameters_sha256=operation_body["parameters_sha256"],
            provider_fingerprint=operation_body["provider_fingerprint"],
            cache_scope=BatchCacheScope(operation_body["cache_scope"]),
        )
        items: list[BatchItem] = []
        for entry in items_body:
            if not isinstance(entry, dict) or set(entry) != {
                "item_id",
                "modality",
                "relative_path",
                "media_type",
                "sha256",
                "cache_key",
                "snapshot",
            }:
                raise ValueError("invalid plan item fields")
            relative = PurePosixPath(entry["relative_path"])
            source_path = input_root.joinpath(*relative.parts)
            if not source_path.is_relative_to(input_root):
                raise ValueError("plan source escapes its configured root")
            snapshot = entry["snapshot"]
            if not isinstance(snapshot, dict) or set(snapshot) != {
                "device",
                "inode",
                "size_bytes",
                "modified_ns",
                "changed_ns",
            }:
                raise ValueError("invalid plan source snapshot fields")
            items.append(
                BatchItem(
                    item_id=entry["item_id"],
                    modality=BatchModality(entry["modality"]),
                    source_path=source_path,
                    relative_path=relative,
                    media_type=entry["media_type"],
                    sha256=entry["sha256"],
                    snapshot=SourceSnapshot(
                        device=snapshot["device"],
                        inode=snapshot["inode"],
                        size_bytes=snapshot["size_bytes"],
                        modified_ns=snapshot["modified_ns"],
                        changed_ns=snapshot["changed_ns"],
                    ),
                )
            )
        rebuilt = BatchPlanner().plan(items, operation)
        if rebuilt.plan_id != plan_id or tuple(entry.cache_key for entry in rebuilt.items) != tuple(
            entry["cache_key"] for entry in items_body
        ):
            raise ValueError("plan receipt identity or cache keys changed")
        if _canonical_json(body) != encoded:
            raise ValueError("plan receipt encoding is not canonical")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("invalid plan receipt") from error
    return rebuilt


def receipt_sha256(encoded: str) -> str:
    """Return the digest used to detect a changed durable receipt."""
    return _sha256(encoded)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_safe_root(input_root: Path) -> None:
    if not input_root.is_absolute():
        raise ValueError("input_root must be absolute")
    if os.name == "nt":
        from capicove._windows_files import validate_windows_path

        validate_windows_path(input_root, directory=True)
    elif any(part.is_symlink() for part in (input_root, *input_root.parents)):
        raise ValueError("input_root must not contain symlinks")
    if not input_root.is_dir():
        raise ValueError("input_root must be an existing directory")
