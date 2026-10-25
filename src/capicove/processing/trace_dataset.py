"""Convert validated Agent trace exports into governed dataset snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)
from capicove.export import decode_trace_export_manifest, validate_trace_export


TRACE_DATASET_SCHEMA = DatasetSchema(
    (
        SchemaField("schema_version", "int64", False),
        SchemaField("record_type", "string", False),
        SchemaField("record_version", "int64", False),
        SchemaField("record_id", "string", False),
        SchemaField("episode_id", "string", False),
        SchemaField("trace_id", "string", False),
        SchemaField("sequence", "int64", False),
        SchemaField("occurred_at", "timestamp_utc", False),
        SchemaField("previous_record_sha256", "string", True),
    )
)


def trace_export_snapshot(
    export_directory: Path,
    /,
    *,
    dataset_id: str,
    version: int,
) -> DatasetSnapshotRef:
    """Validate a version-1 trace export and return its immutable source snapshot."""
    if not isinstance(export_directory, Path) or not export_directory.is_absolute():
        raise ValueError("export_directory must be an absolute Path")
    report = validate_trace_export(export_directory)
    if not report.valid:
        codes = ",".join(issue.code for issue in report.issues[:5])
        raise ValueError(f"trace export validation failed: {codes}")
    manifest_bytes = (export_directory / "manifest.json").read_bytes()
    manifest_value = json.loads(manifest_bytes)
    if not isinstance(manifest_value, dict):  # pragma: no cover - validated above
        raise ValueError("trace export manifest must be an object")
    manifest = decode_trace_export_manifest(manifest_value)
    if not manifest.shards:
        raise ValueError("trace export must contain at least one JSONL shard")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    processing_fingerprint = hashlib.sha256(
        b"capicove.trace-export-dataset.v1\0" + manifest_bytes
    ).hexdigest()
    return DatasetSnapshotRef(
        dataset_id=dataset_id,
        version=version,
        snapshot_id=f"snapshot:{manifest_sha256}",
        uri=export_directory.as_uri(),
        data_format=DataFormat.JSONL,
        schema=TRACE_DATASET_SCHEMA,
        manifest_sha256=manifest_sha256,
        processing_fingerprint=processing_fingerprint,
        parent_snapshot_ids=(),
    )
