"""Optional read-only Spark smoke test for a caller-owned media dataset."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from capicove_execution_spark.backend import SparkExecutionBackend
from capicove_execution_spark.compiler import spark_operator_specs
from capicove_execution_spark.session import SparkBackendConfiguration

from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)
from capicove.processing import LogicalNode, ProcessingContext, ProcessingPlanner, RunState


pytestmark = pytest.mark.spark


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_external_media_directory_is_not_modified(tmp_path: Path) -> None:
    configured = os.environ.get("CAPICOVE_TEST_DATASETS_ROOT")
    if configured is None:
        pytest.skip("CAPICOVE_TEST_DATASETS_ROOT is not configured")
    source_root = Path(configured).resolve(strict=True)
    media = tuple(
        sorted(path for path in source_root.rglob("*") if path.is_file() and not path.is_symlink())[
            :32
        ]
    )
    if not media:
        pytest.skip("external dataset contains no regular files")
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in media}
    manifest = tmp_path / "media-manifest.jsonl"
    lines = [
        json.dumps(
            {
                "relative_path": path.relative_to(source_root).as_posix(),
                "extension": path.suffix.lower() or "<none>",
                "size_bytes": path.stat().st_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for path in media
    ]
    payload = ("\n".join(lines) + "\n").encode()
    manifest.write_bytes(payload)
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:external-media-manifest",
        version=1,
        snapshot_id=f"snapshot:{_digest(payload)}",
        uri=manifest.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=DatasetSchema(
            (
                SchemaField("relative_path", "string", True),
                SchemaField("extension", "string", True),
                SchemaField("size_bytes", "int64", True),
            )
        ),
        manifest_sha256=_digest(b"external-media-manifest"),
        processing_fingerprint=_digest(b"read-only-source"),
        parent_snapshot_ids=(),
    )
    node = LogicalNode(
        node_id="count_extensions",
        operator_id="capicove.data.group-count",
        operator_version=1,
        input_bindings=("source",),
        output_binding="result",
        parameters={"columns": ["extension"], "count_column": "files"},
    )
    plan = ProcessingPlanner(spark_operator_specs()).build(
        input_bindings={"source": snapshot},
        nodes=(node,),
        output_dataset_id="dataset:external-media-summary",
    )
    backend = SparkExecutionBackend(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-external-dataset-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )

    reference = backend.submit(
        plan,
        ProcessingContext(
            run_id="run:external-media-summary-1",
            output_root=(tmp_path / "outputs").resolve(),
            requested_at=datetime.now(UTC),
        ),
    )

    assert backend.poll(reference).state is RunState.SUCCEEDED
    after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in media}
    assert after == before
