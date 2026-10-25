"""Real local Spark tests for explicit JSONL and Parquet dataset I/O."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from capicove_execution_spark.io import SparkDatasetIO, dataset_schema_from_spark
from capicove_execution_spark.session import (
    SparkBackendConfiguration,
    create_spark_session,
)

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)


pytestmark = pytest.mark.spark


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _nested_schema(
    *,
    tags_contains_null: bool = False,
    attributes_value_contains_null: bool = False,
) -> DatasetSchema:
    return DatasetSchema(
        (
            SchemaField(
                "tags",
                ComplexSchemaType(
                    ComplexTypeKind.ARRAY,
                    element_type="string",
                    contains_null=tags_contains_null,
                ),
                False,
            ),
            SchemaField(
                "attributes",
                ComplexSchemaType(
                    ComplexTypeKind.MAP,
                    value_type="string",
                    value_contains_null=attributes_value_contains_null,
                ),
                False,
            ),
            SchemaField(
                "profile",
                ComplexSchemaType(
                    ComplexTypeKind.STRUCT,
                    fields=(SchemaField("name", "string", False),),
                ),
                False,
            ),
        ),
        schema_version=2,
    )


def _nested_json_schema() -> DatasetSchema:
    return DatasetSchema(
        (
            SchemaField(
                "json_array",
                ComplexSchemaType(
                    ComplexTypeKind.ARRAY,
                    element_type="json",
                    contains_null=False,
                ),
                False,
            ),
            SchemaField(
                "json_map",
                ComplexSchemaType(
                    ComplexTypeKind.MAP,
                    value_type="json",
                    value_contains_null=False,
                ),
                False,
            ),
            SchemaField(
                "json_profile",
                ComplexSchemaType(
                    ComplexTypeKind.STRUCT,
                    fields=(SchemaField("metadata", "json", False),),
                ),
                False,
            ),
        ),
        schema_version=2,
    )


def _jsonl_snapshot(source: Path, payload: bytes, schema: DatasetSchema) -> DatasetSnapshotRef:
    return DatasetSnapshotRef(
        dataset_id="dataset:nested-input",
        version=1,
        snapshot_id=f"snapshot:{_digest(payload)}",
        uri=source.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=schema,
        manifest_sha256=_digest(b"manifest"),
        processing_fingerprint=_digest(b"source"),
        parent_snapshot_ids=(),
    )


def test_nested_jsonl_and_parquet_round_trip_preserve_values(tmp_path: Path) -> None:
    source = tmp_path / "nested.jsonl"
    payload = b'{"tags":["alpha","beta"],"attributes":{"tier":"gold"},"profile":{"name":"Ada"}}\n'
    source.write_bytes(payload)
    schema = _nested_schema()
    snapshot = _jsonl_snapshot(source, payload, schema)
    spark = create_spark_session(
        SparkBackendConfiguration(
            app_name="capicove-nested-io-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        io = SparkDatasetIO(spark)

        frame = io.read(snapshot)
        destination = (tmp_path / "parquet").resolve()
        io.write_parquet(frame, destination)

        assert io.validate_parquet(destination, schema)
        assert [row.asDict(recursive=True) for row in frame.collect()] == [
            {
                "tags": ["alpha", "beta"],
                "attributes": {"tier": "gold"},
                "profile": {"name": "Ada"},
            }
        ]
        assert [
            row.asDict(recursive=True) for row in spark.read.parquet(str(destination)).collect()
        ] == [
            {
                "tags": ["alpha", "beta"],
                "attributes": {"tier": "gold"},
                "profile": {"name": "Ada"},
            }
        ]
    finally:
        spark.stop()


def test_nested_json_values_and_empty_complex_containers_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "nested-json.jsonl"
    payload = (
        b'{"json_array":["{\\"tag\\":1}"],'
        b'"json_map":{"entry":"{\\"active\\":true}"},'
        b'"json_profile":{"metadata":"{\\"name\\":\\"Ada\\"}"}}\n'
        b'{"json_array":[],"json_map":{},"json_profile":{"metadata":"{}"}}\n'
    )
    source.write_bytes(payload)
    schema = _nested_json_schema()
    snapshot = _jsonl_snapshot(source, payload, schema)
    spark = create_spark_session(
        SparkBackendConfiguration(
            app_name="capicove-nested-json-io-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        io = SparkDatasetIO(spark)

        frame = io.read(snapshot)
        destination = (tmp_path / "parquet").resolve()
        io.write_parquet(frame, destination)

        rows = [row.asDict(recursive=True) for row in frame.collect()]
        assert rows[0]["json_array"] == ['{"tag":1}']
        assert rows[0]["json_map"] == {"entry": '{"active":true}'}
        assert rows[0]["json_profile"] == {"metadata": '{"name":"Ada"}'}
        assert rows[1]["json_array"] == []
        assert rows[1]["json_map"] == {}
        assert io.validate_parquet(destination, schema)
    finally:
        spark.stop()


@pytest.mark.parametrize(
    ("payload", "schema"),
    (
        (
            b'{"tags":["alpha"],"attributes":{"tier":"gold"},"profile":{"name":null}}\n',
            _nested_schema(),
        ),
        (
            b'{"tags":["alpha",null],"attributes":{"tier":"gold"},"profile":{"name":"Ada"}}\n',
            _nested_schema(),
        ),
        (
            b'{"tags":["alpha"],"attributes":{"tier":null},"profile":{"name":"Ada"}}\n',
            _nested_schema(),
        ),
    ),
    ids=("required-struct-child", "required-array-element", "required-map-value"),
)
def test_jsonl_read_rejects_nested_nullability_violations(
    tmp_path: Path,
    payload: bytes,
    schema: DatasetSchema,
) -> None:
    source = tmp_path / "nested-invalid.jsonl"
    source.write_bytes(payload)
    snapshot = _jsonl_snapshot(source, payload, schema)
    spark = create_spark_session(
        SparkBackendConfiguration(
            app_name="capicove-nested-nullability-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        with pytest.raises(ValueError, match="schema"):
            SparkDatasetIO(spark).read(snapshot)
    finally:
        spark.stop()


def test_jsonl_read_and_parquet_round_trip_use_canonical_schema(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    payload = (
        b'{"episode_id":"episode-1","record_type":"tool.result","score":0.9,'
        b'"metadata":"{\\"source\\":\\"tool\\"}"}\n'
        b'{"episode_id":"episode-2","record_type":"message.observed","score":null,'
        b'"metadata":null}\n'
    )
    source.write_bytes(payload)
    schema = DatasetSchema(
        (
            SchemaField("episode_id", "string", False),
            SchemaField("record_type", "string", False),
            SchemaField("score", "float64", True),
            SchemaField("metadata", "json", True),
        )
    )
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:trace-input",
        version=1,
        snapshot_id=f"snapshot:{_digest(payload)}",
        uri=source.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=schema,
        manifest_sha256=_digest(b"manifest"),
        processing_fingerprint=_digest(b"source"),
        parent_snapshot_ids=(),
    )
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-io-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        io = SparkDatasetIO(spark)

        frame = io.read(snapshot)

        assert frame.count() == 2
        assert tuple(frame.columns) == ("episode_id", "record_type", "score", "metadata")
        destination = (tmp_path / "parquet").resolve()
        io.write_parquet(frame, destination)
        assert io.validate_parquet(destination, schema)
        observed = dataset_schema_from_spark(spark.read.parquet(str(destination)).schema)
        assert tuple((field.name, field.logical_type) for field in observed.fields) == tuple(
            (field.name, field.logical_type) for field in schema.fields
        )
    finally:
        spark.stop()


def test_jsonl_read_fails_on_malformed_record(tmp_path: Path) -> None:
    source = tmp_path / "malformed.jsonl"
    payload = b'{"episode_id":"episode-1"}\nnot-json\n'
    source.write_bytes(payload)
    schema = DatasetSchema((SchemaField("episode_id", "string", True),))
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:malformed",
        version=1,
        snapshot_id=f"snapshot:{_digest(payload)}",
        uri=source.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=schema,
        manifest_sha256=_digest(b"manifest"),
        processing_fingerprint=_digest(b"source"),
        parent_snapshot_ids=(),
    )
    spark = create_spark_session(
        SparkBackendConfiguration(
            app_name="capicove-malformed-jsonl-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        with pytest.raises(Exception, match=r"MALFORMED_RECORD|malformed"):
            SparkDatasetIO(spark).read(snapshot).collect()
    finally:
        spark.stop()


def test_parquet_validation_rejects_null_in_required_column(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            app_name="capicove-nullability-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        from pyspark.sql.types import StringType, StructField, StructType

        frame = spark.createDataFrame(
            [(None,)],
            StructType([StructField("record_id", StringType(), True)]),
        )
        destination = (tmp_path / "parquet").resolve()
        frame.write.parquet(str(destination))
        expected = DatasetSchema((SchemaField("record_id", "string", False),))

        assert not SparkDatasetIO(spark).validate_parquet(destination, expected)
    finally:
        spark.stop()


@pytest.mark.parametrize("record", ("{}\n", '{"record_id":null}\n'))
def test_jsonl_read_rejects_missing_or_null_required_field(
    tmp_path: Path,
    record: str,
) -> None:
    source = tmp_path / "records.jsonl"
    payload = record.encode()
    source.write_bytes(payload)
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:required-jsonl",
        version=1,
        snapshot_id=f"snapshot:{_digest(payload)}",
        uri=source.resolve().as_uri(),
        data_format=DataFormat.JSONL,
        schema=DatasetSchema((SchemaField("record_id", "string", False),)),
        manifest_sha256=_digest(b"manifest"),
        processing_fingerprint=_digest(b"source"),
        parent_snapshot_ids=(),
    )
    spark = create_spark_session(
        SparkBackendConfiguration(
            app_name="capicove-required-jsonl-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        with pytest.raises(ValueError, match="schema"):
            SparkDatasetIO(spark).read(snapshot)
    finally:
        spark.stop()
