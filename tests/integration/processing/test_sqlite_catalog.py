"""Integration tests for local processing metadata persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    LineageEdge,
    SchemaField,
)
from capicove.processing._adapters.sqlite import (
    ProcessingStoreError,
    RunTransitionError,
    SnapshotConflictError,
    SqliteProcessingStore,
)
from capicove.processing.models import ProcessingRun, RunState


@pytest.mark.parametrize("placement", ("database", "parent"))
def test_store_rejects_symlinked_internal_paths(
    tmp_path: Path,
    placement: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    if placement == "database":
        target = outside / "real.sqlite3"
        sqlite3.connect(target).close()
        database_path = tmp_path / "processing.sqlite3"
        database_path.symlink_to(target)
    else:
        linked_parent = tmp_path / ".capicove"
        linked_parent.symlink_to(outside, target_is_directory=True)
        database_path = linked_parent / "processing.sqlite3"

    with pytest.raises(ProcessingStoreError, match="symbolic link"):
        SqliteProcessingStore(database_path)


def test_store_rechecks_database_path_before_each_connection(tmp_path: Path) -> None:
    database_path = tmp_path / "processing.sqlite3"
    store = SqliteProcessingStore(database_path)
    database_path.unlink()
    outside = tmp_path / "outside.sqlite3"
    sqlite3.connect(outside).close()
    database_path.symlink_to(outside)

    with pytest.raises(ProcessingStoreError, match="symbolic link"):
        store.latest("dataset:any")


def _snapshot(*, version: int = 1, manifest: str = "b" * 64) -> DatasetSnapshotRef:
    return DatasetSnapshotRef(
        dataset_id="dataset:curated-traces",
        version=version,
        snapshot_id="snapshot:" + "a" * 64,
        uri=f"file:///tmp/capicove/curated-traces/v{version}",
        data_format=DataFormat.PARQUET,
        schema=DatasetSchema((SchemaField("episode_id", "string", False),)),
        manifest_sha256=manifest,
        processing_fingerprint="c" * 64,
        parent_snapshot_ids=(),
    )


def _complex_snapshot(*, version: int, snapshot_hash: str) -> DatasetSnapshotRef:
    profile = ComplexSchemaType(
        kind=ComplexTypeKind.STRUCT,
        fields=(
            SchemaField("name", "string", False),
            SchemaField(
                "scores",
                ComplexSchemaType(
                    kind=ComplexTypeKind.ARRAY,
                    element_type="float64",
                    contains_null=False,
                ),
                False,
            ),
        ),
    )
    attributes = ComplexSchemaType(
        kind=ComplexTypeKind.MAP,
        value_type="json",
        value_contains_null=True,
    )
    return DatasetSnapshotRef(
        dataset_id="dataset:complex-traces",
        version=version,
        snapshot_id=f"snapshot:{snapshot_hash * 64}",
        uri=f"file:///tmp/capicove/complex-traces/v{version}",
        data_format=DataFormat.PARQUET,
        schema=DatasetSchema(
            (
                SchemaField("profile", profile, False),
                SchemaField("attributes", attributes, True),
            ),
            schema_version=2,
        ),
        manifest_sha256="b" * 64,
        processing_fingerprint="c" * 64,
        parent_snapshot_ids=(),
    )


def _run() -> ProcessingRun:
    now = datetime.now(UTC)
    return ProcessingRun(
        run_id="run:one",
        plan_id="plan:" + "d" * 64,
        backend_id="spark",
        state=RunState.SUBMITTED,
        created_at=now,
        updated_at=now,
    )


def _parented_snapshot() -> DatasetSnapshotRef:
    snapshot = _snapshot()
    return DatasetSnapshotRef(
        dataset_id=snapshot.dataset_id,
        version=snapshot.version,
        snapshot_id=snapshot.snapshot_id,
        uri=snapshot.uri,
        data_format=snapshot.data_format,
        schema=snapshot.schema,
        manifest_sha256=snapshot.manifest_sha256,
        processing_fingerprint=snapshot.processing_fingerprint,
        parent_snapshot_ids=("snapshot:" + "d" * 64, "snapshot:" + "e" * 64),
    )


def _edge(parent: str, output: str, *, run_id: str = "run:one") -> LineageEdge:
    return LineageEdge(
        input_snapshot_id=parent,
        output_snapshot_id=output,
        run_id=run_id,
        operator_ids=("capicove.data.project",),
    )


def test_catalog_commit_is_append_only_and_idempotent(tmp_path: Path) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    snapshot = _snapshot()

    store.commit(snapshot, ())
    store.commit(snapshot, ())

    assert store.get(snapshot.snapshot_id) == snapshot
    assert store.latest(snapshot.dataset_id) == snapshot


def test_catalog_round_trips_version_2_schema_for_commit_and_success(tmp_path: Path) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    committed = _complex_snapshot(version=1, snapshot_hash="a")

    store.commit(committed, ())

    assert store.get(committed.snapshot_id) == committed
    assert store.latest(committed.dataset_id) == committed

    store.create(_run())
    store.transition("run:one", RunState.SUBMITTED, RunState.RUNNING)
    succeeded_snapshot = _complex_snapshot(version=2, snapshot_hash="d")

    succeeded = store.commit_success(
        "run:one",
        succeeded_snapshot,
        (),
        native_run_id="capicove:run:one",
    )

    assert succeeded.output_snapshot_id == succeeded_snapshot.snapshot_id
    assert store.get(succeeded_snapshot.snapshot_id) == succeeded_snapshot
    assert store.latest(succeeded_snapshot.dataset_id) == succeeded_snapshot


def test_catalog_round_trips_schema_with_16_complex_levels(tmp_path: Path) -> None:
    value: str | ComplexSchemaType = "string"
    for _ in range(16):
        value = ComplexSchemaType(
            kind=ComplexTypeKind.ARRAY,
            element_type=value,
            contains_null=False,
        )
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:maximum-depth",
        version=1,
        snapshot_id="snapshot:" + "e" * 64,
        uri="file:///tmp/capicove/maximum-depth/v1",
        data_format=DataFormat.PARQUET,
        schema=DatasetSchema((SchemaField("value", value, False),), schema_version=2),
        manifest_sha256="b" * 64,
        processing_fingerprint="c" * 64,
        parent_snapshot_ids=(),
    )
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())

    store.commit(snapshot, ())

    assert store.get(snapshot.snapshot_id) == snapshot


@pytest.mark.parametrize(
    "corruption",
    (
        "invalid_json",
        "extra_envelope_key",
        "missing_envelope_key",
        "wrong_envelope_type",
        "wrong_schema_type",
        "wrong_field_type",
        "wrong_logical_type",
        "duplicate_top_level_member",
        "duplicate_schema_member",
        "duplicate_field_member",
        "deeply_nested_logical_type",
    ),
)
def test_catalog_rejects_malformed_persisted_snapshot_payloads(
    tmp_path: Path, corruption: str
) -> None:
    database_path = (tmp_path / "processing.sqlite3").resolve()
    store = SqliteProcessingStore(database_path)
    snapshot = _snapshot()
    store.commit(snapshot, ())
    with sqlite3.connect(database_path) as connection:
        persisted_payload = connection.execute(
            "SELECT snapshot_json FROM dataset_snapshots WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        ).fetchone()[0]
        payload = json.loads(persisted_payload)
        if corruption == "invalid_json":
            malformed_payload = "{"
        elif corruption == "duplicate_top_level_member":
            malformed_payload = persisted_payload[:-1] + ',"dataset_id":"dataset:curated-traces"}'
        elif corruption == "duplicate_schema_member":
            malformed_payload = persisted_payload.replace(
                '"schema_version":1}', '"schema_version":1,"schema_version":1}', 1
            )
        elif corruption == "duplicate_field_member":
            malformed_payload = persisted_payload.replace(
                '"nullable":false}', '"nullable":false,"nullable":false}', 1
            )
        elif corruption == "deeply_nested_logical_type":
            logical_type = '"string"'
            for _ in range(1000):
                logical_type = (
                    '{"contains_null":false,"element_type":' + logical_type + ',"kind":"array"}'
                )
            malformed_payload = persisted_payload.replace(
                '"logical_type":"string"', f'"logical_type":{logical_type}', 1
            )
        else:
            if corruption == "extra_envelope_key":
                payload["unexpected"] = "secret-marker"
            elif corruption == "missing_envelope_key":
                del payload["schema"]
            elif corruption == "wrong_envelope_type":
                payload["version"] = "one"
            elif corruption == "wrong_schema_type":
                payload["schema"] = []
            elif corruption == "wrong_field_type":
                payload["schema"]["fields"] = [None]
            else:
                payload["schema"]["fields"][0]["logical_type"] = 1
            malformed_payload = json.dumps(payload)
        connection.execute(
            "UPDATE dataset_snapshots SET snapshot_json = ? WHERE snapshot_id = ?",
            (malformed_payload, snapshot.snapshot_id),
        )

    for reader, identifier in (
        (store.get, snapshot.snapshot_id),
        (store.latest, snapshot.dataset_id),
    ):
        with pytest.raises(ProcessingStoreError, match="snapshot metadata is invalid") as error:
            reader(identifier)
        assert "secret-marker" not in str(error.value)


def test_catalog_rejects_same_snapshot_id_with_different_content(tmp_path: Path) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    store.commit(_snapshot(), ())

    with pytest.raises(SnapshotConflictError):
        store.commit(_snapshot(manifest="e" * 64), ())


def test_run_store_enforces_compare_and_set_transitions(tmp_path: Path) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    store.create(_run())

    running = store.transition(
        "run:one",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="capicove:run:one",
    )

    assert running.state is RunState.RUNNING
    with pytest.raises(RunTransitionError):
        store.transition("run:one", RunState.SUBMITTED, RunState.FAILED)


def test_successful_run_persists_output_snapshot_id(tmp_path: Path) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    store.create(_run())
    store.transition("run:one", RunState.SUBMITTED, RunState.RUNNING)

    succeeded = store.transition(
        "run:one",
        RunState.RUNNING,
        RunState.SUCCEEDED,
        output_snapshot_id="snapshot:" + "a" * 64,
    )

    assert succeeded.output_snapshot_id == "snapshot:" + "a" * 64


def test_atomic_success_rolls_back_snapshot_when_run_is_not_running(tmp_path: Path) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    snapshot = _snapshot()
    store.create(_run())

    with pytest.raises(RunTransitionError):
        store.commit_success(
            "run:one",
            snapshot,
            (),
            native_run_id="capicove:run:one",
        )

    assert store.latest(snapshot.dataset_id) is None
    run = store.get_run("run:one")
    assert run is not None and run.state is RunState.SUBMITTED


@pytest.mark.parametrize(
    "lineage_case",
    ("wrong-output", "missing", "extra", "duplicate"),
)
def test_catalog_commit_rejects_lineage_that_disagrees_with_snapshot(
    tmp_path: Path,
    lineage_case: str,
) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    snapshot = _parented_snapshot()
    first, second = snapshot.parent_snapshot_ids
    output = snapshot.snapshot_id
    lineage: tuple[LineageEdge, ...]
    if lineage_case == "wrong-output":
        lineage = (_edge(first, "snapshot:" + "b" * 64), _edge(second, output))
    elif lineage_case == "missing":
        lineage = (_edge(first, output),)
    elif lineage_case == "extra":
        lineage = (
            _edge(first, output),
            _edge(second, output),
            _edge("snapshot:" + "f" * 64, output),
        )
    else:
        lineage = (_edge(first, output), _edge(first, output), _edge(second, output))

    with pytest.raises(ProcessingStoreError, match="lineage"):
        store.commit(snapshot, lineage)

    assert store.latest(snapshot.dataset_id) is None


@pytest.mark.parametrize(
    "lineage_case",
    ("wrong-output", "wrong-run", "missing", "extra", "duplicate"),
)
def test_atomic_success_rolls_back_snapshot_and_run_for_forged_lineage(
    tmp_path: Path,
    lineage_case: str,
) -> None:
    store = SqliteProcessingStore((tmp_path / "processing.sqlite3").resolve())
    store.create(_run())
    store.transition("run:one", RunState.SUBMITTED, RunState.RUNNING)
    snapshot = _parented_snapshot()
    first, second = snapshot.parent_snapshot_ids
    output = snapshot.snapshot_id
    lineage: tuple[LineageEdge, ...]
    if lineage_case == "wrong-output":
        lineage = (_edge(first, "snapshot:" + "b" * 64), _edge(second, output))
    elif lineage_case == "wrong-run":
        lineage = (_edge(first, output, run_id="run:other"), _edge(second, output))
    elif lineage_case == "missing":
        lineage = (_edge(first, output),)
    elif lineage_case == "extra":
        lineage = (
            _edge(first, output),
            _edge(second, output),
            _edge("snapshot:" + "f" * 64, output),
        )
    else:
        lineage = (_edge(first, output), _edge(first, output), _edge(second, output))

    with pytest.raises(ProcessingStoreError, match="lineage"):
        store.commit_success(
            "run:one",
            snapshot,
            lineage,
            native_run_id="capicove:run:one",
        )

    assert store.latest(snapshot.dataset_id) is None
    run = store.get_run("run:one")
    assert run is not None and run.state is RunState.RUNNING
