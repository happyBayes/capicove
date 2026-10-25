"""Threaded regression tests for local Spark backend ownership."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from capicove_execution_spark.backend import SparkExecutionBackend
from capicove_execution_spark.session import SparkBackendConfiguration

from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)
from capicove.processing.local import LocalProcessingWorkspace
from capicove.processing.models import (
    LogicalNode,
    OperatorSpec,
    ProcessingContext,
    ProcessingPlan,
)
from capicove.processing.planner import ProcessingPlanner


def _plan() -> ProcessingPlan:
    snapshot = DatasetSnapshotRef(
        dataset_id="dataset:source",
        version=1,
        snapshot_id="snapshot:" + "a" * 64,
        uri="file:///tmp/capicove/source.jsonl",
        data_format=DataFormat.JSONL,
        schema=DatasetSchema((SchemaField("record_id", "string", False),)),
        manifest_sha256="b" * 64,
        processing_fingerprint="c" * 64,
        parent_snapshot_ids=(),
    )
    return ProcessingPlanner(
        (OperatorSpec("capicove.data.project", 1, frozenset({"columns", "strict"}), True),)
    ).build(
        input_bindings={"source": snapshot},
        nodes=(
            LogicalNode(
                node_id="project",
                operator_id="capicove.data.project",
                operator_version=1,
                input_bindings=("source",),
                output_binding="result",
                parameters={"columns": ["record_id"], "strict": True},
            ),
        ),
        output_dataset_id="dataset:result",
    )


def _backend(tmp_path: Path) -> SparkExecutionBackend:
    return SparkExecutionBackend(
        SparkBackendConfiguration(local_scratch_root=(tmp_path / "scratch").resolve())
    )


def test_backend_serializes_two_run_execution_lifecycles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    plan = _plan()
    rendezvous = threading.Barrier(2)
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def blocked_execute(
        _plan: ProcessingPlan,
        _context: ProcessingContext,
        _workspace: LocalProcessingWorkspace,
        _native_run_id: str,
    ) -> bool:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            try:
                rendezvous.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass
        finally:
            with state_lock:
                active -= 1
        return False

    monkeypatch.setattr(backend, "_execute", blocked_execute)
    errors: list[BaseException] = []

    def submit(run_id: str, root: Path) -> None:
        try:
            backend.submit(
                plan,
                ProcessingContext(
                    run_id=run_id,
                    output_root=root.resolve(),
                    requested_at=datetime.now(UTC),
                ),
            )
        except (
            BaseException
        ) as error:  # pragma: no cover - assertion reports unexpected thread errors
            errors.append(error)

    threads = (
        threading.Thread(target=submit, args=("run:first", tmp_path / "first")),
        threading.Thread(target=submit, args=("run:second", tmp_path / "second")),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert maximum_active == 1


def test_backend_atomically_binds_one_run_to_one_output_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    plan = _plan()
    rendezvous = threading.Barrier(2)

    def blocked_execute(
        _plan: ProcessingPlan,
        _context: ProcessingContext,
        _workspace: LocalProcessingWorkspace,
        _native_run_id: str,
    ) -> bool:
        try:
            rendezvous.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return False

    monkeypatch.setattr(backend, "_execute", blocked_execute)
    outcomes: list[str] = []
    outcome_lock = threading.Lock()

    def submit(root: Path) -> None:
        try:
            backend.submit(
                plan,
                ProcessingContext(
                    run_id="run:shared",
                    output_root=root.resolve(),
                    requested_at=datetime.now(UTC),
                ),
            )
            outcome = "submitted"
        except ValueError as error:
            outcome = str(error)
        with outcome_lock:
            outcomes.append(outcome)

    threads = (
        threading.Thread(target=submit, args=(tmp_path / "first",)),
        threading.Thread(target=submit, args=(tmp_path / "second",)),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert outcomes.count("submitted") == 1
    assert sum("different output root" in outcome for outcome in outcomes) == 1
