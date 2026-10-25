"""Integration tests for the public local processing workspace."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.data import DatasetSchema, SchemaField
from capicove.processing import (
    LocalProcessingWorkspace,
    ProcessingRecoveryError,
    ProcessingRun,
    RunState,
)


def test_workspace_rejects_a_symlinked_metadata_directory(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    outside = (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    (root / ".capicove").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ProcessingRecoveryError, match="symbolic link"):
        LocalProcessingWorkspace(root)


def test_workspace_rejects_a_symlinked_lock_file(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    LocalProcessingWorkspace(root)
    lock_path = root / ".capicove" / "processing.lock"
    lock_path.unlink()
    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_text("do not lock", encoding="utf-8")
    lock_path.symlink_to(outside_lock)

    with pytest.raises(ProcessingRecoveryError, match="lock"):
        LocalProcessingWorkspace(root)


def _run() -> ProcessingRun:
    now = datetime.now(UTC)
    return ProcessingRun(
        run_id="run:recover",
        plan_id="plan:" + "a" * 64,
        backend_id="capicove.execution.spark",
        state=RunState.SUBMITTED,
        created_at=now,
        updated_at=now,
    )


def test_workspace_recovers_publication_interrupted_after_rename(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    workspace = LocalProcessingWorkspace(root)
    workspace.create(_run())
    workspace.transition(
        "run:recover",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="spark-local:catalog:run:recover",
    )
    staging = workspace.stage("run:recover")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = workspace.publish(
        run_id="run:recover",
        dataset_id="dataset:recovered",
        version=1,
        staging_path=staging,
        schema=DatasetSchema((SchemaField("record_id", "string", False),)),
        processing_fingerprint="b" * 64,
        parent_snapshot_ids=(),
        validator=lambda _: True,
    )

    recovered = LocalProcessingWorkspace(root)

    run = recovered.get_run("run:recover")
    assert run is not None and run.state is RunState.FAILED
    assert run.detail == "RECOVERED_INCOMPLETE_PUBLICATION"
    assert recovered.get(snapshot.snapshot_id) is None
    assert not Path(snapshot.uri.removeprefix("file://")).exists()


def test_workspace_recovers_publication_interrupted_during_staging(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    workspace = LocalProcessingWorkspace(root)
    workspace.create(_run())
    workspace.transition(
        "run:recover",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="spark-local:catalog:run:recover",
    )
    staging = workspace.stage("run:recover")
    (staging / "part-00000.parquet").write_bytes(b"partial-output")

    recovered = LocalProcessingWorkspace(root)

    run = recovered.get_run("run:recover")
    assert run is not None and run.state is RunState.FAILED
    assert run.detail == "RECOVERED_INCOMPLETE_PUBLICATION"
    assert not staging.exists()


def test_workspace_does_not_recover_a_live_locked_execution(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    workspace = LocalProcessingWorkspace(root)
    workspace.create(_run())
    workspace.transition(
        "run:recover",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="spark-local:catalog:run:recover",
    )

    with workspace.execution_lock():
        staging = workspace.stage("run:recover")
        competing = LocalProcessingWorkspace(root)
        run = competing.get_run("run:recover")

        assert run is not None and run.state is RunState.RUNNING
        assert staging.exists()

    recovered = LocalProcessingWorkspace(root)
    run = recovered.get_run("run:recover")
    assert run is not None and run.state is RunState.FAILED
    assert not staging.exists()


@pytest.mark.parametrize("marker_payload", (None, b"{"))
def test_workspace_recovers_an_incomplete_initial_staging_directory(
    tmp_path: Path,
    marker_payload: bytes | None,
) -> None:
    root = (tmp_path / "datasets").resolve()
    workspace = LocalProcessingWorkspace(root)
    workspace.create(_run())
    workspace.transition(
        "run:recover",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="spark-local:catalog:run:recover",
    )
    encoded_run_id = b"run:recover".hex()
    interrupted = root / ".staging" / f".run-{encoded_run_id}.interrupted"
    interrupted.mkdir()
    if marker_payload is not None:
        (interrupted / ".capicove-pending.json").write_bytes(marker_payload)

    recovered = LocalProcessingWorkspace(root)

    run = recovered.get_run("run:recover")
    assert run is not None and run.state is RunState.FAILED
    assert not interrupted.exists()


def test_workspace_commits_and_finalizes_successful_publication(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    workspace = LocalProcessingWorkspace(root)
    workspace.create(_run())
    workspace.transition(
        "run:recover",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="spark-local:catalog:run:recover",
    )
    staging = workspace.stage("run:recover")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = workspace.publish(
        run_id="run:recover",
        dataset_id="dataset:committed",
        version=1,
        staging_path=staging,
        schema=DatasetSchema((SchemaField("record_id", "string", False),)),
        processing_fingerprint="b" * 64,
        parent_snapshot_ids=(),
        validator=lambda _: True,
    )

    succeeded = workspace.commit_success(
        "run:recover",
        snapshot,
        (),
        native_run_id="spark-local:catalog:run:recover",
    )

    assert succeeded.state is RunState.SUCCEEDED
    assert workspace.latest("dataset:committed") == snapshot
    assert workspace.finalize("run:recover", snapshot)
    assert not (Path(snapshot.uri.removeprefix("file://")) / ".capicove-pending.json").exists()


def test_workspace_finalizes_committed_marker_during_restart(tmp_path: Path) -> None:
    root = (tmp_path / "datasets").resolve()
    workspace = LocalProcessingWorkspace(root)
    workspace.create(_run())
    workspace.transition(
        "run:recover",
        RunState.SUBMITTED,
        RunState.RUNNING,
        native_run_id="spark-local:catalog:run:recover",
    )
    staging = workspace.stage("run:recover")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = workspace.publish(
        run_id="run:recover",
        dataset_id="dataset:restart-success",
        version=1,
        staging_path=staging,
        schema=DatasetSchema((SchemaField("record_id", "string", False),)),
        processing_fingerprint="b" * 64,
        parent_snapshot_ids=(),
        validator=lambda _: True,
    )
    workspace.commit_success(
        "run:recover",
        snapshot,
        (),
        native_run_id="spark-local:catalog:run:recover",
    )

    recovered = LocalProcessingWorkspace(root)

    assert recovered.get(snapshot.snapshot_id) == snapshot
    assert not (Path(snapshot.uri.removeprefix("file://")) / ".capicove-pending.json").exists()


@pytest.mark.parametrize("damage", ["part", "manifest", "symlink", "directory", "extra_part"])
def test_workspace_validates_committed_snapshot_before_calling_reader(
    tmp_path: Path, damage: str
) -> None:
    workspace = LocalProcessingWorkspace(tmp_path)
    verify: Any = getattr(workspace, "validate_snapshot", None)
    assert callable(verify), "public committed snapshot validation is required"
    assert verify("snapshot:" + "f" * 64, validator=lambda _: True) is None

    workspace.create(_run())
    workspace.transition("run:recover", RunState.SUBMITTED, RunState.RUNNING)
    staging = workspace.stage("run:recover")
    (staging / "part-00000.parquet").write_bytes(b"PAR1-authored-test-PAR1")
    snapshot = workspace.publish(
        run_id="run:recover",
        dataset_id="dataset:validated",
        version=1,
        staging_path=staging,
        schema=DatasetSchema((SchemaField("record_id", "string", False),)),
        processing_fingerprint="b" * 64,
        parent_snapshot_ids=(),
        validator=lambda _: True,
    )
    workspace.commit_success("run:recover", snapshot, (), native_run_id="local")
    workspace.finalize("run:recover", snapshot)
    visited: list[Path] = []

    def validator(path: Path) -> bool:
        visited.append(path)
        return True

    assert verify(snapshot.snapshot_id, validator=validator) == snapshot
    assert len(visited) == 1
    with pytest.raises(ProcessingRecoveryError):
        verify(snapshot.snapshot_id, validator=lambda _: False)
    visited.clear()
    part = tmp_path / "validated/v1/part-00000.parquet"
    if damage == "part":
        part.write_bytes(b"changed")
    elif damage == "manifest":
        (part.parent / "manifest.json").write_bytes(b"{}")
    elif damage == "symlink":
        outside = tmp_path / "outside.parquet"
        outside.write_bytes(part.read_bytes())
        part.unlink()
        part.symlink_to(outside)
    elif damage == "directory":
        part.unlink()
        part.mkdir()
    else:
        (part.parent / "part-00001.parquet").write_bytes(b"extra")
    with pytest.raises(ProcessingRecoveryError):
        verify(snapshot.snapshot_id, validator=validator)
    assert visited == []
