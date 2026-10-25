"""Ports consumed by backend-neutral data processing orchestration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from capicove.contracts.data import DatasetSnapshotRef, LineageEdge
from capicove.processing.models import (
    BackendRunRef,
    BackendRunStatus,
    OperatorSpec,
    ProcessingContext,
    ProcessingPlan,
    ProcessingRun,
    RunState,
)


@runtime_checkable
class ExecutionBackendOperatorCatalog(Protocol):
    """Expose closed execution-operator metadata without constructing a backend."""

    def operator_specs(self) -> tuple[OperatorSpec, ...]:
        """Return the backend's Agent-readable operator catalog."""


@runtime_checkable
class DatasetCatalog(Protocol):
    """Persist immutable dataset snapshots and lineage."""

    def get(self, snapshot_id: str, /) -> DatasetSnapshotRef | None:
        """Return one snapshot by immutable identity."""

    def latest(self, dataset_id: str, /) -> DatasetSnapshotRef | None:
        """Return the highest committed version for a dataset."""

    def commit(
        self,
        snapshot: DatasetSnapshotRef,
        lineage: tuple[LineageEdge, ...],
        /,
    ) -> None:
        """Append one snapshot and its lineage idempotently."""


@runtime_checkable
class RunStore(Protocol):
    """Persist processing-run lifecycle observations."""

    def create(self, run: ProcessingRun, /) -> None:
        """Create one submitted run idempotently."""

    def get_run(self, run_id: str, /) -> ProcessingRun | None:
        """Return one run by identity."""

    def transition(
        self,
        run_id: str,
        expected: RunState,
        target: RunState,
        /,
        *,
        native_run_id: str | None = None,
        output_snapshot_id: str | None = None,
        detail: str | None = None,
    ) -> ProcessingRun:
        """Compare-and-set one legal lifecycle transition."""


@runtime_checkable
class ExecutionBackend(Protocol):
    """Submit, inspect, and cancel one execution engine's runs."""

    @property
    def backend_id(self) -> str:
        """Return the stable backend identifier."""

    def submit(
        self,
        plan: ProcessingPlan,
        context: ProcessingContext,
        /,
    ) -> BackendRunRef:
        """Submit one plan idempotently."""

    def poll(self, reference: BackendRunRef, /) -> BackendRunStatus:
        """Return the latest backend state."""

    def cancel(self, reference: BackendRunRef, /) -> None:
        """Request cancellation of one known backend run."""
