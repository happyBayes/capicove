"""Synchronous local Spark batch execution with governed publication."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from capicove.contracts.data import LineageEdge
from capicove.processing.local import LocalProcessingWorkspace, ProcessingRecoveryError
from capicove.processing.models import (
    BackendRunRef,
    BackendRunStatus,
    ProcessingContext,
    ProcessingPlan,
    ProcessingRun,
    RunState,
)
from capicove_execution_spark.compiler import SparkPlanCompiler
from capicove_execution_spark.io import SparkDatasetIO, dataset_schema_from_spark
from capicove_execution_spark.session import (
    SparkBackendConfiguration,
    create_spark_session,
)


_SPARK_EXECUTION_LOCK = threading.RLock()


class SparkExecutionBackend:
    """Execute one bounded local Spark batch at a time."""

    def __init__(self, configuration: SparkBackendConfiguration) -> None:
        self._configuration = configuration
        self._workspaces: dict[str, tuple[Path, LocalProcessingWorkspace, str]] = {}
        self._workspace_lock = threading.RLock()

    @property
    def backend_id(self) -> str:
        """Return the stable plugin backend identity."""
        return "capicove.execution.spark"

    def submit(
        self,
        plan: ProcessingPlan,
        context: ProcessingContext,
        /,
    ) -> BackendRunRef:
        """Execute and durably record one idempotent local Spark batch."""
        output_root = context.output_root.resolve()
        with self._workspace_lock:
            bound = self._workspaces.get(context.run_id)
            if bound is not None and bound[0] != output_root:
                raise ValueError("run ID is already bound to a different output root")
            if bound is None:
                workspace = LocalProcessingWorkspace(output_root)
                existing = workspace.get_run(context.run_id)
                catalog_id = hashlib.sha256(output_root.as_uri().encode("utf-8")).hexdigest()[:16]
                native_run_id = (
                    existing.native_run_id
                    if existing is not None and existing.native_run_id is not None
                    else f"spark-local:{catalog_id}:{context.run_id}"
                )
                self._workspaces[context.run_id] = (output_root, workspace, native_run_id)
            else:
                workspace = bound[1]
                native_run_id = bound[2]
        with workspace.execution_lock():
            existing = workspace.get_run(context.run_id)
            if existing is not None:
                if existing.plan_id != plan.plan_id or existing.backend_id != self.backend_id:
                    raise ValueError("run ID is already bound to a different execution")
                if existing.native_run_id is not None:
                    native_run_id = existing.native_run_id
                    with self._workspace_lock:
                        self._workspaces[context.run_id] = (
                            output_root,
                            workspace,
                            native_run_id,
                        )
                reference = BackendRunRef(context.run_id, self.backend_id, native_run_id)
                if existing.state in {RunState.SUBMITTED, RunState.RUNNING}:
                    workspace.transition(
                        context.run_id,
                        existing.state,
                        RunState.FAILED,
                        native_run_id=native_run_id,
                        detail="RECOVERED_INCOMPLETE_EXECUTION",
                    )
                return reference
            reference = BackendRunRef(context.run_id, self.backend_id, native_run_id)
            submitted = ProcessingRun(
                run_id=context.run_id,
                plan_id=plan.plan_id,
                backend_id=self.backend_id,
                state=RunState.SUBMITTED,
                created_at=context.requested_at,
                updated_at=context.requested_at,
            )
            workspace.create(submitted)
            workspace.transition(
                context.run_id,
                RunState.SUBMITTED,
                RunState.RUNNING,
                native_run_id=native_run_id,
            )
            with _SPARK_EXECUTION_LOCK:
                recovery_required = self._execute(plan, context, workspace, native_run_id)
        if recovery_required:
            recovered_workspace = LocalProcessingWorkspace(output_root)
            recovered = recovered_workspace.get_run(context.run_id)
            if recovered is None or recovered.state in {RunState.SUBMITTED, RunState.RUNNING}:
                raise ProcessingRecoveryError("Spark publication recovery is incomplete")
            with self._workspace_lock:
                self._workspaces[context.run_id] = (
                    output_root,
                    recovered_workspace,
                    native_run_id,
                )
        return reference

    def poll(self, reference: BackendRunRef, /) -> BackendRunStatus:
        """Return the latest durable state for a run submitted by this instance."""
        with self._workspace_lock:
            self._validate_reference(reference)
            try:
                workspace = self._workspaces[reference.run_id][1]
            except KeyError as error:
                raise ValueError("Spark run is unknown to this backend instance") from error
        run = workspace.get_run(reference.run_id)
        if run is None:  # pragma: no cover - protected by submit transaction
            raise ValueError("Spark run does not exist")
        return BackendRunStatus(
            reference=reference,
            state=run.state,
            output_snapshot_id=run.output_snapshot_id,
            detail=run.detail,
        )

    def cancel(self, reference: BackendRunRef, /) -> None:
        """Reject cancellation after this synchronous MVP has reached a terminal state."""
        status = self.poll(reference)
        if status.state in {RunState.SUBMITTED, RunState.RUNNING}:
            raise RuntimeError("synchronous Spark cancellation is not available")

    def _execute(
        self,
        plan: ProcessingPlan,
        context: ProcessingContext,
        workspace: LocalProcessingWorkspace,
        native_run_id: str,
    ) -> bool:
        recovery_required = False
        staging = None
        snapshot = None
        spark = None
        try:
            staging = workspace.stage(context.run_id)
            ordered_inputs = tuple(
                (binding, plan.input_bindings[binding]) for binding in sorted(plan.input_bindings)
            )
            unique_input_snapshots = []
            seen_snapshot_ids: set[str] = set()
            for _, input_snapshot in ordered_inputs:
                if input_snapshot.snapshot_id not in seen_snapshot_ids:
                    seen_snapshot_ids.add(input_snapshot.snapshot_id)
                    unique_input_snapshots.append(input_snapshot)
            spark = create_spark_session(self._configuration)
            io = SparkDatasetIO(spark)
            frames = {
                binding: io.read(input_snapshot) for binding, input_snapshot in ordered_inputs
            }
            result = SparkPlanCompiler().compile_nodes(plan.nodes, frames)
            output_schema = dataset_schema_from_spark(result.schema)
            data_path = staging / "data"
            io.write_parquet(result, data_path)
            latest = workspace.latest(plan.output_dataset_id)
            version = 1 if latest is None else latest.version + 1
            snapshot = workspace.publish(
                run_id=context.run_id,
                dataset_id=plan.output_dataset_id,
                version=version,
                staging_path=staging,
                schema=output_schema,
                processing_fingerprint=plan.processing_fingerprint,
                parent_snapshot_ids=tuple(item.snapshot_id for item in unique_input_snapshots),
                validator=lambda path: io.validate_parquet(path / "data", output_schema),
            )
            staging = None
            operator_ids = tuple(dict.fromkeys(node.operator_id for node in plan.nodes))
            lineage = tuple(
                LineageEdge(
                    input_snapshot_id=input_snapshot.snapshot_id,
                    output_snapshot_id=snapshot.snapshot_id,
                    run_id=context.run_id,
                    operator_ids=operator_ids,
                )
                for input_snapshot in unique_input_snapshots
            )
            workspace.commit_success(
                context.run_id,
                snapshot,
                lineage,
                native_run_id=native_run_id,
            )
            workspace.finalize(context.run_id, snapshot)
        except Exception:
            if staging is not None:
                workspace.discard(staging)
            current = workspace.get_run(context.run_id)
            if current is not None and current.state is RunState.SUCCEEDED:
                if snapshot is not None:
                    workspace.finalize(context.run_id, snapshot)
            elif snapshot is not None and not workspace.retract(snapshot):
                recovery_required = True
            elif current is not None and current.state is RunState.RUNNING:
                workspace.transition(
                    context.run_id,
                    RunState.RUNNING,
                    RunState.FAILED,
                    native_run_id=native_run_id,
                    detail="SPARK_BATCH_EXECUTION_FAILED",
                )
        finally:
            if spark is not None:
                try:
                    spark.stop()
                # Cleanup failure must not overwrite the persisted run outcome.
                except Exception:  # nosec B110
                    pass
        return recovery_required

    def _validate_reference(self, reference: BackendRunRef) -> None:
        if reference.backend_id != self.backend_id:
            raise ValueError("backend reference belongs to a different backend")
        with self._workspace_lock:
            bound = self._workspaces.get(reference.run_id)
        if bound is not None and reference.native_run_id != bound[2]:
            raise ValueError("backend reference has a different native run identity")
