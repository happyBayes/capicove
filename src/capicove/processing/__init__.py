"""Multimodal and governed data processing capabilities for Capicove."""

from capicove.processing.local import LocalProcessingWorkspace, ProcessingRecoveryError
from capicove.processing.models import (
    BackendRunRef,
    BackendRunStatus,
    LogicalNode,
    OperatorSpec,
    ProcessingContext,
    ProcessingExecution,
    ProcessingPlan,
    ProcessingRun,
    RunState,
)
from capicove.processing.planner import ProcessingPlanner
from capicove.processing.ports import DatasetCatalog, ExecutionBackend, RunStore
from capicove.processing.trace_dataset import TRACE_DATASET_SCHEMA, trace_export_snapshot


__all__ = [
    "TRACE_DATASET_SCHEMA",
    "BackendRunRef",
    "BackendRunStatus",
    "DatasetCatalog",
    "ExecutionBackend",
    "LocalProcessingWorkspace",
    "LogicalNode",
    "OperatorSpec",
    "ProcessingContext",
    "ProcessingExecution",
    "ProcessingPlan",
    "ProcessingPlanner",
    "ProcessingRecoveryError",
    "ProcessingRun",
    "RunState",
    "RunStore",
    "trace_export_snapshot",
]
