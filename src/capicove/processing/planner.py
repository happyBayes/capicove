"""Deterministic validation and hashing of logical processing plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from capicove.contracts.data import DatasetSnapshotRef
from capicove.processing.models import (
    LogicalNode,
    OperatorSpec,
    ProcessingPlan,
    derive_processing_identity,
)


class ProcessingPlanner:
    """Build closed, topologically valid processing plans."""

    def __init__(self, operators: Iterable[OperatorSpec]) -> None:
        catalog: dict[tuple[str, int], OperatorSpec] = {}
        for operator in operators:
            if operator.registry_key in catalog:
                raise ValueError("operator catalog contains a duplicate version")
            catalog[operator.registry_key] = operator
        if not catalog:
            raise ValueError("operator catalog must not be empty")
        self._operators = catalog

    def build(
        self,
        *,
        input_bindings: Mapping[str, DatasetSnapshotRef],
        nodes: Iterable[LogicalNode],
        output_dataset_id: str,
    ) -> ProcessingPlan:
        """Validate and return one deterministic logical plan."""
        inputs = dict(input_bindings)
        ordered_nodes = tuple(nodes)
        if not inputs:
            raise ValueError("input_bindings must not be empty")
        if not ordered_nodes:
            raise ValueError("nodes must not be empty")
        available = set(inputs)
        node_ids: set[str] = set()
        for node in ordered_nodes:
            if node.node_id in node_ids:
                raise ValueError("node_id values must be unique")
            node_ids.add(node.node_id)
            missing = set(node.input_bindings) - available
            if missing:
                raise ValueError(f"node references unknown input binding: {sorted(missing)[0]}")
            if node.output_binding in available:
                raise ValueError("node output binding must be new")
            try:
                spec = self._operators[(node.operator_id, node.operator_version)]
            except KeyError as error:
                raise ValueError("node references an unsupported operator version") from error
            if not spec.deterministic:
                raise ValueError("node references a non-deterministic operator version")
            if len(node.input_bindings) != spec.input_count:
                raise ValueError("node input count must match the operator specification")
            parameter_names = frozenset(node.parameters)
            if parameter_names != spec.parameter_names:
                raise ValueError("node parameters must exactly match the operator specification")
            available.add(node.output_binding)
        plan_id, processing_fingerprint = derive_processing_identity(
            inputs,
            ordered_nodes,
            output_dataset_id,
        )
        return ProcessingPlan(
            plan_id=plan_id,
            input_bindings=inputs,
            nodes=ordered_nodes,
            output_dataset_id=output_dataset_id,
            processing_fingerprint=processing_fingerprint,
        )
