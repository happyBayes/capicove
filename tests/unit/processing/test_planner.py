"""Tests for deterministic closed processing plans."""

from __future__ import annotations

from dataclasses import replace

import pytest

from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
    SchemaField,
)
from capicove.processing.models import LogicalNode, OperatorSpec
from capicove.processing.planner import ProcessingPlanner


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _snapshot() -> DatasetSnapshotRef:
    return DatasetSnapshotRef(
        dataset_id="dataset:source",
        version=1,
        snapshot_id=f"snapshot:{SHA_A}",
        uri="file:///tmp/capicove/source/v1",
        data_format=DataFormat.JSONL,
        schema=DatasetSchema((SchemaField("episode_id", "string", False),)),
        manifest_sha256=SHA_B,
        processing_fingerprint=SHA_C,
        parent_snapshot_ids=(),
    )


def _planner() -> ProcessingPlanner:
    return ProcessingPlanner(
        (
            OperatorSpec(
                "capicove.data.project",
                1,
                frozenset({"columns", "strict"}),
                True,
            ),
            OperatorSpec(
                "capicove.data.join",
                1,
                frozenset({"how", "left_keys", "right_keys", "right_prefix"}),
                True,
                input_count=2,
            ),
        )
    )


def _node(parameters: dict[str, object]) -> LogicalNode:
    return LogicalNode(
        node_id="project",
        operator_id="capicove.data.project",
        operator_version=1,
        input_bindings=("source",),
        output_binding="projected",
        parameters=parameters,
    )


def test_plan_hash_is_independent_of_parameter_key_order() -> None:
    left = _planner().build(
        input_bindings={"source": _snapshot()},
        nodes=(_node({"columns": ["episode_id"], "strict": True}),),
        output_dataset_id="dataset:curated-traces",
    )
    right = _planner().build(
        input_bindings={"source": _snapshot()},
        nodes=(_node({"strict": True, "columns": ["episode_id"]}),),
        output_dataset_id="dataset:curated-traces",
    )

    assert left.plan_id == right.plan_id
    assert left.processing_fingerprint == right.processing_fingerprint


def test_plan_rejects_forged_identity_after_logical_content_changes() -> None:
    plan = _planner().build(
        input_bindings={"source": _snapshot()},
        nodes=(_node({"columns": ["episode_id"], "strict": True}),),
        output_dataset_id="dataset:curated-traces",
    )

    with pytest.raises(ValueError, match="plan_id"):
        replace(plan, output_dataset_id="dataset:different-output")


def test_plan_rejects_noncanonical_output_dataset_id() -> None:
    with pytest.raises(ValueError, match="output_dataset_id"):
        _planner().build(
            input_bindings={"source": _snapshot()},
            nodes=(_node({"columns": ["episode_id"], "strict": True}),),
            output_dataset_id="dataset:contains/slash",
        )


def test_plan_rejects_node_that_reads_unknown_binding() -> None:
    node = LogicalNode(
        node_id="project",
        operator_id="capicove.data.project",
        operator_version=1,
        input_bindings=("missing",),
        output_binding="projected",
        parameters={"columns": ["episode_id"], "strict": True},
    )
    with pytest.raises(ValueError, match="input binding"):
        _planner().build(
            input_bindings={"source": _snapshot()},
            nodes=(node,),
            output_dataset_id="dataset:curated-traces",
        )


def test_plan_rejects_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="parameters"):
        _planner().build(
            input_bindings={"source": _snapshot()},
            nodes=(_node({"columns": ["episode_id"], "strict": True, "shell": "x"}),),
            output_dataset_id="dataset:curated-traces",
        )


def test_plan_rejects_node_with_wrong_operator_input_count() -> None:
    node = LogicalNode(
        node_id="join",
        operator_id="capicove.data.join",
        operator_version=1,
        input_bindings=("source",),
        output_binding="joined",
        parameters={
            "how": "inner",
            "left_keys": ["episode_id"],
            "right_keys": ["episode_id"],
            "right_prefix": "right_",
        },
    )

    with pytest.raises(ValueError, match="input count"):
        _planner().build(
            input_bindings={"source": _snapshot()},
            nodes=(node,),
            output_dataset_id="dataset:curated-traces",
        )


def test_plan_accepts_binary_node_with_available_bindings() -> None:
    node = LogicalNode(
        node_id="join",
        operator_id="capicove.data.join",
        operator_version=1,
        input_bindings=("left", "right"),
        output_binding="joined",
        parameters={
            "how": "inner",
            "left_keys": ["episode_id"],
            "right_keys": ["episode_id"],
            "right_prefix": "right_",
        },
    )

    plan = _planner().build(
        input_bindings={"left": _snapshot(), "right": _snapshot()},
        nodes=(node,),
        output_dataset_id="dataset:curated-traces",
    )

    assert plan.nodes == (node,)


def test_plan_rejects_operator_declared_non_deterministic() -> None:
    planner = ProcessingPlanner(
        (
            OperatorSpec(
                "capicove.data.arbitrary-survivor",
                1,
                frozenset({"columns"}),
                False,
            ),
        )
    )
    node = LogicalNode(
        node_id="deduplicate",
        operator_id="capicove.data.arbitrary-survivor",
        operator_version=1,
        input_bindings=("source",),
        output_binding="deduplicated",
        parameters={"columns": ["episode_id"]},
    )

    with pytest.raises(ValueError, match="non-deterministic"):
        planner.build(
            input_bindings={"source": _snapshot()},
            nodes=(node,),
            output_dataset_id="dataset:curated-traces",
        )
