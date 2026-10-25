"""Tests for backend-neutral processing values."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from capicove.processing.models import (
    BackendRunRef,
    BackendRunStatus,
    LogicalNode,
    OperatorSpec,
    ProcessingContext,
    ProcessingRun,
    RunState,
)


def test_logical_node_deeply_freezes_parameters() -> None:
    parameters: dict[str, object] = {"columns": ["episode_id"]}
    node = LogicalNode(
        node_id="project",
        operator_id="capicove.data.project",
        operator_version=1,
        input_bindings=("source",),
        output_binding="projected",
        parameters=parameters,
    )

    parameters["columns"] = ["changed"]
    assert node.parameters == {"columns": ("episode_id",)}
    with pytest.raises(TypeError):
        node.parameters["changed"] = True


def test_operator_spec_rejects_boolean_version() -> None:
    with pytest.raises(ValueError, match="operator_version"):
        OperatorSpec("capicove.data.project", True, frozenset({"columns"}), True)


def test_operator_spec_accepts_legacy_positional_construction() -> None:
    spec = OperatorSpec("capicove.data.project", 1, frozenset({"columns"}), True)

    assert spec.input_count == 1
    assert spec.parameter_schema == {}


def test_operator_spec_deeply_freezes_closed_parameter_schema() -> None:
    parameter_schema: dict[str, object] = {
        "type": "object",
        "properties": {"columns": {"type": "array", "items": {"type": "string"}}},
        "required": ["columns"],
        "additionalProperties": False,
    }
    spec = OperatorSpec(
        "capicove.data.project",
        1,
        frozenset({"columns"}),
        True,
        parameter_schema=parameter_schema,
    )

    parameter_schema["properties"] = {}

    frozen_schema = cast(dict[str, object], spec.parameter_schema)
    properties = cast(dict[str, object], frozen_schema["properties"])
    columns = cast(dict[str, object], properties["columns"])
    items = cast(dict[str, str], columns["items"])

    assert items["type"] == "string"
    with pytest.raises(TypeError):
        frozen_schema["type"] = "changed"
    with pytest.raises(TypeError):
        columns["type"] = "changed"


@pytest.mark.parametrize("input_count", (True, 0, 9))
def test_operator_spec_rejects_invalid_input_count(input_count: int) -> None:
    with pytest.raises(ValueError, match="input_count"):
        OperatorSpec(
            "capicove.data.project",
            1,
            frozenset({"columns"}),
            True,
            input_count=input_count,
        )


def test_operator_spec_requires_schema_names_to_match_parameters() -> None:
    with pytest.raises(ValueError, match="parameter_schema"):
        OperatorSpec(
            "capicove.data.project",
            1,
            frozenset({"columns"}),
            True,
            parameter_schema={
                "type": "object",
                "properties": {"wrong": {"type": "array"}},
                "required": ["wrong"],
                "additionalProperties": False,
            },
        )


def test_processing_context_requires_absolute_output_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_root"):
        ProcessingContext("run:one", Path("relative"), datetime.now(UTC))


def test_succeeded_run_requires_output_snapshot() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="output_snapshot_id"):
        ProcessingRun(
            run_id="run:one",
            plan_id="plan:" + "a" * 64,
            backend_id="spark",
            state=RunState.SUCCEEDED,
            created_at=now,
            updated_at=now,
        )


def test_backend_status_requires_matching_reference_state() -> None:
    reference = BackendRunRef("run:one", "spark", "capicove:run:one")
    status = BackendRunStatus(reference, RunState.RUNNING)

    assert status.reference.run_id == "run:one"
