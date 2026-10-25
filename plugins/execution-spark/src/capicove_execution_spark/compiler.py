"""Compile closed Capicove logical operators to Spark DataFrames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, cast

from capicove.processing.models import LogicalNode, thaw_json
from capicove_execution_spark.advanced_operators import compile_advanced_node
from capicove_execution_spark.operator_contracts import (
    spark_operator_specs as _spark_operator_specs,
)


spark_operator_specs = _spark_operator_specs


class UnsupportedSparkOperatorError(ValueError):
    """Report an operator without a reviewed Spark implementation."""


class SparkPlanCompiler:
    """Apply reviewed logical nodes without evaluating caller code or SQL."""

    def compile_nodes(
        self,
        nodes: Sequence[LogicalNode],
        inputs: Mapping[str, Any],
        /,
    ) -> Any:
        """Compile nodes in validated topological order and return the final frame."""
        frames = dict(inputs)
        final: Any | None = None
        for node in nodes:
            try:
                node_frames = tuple(frames[name] for name in node.input_bindings)
            except KeyError as error:
                raise ValueError("node input binding is unavailable") from error
            final = self._compile_node(node, node_frames)
            frames[node.output_binding] = final
        if final is None:
            raise ValueError("nodes must not be empty")
        return final

    def _compile_node(self, node: LogicalNode, frames: tuple[Any, ...]) -> Any:
        advanced = compile_advanced_node(node, frames)
        if advanced is not None:
            return advanced
        frame = _one_frame(frames)
        key = node.operator_id, node.operator_version
        if key == ("capicove.data.project", 1):
            return self._project(node, frame)
        if key == ("capicove.data.filter", 1):
            return self._filter(node, frame)
        if key == ("capicove.data.drop-duplicates", 1):
            columns = _string_sequence(node.parameters["columns"], "columns")
            _require_columns(frame, columns)
            return frame.dropDuplicates(list(columns))
        if key == ("capicove.data.group-count", 1):
            return self._group_count(node, frame)
        if key == ("capicove.agent-trace.project", 1):
            return self._trace_project(node, frame)
        raise UnsupportedSparkOperatorError(
            f"Spark operator is unsupported: {node.operator_id}@{node.operator_version}"
        )

    def _project(self, node: LogicalNode, frame: Any) -> Any:
        columns = _string_sequence(node.parameters["columns"], "columns")
        strict = node.parameters["strict"]
        if not isinstance(strict, bool):
            raise ValueError("strict must be a boolean")
        if strict:
            _require_columns(frame, columns)
            selected = columns
        else:
            selected = tuple(column for column in columns if column in frame.columns)
            if not selected:
                raise ValueError("project has no available columns")
        return frame.select(*(_exact_column(frame, column) for column in selected))

    def _filter(self, node: LogicalNode, frame: Any) -> Any:
        predicate = _mapping(node.parameters["predicate"], "predicate")
        operator = _string(predicate.get("operator"), "predicate.operator")
        column_name = _string(predicate.get("column"), "predicate.column")
        _require_columns(frame, (column_name,))
        column = _exact_column(frame, column_name)
        if operator in {"is_null", "is_not_null"}:
            if set(predicate) != {"column", "operator"}:
                raise ValueError("null predicate accepts only column and operator")
            expression = column.isNull() if operator == "is_null" else column.isNotNull()
        else:
            if set(predicate) != {"column", "operator", "value"}:
                raise ValueError("comparison predicate requires column, operator, and value")
            value = cast(Any, thaw_json(predicate["value"]))  # type: ignore[arg-type]
            if operator == "eq":
                expression = column == value
            elif operator == "ne":
                expression = column != value
            elif operator == "lt":
                expression = column < value
            elif operator == "lte":
                expression = column <= value
            elif operator == "gt":
                expression = column > value
            elif operator == "gte":
                expression = column >= value
            elif operator == "in":
                expression = column.isin(_list(value, "predicate.value", max_items=1000))
            else:
                raise ValueError("predicate.operator is unsupported")
        return frame.filter(expression)

    def _group_count(self, node: LogicalNode, frame: Any) -> Any:
        columns = _string_sequence(node.parameters["columns"], "columns")
        count_column = _string(node.parameters["count_column"], "count_column")
        _require_columns(frame, columns)
        if count_column in columns:
            raise ValueError("count_column must differ from grouping columns")
        grouped = frame.groupBy(*(_exact_column(frame, column) for column in columns)).count()
        return grouped.withColumnRenamed(
            "count",
            count_column,
        )

    def _trace_project(self, node: LogicalNode, frame: Any) -> Any:
        columns = _string_sequence(node.parameters["columns"], "columns")
        _require_columns(frame, columns)
        return frame.select(*(_exact_column(frame, column) for column in columns))


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _one_frame(frames: tuple[Any, ...]) -> Any:
    """Return the sole input for a unary reviewed operator."""
    if len(frames) != 1:
        raise ValueError("Spark unary operators require exactly one input binding")
    return frames[0]


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{field_name} must be a non-empty string array")
    if len(value) > 1000:
        raise ValueError(f"{field_name} must contain at most 1000 values")
    values = tuple(_string(item, field_name) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must contain unique values")
    return values


def _require_columns(frame: Any, columns: tuple[str, ...]) -> None:
    missing = tuple(column for column in columns if column not in frame.columns)
    if missing:
        raise ValueError(f"Spark input is missing column: {missing[0]}")


def _exact_column(frame: Any, name: str) -> Any:
    """Return one exact top-level column without Spark expression parsing."""
    return frame[f"`{name.replace('`', '``')}`"]


def _list(value: object, field_name: str, *, max_items: int | None = None) -> list[object]:
    if (
        not isinstance(value, list)
        or not value
        or (max_items is not None and len(value) > max_items)
    ):
        raise ValueError(f"{field_name} must be a non-empty array")
    if any(not _finite_json_scalar(item) for item in value):
        raise ValueError(f"{field_name} items must be a finite JSON scalar or null")
    return value


def _finite_json_scalar(value: object) -> bool:
    """Return whether a value is a finite JSON scalar or null."""
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and isfinite(value))
    )
