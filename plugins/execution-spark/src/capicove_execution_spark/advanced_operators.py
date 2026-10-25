"""Compile closed Spark join and global ordering operators."""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from functools import reduce
from math import isfinite
from typing import Any

from capicove.contracts.data import ComplexSchemaType, ComplexTypeKind, SchemaValueType
from capicove.processing.models import LogicalNode
from capicove_execution_spark.io import (
    _canonical_schema_field_from_spark,
    _spark_metadata_for_logical_type,
)


_COLUMN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_COLUMNS = 1000
_MAX_COLUMN_NAME_LENGTH = 128
_JOIN_TYPES = frozenset({"inner", "left", "right", "full", "left_semi", "left_anti"})
_JOIN_PARAMETER_NAMES = frozenset({"how", "left_keys", "right_keys", "right_prefix"})
_SORT_PARAMETER_NAMES = frozenset({"orders"})
_ORDER_PARAMETER_NAMES = frozenset({"column", "direction", "nulls"})
_WINDOW_PARAMETER_NAMES = frozenset({"partition_by", "order_by", "functions"})
_AGGREGATE_PARAMETER_NAMES = frozenset({"group_by", "aggregations"})
_STABLE_DEDUPLICATE_PARAMETER_NAMES = frozenset({"columns", "order_by"})
_RANKING_FUNCTIONS = frozenset({"row_number", "rank", "dense_rank"})
_RELATIVE_FUNCTIONS = frozenset({"lag", "lead"})
_AGGREGATE_FUNCTIONS = frozenset(
    {
        "count",
        "count_distinct",
        "sum",
        "avg",
        "min",
        "max",
        "collect_list",
        "collect_set",
    }
)
_MAX_COLLECTION_ITEMS = 10000


def compile_advanced_node(node: LogicalNode, frames: tuple[Any, ...]) -> Any | None:
    """Compile one supported advanced node or return ``None`` for other nodes."""
    key = node.operator_id, node.operator_version
    if key == ("capicove.data.drop-duplicates", 2):
        return _stable_drop_duplicates(node, frames)
    if key == ("capicove.data.join", 1):
        return _join(node, frames)
    if key == ("capicove.data.sort", 1):
        return _sort(node, frames)
    if key == ("capicove.data.window", 1):
        return _window(node, frames)
    if key == ("capicove.data.aggregate", 1):
        return _aggregate(node, frames)
    return None


def _stable_drop_duplicates(node: LogicalNode, frames: tuple[Any, ...]) -> Any:
    """Select one survivor per key using an explicit unique ordering."""
    if len(frames) != 1:
        raise ValueError("drop-duplicates requires exactly one input binding")
    frame = frames[0]
    parameters = _exact_mapping(
        node.parameters,
        _STABLE_DEDUPLICATE_PARAMETER_NAMES,
        "drop-duplicates parameters",
    )
    keys = _column_names(parameters["columns"], "columns")
    _require_columns(frame, keys)
    order_by = _orders(parameters["order_by"], frame, field_name="order_by")
    candidates = frame.dropDuplicates()
    _require_unique_ordering(
        candidates,
        keys,
        order_by,
        error="drop-duplicates order_by must be a unique tie-break within each key",
    )
    from pyspark.sql import Window, functions

    ordinal = _temporary_column_name(candidates, "__capicove_deduplicate_ordinal")
    specification = Window.partitionBy(
        *(_exact_column(candidates, column) for column in keys)
    ).orderBy(*_spark_order_columns(order_by, candidates))
    ranked = candidates.withColumn(ordinal, functions.row_number().over(specification))
    return ranked.filter(ranked[ordinal] == 1).drop(ordinal)


def _join(node: LogicalNode, frames: tuple[Any, ...]) -> Any:
    """Compile the reviewed equality-only join grammar."""
    if len(frames) != 2:
        raise ValueError("join requires exactly two input bindings")
    left, right = frames
    how, left_keys, right_keys, right_prefix = _join_parameters(node.parameters)
    _require_columns(left, left_keys)
    _require_columns(right, right_keys)
    join_key_types = _join_key_types(left, right, left_keys, right_keys)
    if how in {"left_semi", "left_anti"}:
        left_alias = left.alias("__capicove_left")
        right_alias = right.alias("__capicove_right")
        joined = _joined_frame(left_alias, right_alias, left_keys, right_keys, how)
        return joined.select(*(_exact_column(left_alias, column) for column in left.columns))

    right_non_keys = tuple(column for column in right.columns if column not in right_keys)
    output_names = (*left.columns, *(right_prefix + column for column in right_non_keys))
    _require_unique_output_names(output_names)
    left_alias = left.alias("__capicove_left")
    right_alias = right.alias("__capicove_right")
    joined = _joined_frame(left_alias, right_alias, left_keys, right_keys, how)
    from pyspark.sql import functions

    right_key_by_left = dict(zip(left_keys, right_keys, strict=True))
    output_columns = [
        (
            functions.coalesce(
                _exact_column(left_alias, column),
                _exact_column(right_alias, right_key_by_left[column]),
            ).alias(
                column,
                metadata=_spark_metadata_for_logical_type(join_key_types[column]),
            )
            if column in right_key_by_left
            else _exact_column(left_alias, column)
        )
        for column in left.columns
    ]
    output_columns.extend(
        _exact_column(right_alias, column).alias(
            right_prefix + column,
            metadata=_spark_metadata_for_logical_type(
                _canonical_schema_field_from_spark(right.schema[column]).logical_type
            ),
        )
        for column in right_non_keys
    )
    return joined.select(*output_columns)


def _joined_frame(
    left: Any,
    right: Any,
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
    how: str,
) -> Any:
    """Join two frames using null-intolerant equality for every requested key."""
    conditions = tuple(
        _exact_column(left, left_key) == _exact_column(right, right_key)
        for left_key, right_key in zip(left_keys, right_keys, strict=True)
    )
    condition = reduce(operator.and_, conditions)
    return left.join(right, condition, how)


def _sort(node: LogicalNode, frames: tuple[Any, ...]) -> Any:
    """Compile a reviewed global ordering with explicit null behavior."""
    if len(frames) != 1:
        raise ValueError("sort requires exactly one input binding")
    frame = frames[0]
    parameters = _exact_mapping(node.parameters, _SORT_PARAMETER_NAMES, "sort parameters")
    orders = _orders(parameters["orders"], frame)
    return frame.orderBy(*_spark_order_columns(orders, frame))


def _window(node: LogicalNode, frames: tuple[Any, ...]) -> Any:
    """Compile closed ranking and relative-row window functions."""
    if len(frames) != 1:
        raise ValueError("window requires exactly one input binding")
    frame = frames[0]
    partition_by, order_by, window_functions = _window_parameters(node.parameters, frame)
    if any(
        function["function"] in ({"row_number"} | _RELATIVE_FUNCTIONS)
        for function in window_functions
    ):
        _require_unique_ordering(
            frame,
            partition_by,
            order_by,
            error="order-sensitive window functions require unique ordering within each partition",
        )
    from pyspark.sql import Window, functions

    specification = Window.partitionBy(
        *(_exact_column(frame, column) for column in partition_by)
    ).orderBy(*_spark_order_columns(order_by, frame))
    result = frame
    for function in window_functions:
        function_name = function["function"]
        output_column = function["output_column"]
        if function_name == "row_number":
            expression = functions.row_number()
        elif function_name == "rank":
            expression = functions.rank()
        elif function_name == "dense_rank":
            expression = functions.dense_rank()
        elif function_name == "lag":
            expression = functions.lag(
                _exact_column(frame, function["input_column"]),
                function["offset"],
                function["default"],
            )
        else:
            expression = functions.lead(
                _exact_column(frame, function["input_column"]),
                function["offset"],
                function["default"],
            )
        metadata: dict[str, object] = {}
        if function_name in _RELATIVE_FUNCTIONS:
            logical_type = _canonical_schema_field_from_spark(
                frame.schema[function["input_column"]]
            ).logical_type
            metadata = _spark_metadata_for_logical_type(logical_type)
        result = result.withColumn(
            output_column,
            expression.over(specification).alias(output_column, metadata=metadata),
        )
    return result


def _require_unique_ordering(
    frame: Any,
    partition_by: Sequence[str],
    order_by: Sequence[tuple[str, str, str]],
    *,
    error: str,
) -> None:
    """Reject ordering ties before an order-sensitive deterministic operation."""
    from pyspark.sql import functions

    names = tuple(dict.fromkeys((*partition_by, *(order[0] for order in order_by))))
    count_name = _temporary_column_name(frame, "__capicove_tie_count")
    tied = (
        frame.groupBy(*(_exact_column(frame, name) for name in names))
        .agg(functions.count(functions.lit(1)).alias(count_name))
        .filter(functions.column(count_name) > 1)
        .limit(1)
        .count()
    )
    if tied:
        raise ValueError(error)


def _temporary_column_name(frame: Any, prefix: str) -> str:
    """Return a private column name that does not collide with input fields."""
    candidate = prefix
    suffix = 0
    while candidate in frame.columns:
        suffix += 1
        candidate = f"{prefix}_{suffix}"
    return candidate


def _aggregate(node: LogicalNode, frames: tuple[Any, ...]) -> Any:
    """Compile closed grouped or global aggregate expressions."""
    if len(frames) != 1:
        raise ValueError("aggregate requires exactly one input binding")
    frame = frames[0]
    group_by, aggregations = _aggregate_parameters(node.parameters, frame)
    scalar_aggregations = tuple(
        (index, aggregation)
        for index, aggregation in enumerate(aggregations)
        if aggregation["function"] not in {"collect_list", "collect_set"}
    )
    result = _aggregate_groups(frame, group_by)
    if scalar_aggregations:
        expressions = tuple(
            _aggregate_expression(aggregation, index, frame)
            for index, aggregation in scalar_aggregations
        )
        scalar_result = (
            frame.groupBy(*(_exact_column(frame, column) for column in group_by)).agg(*expressions)
            if group_by
            else frame.agg(*expressions)
        )
        result = _merge_aggregate_result(
            result,
            scalar_result,
            group_by,
            tuple(aggregation["output_column"] for _, aggregation in scalar_aggregations),
            require_match=True,
        )
    for index, aggregation in enumerate(aggregations):
        if aggregation["function"] not in {"collect_list", "collect_set"}:
            continue
        collection_result = _bounded_collection_aggregate(frame, group_by, aggregation, index)
        result = _merge_aggregate_result(
            result,
            collection_result,
            group_by,
            (aggregation["output_column"],),
        )
        result = _fill_empty_collection(result, aggregation["output_column"])
    output_columns = (*group_by, *(aggregation["output_column"] for aggregation in aggregations))
    return result.select(*(_exact_column(result, column) for column in output_columns))


def _aggregate_groups(frame: Any, group_by: tuple[str, ...]) -> Any:
    """Return one row for every group, including one global empty key."""
    if group_by:
        return frame.select(*(_exact_column(frame, column) for column in group_by)).dropDuplicates()
    return frame.sparkSession.range(1).select()


def _bounded_collection_aggregate(
    frame: Any,
    group_by: tuple[str, ...],
    aggregation: Mapping[str, Any],
    index: int,
) -> Any:
    """Limit each group before a collection aggregate materializes its array."""
    from pyspark.sql import Window, functions

    input_column = aggregation["input_column"]
    maximum = aggregation["max_items"]
    ordinal = _temporary_column_name(frame, f"__capicove_collection_{index}_ordinal")
    if aggregation["function"] == "collect_list":
        order_by = tuple(aggregation["order_by"])
        if input_column not in {order[0] for order in order_by}:
            order_by = (*order_by, (input_column, "asc", "last"))
        candidates = frame
    else:
        order_by = ((input_column, "asc", "last"),)
        candidates = (
            frame.filter(_exact_column(frame, input_column).isNotNull())
            .select(*(_exact_column(frame, column) for column in (*group_by, input_column)))
            .dropDuplicates()
        )
    specification = Window.partitionBy(
        *(_exact_column(candidates, column) for column in group_by)
    ).orderBy(*_spark_order_columns(order_by, candidates))
    ranked = candidates.withColumn(ordinal, functions.row_number().over(specification))
    limited = ranked.filter(ranked[ordinal] <= maximum).drop(ordinal)
    if aggregation["function"] == "collect_list":
        input_field = _canonical_schema_field_from_spark(frame.schema[input_column])
        array_type = ComplexSchemaType(
            ComplexTypeKind.ARRAY,
            element_type=input_field.logical_type,
            contains_null=input_field.nullable,
        )
        expression = _stable_collect_list(aggregation, index, limited).alias(
            aggregation["output_column"],
            metadata=_spark_metadata_for_logical_type(array_type),
        )
    else:
        input_field = _canonical_schema_field_from_spark(frame.schema[input_column])
        array_type = ComplexSchemaType(
            ComplexTypeKind.ARRAY,
            element_type=input_field.logical_type,
            contains_null=False,
        )
        expression = functions.sort_array(
            functions.collect_list(_exact_column(limited, input_column))
        ).alias(
            aggregation["output_column"],
            metadata=_spark_metadata_for_logical_type(array_type),
        )
    return (
        limited.groupBy(*(_exact_column(limited, column) for column in group_by)).agg(expression)
        if group_by
        else limited.agg(expression)
    )


def _merge_aggregate_result(
    result: Any,
    addition: Any,
    group_by: tuple[str, ...],
    output_columns: tuple[str, ...],
    *,
    require_match: bool = False,
) -> Any:
    """Merge one aggregate branch without losing null-valued group keys."""
    if not group_by:
        return result.crossJoin(addition)
    left = result.alias("__capicove_aggregate_left")
    right = addition.alias("__capicove_aggregate_right")
    condition = reduce(
        operator.and_,
        (
            _exact_column(left, column).eqNullSafe(_exact_column(right, column))
            for column in group_by
        ),
    )
    joined = left.join(right, condition, "inner" if require_match else "left")
    return joined.select(
        *(_exact_column(left, column) for column in result.columns),
        *(_exact_column(right, column) for column in output_columns),
    )


def _fill_empty_collection(frame: Any, output_column: str) -> Any:
    """Represent groups without collected values as a typed empty array."""
    from pyspark.sql import functions

    field = frame.schema[output_column]
    metadata = dict(field.metadata)
    return frame.withColumn(
        output_column,
        functions.coalesce(
            _exact_column(frame, output_column),
            functions.array().cast(field.dataType),
        ).alias(output_column, metadata=metadata),
    )


def _window_parameters(
    parameters: Mapping[str, object], frame: Any
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...], tuple[Mapping[str, Any], ...]]:
    """Validate and return window partitioning, ordering, and function requests."""
    values = _exact_mapping(parameters, _WINDOW_PARAMETER_NAMES, "window parameters")
    partition_by = _column_names_or_empty(values["partition_by"], "partition_by")
    order_by = _orders(values["order_by"], frame, field_name="order_by")
    _require_columns(frame, partition_by)
    requests = _nonempty_sequence(values["functions"], "functions")
    functions: list[Mapping[str, Any]] = []
    output_columns: list[str] = []
    for request in requests:
        values = _mapping_with_string_keys(request, "window function")
        function_name = values.get("function")
        if not isinstance(function_name, str) or function_name not in (
            _RANKING_FUNCTIONS | _RELATIVE_FUNCTIONS
        ):
            raise ValueError("window function is unsupported")
        expected = (
            frozenset({"function", "output_column"})
            if function_name in _RANKING_FUNCTIONS
            else frozenset({"function", "output_column", "input_column", "offset", "default"})
        )
        values = _exact_mapping(values, expected, "window function")
        output_column = _column_name(values["output_column"], "window output_column")
        request_values: dict[str, Any] = {"function": function_name, "output_column": output_column}
        if function_name in _RELATIVE_FUNCTIONS:
            input_column = _column_name(values["input_column"], "window input_column")
            _require_columns(frame, (input_column,))
            offset = values["offset"]
            if isinstance(offset, bool) or not isinstance(offset, int) or not 1 <= offset <= 10000:
                raise ValueError("window offset must be an integer from 1 through 10000")
            default = values["default"]
            if not _json_scalar(default):
                raise ValueError("window default must be a finite JSON scalar or null")
            _require_compatible_window_default(frame, input_column, default)
            request_values.update(input_column=input_column, offset=offset, default=default)
        functions.append(request_values)
        output_columns.append(output_column)
    _require_new_output_columns(frame, output_columns, "window")
    return partition_by, order_by, tuple(functions)


def _aggregate_parameters(
    parameters: Mapping[str, object], frame: Any
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    """Validate and return closed grouped/global aggregate requests."""
    values = _exact_mapping(parameters, _AGGREGATE_PARAMETER_NAMES, "aggregate parameters")
    group_by = _column_names_or_empty(values["group_by"], "group_by")
    _require_columns(frame, group_by)
    requests = _nonempty_sequence(values["aggregations"], "aggregations")
    aggregations: list[Mapping[str, Any]] = []
    output_columns: list[str] = []
    for request in requests:
        values = _mapping_with_string_keys(request, "aggregation")
        function_name = values.get("function")
        if not isinstance(function_name, str) or function_name not in _AGGREGATE_FUNCTIONS:
            raise ValueError("aggregation function is unsupported")
        expected = _aggregate_parameter_names(function_name, values)
        values = _exact_mapping(values, expected, "aggregation")
        output_column = _column_name(values["output_column"], "aggregation output_column")
        aggregation: dict[str, Any] = {"function": function_name, "output_column": output_column}
        if "input_column" in values:
            input_column = _column_name(values["input_column"], "aggregation input_column")
            _require_columns(frame, (input_column,))
            aggregation["input_column"] = input_column
        if function_name in {"sum", "avg"}:
            _require_numeric_column(frame, aggregation["input_column"])
        if function_name in {"min", "max"}:
            _require_orderable_column(frame, aggregation["input_column"])
        if function_name == "collect_list":
            aggregation["max_items"] = _max_items(values["max_items"])
            order_by = _orders(values["order_by"], frame, field_name="order_by", scalar_only=True)
            _require_orderable_column(frame, aggregation["input_column"], scalar_only=True)
            for column, _, _ in order_by:
                _require_orderable_column(frame, column, scalar_only=True)
            aggregation["order_by"] = order_by
        elif function_name == "collect_set":
            aggregation["max_items"] = _max_items(values["max_items"])
            _require_orderable_column(frame, aggregation["input_column"], scalar_only=True)
        aggregations.append(aggregation)
        output_columns.append(output_column)
    if set(group_by).intersection(output_columns):
        raise ValueError("aggregation output columns must not collide with group_by columns")
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("aggregation output columns must be unique")
    return group_by, tuple(aggregations)


def _aggregate_parameter_names(function_name: str, values: Mapping[str, object]) -> frozenset[str]:
    """Return the exact permitted property set for one aggregate function."""
    if function_name == "count":
        if "input_column" in values:
            return frozenset({"function", "input_column", "output_column"})
        return frozenset({"function", "output_column"})
    if function_name in {"count_distinct", "sum", "avg", "min", "max"}:
        return frozenset({"function", "input_column", "output_column"})
    if function_name == "collect_list":
        return frozenset({"function", "input_column", "order_by", "max_items", "output_column"})
    return frozenset({"function", "input_column", "max_items", "output_column"})


def _aggregate_expression(aggregation: Mapping[str, Any], index: int, frame: Any) -> Any:
    """Build one reviewed aggregate expression from validated parameters."""
    from pyspark.sql import functions

    function_name = aggregation["function"]
    output_column = aggregation["output_column"]
    if function_name == "count":
        input_name = aggregation.get("input_column")
        column = functions.lit(1) if input_name is None else _exact_column(frame, input_name)
        return functions.count(column).alias(output_column)
    input_column = aggregation["input_column"]
    if function_name == "count_distinct":
        return functions.count_distinct(_exact_column(frame, input_column)).alias(output_column)
    if function_name == "sum":
        return functions.sum(_exact_column(frame, input_column)).alias(output_column)
    if function_name == "avg":
        return functions.avg(_exact_column(frame, input_column)).alias(output_column)
    if function_name == "min":
        expression = functions.min(_exact_column(frame, input_column))
        logical_type = _canonical_schema_field_from_spark(frame.schema[input_column]).logical_type
        return expression.alias(
            output_column,
            metadata=_spark_metadata_for_logical_type(logical_type),
        )
    if function_name == "max":
        expression = functions.max(_exact_column(frame, input_column))
        logical_type = _canonical_schema_field_from_spark(frame.schema[input_column]).logical_type
        return expression.alias(
            output_column,
            metadata=_spark_metadata_for_logical_type(logical_type),
        )
    raise ValueError("collection aggregations require bounded pre-aggregation")


def _stable_collect_list(aggregation: Mapping[str, Any], index: int, frame: Any) -> Any:
    """Collect values in reviewed explicit order with a value tie-breaker."""
    from pyspark.sql import functions

    order_by = aggregation["order_by"]
    value_alias = f"__capicove_aggregate_{index}_value"
    order_aliases = tuple(
        f"__capicove_aggregate_{index}_order_{position}" for position, _ in enumerate(order_by)
    )
    fields = [
        _exact_column(frame, column).alias(alias)
        for alias, (column, _, _) in zip(order_aliases, order_by, strict=True)
    ]
    fields.append(_exact_column(frame, aggregation["input_column"]).alias(value_alias))
    collected = functions.collect_list(functions.struct(*fields))
    ordered = functions.array_sort(
        collected,
        lambda left, right: _struct_order_comparison(
            left, right, order_by, order_aliases, value_alias
        ),
    )
    return functions.transform(
        functions.slice(ordered, 1, aggregation["max_items"]),
        lambda item: item[value_alias],
    )


def _struct_order_comparison(
    left: Any,
    right: Any,
    order_by: Sequence[tuple[str, str, str]],
    order_aliases: Sequence[str],
    value_alias: str,
) -> Any:
    """Compare collected structs by explicit keys then the value tie-breaker."""
    from pyspark.sql import functions

    comparison = _column_comparison(left[value_alias], right[value_alias], "asc", "first")
    for alias, (_, direction, nulls) in reversed(tuple(zip(order_aliases, order_by, strict=True))):
        next_comparison = _column_comparison(left[alias], right[alias], direction, nulls)
        comparison = functions.when(next_comparison != 0, next_comparison).otherwise(comparison)
    return comparison


def _column_comparison(left: Any, right: Any, direction: str, nulls: str) -> Any:
    """Return a Spark integer comparator for two orderable scalar columns."""
    from pyspark.sql import functions

    null_first = -1 if nulls == "first" else 1
    null_last = -null_first
    ascending = functions.when(left < right, -1).when(left > right, 1).otherwise(functions.lit(0))
    if direction == "desc":
        ascending = -ascending
    return (
        functions.when(left.isNull() & right.isNull(), functions.lit(0))
        .when(left.isNull(), functions.lit(null_first))
        .when(right.isNull(), functions.lit(null_last))
        .otherwise(ascending)
    )


def _spark_order_columns(orders: Sequence[tuple[str, str, str]], frame: Any) -> list[Any]:
    """Build explicit reviewed Spark sort columns from validated order clauses."""
    columns = []
    for column, direction, nulls in orders:
        spark_column = _exact_column(frame, column)
        if direction == "asc" and nulls == "first":
            columns.append(spark_column.asc_nulls_first())
        elif direction == "asc" and nulls == "last":
            columns.append(spark_column.asc_nulls_last())
        elif direction == "desc" and nulls == "first":
            columns.append(spark_column.desc_nulls_first())
        else:
            columns.append(spark_column.desc_nulls_last())
    return columns


def _exact_column(frame: Any, name: str) -> Any:
    """Return one exact top-level field without Spark identifier expansion."""
    return frame[f"`{name.replace('`', '``')}`"]


def _join_parameters(
    parameters: Mapping[str, object],
) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    """Validate and return the exact equality-join parameter shape."""
    values = _exact_mapping(parameters, _JOIN_PARAMETER_NAMES, "join parameters")
    how = values["how"]
    if not isinstance(how, str) or how not in _JOIN_TYPES:
        raise ValueError("join how is unsupported")
    left_keys = _column_names(values["left_keys"], "left_keys")
    right_keys = _column_names(values["right_keys"], "right_keys")
    if len(left_keys) != len(right_keys):
        raise ValueError("join key arrays must have equal length")
    right_prefix = values["right_prefix"]
    if not isinstance(right_prefix, str) or not _COLUMN_NAME.fullmatch(right_prefix):
        raise ValueError("right_prefix must be a canonical non-blank prefix")
    if len(right_prefix) > _MAX_COLUMN_NAME_LENGTH:
        raise ValueError("right_prefix is too long")
    return how, left_keys, right_keys, right_prefix


def _join_key_types(
    left: Any,
    right: Any,
    left_keys: tuple[str, ...],
    right_keys: tuple[str, ...],
) -> dict[str, SchemaValueType]:
    """Prevalidate each equality key pair by canonical logical type."""
    key_types: dict[str, SchemaValueType] = {}
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        left_type = _canonical_schema_field_from_spark(left.schema[left_key]).logical_type
        right_type = _canonical_schema_field_from_spark(right.schema[right_key]).logical_type
        if left_type != right_type:
            raise ValueError("join key pairs must have identical canonical logical types")
        if not _logical_type_supports_equality(left_type):
            raise ValueError("join key canonical logical types must support equality")
        key_types[left_key] = left_type
    return key_types


def _logical_type_supports_equality(schema_type: SchemaValueType) -> bool:
    """Return whether Spark equality supports one reviewed canonical type."""
    if isinstance(schema_type, str):
        return True
    if schema_type.kind is ComplexTypeKind.ARRAY:
        return schema_type.element_type is not None and _logical_type_supports_equality(
            schema_type.element_type
        )
    if schema_type.kind is ComplexTypeKind.STRUCT:
        return all(
            _logical_type_supports_equality(field.logical_type) for field in schema_type.fields
        )
    return False


def _orders(
    value: object, frame: Any, *, field_name: str = "orders", scalar_only: bool = False
) -> tuple[tuple[str, str, str], ...]:
    """Validate and return the exact global-ordering clauses."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    if len(value) > _MAX_COLUMNS:
        raise ValueError(f"{field_name} exceeds the maximum item count")
    orders: list[tuple[str, str, str]] = []
    for item in value:
        values = _exact_mapping(item, _ORDER_PARAMETER_NAMES, "order")
        column = _column_name(values["column"], "order column")
        direction = values["direction"]
        if not isinstance(direction, str) or direction not in {"asc", "desc"}:
            raise ValueError("order direction is unsupported")
        nulls = values["nulls"]
        if not isinstance(nulls, str) or nulls not in {"first", "last"}:
            raise ValueError("order nulls policy is unsupported")
        orders.append((column, direction, nulls))
    columns = tuple(order[0] for order in orders)
    if len(set(columns)) != len(columns):
        raise ValueError("order columns must be unique")
    _require_columns(frame, columns)
    for column in columns:
        _require_orderable_column(frame, column, scalar_only=scalar_only)
    return tuple(orders)


def _exact_mapping(
    value: object, expected_names: frozenset[str], field_name: str
) -> Mapping[str, object]:
    """Require a mapping with exactly the reviewed parameter names."""
    if not isinstance(value, Mapping) or set(value) != expected_names:
        raise ValueError(f"{field_name} must contain exactly the declared keys")
    return value


def _column_name(value: object, field_name: str) -> str:
    """Require one bounded exact top-level column name."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > _MAX_COLUMN_NAME_LENGTH
    ):
        raise ValueError(f"{field_name} must be a bounded column name")
    return value


def _column_names(value: object, field_name: str) -> tuple[str, ...]:
    """Require a non-empty, bounded, unique sequence of column names."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or len(value) > _MAX_COLUMNS
    ):
        raise ValueError(f"{field_name} must be a non-empty column array")
    names = tuple(_column_name(item, field_name) for item in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must contain unique column names")
    return names


def _column_names_or_empty(value: object, field_name: str) -> tuple[str, ...]:
    """Require a bounded unique sequence of column names that may be empty."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) > _MAX_COLUMNS
    ):
        raise ValueError(f"{field_name} must be a bounded column array")
    names = tuple(_column_name(item, field_name) for item in value)
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must contain unique column names")
    return names


def _nonempty_sequence(value: object, field_name: str) -> Sequence[object]:
    """Require a bounded non-string sequence with at least one element."""
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or len(value) > _MAX_COLUMNS
    ):
        raise ValueError(f"{field_name} must be a non-empty array")
    return value


def _mapping_with_string_keys(value: object, field_name: str) -> Mapping[str, object]:
    """Require a mapping before checking its exact closed property set."""
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return value


def _json_scalar(value: object) -> bool:
    """Return whether a value is a finite JSON scalar or null."""
    return (
        value is None
        or isinstance(value, (str, bool, int))
        or (isinstance(value, float) and isfinite(value))
    )


def _max_items(value: object) -> int:
    """Require a collection output cap within the reviewed resource boundary."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_COLLECTION_ITEMS
    ):
        raise ValueError("aggregation max_items must be an integer from 1 through 10000")
    return value


def _require_columns(frame: Any, columns: tuple[str, ...]) -> None:
    """Reject a request that references an absent frame column."""
    available = set(frame.columns)
    for column in columns:
        if column not in available:
            raise ValueError(f"Spark input is missing column: {column}")


def _require_new_output_columns(frame: Any, names: Sequence[str], operator_name: str) -> None:
    """Reject duplicate or existing output names before compiling an operator."""
    if len(set(names)) != len(names):
        raise ValueError(f"{operator_name} output columns must be unique")
    if any(name in frame.columns for name in names):
        raise ValueError(f"{operator_name} output columns must not exist in the input")


def _require_numeric_column(frame: Any, column: str) -> None:
    """Require an actual Spark numeric data type for a numeric aggregate."""
    from pyspark.sql.types import NumericType

    if not isinstance(frame.schema[column].dataType, NumericType):
        raise ValueError("aggregation input_column must have a numeric Spark data type")


def _require_orderable_column(frame: Any, column: str, *, scalar_only: bool = False) -> None:
    """Require a reviewed Spark orderable type, optionally limited to scalars."""
    data_type = frame.schema[column].dataType
    is_orderable = _is_orderable_scalar_type if scalar_only else _is_spark_orderable_type
    if not is_orderable(data_type):
        type_name = "orderable scalar" if scalar_only else "orderable"
        raise ValueError(f"Spark input column must have an {type_name} Spark data type")


def _is_orderable_scalar_type(data_type: object) -> bool:
    """Return whether a Spark type is an explicitly supported orderable scalar."""
    from pyspark.sql.types import (
        BinaryType,
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        TimestampType,
    )

    return isinstance(
        data_type,
        (
            BinaryType,
            BooleanType,
            ByteType,
            DoubleType,
            FloatType,
            IntegerType,
            LongType,
            ShortType,
            StringType,
            TimestampType,
        ),
    )


def _is_spark_orderable_type(data_type: object) -> bool:
    """Return whether a Spark type is recursively orderable under the reviewed subset."""
    from pyspark.sql.types import ArrayType, StructType

    if _is_orderable_scalar_type(data_type):
        return True
    if isinstance(data_type, ArrayType):
        return _is_spark_orderable_type(data_type.elementType)
    if isinstance(data_type, StructType):
        return all(_is_spark_orderable_type(field.dataType) for field in data_type.fields)
    return False


def _require_compatible_window_default(frame: Any, column: str, default: object) -> None:
    """Reject non-null lag/lead defaults incompatible with the input Spark type."""
    if default is None:
        return
    data_type = frame.schema[column].dataType
    if not _window_default_is_compatible(data_type, default):
        raise ValueError("window default is incompatible with the input Spark data type")


def _window_default_is_compatible(data_type: object, default: object) -> bool:
    """Return whether a finite JSON default fits one canonical Spark scalar type."""
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        TimestampType,
    )

    if isinstance(data_type, BooleanType):
        return isinstance(default, bool)
    if isinstance(data_type, ByteType):
        return _is_bounded_integral_default(default, -(2**7), 2**7 - 1)
    if isinstance(data_type, ShortType):
        return _is_bounded_integral_default(default, -(2**15), 2**15 - 1)
    if isinstance(data_type, IntegerType):
        return _is_bounded_integral_default(default, -(2**31), 2**31 - 1)
    if isinstance(data_type, LongType):
        return _is_bounded_integral_default(default, -(2**63), 2**63 - 1)
    if isinstance(data_type, FloatType):
        return _is_finite_float32_literal(default)
    if isinstance(data_type, DoubleType):
        return _is_finite_spark_float_literal(default)
    if isinstance(data_type, StringType):
        return isinstance(default, str)
    if isinstance(data_type, TimestampType):
        return _is_utc_timestamp_string(default)
    return False


def _is_bounded_integral_default(value: object, lower: int, upper: int) -> bool:
    """Return whether a non-boolean integer fits one signed Spark integral type."""
    return isinstance(value, int) and not isinstance(value, bool) and lower <= value <= upper


def _is_finite_spark_float_literal(value: object) -> bool:
    """Return whether a finite JSON number can safely become a Spark floating literal."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return isfinite(float(value))
    except OverflowError:
        return False


def _is_finite_float32_literal(value: object) -> bool:
    """Return whether a finite JSON number fits an IEEE-754 single-precision literal."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        converted = float(value)
    except OverflowError:
        return False
    return isfinite(converted) and abs(converted) <= 3.4028234663852886e38


def _is_utc_timestamp_string(value: object) -> bool:
    """Return whether a value is a safely parseable ISO-8601 UTC timestamp string."""
    if not isinstance(value, str) or "T" not in value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.utcoffset() == timedelta(0)


def _require_unique_output_names(names: Sequence[str]) -> None:
    """Reject unsafe or ambiguous join output names before running Spark."""
    if any(
        not name.strip() or "\x00" in name or len(name) > _MAX_COLUMN_NAME_LENGTH for name in names
    ):
        raise ValueError("join output columns must be bounded")
    if len(set(names)) != len(names):
        raise ValueError("join output column collision")
