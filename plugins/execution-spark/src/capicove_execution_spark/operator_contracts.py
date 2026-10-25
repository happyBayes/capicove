"""Dependency-light Spark operator metadata for Agent plan construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from capicove.processing.models import OperatorSpec


JOIN_PARAMETERS = frozenset({"how", "left_keys", "right_keys", "right_prefix"})
WINDOW_PARAMETERS = frozenset({"partition_by", "order_by", "functions"})
AGGREGATE_PARAMETERS = frozenset({"group_by", "aggregations"})


def _closed_object(
    properties: Mapping[str, object], required: Sequence[str] | None = None
) -> dict[str, object]:
    """Build a JSON Schema object that rejects undeclared properties."""
    names = tuple(properties) if required is None else tuple(required)
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(names),
        "additionalProperties": False,
    }


def _string_enum(*values: str) -> dict[str, object]:
    """Build a closed string enumeration."""
    return {"type": "string", "enum": list(values)}


def _column_array(*, minimum: int) -> dict[str, object]:
    """Build metadata for a bounded-or-empty array of top-level columns."""
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": minimum,
        "maxItems": 1000,
    }


def _order_array() -> dict[str, object]:
    """Build metadata for one or more explicit global ordering clauses."""
    return {
        "type": "array",
        "items": _closed_object(
            {
                "column": {"type": "string"},
                "direction": _string_enum("asc", "desc"),
                "nulls": _string_enum("first", "last"),
            }
        ),
        "minItems": 1,
        "maxItems": 1000,
    }


def _json_scalar_schema() -> dict[str, object]:
    """Build metadata for a finite JSON scalar or null.

    JSON Schema has no portable finite-number keyword. OperatorSpec's JSON freezer
    rejects non-finite Python floats before this metadata can be published or used.
    """
    return {"type": ["string", "number", "boolean", "null"]}


def _predicate_schema() -> dict[str, object]:
    """Build metadata for the existing closed structured-filter grammar."""
    null_check = _closed_object(
        {
            "column": {"type": "string"},
            "operator": _string_enum("is_null", "is_not_null"),
        }
    )
    comparison = _closed_object(
        {
            "column": {"type": "string"},
            "operator": _string_enum("eq", "ne", "lt", "lte", "gt", "gte"),
            "value": _json_scalar_schema(),
        }
    )
    membership = _closed_object(
        {
            "column": {"type": "string"},
            "operator": _string_enum("in"),
            "value": {
                "type": "array",
                "items": _json_scalar_schema(),
                "minItems": 1,
                "maxItems": 1000,
            },
        }
    )
    return {"oneOf": [null_check, comparison, membership]}


def _window_function_schema() -> dict[str, object]:
    """Build metadata for the reviewed ranking and relative-row functions."""
    ranking = tuple(
        _closed_object(
            {
                "function": _string_enum(function_name),
                "output_column": {"type": "string"},
            }
        )
        for function_name in ("row_number", "rank", "dense_rank")
    )
    relative = tuple(
        _closed_object(
            {
                "function": _string_enum(function_name),
                "output_column": {"type": "string"},
                "input_column": {"type": "string"},
                "offset": {"type": "integer", "minimum": 1, "maximum": 10000},
                "default": _json_scalar_schema(),
            }
        )
        for function_name in ("lag", "lead")
    )
    return {"oneOf": [*ranking, *relative]}


def _aggregation_schema() -> dict[str, object]:
    """Build metadata for the reviewed aggregate-function union."""
    output = {"output_column": {"type": "string"}}
    count = _closed_object({"function": _string_enum("count"), **output})
    count_with_column = _closed_object(
        {
            "function": _string_enum("count"),
            "input_column": {"type": "string"},
            **output,
        }
    )
    scalar = tuple(
        _closed_object(
            {
                "function": _string_enum(function_name),
                "input_column": {"type": "string"},
                **output,
            }
        )
        for function_name in ("count_distinct", "sum", "avg", "min", "max")
    )
    collect_list = _closed_object(
        {
            "function": _string_enum("collect_list"),
            "input_column": {"type": "string"},
            "order_by": _order_array(),
            "max_items": {"type": "integer", "minimum": 1, "maximum": 10000},
            **output,
        }
    )
    collect_set = _closed_object(
        {
            "function": _string_enum("collect_set"),
            "input_column": {"type": "string"},
            "max_items": {"type": "integer", "minimum": 1, "maximum": 10000},
            **output,
        }
    )
    return {"oneOf": [count, count_with_column, *scalar, collect_list, collect_set]}


JOIN_SCHEMA = _closed_object(
    {
        "how": _string_enum("inner", "left", "right", "full", "left_semi", "left_anti"),
        "left_keys": _column_array(minimum=1),
        "right_keys": _column_array(minimum=1),
        "right_prefix": {"type": "string"},
    }
)
SORT_SCHEMA = _closed_object({"orders": _order_array()})
WINDOW_SCHEMA = _closed_object(
    {
        "partition_by": _column_array(minimum=0),
        "order_by": _order_array(),
        "functions": {
            "type": "array",
            "items": _window_function_schema(),
            "minItems": 1,
            "maxItems": 1000,
        },
    }
)
AGGREGATE_SCHEMA = _closed_object(
    {
        "group_by": _column_array(minimum=0),
        "aggregations": {
            "type": "array",
            "items": _aggregation_schema(),
            "minItems": 1,
            "maxItems": 1000,
        },
    }
)


def spark_operator_specs() -> tuple[OperatorSpec, ...]:
    """Return every closed Spark operator without importing PySpark."""
    return (
        OperatorSpec(
            "capicove.data.project",
            1,
            frozenset({"columns", "strict"}),
            True,
            parameter_schema=_closed_object(
                {"columns": _column_array(minimum=1), "strict": {"type": "boolean"}}
            ),
        ),
        OperatorSpec(
            "capicove.data.filter",
            1,
            frozenset({"predicate"}),
            True,
            parameter_schema=_closed_object({"predicate": _predicate_schema()}),
        ),
        OperatorSpec(
            "capicove.data.drop-duplicates",
            1,
            frozenset({"columns"}),
            False,
            parameter_schema=_closed_object({"columns": _column_array(minimum=1)}),
        ),
        OperatorSpec(
            "capicove.data.drop-duplicates",
            2,
            frozenset({"columns", "order_by"}),
            True,
            parameter_schema=_closed_object(
                {
                    "columns": _column_array(minimum=1),
                    "order_by": _order_array(),
                }
            ),
        ),
        OperatorSpec(
            "capicove.data.group-count",
            1,
            frozenset({"columns", "count_column"}),
            True,
            parameter_schema=_closed_object(
                {"columns": _column_array(minimum=1), "count_column": {"type": "string"}}
            ),
        ),
        OperatorSpec(
            "capicove.agent-trace.project",
            1,
            frozenset({"columns"}),
            True,
            parameter_schema=_closed_object({"columns": _column_array(minimum=1)}),
        ),
        OperatorSpec(
            "capicove.data.join",
            1,
            JOIN_PARAMETERS,
            True,
            input_count=2,
            parameter_schema=JOIN_SCHEMA,
        ),
        OperatorSpec(
            "capicove.data.sort",
            1,
            frozenset({"orders"}),
            True,
            parameter_schema=SORT_SCHEMA,
        ),
        OperatorSpec(
            "capicove.data.window",
            1,
            WINDOW_PARAMETERS,
            True,
            parameter_schema=WINDOW_SCHEMA,
        ),
        OperatorSpec(
            "capicove.data.aggregate",
            1,
            AGGREGATE_PARAMETERS,
            True,
            parameter_schema=AGGREGATE_SCHEMA,
        ),
    )
