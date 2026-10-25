"""Tests for the closed logical Spark operator compiler."""

from __future__ import annotations

import pytest
from capicove_execution_spark.compiler import (
    SparkPlanCompiler,
    UnsupportedSparkOperatorError,
    _list,
    spark_operator_specs,
)
from capicove_execution_spark.operator_contracts import spark_operator_specs as contract_specs
from pyspark.sql.types import LongType, StringType, StructField, StructType

from capicove.processing.models import LogicalNode


class _RecordingFrame:
    def __init__(
        self,
        operations: list[tuple[str, tuple[object, ...]]] | None = None,
        columns: list[str] | None = None,
    ) -> None:
        self.operations = [] if operations is None else operations
        self.columns = ["episode_id", "record_type"] if columns is None else columns
        self.schema = StructType(
            [
                StructField(
                    column,
                    LongType() if column == "episode_id" else StringType(),
                    nullable=True,
                )
                for column in self.columns
            ]
        )

    def __getitem__(self, name: str) -> _RecordingColumn:
        exact_name = name[1:-1].replace("``", "`") if name.startswith("`") else name
        return _RecordingColumn(exact_name)

    def select(self, *columns: object) -> _RecordingFrame:
        return _RecordingFrame([*self.operations, ("select", columns)], self.columns)

    def dropDuplicates(self, columns: list[str]) -> _RecordingFrame:
        return _RecordingFrame([*self.operations, ("dropDuplicates", tuple(columns))], self.columns)

    def groupBy(self, *columns: str) -> _RecordingGroup:
        return _RecordingGroup(self.operations, columns)

    def withColumnRenamed(self, source: str, destination: str) -> _RecordingFrame:
        columns = [destination if column == source else column for column in self.columns]
        return _RecordingFrame(
            [*self.operations, ("withColumnRenamed", (source, destination))],
            columns,
        )

    def filter(self, expression: object) -> _RecordingFrame:
        return _RecordingFrame([*self.operations, ("filter", (expression,))], self.columns)


class _RecordingGroup:
    def __init__(
        self,
        operations: list[tuple[str, tuple[object, ...]]],
        columns: tuple[str, ...],
    ) -> None:
        self._operations = operations
        self._columns = columns

    def count(self) -> _RecordingFrame:
        return _RecordingFrame(
            [*self._operations, ("groupBy", self._columns), ("count", ())],
            [*self._columns, "count"],
        )


class _RecordingColumn(str):
    def __new__(cls, value: str = "") -> _RecordingColumn:
        return super().__new__(cls, value)

    def isin(self, values: list[object]) -> tuple[str, tuple[object, ...]]:
        return "isin", tuple(values)


def _node(
    node_id: str,
    operator_id: str,
    input_binding: str,
    output_binding: str,
    parameters: dict[str, object],
) -> LogicalNode:
    return LogicalNode(
        node_id=node_id,
        operator_id=operator_id,
        operator_version=1,
        input_bindings=(input_binding,),
        output_binding=output_binding,
        parameters=parameters,
    )


def _join_node(parameters: dict[str, object]) -> LogicalNode:
    return LogicalNode(
        node_id="join",
        operator_id="capicove.data.join",
        operator_version=1,
        input_bindings=("left", "right"),
        output_binding="joined",
        parameters=parameters,
    )


def _sort_node(parameters: dict[str, object]) -> LogicalNode:
    return _node("sort", "capicove.data.sort", "source", "sorted", parameters)


def _window_node(parameters: dict[str, object]) -> LogicalNode:
    return _node("window", "capicove.data.window", "source", "windowed", parameters)


def _aggregate_node(parameters: dict[str, object]) -> LogicalNode:
    return _node("aggregate", "capicove.data.aggregate", "source", "aggregated", parameters)


def test_compiler_applies_nodes_in_topological_order() -> None:
    nodes = (
        _node(
            "project",
            "capicove.data.project",
            "source",
            "projected",
            {"columns": ["episode_id", "record_type"], "strict": True},
        ),
        _node(
            "deduplicate",
            "capicove.data.drop-duplicates",
            "projected",
            "result",
            {"columns": ["episode_id", "record_type"]},
        ),
    )

    result = SparkPlanCompiler().compile_nodes(nodes, {"source": _RecordingFrame()})

    assert result.operations == [
        ("select", ("episode_id", "record_type")),
        ("dropDuplicates", ("episode_id", "record_type")),
    ]


def test_compiler_rejects_unregistered_operator() -> None:
    node = _node("shell", "capicove.data.shell", "source", "result", {})

    with pytest.raises(UnsupportedSparkOperatorError, match=r"capicove\.data\.shell"):
        SparkPlanCompiler().compile_nodes((node,), {"source": _RecordingFrame()})


def test_project_rejects_missing_strict_column() -> None:
    node = _node(
        "project",
        "capicove.data.project",
        "source",
        "result",
        {"columns": ["missing"], "strict": True},
    )

    with pytest.raises(ValueError, match="missing"):
        SparkPlanCompiler().compile_nodes((node,), {"source": _RecordingFrame()})


def test_operator_catalog_does_not_advertise_unrepresentable_explode() -> None:
    operator_ids = {spec.operator_id for spec in spark_operator_specs()}

    assert "capicove.data.explode" not in operator_ids


def test_compiler_reexports_dependency_light_operator_catalog() -> None:
    assert spark_operator_specs is contract_specs


def test_list_validator_enforces_optional_maximum() -> None:
    values = list(range(1000))

    assert _list(values, "predicate.value", max_items=1000) == values
    with pytest.raises(ValueError, match=r"predicate\.value must be a non-empty array"):
        _list([*values, 1000], "predicate.value", max_items=1000)


@pytest.mark.parametrize(
    "operator_id",
    (
        "capicove.data.project",
        "capicove.data.drop-duplicates",
        "capicove.data.group-count",
        "capicove.agent-trace.project",
    ),
)
def test_unary_column_arrays_enforce_the_published_1000_item_bound(
    operator_id: str,
) -> None:
    columns = [f"column_{index}" for index in range(1001)]
    parameters: dict[str, object] = {"columns": columns[:1000]}
    if operator_id == "capicove.data.project":
        parameters["strict"] = True
    elif operator_id == "capicove.data.group-count":
        parameters["count_column"] = "records"
    accepted = _node("accepted", operator_id, "source", "accepted_output", parameters)
    oversized_parameters = dict(parameters)
    oversized_parameters["columns"] = columns
    oversized = _node(
        "oversized",
        operator_id,
        "source",
        "oversized_output",
        oversized_parameters,
    )
    frame = _RecordingFrame(columns=columns)

    SparkPlanCompiler().compile_nodes((accepted,), {"source": frame})
    with pytest.raises(ValueError, match=r"columns.*1000"):
        SparkPlanCompiler().compile_nodes((oversized,), {"source": frame})


def test_filter_in_accepts_1000_json_scalars_and_rejects_nested_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accepted_values: list[object] = [None, True, 1, 1.5, "value", *range(995)]
    accepted = _node(
        "accepted",
        "capicove.data.filter",
        "source",
        "accepted_output",
        {
            "predicate": {
                "column": "record_type",
                "operator": "in",
                "value": accepted_values,
            }
        },
    )
    nested = _node(
        "nested",
        "capicove.data.filter",
        "source",
        "nested_output",
        {
            "predicate": {
                "column": "record_type",
                "operator": "in",
                "value": ["value", {"nested": "object"}],
            }
        },
    )
    monkeypatch.setattr("pyspark.sql.functions.col", lambda _: _RecordingColumn())
    frame = _RecordingFrame()

    SparkPlanCompiler().compile_nodes((accepted,), {"source": frame})
    with pytest.raises(ValueError, match="finite JSON scalar or null"):
        SparkPlanCompiler().compile_nodes((nested,), {"source": frame})
    with pytest.raises(ValueError, match="finite JSON scalar or null"):
        _list([float("inf")], "predicate.value", max_items=1000)


def test_compiler_passes_declared_join_inputs_to_advanced_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = _RecordingFrame()
    right = _RecordingFrame()
    node = _join_node(
        {
            "how": "inner",
            "left_keys": ["episode_id"],
            "right_keys": ["episode_id"],
            "right_prefix": "right_",
        }
    )
    received: list[tuple[object, ...]] = []

    def compile_advanced_node(_: LogicalNode, frames: tuple[object, ...]) -> _RecordingFrame:
        received.append(frames)
        return left

    monkeypatch.setattr(
        "capicove_execution_spark.compiler.compile_advanced_node", compile_advanced_node
    )

    result = SparkPlanCompiler().compile_nodes((node,), {"left": left, "right": right})

    assert result is left
    assert received == [(left, right)]


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        (
            {
                "how": "inner",
                "left_keys": ["episode_id"],
                "right_keys": ["episode_id", "record_type"],
                "right_prefix": "right_",
            },
            "equal length",
        ),
        (
            {
                "how": "inner",
                "left_keys": ["episode_id", "episode_id"],
                "right_keys": ["episode_id", "record_type"],
                "right_prefix": "right_",
            },
            "unique",
        ),
        (
            {
                "how": "cross",
                "left_keys": ["episode_id"],
                "right_keys": ["episode_id"],
                "right_prefix": "right_",
            },
            "how",
        ),
        (
            {
                "how": "inner",
                "left_keys": ["episode_id"],
                "right_keys": ["episode_id"],
                "right_prefix": " ",
            },
            "right_prefix",
        ),
        (
            {
                "how": "inner",
                "left_keys": ["episode_id"],
                "right_keys": ["episode_id"],
                "right_prefix": "not-valid-",
            },
            "right_prefix",
        ),
    ),
)
def test_join_rejects_invalid_closed_parameters(
    parameters: dict[str, object], message: str
) -> None:
    from capicove_execution_spark.advanced_operators import compile_advanced_node

    with pytest.raises(ValueError, match=message):
        compile_advanced_node(
            _join_node(parameters),
            (_RecordingFrame(), _RecordingFrame()),
        )


def test_join_rejects_missing_key_and_prefixed_output_collision() -> None:
    from capicove_execution_spark.advanced_operators import compile_advanced_node

    missing_key = _join_node(
        {
            "how": "inner",
            "left_keys": ["missing"],
            "right_keys": ["episode_id"],
            "right_prefix": "right_",
        }
    )
    collision = _join_node(
        {
            "how": "inner",
            "left_keys": ["episode_id"],
            "right_keys": ["episode_id"],
            "right_prefix": "right_",
        }
    )

    with pytest.raises(ValueError, match="missing"):
        compile_advanced_node(missing_key, (_RecordingFrame(), _RecordingFrame()))
    with pytest.raises(ValueError, match="collision"):
        compile_advanced_node(
            collision,
            (
                _RecordingFrame(columns=["episode_id", "right_record_type"]),
                _RecordingFrame(columns=["episode_id", "record_type"]),
            ),
        )


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        ({"orders": []}, "orders"),
        (
            {
                "orders": [
                    {"column": "episode_id", "direction": "asc", "nulls": "first"},
                    {"column": "episode_id", "direction": "desc", "nulls": "last"},
                ]
            },
            "unique",
        ),
        (
            {"orders": [{"column": "episode_id", "direction": "sideways", "nulls": "first"}]},
            "direction",
        ),
        (
            {"orders": [{"column": "episode_id", "direction": "asc", "nulls": "middle"}]},
            "nulls",
        ),
        (
            {
                "orders": [
                    {
                        "column": "episode_id",
                        "direction": "asc",
                        "nulls": "first",
                        "extra": True,
                    }
                ]
            },
            "exactly",
        ),
        (
            {"orders": [{"column": "missing", "direction": "asc", "nulls": "first"}]},
            "missing",
        ),
        (
            {
                "orders": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "extra": True,
            },
            "exactly",
        ),
    ),
)
def test_sort_rejects_invalid_closed_orders(parameters: dict[str, object], message: str) -> None:
    from capicove_execution_spark.advanced_operators import compile_advanced_node

    with pytest.raises(ValueError, match=message):
        compile_advanced_node(_sort_node(parameters), (_RecordingFrame(),))


@pytest.mark.parametrize(
    "parameters",
    (
        {
            "partition_by": [],
            "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
            "functions": [
                {"function": "row_number", "output_column": "row_number"},
                {"function": "rank", "output_column": "rank"},
                {"function": "dense_rank", "output_column": "dense_rank"},
                {
                    "function": "lag",
                    "output_column": "previous_id",
                    "input_column": "episode_id",
                    "offset": 1,
                    "default": None,
                },
                {
                    "function": "lead",
                    "output_column": "next_id",
                    "input_column": "episode_id",
                    "offset": 10000,
                    "default": 0,
                },
            ],
        },
    ),
)
def test_window_accepts_every_supported_function(parameters: dict[str, object]) -> None:
    from capicove_execution_spark.advanced_operators import _window_parameters

    _, _, functions = _window_parameters(parameters, _RecordingFrame())

    assert [function["function"] for function in functions] == [
        "row_number",
        "rank",
        "dense_rank",
        "lag",
        "lead",
    ]


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        (
            {"partition_by": [], "order_by": [], "functions": []},
            "order_by",
        ),
        (
            {
                "partition_by": ["episode_id", "episode_id"],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "row_number", "output_column": "row"}],
            },
            "partition_by",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "missing", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "row_number", "output_column": "row"}],
            },
            "missing",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {"function": "rank", "output_column": "same"},
                    {"function": "dense_rank", "output_column": "same"},
                ],
            },
            "unique",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "row_number", "output_column": "row", "offset": 1}],
            },
            "exactly",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "percent_rank", "output_column": "rank"}],
            },
            "unsupported",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous",
                        "input_column": "episode_id",
                        "offset": True,
                        "default": None,
                    }
                ],
            },
            "offset",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lead",
                        "output_column": "next",
                        "input_column": "episode_id",
                        "offset": 1,
                        "default": [],
                    }
                ],
            },
            "default",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous",
                        "input_column": "episode_id",
                        "offset": 0,
                        "default": None,
                    }
                ],
            },
            "offset",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lead",
                        "output_column": "next",
                        "input_column": "episode_id",
                        "offset": 10001,
                        "default": None,
                    }
                ],
            },
            "offset",
        ),
        (
            {
                "partition_by": [],
                "order_by": [{"column": "episode_id", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "rank", "output_column": "rank"}],
                "extra": True,
            },
            "exactly",
        ),
    ),
)
def test_window_rejects_invalid_closed_parameters(
    parameters: dict[str, object], message: str
) -> None:
    from capicove_execution_spark.advanced_operators import compile_advanced_node

    with pytest.raises(ValueError, match=message):
        compile_advanced_node(_window_node(parameters), (_RecordingFrame(),))


@pytest.mark.parametrize(
    "parameters",
    (
        {
            "group_by": [],
            "aggregations": [
                {"function": "count", "output_column": "rows"},
                {"function": "count", "input_column": "episode_id", "output_column": "ids"},
                {
                    "function": "count_distinct",
                    "input_column": "episode_id",
                    "output_column": "unique_ids",
                },
                {"function": "min", "input_column": "episode_id", "output_column": "first_id"},
                {"function": "max", "input_column": "episode_id", "output_column": "last_id"},
            ],
        },
    ),
)
def test_aggregate_accepts_grouped_and_global_count_variants(parameters: dict[str, object]) -> None:
    from capicove_execution_spark.advanced_operators import _aggregate_parameters

    group_by, aggregations = _aggregate_parameters(parameters, _RecordingFrame())

    assert group_by == ()
    assert [aggregation["function"] for aggregation in aggregations] == [
        "count",
        "count",
        "count_distinct",
        "min",
        "max",
    ]


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        ({"group_by": [], "aggregations": []}, "aggregations"),
        (
            {
                "group_by": ["episode_id"],
                "aggregations": [{"function": "count", "output_column": "episode_id"}],
            },
            "collide",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {"function": "count", "output_column": "same"},
                    {"function": "count", "output_column": "same"},
                ],
            },
            "unique",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {"function": "sum", "input_column": "missing", "output_column": "total"}
                ],
            },
            "missing",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {"function": "median", "input_column": "episode_id", "output_column": "x"}
                ],
            },
            "unsupported",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "count",
                        "input_column": "episode_id",
                        "max_items": 1,
                        "output_column": "rows",
                    }
                ],
            },
            "exactly",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_list",
                        "input_column": "episode_id",
                        "max_items": 1,
                        "output_column": "items",
                    }
                ],
            },
            "exactly",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_set",
                        "input_column": "episode_id",
                        "max_items": True,
                        "output_column": "items",
                    }
                ],
            },
            "max_items",
        ),
        (
            {
                "group_by": [],
                "aggregations": [{"function": "count", "output_column": "rows"}],
                "extra": True,
            },
            "exactly",
        ),
        (
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_set",
                        "input_column": "episode_id",
                        "max_items": 10001,
                        "output_column": "items",
                    }
                ],
            },
            "max_items",
        ),
    ),
)
def test_aggregate_rejects_invalid_closed_parameters(
    parameters: dict[str, object], message: str
) -> None:
    from capicove_execution_spark.advanced_operators import compile_advanced_node

    with pytest.raises(ValueError, match=message):
        compile_advanced_node(_aggregate_node(parameters), (_RecordingFrame(),))


def test_orderable_scalar_type_classifier_accepts_only_canonical_scalars() -> None:
    from capicove_execution_spark.advanced_operators import _is_orderable_scalar_type
    from pyspark.sql.types import (
        ArrayType,
        BinaryType,
        BooleanType,
        DateType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    assert _is_orderable_scalar_type(BooleanType())
    assert _is_orderable_scalar_type(LongType())
    assert _is_orderable_scalar_type(StringType())
    assert _is_orderable_scalar_type(BinaryType())
    assert _is_orderable_scalar_type(TimestampType())
    assert not _is_orderable_scalar_type(ArrayType(StringType()))
    assert not _is_orderable_scalar_type(MapType(StringType(), StringType()))
    assert not _is_orderable_scalar_type(StructType([StructField("value", StringType())]))
    assert not _is_orderable_scalar_type(DateType())


@pytest.mark.parametrize(
    ("data_type", "accepted", "rejected"),
    (
        ("ByteType", (-128, 127), (-129, 128)),
        ("ShortType", (-32768, 32767), (-32769, 32768)),
        ("IntegerType", (-2147483648, 2147483647), (-2147483649, 2147483648)),
        (
            "LongType",
            (-(2**63), 2**63 - 1),
            (-(2**63) - 1, 2**63),
        ),
    ),
)
def test_window_default_compatibility_enforces_integral_spark_bounds(
    data_type: str, accepted: tuple[int, int], rejected: tuple[int, int]
) -> None:
    import pyspark.sql.types as spark_types
    from capicove_execution_spark.advanced_operators import _window_default_is_compatible

    spark_type = getattr(spark_types, data_type)()

    assert all(_window_default_is_compatible(spark_type, value) for value in accepted)
    assert not any(_window_default_is_compatible(spark_type, value) for value in rejected)
    assert not _window_default_is_compatible(spark_type, True)


def test_window_default_compatibility_rejects_unrepresentable_floating_values() -> None:
    from capicove_execution_spark.advanced_operators import _window_default_is_compatible
    from pyspark.sql.types import DecimalType, DoubleType, FloatType

    assert _window_default_is_compatible(DoubleType(), 1)
    assert _window_default_is_compatible(DoubleType(), 1.5)
    assert not _window_default_is_compatible(DoubleType(), 10**400)
    assert _window_default_is_compatible(FloatType(), 3.4e38)
    assert not _window_default_is_compatible(FloatType(), 3.5e38)
    assert not _window_default_is_compatible(FloatType(), True)
    assert not _window_default_is_compatible(FloatType(), "1.0")
    assert not _window_default_is_compatible(DecimalType(5, 2), 1)
