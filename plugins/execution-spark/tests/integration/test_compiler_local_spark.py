"""Real local Spark tests for the closed operator compiler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from capicove_execution_spark.compiler import SparkPlanCompiler
from capicove_execution_spark.io import dataset_schema_from_spark, spark_schema
from capicove_execution_spark.session import (
    SparkBackendConfiguration,
    create_spark_session,
)

from capicove.contracts.data import (
    ComplexSchemaType,
    ComplexTypeKind,
    DatasetSchema,
    SchemaField,
)
from capicove.processing.models import LogicalNode


pytestmark = pytest.mark.spark


def _node(
    node_id: str,
    operator_id: str,
    input_binding: str,
    output_binding: str,
    parameters: dict[str, object],
    *,
    operator_version: int = 1,
) -> LogicalNode:
    return LogicalNode(
        node_id=node_id,
        operator_id=operator_id,
        operator_version=operator_version,
        input_bindings=(input_binding,),
        output_binding=output_binding,
        parameters=parameters,
    )


def _advanced_node(
    node_id: str,
    operator_id: str,
    input_bindings: tuple[str, ...],
    output_binding: str,
    parameters: dict[str, object],
) -> LogicalNode:
    return LogicalNode(
        node_id=node_id,
        operator_id=operator_id,
        operator_version=1,
        input_bindings=input_bindings,
        output_binding=output_binding,
        parameters=parameters,
    )


def test_compiler_runs_project_filter_deduplicate_and_group_count(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-compiler-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        assert spark.sparkContext.pythonExec == sys.executable
        effective_configuration = dict(spark.sparkContext.getConf().getAll())
        assert effective_configuration["spark.driver.host"] == "127.0.0.1"
        assert effective_configuration["spark.driver.bindAddress"] == "127.0.0.1"
        source = spark.createDataFrame(
            [
                ("episode-1", "tool.result", "image"),
                ("episode-1", "tool.result", "image"),
                ("episode-2", "message.observed", "text"),
            ],
            ("episode_id", "record_type", "modality"),
        )
        nodes = (
            _node(
                "filter_results",
                "capicove.data.filter",
                "source",
                "filtered",
                {
                    "predicate": {
                        "column": "record_type",
                        "operator": "eq",
                        "value": "tool.result",
                    }
                },
            ),
            _node(
                "deduplicate",
                "capicove.data.drop-duplicates",
                "filtered",
                "unique",
                {"columns": ["episode_id", "record_type", "modality"]},
            ),
            _node(
                "group_count",
                "capicove.data.group-count",
                "unique",
                "counts",
                {"columns": ["modality"], "count_column": "records"},
            ),
            _node(
                "project",
                "capicove.data.project",
                "counts",
                "result",
                {"columns": ["modality", "records"], "strict": True},
            ),
        )

        result = SparkPlanCompiler().compile_nodes(nodes, {"source": source})

        assert [row.asDict(recursive=True) for row in result.collect()] == [
            {"modality": "image", "records": 1}
        ]
    finally:
        spark.stop()


def test_compiler_runs_every_closed_filter_predicate(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-filter-predicate-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("null", None), ("one", 1), ("two", 2), ("three", 3)],
            "id string, value int",
        )
        cases = (
            ({"column": "value", "operator": "is_null"}, ["null"]),
            (
                {"column": "value", "operator": "is_not_null"},
                ["one", "three", "two"],
            ),
            ({"column": "value", "operator": "eq", "value": 2}, ["two"]),
            (
                {"column": "value", "operator": "ne", "value": 2},
                ["one", "three"],
            ),
            ({"column": "value", "operator": "lt", "value": 2}, ["one"]),
            (
                {"column": "value", "operator": "lte", "value": 2},
                ["one", "two"],
            ),
            ({"column": "value", "operator": "gt", "value": 2}, ["three"]),
            (
                {"column": "value", "operator": "gte", "value": 2},
                ["three", "two"],
            ),
            (
                {"column": "value", "operator": "in", "value": [1, 3]},
                ["one", "three"],
            ),
        )

        for predicate, expected_ids in cases:
            node = _node(
                f"filter_{predicate['operator']}",
                "capicove.data.filter",
                "source",
                "result",
                {"predicate": predicate},
            )

            result = SparkPlanCompiler().compile_nodes((node,), {"source": source})

            assert sorted(row["id"] for row in result.collect()) == expected_ids
    finally:
        spark.stop()


def test_stable_deduplicate_selects_same_survivor_after_repartition(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-stable-deduplicate-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("a", 2, "later"), ("a", 1, "winner"), ("b", 1, "only")],
            ("key", "priority", "value"),
        )
        node = _node(
            "deduplicate",
            "capicove.data.drop-duplicates",
            "source",
            "result",
            {
                "columns": ["key"],
                "order_by": [{"column": "priority", "direction": "asc", "nulls": "last"}],
            },
            operator_version=2,
        )

        observed = []
        for partitions in (2, 3, 4):
            result = SparkPlanCompiler().compile_nodes(
                (node,),
                {"source": source.repartition(partitions)},
            )
            observed.append(
                sorted(
                    (row.asDict(recursive=True) for row in result.collect()),
                    key=lambda row: row["key"],
                )
            )

        assert (
            observed
            == [
                [
                    {"key": "a", "priority": 1, "value": "winner"},
                    {"key": "b", "priority": 1, "value": "only"},
                ]
            ]
            * 3
        )
    finally:
        spark.stop()


def test_stable_deduplicate_rejects_ambiguous_ordering_ties(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-ambiguous-deduplicate-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("a", 1, "first"), ("a", 1, "second")],
            ("key", "priority", "value"),
        )
        node = _node(
            "deduplicate",
            "capicove.data.drop-duplicates",
            "source",
            "result",
            {
                "columns": ["key"],
                "order_by": [{"column": "priority", "direction": "asc", "nulls": "last"}],
            },
            operator_version=2,
        )

        with pytest.raises(ValueError, match="unique tie-break"):
            SparkPlanCompiler().compile_nodes((node,), {"source": source})
    finally:
        spark.stop()


def test_compiler_rejects_malformed_filter_predicates(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-filter-rejection-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame([(1,)], ("value",))
        cases = (
            ({"column": "value", "operator": "is_null", "value": None}, "null predicate"),
            ({"column": "value", "operator": "eq"}, "comparison predicate"),
            (
                {"column": "value", "operator": "unsupported", "value": 1},
                "unsupported",
            ),
            (["not", "an", "object"], "predicate must be an object"),
            ({"column": "value", "operator": "in", "value": 1}, "non-empty array"),
        )

        for predicate, message in cases:
            node = _node(
                "filter_invalid",
                "capicove.data.filter",
                "source",
                "result",
                {"predicate": predicate},
            )

            with pytest.raises(ValueError, match=message):
                SparkPlanCompiler().compile_nodes((node,), {"source": source})
    finally:
        spark.stop()


@pytest.mark.parametrize(
    ("how", "expected_columns", "expected_rows"),
    (
        (
            "inner",
            ["id", "left_value", "right_note"],
            [(1, "left-one", "right-one")],
        ),
        (
            "left",
            ["id", "left_value", "right_note"],
            [
                (1, "left-one", "right-one"),
                (2, "left-two", None),
                (None, "left-null", None),
            ],
        ),
        (
            "right",
            ["id", "left_value", "right_note"],
            [
                (1, "left-one", "right-one"),
                (3, None, "right-three"),
                (None, None, "right-null"),
            ],
        ),
        (
            "full",
            ["id", "left_value", "right_note"],
            [
                (1, "left-one", "right-one"),
                (2, "left-two", None),
                (3, None, "right-three"),
                (None, "left-null", None),
                (None, None, "right-null"),
            ],
        ),
        (
            "left_semi",
            ["id", "left_value"],
            [(1, "left-one")],
        ),
        (
            "left_anti",
            ["id", "left_value"],
            [(2, "left-two"), (None, "left-null")],
        ),
    ),
)
def test_compiler_runs_closed_join_types_with_null_and_right_only_keys(
    tmp_path: Path,
    how: str,
    expected_columns: list[str],
    expected_rows: list[tuple[object, ...]],
) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name=f"capicove-join-{how}-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        left = spark.createDataFrame(
            [(1, "left-one"), (2, "left-two"), (None, "left-null")],
            ("id", "left_value"),
        )
        right = spark.createDataFrame(
            [(1, "right-one"), (3, "right-three"), (None, "right-null")],
            ("right_id", "note"),
        )
        node = _advanced_node(
            "join",
            "capicove.data.join",
            ("left", "right"),
            "joined",
            {
                "how": how,
                "left_keys": ["id"],
                "right_keys": ["right_id"],
                "right_prefix": "right_",
            },
        )

        result = SparkPlanCompiler().compile_nodes((node,), {"left": left, "right": right})

        assert result.columns == expected_columns
        assert sorted(result.collect(), key=lambda row: repr(tuple(row))) == sorted(
            expected_rows, key=repr
        )
    finally:
        spark.stop()


def test_join_rejects_mismatched_canonical_logical_key_types_before_spark_join(
    tmp_path: Path,
) -> None:
    from pyspark.sql.types import LongType, StringType, StructField, StructType

    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-join-logical-key-types-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        integer_key = spark.createDataFrame(
            [(1,)], StructType([StructField("key", LongType(), False)])
        )
        string_key = spark.createDataFrame(
            [("1",)], StructType([StructField("key", StringType(), False)])
        )
        json_key = spark.createDataFrame(
            [('{"id":1}',)],
            spark_schema(DatasetSchema((SchemaField("key", "json", False),))),
        )
        node = _advanced_node(
            "join",
            "capicove.data.join",
            ("left", "right"),
            "joined",
            {
                "how": "inner",
                "left_keys": ["key"],
                "right_keys": ["key"],
                "right_prefix": "right_",
            },
        )

        with pytest.raises(ValueError, match="canonical logical types"):
            SparkPlanCompiler().compile_nodes((node,), {"left": integer_key, "right": string_key})
        with pytest.raises(ValueError, match="canonical logical types"):
            SparkPlanCompiler().compile_nodes((node,), {"left": json_key, "right": string_key})
    finally:
        spark.stop()


def test_join_accepts_identical_recursive_key_types_supported_by_equality(
    tmp_path: Path,
) -> None:
    from pyspark.sql import Row
    from pyspark.sql.types import ArrayType, LongType, StructField, StructType

    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-join-recursive-key-types-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        key_type = StructType(
            [
                StructField("identifier", LongType(), False),
                StructField("versions", ArrayType(LongType(), False), False),
            ]
        )
        schema = StructType(
            [
                StructField("key", key_type, False),
                StructField("value", LongType(), False),
            ]
        )
        left = spark.createDataFrame([(Row(identifier=1, versions=[1, 2]), 10)], schema)
        right = spark.createDataFrame([(Row(identifier=1, versions=[1, 2]), 20)], schema)
        node = _advanced_node(
            "join",
            "capicove.data.join",
            ("left", "right"),
            "joined",
            {
                "how": "inner",
                "left_keys": ["key"],
                "right_keys": ["key"],
                "right_prefix": "right_",
            },
        )

        result = SparkPlanCompiler().compile_nodes((node,), {"left": left, "right": right})

        assert [row.asDict(recursive=True) for row in result.collect()] == [
            {
                "key": {"identifier": 1, "versions": [1, 2]},
                "value": 10,
                "right_value": 20,
            }
        ]
    finally:
        spark.stop()


def test_join_rejects_identical_map_key_types_not_supported_by_equality(
    tmp_path: Path,
) -> None:
    from pyspark.sql.types import MapType, StringType, StructField, StructType

    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-join-map-key-types-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        schema = StructType([StructField("key", MapType(StringType(), StringType(), False), False)])
        left = spark.createDataFrame([({"id": "one"},)], schema)
        right = spark.createDataFrame([({"id": "one"},)], schema)
        node = _advanced_node(
            "join",
            "capicove.data.join",
            ("left", "right"),
            "joined",
            {
                "how": "inner",
                "left_keys": ["key"],
                "right_keys": ["key"],
                "right_prefix": "right_",
            },
        )

        with pytest.raises(ValueError, match="support equality"):
            SparkPlanCompiler().compile_nodes((node,), {"left": left, "right": right})
    finally:
        spark.stop()


def test_expression_outputs_keep_governed_json_logical_metadata(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-json-expression-metadata-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        left = spark.createDataFrame(
            [('{"id":1}', 1), ('{"id":2}', 2)],
            spark_schema(
                DatasetSchema(
                    (
                        SchemaField("key", "json", False),
                        SchemaField("position", "int64", False),
                    )
                )
            ),
        )
        right = spark.createDataFrame(
            [('{"id":1}',), ('{"id":2}',)],
            spark_schema(DatasetSchema((SchemaField("key", "json", False),))),
        )
        join = _advanced_node(
            "join",
            "capicove.data.join",
            ("left", "right"),
            "joined",
            {
                "how": "inner",
                "left_keys": ["key"],
                "right_keys": ["key"],
                "right_prefix": "right_",
            },
        )
        window = _advanced_node(
            "window",
            "capicove.data.window",
            ("joined",),
            "windowed",
            {
                "partition_by": [],
                "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous_key",
                        "input_column": "key",
                        "offset": 1,
                        "default": None,
                    },
                    {
                        "function": "lead",
                        "output_column": "next_key",
                        "input_column": "key",
                        "offset": 1,
                        "default": None,
                    },
                ],
            },
        )
        aggregate = _advanced_node(
            "aggregate",
            "capicove.data.aggregate",
            ("windowed",),
            "aggregated",
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_list",
                        "input_column": "key",
                        "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                        "max_items": 10,
                        "output_column": "listed_keys",
                    },
                    {
                        "function": "collect_set",
                        "input_column": "key",
                        "max_items": 10,
                        "output_column": "unique_keys",
                    },
                ],
            },
        )

        joined = SparkPlanCompiler().compile_nodes((join,), {"left": left, "right": right})
        windowed = SparkPlanCompiler().compile_nodes((window,), {"joined": joined})
        aggregated = SparkPlanCompiler().compile_nodes((aggregate,), {"windowed": windowed})

        assert dataset_schema_from_spark(joined.schema).fields[0] == SchemaField(
            "key", "json", False
        )
        assert dataset_schema_from_spark(windowed.schema).fields[-2:] == (
            SchemaField("previous_key", "json", True),
            SchemaField("next_key", "json", True),
        )
        json_array = ComplexSchemaType(
            ComplexTypeKind.ARRAY,
            element_type="json",
            contains_null=False,
        )
        assert dataset_schema_from_spark(aggregated.schema) == DatasetSchema(
            (
                SchemaField("listed_keys", json_array, False),
                SchemaField("unique_keys", json_array, False),
            ),
            schema_version=2,
        )
        assert [row.asDict(recursive=True) for row in windowed.orderBy("position").collect()] == [
            {
                "key": '{"id":1}',
                "position": 1,
                "previous_key": None,
                "next_key": '{"id":2}',
            },
            {
                "key": '{"id":2}',
                "position": 2,
                "previous_key": '{"id":1}',
                "next_key": None,
            },
        ]
        assert [row.asDict(recursive=True) for row in aggregated.collect()] == [
            {
                "listed_keys": ['{"id":1}', '{"id":2}'],
                "unique_keys": ['{"id":1}', '{"id":2}'],
            }
        ]
    finally:
        spark.stop()


def test_compiler_runs_global_sort_with_explicit_multi_column_null_policies(
    tmp_path: Path,
) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-sort-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("a", 2), ("a", None), ("b", 1), ("b", None), (None, 9)],
            ("group", "score"),
        )
        node = _advanced_node(
            "sort",
            "capicove.data.sort",
            ("source",),
            "sorted",
            {
                "orders": [
                    {"column": "group", "direction": "asc", "nulls": "last"},
                    {"column": "score", "direction": "desc", "nulls": "first"},
                ]
            },
        )

        result = SparkPlanCompiler().compile_nodes((node,), {"source": source})

        assert [tuple(row) for row in result.collect()] == [
            ("a", None),
            ("a", 2),
            ("b", None),
            ("b", 1),
            (None, 9),
        ]
    finally:
        spark.stop()


@pytest.mark.parametrize(
    ("how", "expected_rows"),
    (
        ("left_semi", [(1, "left")]),
        ("left_anti", [(2, "left")]),
    ),
)
def test_compiler_qualifies_shared_lineage_semi_and_anti_join_keys(
    tmp_path: Path,
    how: str,
    expected_rows: list[tuple[object, ...]],
) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name=f"capicove-shared-lineage-{how}-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [(1, "left"), (2, "left"), (1, "right"), (3, "right")],
            ("id", "side"),
        )
        left = source.filter(source["side"] == "left")
        right = source.filter(source["side"] == "right")
        node = _advanced_node(
            "join",
            "capicove.data.join",
            ("left", "right"),
            "joined",
            {
                "how": how,
                "left_keys": ["id"],
                "right_keys": ["id"],
                "right_prefix": "right_",
            },
        )

        result = SparkPlanCompiler().compile_nodes((node,), {"left": left, "right": right})

        assert [tuple(row) for row in result.collect()] == expected_rows
    finally:
        spark.stop()


def test_compiler_runs_closed_window_functions_and_multi_aggregates(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-window-aggregate-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [
                ("a", 2, 10, "b"),
                ("a", 1, 20, "a"),
                ("a", 3, 30, "a"),
                ("b", 2, 5, "d"),
                ("b", 1, 7, "c"),
            ],
            ("group", "position", "score", "value"),
        )
        window = _advanced_node(
            "window",
            "capicove.data.window",
            ("source",),
            "windowed",
            {
                "partition_by": ["group"],
                "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                "functions": [
                    {"function": "row_number", "output_column": "row_number"},
                    {"function": "rank", "output_column": "rank"},
                    {"function": "dense_rank", "output_column": "dense_rank"},
                    {
                        "function": "lag",
                        "output_column": "previous_value",
                        "input_column": "value",
                        "offset": 1,
                        "default": "start",
                    },
                    {
                        "function": "lead",
                        "output_column": "next_value",
                        "input_column": "value",
                        "offset": 1,
                        "default": "end",
                    },
                ],
            },
        )
        aggregate = _advanced_node(
            "aggregate",
            "capicove.data.aggregate",
            ("windowed",),
            "aggregated",
            {
                "group_by": ["group"],
                "aggregations": [
                    {"function": "count", "output_column": "rows"},
                    {"function": "count", "input_column": "score", "output_column": "scored_rows"},
                    {
                        "function": "count_distinct",
                        "input_column": "value",
                        "output_column": "unique_values",
                    },
                    {"function": "sum", "input_column": "score", "output_column": "total_score"},
                    {"function": "avg", "input_column": "score", "output_column": "average_score"},
                    {"function": "min", "input_column": "value", "output_column": "minimum_value"},
                    {"function": "max", "input_column": "value", "output_column": "maximum_value"},
                    {
                        "function": "collect_list",
                        "input_column": "value",
                        "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                        "max_items": 2,
                        "output_column": "ordered_values",
                    },
                    {
                        "function": "collect_set",
                        "input_column": "value",
                        "max_items": 2,
                        "output_column": "distinct_values",
                    },
                ],
            },
        )
        global_aggregate = _advanced_node(
            "global_aggregate",
            "capicove.data.aggregate",
            ("source",),
            "global",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "count", "output_column": "rows"},
                    {"function": "sum", "input_column": "score", "output_column": "total_score"},
                ],
            },
        )

        windowed = SparkPlanCompiler().compile_nodes((window,), {"source": source})
        result = SparkPlanCompiler().compile_nodes((aggregate,), {"windowed": windowed})
        global_result = SparkPlanCompiler().compile_nodes((global_aggregate,), {"source": source})

        assert windowed.columns[-5:] == [
            "row_number",
            "rank",
            "dense_rank",
            "previous_value",
            "next_value",
        ]
        assert [tuple(row) for row in windowed.orderBy("group", "position").collect()] == [
            ("a", 1, 20, "a", 1, 1, 1, "start", "b"),
            ("a", 2, 10, "b", 2, 2, 2, "a", "a"),
            ("a", 3, 30, "a", 3, 3, 3, "b", "end"),
            ("b", 1, 7, "c", 1, 1, 1, "start", "d"),
            ("b", 2, 5, "d", 2, 2, 2, "c", "end"),
        ]
        assert result.columns == [
            "group",
            "rows",
            "scored_rows",
            "unique_values",
            "total_score",
            "average_score",
            "minimum_value",
            "maximum_value",
            "ordered_values",
            "distinct_values",
        ]
        assert [row.asDict(recursive=True) for row in result.orderBy("group").collect()] == [
            {
                "group": "a",
                "rows": 3,
                "scored_rows": 3,
                "unique_values": 2,
                "total_score": 60,
                "average_score": 20.0,
                "minimum_value": "a",
                "maximum_value": "b",
                "ordered_values": ["a", "b"],
                "distinct_values": ["a", "b"],
            },
            {
                "group": "b",
                "rows": 2,
                "scored_rows": 2,
                "unique_values": 2,
                "total_score": 12,
                "average_score": 6.0,
                "minimum_value": "c",
                "maximum_value": "d",
                "ordered_values": ["c", "d"],
                "distinct_values": ["c", "d"],
            },
        ]
        assert [tuple(row) for row in global_result.collect()] == [(5, 72)]
    finally:
        spark.stop()


def test_compiler_rejects_non_numeric_and_non_orderable_aggregate_inputs(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-aggregate-types-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame([("not-a-number", ["a"])], ("text", "items"))
        nonnumeric = _advanced_node(
            "nonnumeric",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "sum", "input_column": "text", "output_column": "total"}
                ],
            },
        )
        nonnumeric_average = _advanced_node(
            "nonnumeric_average",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "avg", "input_column": "text", "output_column": "average"}
                ],
            },
        )
        nonorderable = _advanced_node(
            "nonorderable",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_set",
                        "input_column": "items",
                        "max_items": 1,
                        "output_column": "values",
                    }
                ],
            },
        )
        nonorderable_order_key = _advanced_node(
            "nonorderable_order_key",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_list",
                        "input_column": "text",
                        "order_by": [{"column": "items", "direction": "asc", "nulls": "first"}],
                        "max_items": 1,
                        "output_column": "values",
                    }
                ],
            },
        )

        with pytest.raises(ValueError, match="numeric"):
            SparkPlanCompiler().compile_nodes((nonnumeric,), {"source": source})
        with pytest.raises(ValueError, match="numeric"):
            SparkPlanCompiler().compile_nodes((nonnumeric_average,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((nonorderable,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((nonorderable_order_key,), {"source": source})
    finally:
        spark.stop()


def test_compiler_collect_list_honors_direction_null_policy_and_value_ties(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-collect-list-order-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [
                ("group", 1, "z", "ignored", "ignored"),
                ("group", 2, "b", "ignored", "ignored"),
                ("group", 2, "a", "ignored", "ignored"),
                ("group", None, "n", "ignored", "ignored"),
            ],
            (
                "group",
                "position",
                "value",
                "__capicove_aggregate_0_order_0",
                "__capicove_aggregate_0_value",
            ),
        )
        node = _advanced_node(
            "aggregate",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": ["group"],
                "aggregations": [
                    {
                        "function": "collect_list",
                        "input_column": "value",
                        "order_by": [{"column": "position", "direction": "desc", "nulls": "last"}],
                        "max_items": 4,
                        "output_column": "values",
                    }
                ],
            },
        )

        result = SparkPlanCompiler().compile_nodes((node,), {"source": source})

        assert [row.asDict(recursive=True) for row in result.collect()] == [
            {"group": "group", "values": ["a", "b", "z", "n"]}
        ]
    finally:
        spark.stop()


def test_collection_aggregates_limit_rows_before_materializing_arrays(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-bounded-collection-aggregate-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.range(5000).selectExpr("'group' AS group", "id AS position", "id AS value")
        node = _advanced_node(
            "aggregate",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": ["group"],
                "aggregations": [
                    {
                        "function": "collect_list",
                        "input_column": "value",
                        "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                        "max_items": 3,
                        "output_column": "first_values",
                    },
                    {
                        "function": "collect_set",
                        "input_column": "value",
                        "max_items": 3,
                        "output_column": "smallest_values",
                    },
                ],
            },
        )

        result = SparkPlanCompiler().compile_nodes((node,), {"source": source})
        rows = result.collect()
        executed_plan = result._jdf.queryExecution().executedPlan().toString()

        assert [row.asDict(recursive=True) for row in rows] == [
            {
                "group": "group",
                "first_values": [0, 1, 2],
                "smallest_values": [0, 1, 2],
            }
        ]
        assert executed_plan.count("TakeOrderedAndProject(limit=3") >= 2
        assert executed_plan.count("__capicove_collection_") >= 2
    finally:
        spark.stop()


def test_compiler_preserves_ranking_ties_and_count_null_semantics(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-window-ties-count-nulls-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("group", 1, "a"), ("group", 1, None), ("group", 2, "a"), ("group", 3, None)],
            ("group", "position", "value"),
        )
        window = _advanced_node(
            "window",
            "capicove.data.window",
            ("source",),
            "windowed",
            {
                "partition_by": ["group"],
                "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                "functions": [
                    {"function": "rank", "output_column": "rank"},
                    {"function": "dense_rank", "output_column": "dense_rank"},
                ],
            },
        )
        aggregate = _advanced_node(
            "aggregate",
            "capicove.data.aggregate",
            ("source",),
            "aggregated",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "count", "output_column": "all_rows"},
                    {
                        "function": "count",
                        "input_column": "value",
                        "output_column": "non_null_values",
                    },
                    {
                        "function": "count_distinct",
                        "input_column": "value",
                        "output_column": "distinct_values",
                    },
                ],
            },
        )

        windowed = SparkPlanCompiler().compile_nodes((window,), {"source": source})
        aggregated = SparkPlanCompiler().compile_nodes((aggregate,), {"source": source})

        ordered_rows = windowed.orderBy("position", "value").collect()
        assert [row["rank"] for row in ordered_rows] == [1, 1, 3, 4]
        assert [row["dense_rank"] for row in ordered_rows] == [1, 1, 2, 3]
        assert [tuple(row) for row in aggregated.collect()] == [(4, 2, 1)]
    finally:
        spark.stop()


def test_window_rejects_ambiguous_ties_for_order_sensitive_functions(tmp_path: Path) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-window-ambiguous-ties-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("group", 1, "first"), ("group", 1, "second")],
            ("group", "position", "value"),
        )
        node = _advanced_node(
            "window",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": ["group"],
                "order_by": [{"column": "position", "direction": "asc", "nulls": "last"}],
                "functions": [{"function": "row_number", "output_column": "ordinal"}],
            },
        )

        with pytest.raises(ValueError, match="unique ordering"):
            SparkPlanCompiler().compile_nodes((node,), {"source": source.repartition(2)})
    finally:
        spark.stop()


def test_compiler_validates_relative_defaults_and_orderable_types(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-window-default-types-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [
                (
                    1,
                    True,
                    1.5,
                    "one",
                    b"one",
                    datetime(2026, 1, 1, tzinfo=UTC),
                    {"x": "one"},
                ),
                (
                    2,
                    False,
                    2.5,
                    "two",
                    b"two",
                    datetime(2026, 1, 2, tzinfo=UTC),
                    {"x": "two"},
                ),
            ],
            ("id", "enabled", "score", "text", "payload", "observed_at", "attributes"),
        )
        valid = _advanced_node(
            "valid_defaults",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous_id",
                        "input_column": "id",
                        "offset": 1,
                        "default": 0,
                    },
                    {
                        "function": "lag",
                        "output_column": "previous_enabled",
                        "input_column": "enabled",
                        "offset": 1,
                        "default": False,
                    },
                    {
                        "function": "lag",
                        "output_column": "previous_score",
                        "input_column": "score",
                        "offset": 1,
                        "default": 0.0,
                    },
                    {
                        "function": "lag",
                        "output_column": "previous_text",
                        "input_column": "text",
                        "offset": 1,
                        "default": "start",
                    },
                    {
                        "function": "lag",
                        "output_column": "previous_time",
                        "input_column": "observed_at",
                        "offset": 1,
                        "default": "2025-12-31T00:00:00Z",
                    },
                ],
            },
        )
        invalid_long_default = _advanced_node(
            "invalid_long_default",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous_id",
                        "input_column": "id",
                        "offset": 1,
                        "default": "bad",
                    }
                ],
            },
        )
        invalid_boolean_default = _advanced_node(
            "invalid_boolean_default",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lead",
                        "output_column": "next_enabled",
                        "input_column": "enabled",
                        "offset": 1,
                        "default": 1,
                    }
                ],
            },
        )
        invalid_binary_default = _advanced_node(
            "invalid_binary_default",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous_payload",
                        "input_column": "payload",
                        "offset": 1,
                        "default": "not-bytes",
                    }
                ],
            },
        )
        invalid_huge_double_default = _advanced_node(
            "invalid_huge_double_default",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous_score",
                        "input_column": "score",
                        "offset": 1,
                        "default": 10**400,
                    }
                ],
            },
        )
        map_sort = _advanced_node(
            "map_sort",
            "capicove.data.sort",
            ("source",),
            "result",
            {"orders": [{"column": "attributes", "direction": "asc", "nulls": "first"}]},
        )
        map_window = _advanced_node(
            "map_window",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "attributes", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "rank", "output_column": "rank"}],
            },
        )
        map_aggregate = _advanced_node(
            "map_aggregate",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "min", "input_column": "attributes", "output_column": "minimum"}
                ],
            },
        )
        binary_collection = _advanced_node(
            "binary_collection",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {
                        "function": "collect_set",
                        "input_column": "payload",
                        "max_items": 1,
                        "output_column": "payloads",
                    }
                ],
            },
        )

        result = SparkPlanCompiler().compile_nodes((valid,), {"source": source})

        valid_rows = result.orderBy("id").collect()
        assert valid_rows[0]["previous_id"] == 0
        assert valid_rows[0]["previous_enabled"] is False
        assert valid_rows[0]["previous_score"] == 0.0
        assert valid_rows[0]["previous_text"] == "start"
        assert valid_rows[0]["previous_time"] < valid_rows[0]["observed_at"]
        assert valid_rows[1]["previous_id"] == valid_rows[0]["id"]
        assert valid_rows[1]["previous_enabled"] == valid_rows[0]["enabled"]
        assert valid_rows[1]["previous_score"] == valid_rows[0]["score"]
        assert valid_rows[1]["previous_text"] == valid_rows[0]["text"]
        assert valid_rows[1]["previous_time"] == valid_rows[0]["observed_at"]
        with pytest.raises(ValueError, match="incompatible"):
            SparkPlanCompiler().compile_nodes((invalid_long_default,), {"source": source})
        with pytest.raises(ValueError, match="incompatible"):
            SparkPlanCompiler().compile_nodes((invalid_boolean_default,), {"source": source})
        with pytest.raises(ValueError, match="incompatible"):
            SparkPlanCompiler().compile_nodes((invalid_binary_default,), {"source": source})
        with pytest.raises(ValueError, match="incompatible"):
            SparkPlanCompiler().compile_nodes((invalid_huge_double_default,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((map_sort,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((map_window,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((map_aggregate,), {"source": source})
        assert [
            tuple(row)
            for row in SparkPlanCompiler()
            .compile_nodes((binary_collection,), {"source": source})
            .collect()
        ] == [([b"one"],)]
    finally:
        spark.stop()


def test_compiler_treats_spark_metacharacters_as_exact_top_level_columns(
    tmp_path: Path,
) -> None:
    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-exact-column-reference-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        source = spark.createDataFrame(
            [("group", 2, "later", "discard"), ("group", 1, "first", "discard")],
            ("*", "a.b", "`tick`", "unrequested"),
        )
        nodes = (
            _node(
                "project",
                "capicove.data.project",
                "source",
                "projected",
                {"columns": ["*", "a.b", "`tick`"], "strict": True},
            ),
            _node(
                "filter",
                "capicove.data.filter",
                "projected",
                "filtered",
                {"predicate": {"column": "a.b", "operator": "gte", "value": 1}},
            ),
            _advanced_node(
                "sort",
                "capicove.data.sort",
                ("filtered",),
                "sorted",
                {"orders": [{"column": "a.b", "direction": "asc", "nulls": "last"}]},
            ),
            _advanced_node(
                "aggregate",
                "capicove.data.aggregate",
                ("sorted",),
                "result",
                {
                    "group_by": ["*"],
                    "aggregations": [
                        {
                            "function": "min",
                            "input_column": "`tick`",
                            "output_column": "first_tick",
                        }
                    ],
                },
            ),
        )

        result = SparkPlanCompiler().compile_nodes(nodes, {"source": source})

        assert [row.asDict(recursive=True) for row in result.collect()] == [
            {"*": "group", "first_tick": "first"}
        ]

        right = spark.createDataFrame([(1, "joined")], ("right.key", "right`value"))
        exact_advanced_nodes = (
            _advanced_node(
                "join",
                "capicove.data.join",
                ("source", "right"),
                "joined",
                {
                    "how": "inner",
                    "left_keys": ["a.b"],
                    "right_keys": ["right.key"],
                    "right_prefix": "right_",
                },
            ),
            _advanced_node(
                "window",
                "capicove.data.window",
                ("joined",),
                "windowed",
                {
                    "partition_by": ["*"],
                    "order_by": [{"column": "a.b", "direction": "asc", "nulls": "last"}],
                    "functions": [{"function": "row_number", "output_column": "row.number"}],
                },
            ),
            LogicalNode(
                node_id="deduplicate",
                operator_id="capicove.data.drop-duplicates",
                operator_version=2,
                input_bindings=("windowed",),
                output_binding="deduplicated",
                parameters={
                    "columns": ["*"],
                    "order_by": [
                        {
                            "column": "row.number",
                            "direction": "asc",
                            "nulls": "last",
                        }
                    ],
                },
            ),
        )

        exact_advanced = SparkPlanCompiler().compile_nodes(
            exact_advanced_nodes,
            {"source": source, "right": right},
        )

        assert [row.asDict(recursive=True) for row in exact_advanced.collect()] == [
            {
                "*": "group",
                "a.b": 1,
                "`tick`": "first",
                "unrequested": "discard",
                "right_right`value": "joined",
                "row.number": 1,
            }
        ]
    finally:
        spark.stop()


def test_compiler_accepts_recursive_orderable_types_but_rejects_nested_maps(tmp_path: Path) -> None:
    from pyspark.sql import Row
    from pyspark.sql.types import (
        ArrayType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
    )

    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-recursive-orderable-types-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        profile_type = StructType(
            [StructField("name", StringType()), StructField("score", LongType())]
        )
        nested_type = StructType([StructField("attributes", MapType(StringType(), StringType()))])
        source = spark.createDataFrame(
            [
                (1, ["b"], Row(name="b", score=2), Row(attributes={"x": "two"})),
                (2, ["a"], Row(name="a", score=1), Row(attributes={"x": "one"})),
            ],
            StructType(
                [
                    StructField("id", LongType()),
                    StructField("tags", ArrayType(StringType())),
                    StructField("profile", profile_type),
                    StructField("nested", nested_type),
                ]
            ),
        )
        sort = _advanced_node(
            "sort",
            "capicove.data.sort",
            ("source",),
            "sorted",
            {"orders": [{"column": "tags", "direction": "asc", "nulls": "first"}]},
        )
        window = _advanced_node(
            "window",
            "capicove.data.window",
            ("source",),
            "windowed",
            {
                "partition_by": [],
                "order_by": [{"column": "profile", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "rank", "output_column": "rank"}],
            },
        )
        aggregate = _advanced_node(
            "aggregate",
            "capicove.data.aggregate",
            ("source",),
            "aggregated",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "min", "input_column": "tags", "output_column": "minimum_tags"},
                    {
                        "function": "max",
                        "input_column": "profile",
                        "output_column": "maximum_profile",
                    },
                ],
            },
        )
        nested_sort = _advanced_node(
            "nested_sort",
            "capicove.data.sort",
            ("source",),
            "result",
            {"orders": [{"column": "nested", "direction": "asc", "nulls": "first"}]},
        )
        nested_window = _advanced_node(
            "nested_window",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "nested", "direction": "asc", "nulls": "first"}],
                "functions": [{"function": "rank", "output_column": "rank"}],
            },
        )
        nested_aggregate = _advanced_node(
            "nested_aggregate",
            "capicove.data.aggregate",
            ("source",),
            "result",
            {
                "group_by": [],
                "aggregations": [
                    {"function": "min", "input_column": "nested", "output_column": "minimum"}
                ],
            },
        )

        assert [
            row["id"]
            for row in SparkPlanCompiler().compile_nodes((sort,), {"source": source}).collect()
        ] == [2, 1]
        assert [
            row["rank"]
            for row in SparkPlanCompiler()
            .compile_nodes((window,), {"source": source})
            .orderBy("id")
            .collect()
        ] == [2, 1]
        assert [
            row.asDict(recursive=True)
            for row in SparkPlanCompiler().compile_nodes((aggregate,), {"source": source}).collect()
        ] == [{"minimum_tags": ["a"], "maximum_profile": {"name": "b", "score": 2}}]
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((nested_sort,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((nested_window,), {"source": source})
        with pytest.raises(ValueError, match="orderable"):
            SparkPlanCompiler().compile_nodes((nested_aggregate,), {"source": source})
    finally:
        spark.stop()


def test_compiler_rejects_float_and_decimal_relative_default_overflow_before_actions(
    tmp_path: Path,
) -> None:
    from decimal import Decimal

    from pyspark.sql.types import DecimalType, FloatType, LongType, StructField, StructType

    spark = create_spark_session(
        SparkBackendConfiguration(
            master="local[2]",
            app_name="capicove-float-decimal-defaults-test",
            local_scratch_root=(tmp_path / "spark").resolve(),
        )
    )
    try:
        float_source = spark.createDataFrame(
            [(1, 1.0)],
            StructType([StructField("id", LongType()), StructField("value", FloatType())]),
        )
        decimal_source = spark.createDataFrame(
            [(1, Decimal("1.00"))],
            StructType([StructField("id", LongType()), StructField("value", DecimalType(5, 2))]),
        )
        float_overflow = _advanced_node(
            "float_overflow",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lag",
                        "output_column": "previous_value",
                        "input_column": "value",
                        "offset": 1,
                        "default": 1e100,
                    }
                ],
            },
        )
        decimal_default = _advanced_node(
            "decimal_default",
            "capicove.data.window",
            ("source",),
            "result",
            {
                "partition_by": [],
                "order_by": [{"column": "id", "direction": "asc", "nulls": "first"}],
                "functions": [
                    {
                        "function": "lead",
                        "output_column": "next_value",
                        "input_column": "value",
                        "offset": 1,
                        "default": 1,
                    }
                ],
            },
        )

        with pytest.raises(ValueError, match="incompatible"):
            SparkPlanCompiler().compile_nodes((float_overflow,), {"source": float_source})
        with pytest.raises(ValueError, match="incompatible"):
            SparkPlanCompiler().compile_nodes((decimal_default,), {"source": decimal_source})
    finally:
        spark.stop()
