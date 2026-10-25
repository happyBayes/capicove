"""Tests for the lazy Spark execution plugin package."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from capicove_execution_spark import spark_execution_backend_plugin
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from capicove.plugins import PluginAvailability


PLUGIN_SRC = Path(__file__).parents[2] / "src"


def test_plugin_operator_catalog_does_not_import_pyspark() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PLUGIN_SRC), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from capicove_execution_spark import spark_execution_backend_plugin; "
                "spark_execution_backend_plugin().operator_specs(); "
                "import sys; assert 'pyspark' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_plugin_declares_offline_execution_contract() -> None:
    plugin = spark_execution_backend_plugin()

    assert plugin.metadata.plugin_id == "capicove.execution.spark"
    assert plugin.metadata.network_access is False
    assert plugin.metadata.optional_packages == ("pyspark",)
    assert isinstance(plugin.availability(), PluginAvailability)


def test_plugin_operator_catalog_is_dependency_light_and_closed() -> None:
    plugin = spark_execution_backend_plugin()

    specifications = {spec.operator_id: spec for spec in plugin.operator_specs()}
    join = specifications["capicove.data.join"]

    assert join.input_count == 2
    for specification in specifications.values():
        assert specification.parameter_schema["type"] == "object"
        properties = cast(Mapping[str, object], specification.parameter_schema["properties"])
        required = cast(tuple[str, ...], specification.parameter_schema["required"])
        assert set(properties) == specification.parameter_names
        assert set(required) == specification.parameter_names
        assert specification.parameter_schema["additionalProperties"] is False


def test_drop_duplicates_versions_expose_honest_determinism_contracts() -> None:
    specifications = {
        (spec.operator_id, spec.operator_version): spec
        for spec in spark_execution_backend_plugin().operator_specs()
    }

    legacy = specifications[("capicove.data.drop-duplicates", 1)]
    stable = specifications[("capicove.data.drop-duplicates", 2)]

    assert legacy.deterministic is False
    assert stable.deterministic is True
    assert stable.parameter_names == frozenset({"columns", "order_by"})


def test_window_schema_accepts_an_integer_lag_default() -> None:
    specifications = {
        spec.operator_id: spec for spec in spark_execution_backend_plugin().operator_specs()
    }
    schema = specifications["capicove.data.window"].parameter_schema
    instance = {
        "partition_by": [],
        "order_by": [{"column": "sequence", "direction": "asc", "nulls": "last"}],
        "functions": [
            {
                "function": "lag",
                "output_column": "previous_sequence",
                "input_column": "sequence",
                "offset": 1,
                "default": 0,
            }
        ],
    }

    assert Draft202012Validator(schema).is_valid(instance)


def test_filter_schema_is_closed_and_represents_reviewed_predicates() -> None:
    specifications = {
        spec.operator_id: spec for spec in spark_execution_backend_plugin().operator_specs()
    }
    schema = cast(Mapping[str, Any], specifications["capicove.data.filter"].parameter_schema)
    properties = cast(Mapping[str, Any], schema["properties"])
    predicate_schema = cast(Mapping[str, Any], properties["predicate"])
    validator = Draft202012Validator(schema)
    membership_schema = predicate_schema["oneOf"][2]["properties"]["value"]

    assert all(branch["additionalProperties"] is False for branch in predicate_schema["oneOf"])
    assert membership_schema["maxItems"] == 1000
    assert validator.is_valid({"predicate": {"column": "score", "operator": "is_null"}})
    assert validator.is_valid({"predicate": {"column": "score", "operator": "gte", "value": 1}})
    assert validator.is_valid({"predicate": {"column": "score", "operator": "in", "value": [1, 2]}})
    assert validator.is_valid(
        {"predicate": {"column": "score", "operator": "in", "value": list(range(1000))}}
    )
    assert not validator.is_valid(
        {
            "predicate": {
                "column": "score",
                "operator": "in",
                "value": list(range(1001)),
            }
        }
    )
    assert not validator.is_valid(
        {"predicate": {"column": "score", "operator": "is_null", "value": None}}
    )
    assert not validator.is_valid({"predicate": {"column": "score", "operator": "gte"}})
    assert not validator.is_valid(
        {"predicate": {"column": "score", "operator": "shell", "value": "x"}}
    )
    assert not validator.is_valid(
        {
            "predicate": {
                "column": "score",
                "operator": "gte",
                "value": {"shell": "x"},
            }
        }
    )


def test_plugin_rejects_java_older_than_17(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "capicove_execution_spark.plugin.importlib.util.find_spec", lambda _: object()
    )
    monkeypatch.setattr("capicove_execution_spark.plugin.shutil.which", lambda _: "/usr/bin/java")
    monkeypatch.setattr(
        "capicove_execution_spark.plugin.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            "",
            'openjdk version "11.0.24"',
        ),
    )

    availability = spark_execution_backend_plugin().availability()

    assert not availability.available
    assert availability.detail is not None and "Java 17" in availability.detail
