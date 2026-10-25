"""Tests for allowlisted Spark session configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from capicove_execution_spark.session import (
    SparkBackendConfiguration,
    _validate_effective_configuration,
    create_spark_session,
)


def test_configuration_normalizes_allowlisted_values(tmp_path: Path) -> None:
    configuration = SparkBackendConfiguration.from_mapping(
        {
            "master": "local[2]",
            "app_name": "capicove-test",
            "driver_memory": "1g",
            "executor_memory": "1g",
            "executor_cores": 1,
            "shuffle_partitions": 2,
            "local_scratch_root": str(tmp_path.resolve()),
        }
    )

    assert configuration.master == "local[2]"
    assert configuration.local_scratch_root == tmp_path.resolve()


def test_configuration_rejects_unknown_spark_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="configuration key"):
        SparkBackendConfiguration.from_mapping(
            {"master": "local[1]", "spark.jars": "/unreviewed/code.jar"}
        )


@pytest.mark.parametrize("master", ("local[*]", "local[0]", "local[9]", "spark://host"))
def test_configuration_rejects_unbounded_or_remote_master(master: str) -> None:
    with pytest.raises(ValueError, match="master"):
        SparkBackendConfiguration.from_mapping({"master": master})


def test_configuration_rejects_comma_separated_scratch_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="comma"):
        SparkBackendConfiguration(local_scratch_root=tmp_path / "first,second")


def test_configuration_rejects_a_scratch_root_through_a_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        SparkBackendConfiguration(local_scratch_root=linked / "spark")


def test_session_rechecks_scratch_root_before_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    configuration = SparkBackendConfiguration(local_scratch_root=scratch)
    outside = tmp_path / "outside"
    outside.mkdir()
    scratch.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        "capicove_execution_spark.plugin.approved_java_executable",
        lambda: Path("/approved/java"),
    )

    with pytest.raises(ValueError, match="symbolic link"):
        create_spark_session(configuration)


@pytest.mark.parametrize(
    "name",
    (
        "CLASSPATH",
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "PYSPARK_GATEWAY_PORT",
        "PYSPARK_SUBMIT_ARGS",
        "HADOOP_CONF_DIR",
        "SPARK_DIST_CLASSPATH",
        "SPARK_CONF_DIR",
        "SPARK_HOME",
        "SPARK_LOCAL_IP",
        "YARN_CONF_DIR",
    ),
)
def test_session_rejects_spark_submit_environment_injection(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "attacker-value")

    with pytest.raises(ValueError, match="environment"):
        create_spark_session(SparkBackendConfiguration())


@pytest.mark.parametrize(
    "key",
    (
        "spark.jars.packages",
        "spark.executorEnv.CREDENTIAL",
        "spark.sql.extensions",
        "spark.driver.extraClassPath",
        "spark.submit.pyFiles",
    ),
)
def test_effective_configuration_rejects_runtime_injection(key: str) -> None:
    with pytest.raises(ValueError, match="effective Spark configuration"):
        _validate_effective_configuration({"spark.master": "local[1]", key: "attacker-value"})


def test_effective_configuration_accepts_only_exact_loopback_driver_networking() -> None:
    _validate_effective_configuration(
        {
            "spark.driver.bindAddress": "127.0.0.1",
            "spark.driver.host": "127.0.0.1",
            "spark.master": "local[1]",
        }
    )

    for key in ("spark.driver.host", "spark.driver.bindAddress"):
        configuration = {
            "spark.driver.bindAddress": "127.0.0.1",
            "spark.driver.host": "127.0.0.1",
            "spark.master": "local[1]",
        }
        configuration[key] = "0.0.0.0"
        with pytest.raises(ValueError, match=key):
            _validate_effective_configuration(configuration)


@pytest.mark.parametrize(
    "key",
    ("spark.driver.extraJavaOptions", "spark.executor.extraJavaOptions"),
)
def test_effective_configuration_rejects_unreviewed_java_options(key: str) -> None:
    with pytest.raises(ValueError, match="effective Spark configuration"):
        _validate_effective_configuration({key: "-javaagent:/tmp/attacker.jar"})


def test_session_rejects_a_non_java_17_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    java = tmp_path / "bin" / "java"
    java.parent.mkdir()
    java.write_text("not a Java runtime", encoding="utf-8")
    java.chmod(0o700)
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))

    with pytest.raises(ValueError, match="Java 17"):
        create_spark_session(SparkBackendConfiguration())


def test_session_rejects_an_existing_spark_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyspark import SparkContext

    monkeypatch.setattr(
        "capicove_execution_spark.plugin.approved_java_executable",
        lambda: Path("/approved/java"),
    )
    monkeypatch.setattr(SparkContext, "_active_spark_context", object())

    with pytest.raises(RuntimeError, match="existing Spark context"):
        create_spark_session(SparkBackendConfiguration())
