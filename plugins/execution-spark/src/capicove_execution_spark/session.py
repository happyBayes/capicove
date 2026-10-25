"""Allowlisted configuration and lazy construction of local Spark sessions."""

from __future__ import annotations

import os
import re
import shlex
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_KEYS = frozenset(
    {
        "master",
        "app_name",
        "driver_memory",
        "executor_memory",
        "executor_cores",
        "shuffle_partitions",
        "local_scratch_root",
    }
)
_MEMORY = re.compile(r"^[1-9][0-9]*[mg]$")
_DENIED_EFFECTIVE_KEYS = frozenset(
    {
        "spark.archives",
        "spark.files",
        "spark.jars",
        "spark.jars.packages",
        "spark.driver.extraClassPath",
        "spark.driver.userClassPathFirst",
        "spark.executor.extraClassPath",
        "spark.executor.userClassPathFirst",
        "spark.extraListeners",
        "spark.plugins",
        "spark.sql.extensions",
        "spark.submit.pyFiles",
    }
)
_DENIED_EFFECTIVE_PREFIXES = (
    "spark.executorEnv.",
    "spark.kubernetes.driverEnv.",
    "spark.yarn.appMasterEnv.",
)
_DENIED_ENVIRONMENT = (
    "CLASSPATH",
    "HADOOP_CONF_DIR",
    "JAVA_TOOL_OPTIONS",
    "JDK_JAVA_OPTIONS",
    "PYSPARK_DRIVER_PYTHON",
    "PYSPARK_GATEWAY_PORT",
    "PYSPARK_GATEWAY_SECRET",
    "PYSPARK_PYTHON",
    "PYSPARK_SUBMIT_ARGS",
    "SPARK_DIST_CLASSPATH",
    "SPARK_CONF_DIR",
    "SPARK_HOME",
    "SPARK_LOCAL_IP",
    "SPARK_SUBMIT_OPTS",
    "YARN_CONF_DIR",
    "_JAVA_OPTIONS",
)
_JAVA_OPTION_KEYS = frozenset({"spark.driver.extraJavaOptions", "spark.executor.extraJavaOptions"})
_SAFE_JAVA_OPTIONS = frozenset(
    {
        "--add-modules=jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED",
        "--sun-misc-unsafe-memory-access=allow",
        "-Dio.netty.allocator.type=pooled",
        "-Dio.netty.handler.ssl.defaultEndpointVerificationAlgorithm=NONE",
        "-Dio.netty.tryReflectionSetAccessible=true",
        "-Djava.net.preferIPv6Addresses=false",
        "-XX:+IgnoreUnrecognizedVMOptions",
    }
)
_SAFE_JAVA_OPEN = re.compile(
    r"--add-opens=(?:java|jdk)\.[A-Za-z0-9.]+/[A-Za-z0-9.]+(?:=ALL-UNNAMED)?"
)
_ALLOWED_EFFECTIVE_KEYS = frozenset(
    {
        "spark.app.id",
        "spark.app.name",
        "spark.app.startTime",
        "spark.app.submitTime",
        "spark.driver.extraJavaOptions",
        "spark.driver.bindAddress",
        "spark.driver.host",
        "spark.driver.memory",
        "spark.driver.port",
        "spark.executor.cores",
        "spark.executor.extraJavaOptions",
        "spark.executor.id",
        "spark.executor.memory",
        "spark.hadoop.fs.s3a.vectored.read.max.merged.size",
        "spark.hadoop.fs.s3a.vectored.read.min.seek.size",
        "spark.local.dir",
        "spark.master",
        "spark.pyspark.driver.python",
        "spark.pyspark.python",
        "spark.rdd.compress",
        "spark.serializer.objectStreamReset",
        "spark.sql.artifact.isolation.enabled",
        "spark.sql.session.timeZone",
        "spark.sql.shuffle.partitions",
        "spark.submit.deployMode",
        "spark.ui.enabled",
        "spark.ui.showConsoleProgress",
    }
)


@dataclass(frozen=True, slots=True)
class SparkBackendConfiguration:
    """Contain only reviewed local Spark settings."""

    master: str = "local[1]"
    app_name: str = "capicove-spark"
    driver_memory: str = "1g"
    executor_memory: str = "1g"
    executor_cores: int = 1
    shuffle_partitions: int = 2
    local_scratch_root: Path = Path(tempfile.gettempdir()).resolve() / "capicove-spark"

    def __post_init__(self) -> None:
        match = re.fullmatch(r"local\[([1-8])\]", self.master)
        if match is None:
            raise ValueError("master must be local[N] with 1 <= N <= 8")
        if not self.app_name.strip() or len(self.app_name) > 100:
            raise ValueError("app_name must be non-blank and bounded")
        if _MEMORY.fullmatch(self.driver_memory) is None:
            raise ValueError("driver_memory must use a positive m or g suffix")
        if _MEMORY.fullmatch(self.executor_memory) is None:
            raise ValueError("executor_memory must use a positive m or g suffix")
        _require_positive("executor_cores", self.executor_cores, maximum=64)
        _require_positive("shuffle_partitions", self.shuffle_partitions, maximum=10000)
        if (
            not isinstance(self.local_scratch_root, Path)
            or not self.local_scratch_root.is_absolute()
        ):
            raise ValueError("local_scratch_root must be an absolute Path")
        if "," in os.fspath(self.local_scratch_root):
            raise ValueError("local_scratch_root must not contain a comma")
        _reject_symlink_components(self.local_scratch_root)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SparkBackendConfiguration:
        """Validate and normalize a closed configuration mapping."""
        unknown = set(value) - _KEYS
        if unknown:
            raise ValueError(f"unsupported Spark configuration key: {sorted(unknown)[0]}")
        normalized = dict(value)
        if "local_scratch_root" in normalized:
            normalized["local_scratch_root"] = Path(
                _string(normalized["local_scratch_root"], "local_scratch_root")
            )
        return cls(**normalized)  # type: ignore[arg-type]


def create_spark_session(configuration: SparkBackendConfiguration) -> Any:
    """Create one owned local Spark session after explicit backend selection."""
    if any(os.environ.get(name, "").strip() for name in _DENIED_ENVIRONMENT):
        raise ValueError("Spark submit environment injection is not allowed")
    from capicove_execution_spark.plugin import approved_java_executable

    if approved_java_executable() is None:
        raise ValueError("Spark execution requires an approved Java 17 runtime")
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession

    if getattr(SparkContext, "_active_spark_context", None) is not None:
        raise RuntimeError("an existing Spark context cannot be reused")

    _ensure_private_directory(configuration.local_scratch_root)
    spark_configuration = (
        SparkConf(loadDefaults=False)
        .setMaster(configuration.master)
        .setAppName(configuration.app_name)
        .set("spark.driver.host", "127.0.0.1")
        .set("spark.driver.bindAddress", "127.0.0.1")
        .set("spark.driver.memory", configuration.driver_memory)
        .set("spark.executor.memory", configuration.executor_memory)
        .set("spark.executor.cores", str(configuration.executor_cores))
        .set("spark.sql.shuffle.partitions", str(configuration.shuffle_partitions))
        .set("spark.sql.session.timeZone", "UTC")
        .set("spark.local.dir", str(configuration.local_scratch_root))
        .set("spark.ui.enabled", "false")
        .set("spark.pyspark.python", sys.executable)
        .set("spark.pyspark.driver.python", sys.executable)
    )
    session = SparkSession.builder.config(conf=spark_configuration).getOrCreate()
    try:
        _validate_effective_configuration(dict(session.sparkContext.getConf().getAll()))
    except Exception:
        session.stop()
        raise
    session.sparkContext.pythonExec = sys.executable
    return session


def _validate_effective_configuration(configuration: Mapping[str, str]) -> None:
    for key, value in sorted(configuration.items()):
        denied = key in _DENIED_EFFECTIVE_KEYS or any(
            key.startswith(prefix) for prefix in _DENIED_EFFECTIVE_PREFIXES
        )
        if value.strip() and (denied or key not in _ALLOWED_EFFECTIVE_KEYS):
            raise ValueError(f"effective Spark configuration is forbidden: {key}")
        if key in _JAVA_OPTION_KEYS:
            _validate_java_options(key, value)
    for key in ("spark.driver.host", "spark.driver.bindAddress"):
        if configuration.get(key) != "127.0.0.1":
            raise ValueError(f"effective Spark configuration must use loopback: {key}")


def _ensure_private_directory(path: Path) -> None:
    """Create one local directory without traversing symbolic-link components."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("local_scratch_root must not contain a symbolic link")
        if not stat.S_ISDIR(mode):
            raise ValueError("local_scratch_root components must be directories")


def _reject_symlink_components(path: Path) -> None:
    """Reject existing symbolic-link components without creating the path."""
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("local_scratch_root must not contain a symbolic link")


def _validate_java_options(key: str, value: str) -> None:
    try:
        options = shlex.split(value)
    except ValueError as error:
        raise ValueError(f"effective Spark configuration is forbidden: {key}") from error
    if any(
        option not in _SAFE_JAVA_OPTIONS and not _SAFE_JAVA_OPEN.fullmatch(option)
        for option in options
    ):
        raise ValueError(f"effective Spark configuration is forbidden: {key}")


def _require_positive(field_name: str, value: int, *, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{field_name} must be an integer from 1 to {maximum}")


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value
