"""Spark plugin metadata, availability, and explicit backend construction."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess  # nosec B404
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from capicove.plugins import PluginAvailability, PluginMetadata
from capicove.processing.models import OperatorSpec
from capicove.processing.ports import ExecutionBackend
from capicove_execution_spark.operator_contracts import spark_operator_specs


@dataclass(frozen=True, slots=True)
class SparkExecutionBackendPlugin:
    """Construct the Spark backend only after explicit selection."""

    @property
    def metadata(self) -> PluginMetadata:
        """Return static offline plugin metadata."""
        return PluginMetadata(
            plugin_id="capicove.execution.spark",
            plugin_version="0.1.0",
            capability="execution_backend",
            contract_min=1,
            contract_max=1,
            optional_packages=("pyspark",),
            external_binaries=("java",),
            network_access=False,
        )

    def availability(self) -> PluginAvailability:
        """Check PySpark and Java without creating a Spark session."""
        if importlib.util.find_spec("pyspark") is None:
            return PluginAvailability(False, "install pyspark>=4.1,<4.3")
        if approved_java_executable() is None:
            return PluginAvailability(False, "install Java 17 and add java to PATH")
        return PluginAvailability(True)

    def operator_specs(self) -> tuple[OperatorSpec, ...]:
        """Return closed operator metadata without importing PySpark."""
        return spark_operator_specs()

    def create(self, configuration: Mapping[str, object]) -> ExecutionBackend:
        """Create one explicitly configured Spark execution backend."""
        from capicove_execution_spark.backend import SparkExecutionBackend
        from capicove_execution_spark.session import SparkBackendConfiguration

        return SparkExecutionBackend(SparkBackendConfiguration.from_mapping(configuration))


def spark_execution_backend_plugin() -> SparkExecutionBackendPlugin:
    """Return the dependency-light Spark execution backend plugin."""
    return SparkExecutionBackendPlugin()


def approved_java_executable() -> Path | None:
    """Return the currently selected executable only when it is Java 17."""
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        home = Path(java_home)
        if not home.is_absolute():
            return None
        java = home / "bin" / "java"
        if not java.is_file() or not os.access(java, os.X_OK):
            return None
    else:
        discovered = shutil.which("java")
        if discovered is None:
            return None
        java = Path(discovered)
    try:
        # Arguments are fixed and no shell is involved.
        completed = subprocess.run(  # nosec B603
            (str(java.resolve()), "-version"),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    version_text = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r'version "(?:1\.)?([0-9]+)', version_text)
    if completed.returncode != 0 or match is None or int(match.group(1)) != 17:
        return None
    return java.resolve()
