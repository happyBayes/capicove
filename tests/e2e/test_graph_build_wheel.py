"""Verify graph contracts and resources ship without eager optional runtimes."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = pytest.mark.e2e


@pytest.mark.slow
def test_graph_build_wheel_resources_and_lazy_base_import(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    wheel = wheel_cache.wheel()

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode("utf-8")

    assert {
        "capicove/graph_build/NETWORKX_LICENSE.txt",
        "capicove/graph_build/schemas/diagnostics-v1.json",
        "capicove/graph_build/schemas/extraction-v1.json",
        "capicove/graph_build/schemas/report-v1.json",
    } <= names
    for extra in ("graph-community", "graph-data", "graph-models", "text-embedding"):
        assert f"Provides-Extra: {extra}" in metadata
    assert not any(name.startswith(("tests/", "reference/")) for name in names)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(wheel)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import capicove.graph_build as graph; "
                "assert graph.GraphBuildService; "
                "blocked = {'networkx', 'pyarrow', 'sentence_transformers', "
                "'torch', 'transformers'}; "
                "loaded = {name.split('.', 1)[0] for name in sys.modules}; "
                "assert not blocked & loaded, sorted(blocked & loaded)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert imported.returncode == 0, imported.stderr
