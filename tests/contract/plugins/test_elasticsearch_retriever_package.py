"""Packaging contract for the Elasticsearch retriever plugin."""

from __future__ import annotations

import tomllib
from pathlib import Path


PLUGIN_ROOT = Path("plugins/retriever-elasticsearch")


def test_elasticsearch_retriever_distribution_metadata() -> None:
    configuration = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert configuration["project"]["name"] == "capicove-retriever-elasticsearch"
    assert configuration["project"]["requires-python"] == ">=3.11"
    assert "capicove>=0.1.0,<0.2" in configuration["project"]["dependencies"]
    assert "elasticsearch>=8.19,<9" in configuration["project"]["dependencies"]
    assert configuration["project"]["entry-points"]["capicove.retrievers"] == {
        "elasticsearch": ("capicove_retriever_elasticsearch:elasticsearch_retriever_plugin")
    }


def test_core_distribution_does_not_depend_on_elasticsearch() -> None:
    configuration = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = configuration["project"]["dependencies"]

    assert all(not dependency.startswith("elasticsearch") for dependency in dependencies)
