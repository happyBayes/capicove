"""Distribution contract for the independent OpenSearch retriever."""

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_opensearch_distribution_declares_independent_retriever_entry_point() -> None:
    project = tomllib.loads(
        (ROOT / "plugins/retriever-opensearch/pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert project["name"] == "capicove-retriever-opensearch"
    assert project["entry-points"]["capicove.retrievers"] == {
        "opensearch": "capicove_retriever_opensearch:opensearch_retriever_plugin"
    }
    assert "opensearch-py>=3,<4" in project["dependencies"]
