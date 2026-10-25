"""Distribution contract for the independent FAISS retriever plugin."""

import tomllib
from pathlib import Path


ROOT = Path(__file__).parents[3]


def test_faiss_distribution_declares_v3_retriever_entry_point() -> None:
    configuration = tomllib.loads(
        (ROOT / "plugins/retriever-faiss/pyproject.toml").read_text(encoding="utf-8")
    )
    project = configuration["project"]

    assert project["name"] == "capicove-retriever-faiss"
    assert project["version"] == "0.1.0"
    assert project["entry-points"]["capicove.retrievers"] == {
        "faiss": "capicove_retriever_faiss:faiss_retriever_plugin"
    }
    assert "faiss-cpu>=1.9,<2" in project["dependencies"]
    assert "numpy>=1.26,<3" in project["dependencies"]


def test_wheel_configuration_includes_only_plugin_source_package() -> None:
    configuration = tomllib.loads(
        (ROOT / "plugins/retriever-faiss/pyproject.toml").read_text(encoding="utf-8")
    )

    assert configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/capicove_retriever_faiss"
    ]
