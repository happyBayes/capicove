"""Embedding dependency direction and optional runtime distribution contract."""

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


def imports(path: Path) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, path
            result.add(node.module or "")
    return result


@pytest.mark.slow
def test_graph_bridge_only_consumes_public_retrieval_service() -> None:
    bridge = Path("src/capicove/graph_build/_adapters/retrieval_embedding.py")
    assert "capicove.retrieval" in imports(bridge)
    for path in Path("src/capicove/graph_build").rglob("*.py"):
        assert not any(name.startswith("capicove.retrieval._") for name in imports(path)), path
        assert not any(
            name.startswith("capicove.retrieval.")
            and not name.startswith("capicove.retrieval.contracts")
            for name in imports(path)
        ), path
    for path in Path("src/capicove/retrieval").rglob("*.py"):
        assert not any(name.startswith("capicove.graph_build") for name in imports(path)), path


def test_graph_ports_stay_dependency_light() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.graph_build.ports; import sys; assert not {'numpy', 'sentence_transformers', 'torch', 'transformers', 'capicove.retrieval'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_text_embedding_extra_declares_compatible_optional_runtime() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    optional = {
        requirement.name: requirement
        for raw in project["optional-dependencies"]["text-embedding"]
        if (requirement := Requirement(raw))
    }
    assert {"sentence-transformers", "torch", "transformers"} <= optional.keys()
    assert "5.6.0" in optional["sentence-transformers"].specifier
    assert "5.2.3" not in optional["sentence-transformers"].specifier
    assert "5.7.0" not in optional["sentence-transformers"].specifier
    assert "5.15.0" in optional["transformers"].specifier
    assert "6.0.0" not in optional["transformers"].specifier
    assert not {"sentence-transformers", "torch", "transformers"} & {
        Requirement(raw).name for raw in project["dependencies"]
    }
