"""Dependency checks including relative-import resolution."""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path


def imports(source: str, package: str) -> set[str]:
    result: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = "." * node.level + (node.module or "")
            resolved = importlib.util.resolve_name(name, package) if node.level else name
            result.add(resolved)
            result.update(resolved + "." + alias.name for alias in node.names)
    return result


def test_scanner_detects_relative_and_absolute_imports() -> None:
    result = imports(
        "from .. import runtime\nimport torch\nfrom ._adapters import sqlite_index", "capicove.wiki"
    )
    assert "capicove.runtime" in result
    assert "capicove.wiki._adapters.sqlite_index" in result
    assert "torch" in result


def test_wiki_boundaries() -> None:
    violations = []
    for capability in ("wiki", "books"):
        for path in (Path("src/capicove") / capability).rglob("*.py"):
            package = ".".join(path.parent.relative_to("src").parts)
            for target in imports(path.read_text(), package):
                forbidden = (
                    ("capicove.runtime", "torch", "transformers", "openai")
                    if capability == "wiki"
                    else ("capicove.wiki._adapters",)
                )
                if target.startswith(forbidden):
                    violations.append((str(path), target))
    assert not violations


def test_public_maintenance_import_is_dependency_light() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.wiki.maintenance; import sys; assert not {'torch', 'transformers', 'openai', 'yaml'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
