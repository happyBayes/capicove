"""Graph slow and qualification checks have independent explicit opt-ins."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("arguments", "summary"),
    [
        ([], "3 passed, 5 deselected"),
        (["--run-graph-slow"], "7 passed, 1 deselected"),
        (["--run-graph-slow", "-m", "graph_slow"], "4 passed, 4 deselected"),
        (["--run-qualification", "-m", "qualification"], "1 passed, 7 deselected"),
    ],
)
def test_graph_slow_checks_require_opt_in_and_remain_selectable(
    tmp_path: Path, arguments: list[str], summary: str
) -> None:
    repository_tests = Path(__file__).parents[1]
    tests = tmp_path / "tests"
    tests.mkdir()
    for filename in ("conftest.py", "deadline_fixtures.py"):
        (tests / filename).write_text(
            (repository_tests / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests plugins\nmarkers =\n"
        "    slow: Slow tests\n    graph_slow: Manual graph checks\n    qualification: Capacity checks\n",
        encoding="utf-8",
    )
    modules = {
        "qualification/test_capacity.py": (
            "@pytest.mark.qualification\ndef test_capacity():\n    pass\n"
        ),
        "integration/graph_build/test_graph.py": (
            "def test_fast_graph():\n    pass\n\n"
            "@pytest.mark.slow\ndef test_slow_graph():\n    pass\n"
        ),
        "unit/graph_build/test_graph.py": (
            "@pytest.mark.slow\ndef test_slow_graph_unit():\n    pass\n"
        ),
        "e2e/test_document_graph_pipeline.py": (
            "@pytest.mark.slow\ndef test_slow_graph_pipeline():\n    pass\n"
        ),
        "e2e/test_graph_build_wheel.py": (
            "@pytest.mark.slow\ndef test_slow_graph_wheel():\n    pass\n"
        ),
        "integration/other/test_other.py": (
            "@pytest.mark.slow\ndef test_unrelated_slow():\n    pass\n"
        ),
    }
    for relative_path, source in modules.items():
        module = tests / relative_path
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("import pytest\n\n" + source, encoding="utf-8")
        parent = module.parent
        while parent != tmp_path:
            (parent / "__init__.py").touch()
            parent = parent.parent
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "test_plugin.py").write_text("def test_plugin():\n    pass\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--no-cov", "-q", *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert summary in result.stdout, result.stdout
