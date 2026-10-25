"""Protect graph evidence imports from optional runtimes and private adapters."""

import ast
import subprocess
import sys
from pathlib import Path


def test_graph_imports_do_not_load_optional_runtimes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.contracts.graph_build; "
            "import capicove.graph_build; import capicove.graph_build.communities; "
            "import capicove.graph_build.profiles; import sys; "
            "assert not {'torch', 'transformers', 'igraph', 'leidenalg', 'networkx', 'pyarrow', "
            "'sqlite3', 'numpy', 'pyspark'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_graph_contracts_only_depend_on_stdlib_and_canonical_contracts() -> None:
    paths = list(Path("src/capicove/contracts/graph_build").rglob("*.py"))
    assert paths
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, path
                modules = [node.module or ""]
            for module in modules:
                assert module.split(".")[0] in sys.stdlib_module_names or module.startswith(
                    "capicove.contracts."
                ), (path, module)
