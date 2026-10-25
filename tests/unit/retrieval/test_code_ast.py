"""S2 optional Tree-sitter structure: symbols, spans, fallback and isolation (RFC-0033)."""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from capicove.retrieval.code_ast import (
    AST_POLICY_VERSION,
    ParseStatus,
    ast_chunks,
    parse_source,
    parser_identity,
)


pytest.importorskip("tree_sitter", reason="S2 requires the optional code-ast extra")

PYTHON = (
    "import os\n"
    "\n"
    "@decorator(1)\n"
    "class Planner(Base):\n"
    '    """Plan things."""\n'
    "\n"
    "    def plan(self, goal: str) -> list[str]:\n"
    "        def inner():\n"
    "            return goal\n"
    "        return [inner()]\n"
    "\n"
    "    async def run(self):\n"
    "        pass\n"
    "\n"
    "\n"
    "def helper(x, *, y=2):\n"
    "    return x + y\n"
)


def _symbols(language: str, source: str) -> dict[str, object]:
    outcome = parse_source(language, source)
    assert outcome.status is ParseStatus.PARSED, outcome.reason
    return {symbol.qualified_name: symbol for symbol in outcome.symbols}


def test_python_symbols_scopes_signatures_and_spans_match_stdlib_ast() -> None:
    symbols = _symbols("python", PYTHON)
    assert set(symbols) == {
        "Planner",
        "Planner.plan",
        "Planner.plan.inner",
        "Planner.run",
        "helper",
    }
    expected = {}
    for node in ast.walk(ast.parse(PYTHON)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            expected[node.name] = (start, node.end_lineno)
    for qualified, symbol in symbols.items():
        name = qualified.rsplit(".", 1)[-1]
        assert (symbol.start_line, symbol.end_line) == expected[name], qualified  # type: ignore[attr-defined]
        assert PYTHON[symbol.start_char : symbol.end_char].splitlines()[0].strip()  # type: ignore[attr-defined]
    assert symbols["Planner"].kind == "class"  # type: ignore[attr-defined]
    assert symbols["Planner.plan"].signature == "def plan(self, goal: str) -> list[str]"  # type: ignore[attr-defined]
    assert symbols["Planner.plan.inner"].parent == "Planner.plan"  # type: ignore[attr-defined]
    assert symbols["Planner"].decorators == ("@decorator(1)",)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("language", "source", "expected"),
    [
        (
            "javascript",
            "export function render(node) {\n  return node;\n}\nclass View {\n  draw() { return 1; }\n}\nconst parse = (a) => a;\n",
            {"render": "function", "View": "class", "View.draw": "method", "parse": "function"},
        ),
        (
            "typescript",
            "export interface Retry { max: number }\ntype H = () => void;\nexport class Client {\n  fetch(id: string): Promise<void> { return Promise.resolve(); }\n}\nenum Mode { A }\n",
            {
                "Retry": "interface",
                "H": "type",
                "Client": "class",
                "Client.fetch": "method",
                "Mode": "enum",
            },
        ),
        (
            "tsx",
            "export function Card(props: {title: string}) {\n  return <div>{props.title}</div>;\n}\n",
            {"Card": "function"},
        ),
        (
            "jsx",
            "export const App = () => <main>hi</main>;\n",
            {"App": "function"},
        ),
    ],
)
def test_javascript_typescript_jsx_tsx_symbols(
    language: str, source: str, expected: dict[str, str]
) -> None:
    symbols = _symbols(language, source)
    assert {name: symbols[name].kind for name in expected} == expected  # type: ignore[attr-defined]
    for symbol in symbols.values():
        assert source[symbol.start_char : symbol.end_char]  # type: ignore[attr-defined]


def test_utf8_bytes_map_to_exact_characters_and_lines_with_crlf() -> None:
    source = "# 中文注释\r\ndef 计算(值):\r\n    return '小宁' + 值\r\n\r\nclass Ünïcode:\r\n    pass\r\n"
    symbols = _symbols("python", source)
    compute = symbols["计算"]
    assert source[compute.start_char : compute.end_char].startswith("def 计算(值):")  # type: ignore[attr-defined]
    assert (compute.start_line, compute.end_line) == (2, 3)  # type: ignore[attr-defined]
    assert symbols["Ünïcode"].start_line == 5  # type: ignore[attr-defined]


def test_chunks_cover_every_character_and_keep_parent_identity_for_oversized_symbols() -> None:
    body = "".join(f"    value_{i} = compute({i})  # {'x' * 30}\n" for i in range(150))
    source = (
        "import json\n\nTOP = 1\n\ndef huge():\n"
        + body
        + "    return value_0\n\n# trailing comment\n"
    )
    outcome = parse_source("python", source)
    chunks = ast_chunks(source, outcome)
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start_char, chunk.end_char))
    assert covered == set(range(len(source)))
    huge = [chunk for chunk in chunks if chunk.symbol == "huge"]
    assert len(huge) >= 2
    assert all(chunk.structure == "ast" for chunk in huge)
    regions = [chunk for chunk in chunks if chunk.symbol is None]
    assert any("TOP = 1" in source[c.start_char : c.end_char] for c in regions)
    assert any("trailing comment" in source[c.start_char : c.end_char] for c in regions)


def test_missing_grammar_syntax_error_and_budget_use_recorded_line_fallback() -> None:
    missing = parse_source("go", "func main() {}\n")
    assert missing.status is ParseStatus.FALLBACK and missing.reason == "missing_grammar"
    broken = parse_source("python", "def broken(:\n    pass\n")
    assert broken.status is ParseStatus.FALLBACK and broken.reason == "syntax_error"
    budget = parse_source("python", "x = 1\n" * 10, max_bytes=8)
    assert budget.status is ParseStatus.FALLBACK and budget.reason == "budget_exceeded"
    for outcome, source in ((missing, "func main() {}\n"), (broken, "def broken(:\n    pass\n")):
        assert outcome.symbols == ()
        chunks = ast_chunks(source, outcome)
        assert all(chunk.structure == "line-fallback" for chunk in chunks)
        assert "".join(source[c.start_char : c.end_char] for c in chunks) == source


def test_parser_identity_names_policy_runtime_and_grammars() -> None:
    identity = parser_identity()
    assert identity["policy"] == AST_POLICY_VERSION == "code-ast-v1"
    assert identity["tree_sitter"] == "0.25.2"
    assert identity["grammars"] == {
        "python": "tree-sitter-python 0.25.0",
        "javascript": "tree-sitter-javascript 0.25.0",
        "typescript": "tree-sitter-typescript 0.23.2",
    }


def test_base_import_and_s1_runtime_do_not_import_tree_sitter() -> None:
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0].startswith('tree_sitter'):\n"
        "            raise ImportError('blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import capicove, capicove.runtime.code_retrieval, capicove.retrieval.code_ast\n"
        "assert not any(m.startswith('tree_sitter') for m in sys.modules)\n"
        "from capicove.retrieval.code_ast import parse_source\n"
        "print(parse_source('python', 'def f():\\n    pass\\n').reason)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "dependency_unavailable"


def test_adjacent_small_symbols_pack_without_splitting_any_symbol() -> None:
    source = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(60))
    outcome = parse_source("python", source)
    chunks = ast_chunks(source, outcome)
    assert len(chunks) < 10
    starts = {symbol.start_char for symbol in outcome.symbols}
    for symbol in outcome.symbols:
        containing = [c for c in chunks if c.start_char <= symbol.start_char < c.end_char]
        assert len(containing) == 1 and containing[0].end_char >= symbol.end_char
    assert starts
    assert "".join(source[c.start_char : c.end_char] for c in chunks) == source
