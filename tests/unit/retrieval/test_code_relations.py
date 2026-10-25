"""S4 bounded relation facts against independently hand-labelled edges (RFC-0033 REL-01)."""

from __future__ import annotations

import pytest

from capicove.retrieval.code_relations import (
    MAX_EDGES,
    RELATIONS_POLICY_VERSION,
    Relation,
    RelationBudgetError,
    extract_relations,
)


pytest.importorskip("tree_sitter", reason="S4 requires the optional code-ast extra")

FILES = {
    "pkg/util.py": ("def helper(x):\n    return x\n\ndef shared():\n    return 1\n"),
    "pkg/other.py": "def shared():\n    return 2\n",
    "pkg/app.py": (
        "import os\n"
        "from pkg.util import helper\n"
        "from pkg import util as tools\n"
        "\n"
        "class Service:\n"
        "    def run(self, value):\n"
        "        local = helper(value)\n"
        "        tools.helper(local)\n"
        "        os.path.join('a', 'b')\n"
        "        self.finish()\n"
        "        dynamic = getattr(self, 'x')\n"
        "        return shared()\n"
        "\n"
        "    def finish(self):\n"
        "        return None\n"
        "\n"
        "def helper_shadow():\n"
        "    helper = lambda v: v\n"
        "    return helper(3)\n"
    ),
    "web/api.ts": "export function fetchUser(id: string) { return id; }\n",
    "web/view.ts": (
        "import { fetchUser } from './api';\n"
        "import { format } from 'date-fns';\n"
        "export function render() {\n"
        "  fetchUser('1');\n"
        "  return format(new Date(), 'yyyy');\n"
        "}\n"
    ),
}


def _edges() -> set[tuple[str, str, str, str]]:
    relations = extract_relations({path: (lang, text) for path, (lang, text) in _inputs().items()})
    return {
        (r.kind, f"{r.source_path}:{r.source_symbol}", r.target or "", r.status) for r in relations
    }


def _inputs() -> dict[str, tuple[str, str]]:
    return {
        path: ("typescript" if path.endswith(".ts") else "python", text)
        for path, text in FILES.items()
    }


def test_containment_imports_and_unambiguous_calls_match_labels() -> None:
    edges = _edges()
    expected = {
        ("contains", "pkg/app.py:Service", "pkg/app.py:Service.run", "resolved"),
        ("contains", "pkg/app.py:Service", "pkg/app.py:Service.finish", "resolved"),
        ("imports", "pkg/app.py:", "pkg/util.py", "resolved"),
        ("imports", "pkg/app.py:", "os", "external"),
        ("imports", "web/view.ts:", "web/api.ts", "resolved"),
        ("imports", "web/view.ts:", "date-fns", "external"),
        ("calls", "pkg/app.py:Service.run", "pkg/util.py:helper", "resolved"),
        ("calls", "pkg/app.py:Service.run", "pkg/util.py:helper", "resolved"),
        ("calls", "pkg/app.py:Service.run", "pkg/app.py:Service.finish", "resolved"),
        ("calls", "web/view.ts:render", "web/api.ts:fetchUser", "resolved"),
    }
    assert expected <= edges


def test_ambiguous_dynamic_external_and_shadowed_targets_stay_unresolved() -> None:
    edges = _edges()
    assert ("calls", "pkg/app.py:Service.run", "shared", "ambiguous") in edges
    assert ("calls", "pkg/app.py:Service.run", "getattr", "external") in edges
    assert ("calls", "pkg/app.py:Service.run", "os.path.join", "external") in edges
    assert ("calls", "web/view.ts:render", "format", "external") in edges
    assert ("calls", "pkg/app.py:helper_shadow", "helper", "unresolved") in edges
    assert not any(
        edge[0] == "calls" and edge[1] == "pkg/app.py:helper_shadow" and edge[3] == "resolved"
        for edge in edges
    )


def test_relation_records_carry_site_hash_method_and_policy() -> None:
    relations = extract_relations(_inputs())
    call = next(
        r for r in relations if r.kind == "calls" and r.target == "pkg/app.py:Service.finish"
    )
    assert isinstance(call, Relation)
    assert call.source_line == 10 and call.method == "tree-sitter-v1"
    assert call.policy == RELATIONS_POLICY_VERSION == "code-relations-v1"
    text = FILES["pkg/app.py"]
    assert text[call.start_char : call.end_char] == "self.finish()"
    assert len(call.source_sha256) == 64


def test_edge_budget_overflow_fails_instead_of_dropping_facts() -> None:
    many = "".join(f"def f{i}():\n    return f{i + 1}()\n" for i in range(50))
    with pytest.raises(RelationBudgetError):
        extract_relations({"big.py": ("python", many)}, max_edges=10)
    assert MAX_EDGES == 100_000
