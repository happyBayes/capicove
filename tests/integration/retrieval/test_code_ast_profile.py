"""S2 ``code-ast-v1`` inside the lexical code profile: structure, fallback, identity."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from capicove.runtime import code_retrieval as runtime
from tests.integration.retrieval.test_code_lexical import (
    _context,
    _git,
    _host,
    _repository,
    _search,
)


GIT = shutil.which("git")
pytestmark = [
    pytest.mark.skipif(os.name != "posix" or GIT is None, reason="requires POSIX and Git"),
]
pytest.importorskip("tree_sitter", reason="S2 requires the optional code-ast extra")


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = (tmp_path / "w").resolve()
    root.mkdir()
    repo = _repository(root)
    (repo / "src" / "app" / "server.go").write_text("func Serve() {}\n", encoding="utf-8")
    (repo / "src" / "app" / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    _git(repo, "add", "src/app/server.go", "src/app/broken.py")
    return root, repo


def test_ast_definitions_have_scope_signature_and_method(tmp_path: Path) -> None:
    root, repo = _workspace(tmp_path)
    host = _host(root, repo, code_policy_ref="code-ast-v1")
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host)
    service = runtime.open_code_retrieval(host)
    hit = _search(service, "HttpClient.fetch_user_profile", top_k=1).hits[0]
    meta = service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"]
    assert meta["relative_path"] == "src/app/client.py"
    assert meta["parse_status"] == "parsed" and meta["structure"] == "ast"
    definition = next(
        d for d in meta["definitions"] if d["qualified_name"] == "HttpClient.fetch_user_profile"
    )
    assert definition["method"] == "tree-sitter-v1"
    assert definition["signature"] == "def fetch_user_profile(self, user_id)"
    ts_hit = _search(service, "renderProfile", top_k=1).hits[0]
    ts_meta = service.read_evidence(_context(), ts_hit.evidence_id, ("metadata",))["metadata"]
    assert ts_meta["relative_path"] == "src/app/views.ts"
    assert any(
        d["name"] == "renderProfile" and d["method"] == "tree-sitter-v1"
        for d in ts_meta["definitions"]
    )


def test_missing_grammar_and_syntax_errors_fall_back_visibly(tmp_path: Path) -> None:
    root, repo = _workspace(tmp_path)
    host = _host(root, repo, code_policy_ref="code-ast-v1")
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    built = runtime.build_index(host)
    assert built["parse_outcomes"] == {
        "fallback:missing_grammar": 1,
        "fallback:syntax_error": 1,
        "not_applicable": 1,
        "parsed": 3,
    }
    service = runtime.open_code_retrieval(host)
    go = _search(service, "Serve", top_k=1).hits[0]
    meta = service.read_evidence(_context(), go.evidence_id, ("metadata",))["metadata"]
    assert meta["relative_path"] == "src/app/server.go"
    assert (meta["parse_status"], meta["parse_reason"], meta["structure"]) == (
        "fallback",
        "missing_grammar",
        "line-fallback",
    )
    assert all(d["method"] == "lexical-heuristic-v1" for d in meta.get("definitions", ()))


def test_policy_and_parser_identity_change_generation_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, repo = _workspace(tmp_path)
    assert GIT
    lexical = _host(root, repo)
    runtime.import_repository(lexical, "repo", git=GIT)
    s1 = runtime.build_index(lexical)["generation_id"]
    ast_host = _host(root, repo, code_policy_ref="code-ast-v1")
    s2 = runtime.build_index(ast_host)["generation_id"]
    assert s1 != s2
    from capicove.retrieval import code_ast

    original = code_ast.parser_identity()
    monkeypatch.setattr(
        runtime, "parser_identity", lambda: {**original, "tree_sitter": "0.0.0-different"}
    )
    assert runtime.build_index(ast_host)["generation_id"] != s2


def test_ast_policy_requires_the_lexical_code_profile(tmp_path: Path) -> None:
    root, repo = _workspace(tmp_path)
    with pytest.raises(runtime.CodeRetrievalError):
        runtime.code_settings(_host(root, repo, code_policy_ref="code-relations-v1"))
    assert (
        runtime.code_settings(_host(root, repo, code_policy_ref="code-ast-v1")).code_policy
        == "code-ast-v1"
    )
