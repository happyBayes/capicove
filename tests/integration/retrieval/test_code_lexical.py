"""S1 model-free code search: intake, build, ranking and exact evidence (RFC-0033).

Fixtures are authored repositories created with the real Git executable; the
expected definitions, lines and paths are independent oracles.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import (
    SearchFilters,
    SearchRequest,
    SearchStatus,
    TargetType,
    TemporalRange,
    TextOrigin,
    TextQuery,
)
from capicove.runtime import code_retrieval as runtime
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config


GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(os.name != "posix" or GIT is None, reason="requires POSIX and Git")
PROFILE = "local-code-lexical-v1"


def _git(root: Path, *arguments: str) -> None:
    assert GIT
    subprocess.run(
        [GIT, "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(root), "GIT_CONFIG_NOSYSTEM": "1"},
    )


def _repository(root: Path) -> Path:
    repo = root / "repo"
    (repo / "src" / "app").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "app" / "client.py").write_text(
        "import json\n"
        "\n"
        "\n"
        "class HttpClient:\n"
        "    def fetch_user_profile(self, user_id):\n"
        "        return json.loads(self._get(user_id))\n"
        "\n"
        "\n"
        "def retry_with_backoff(call):\n"
        "    return call()\n",
        encoding="utf-8",
    )
    (repo / "src" / "app" / "views.ts").write_text(
        "import { HttpClient } from './client';\n"
        "\n"
        "export function renderProfile(client: HttpClient) {\n"
        "  // fetch_user_profile is called for the header\n"
        "  return client.fetch_user_profile(1);\n"
        "}\n",
        encoding="utf-8",
    )
    (repo / "src" / "app" / "textFusion.py").write_text(
        "def combine(lists):\n    return sum(lists, [])\n", encoding="utf-8"
    )
    (repo / "docs" / "guide.md").write_text(
        "# Guide\n\nThe quokka cache keeps hot entries in memory.\n", encoding="utf-8"
    )
    (repo / "ignored.py").write_text("SECRET_FLAG = 1\n", encoding="utf-8")
    (repo / ".gitignore").write_text("ignored.py\nbuild/\n", encoding="utf-8")
    # Synthetic key marker verifies that secret-bearing sources are rejected.
    (repo / "keys.py").write_text(
        "KEY = '-----BEGIN RSA PRIVATE KEY-----'\n",  # pragma: allowlist secret
        encoding="utf-8",
    )
    (repo / "package-lock.json").write_text("{}", encoding="utf-8")
    (repo / "data.bin").write_bytes(b"\x00\x01")
    (repo / "link.py").symlink_to(repo / "src" / "app" / "client.py")
    _git(repo, "init", "-q")
    _git(
        repo,
        "add",
        "src",
        "docs",
        ".gitignore",
        "keys.py",
        "package-lock.json",
        "data.bin",
        "link.py",
    )
    _git(repo, "add", "-f", "ignored.py")
    return repo


@pytest.fixture
def host(tmp_path: Path) -> WorkspaceHostConfiguration:
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    root.chmod(0o700)
    repo = _repository(root)
    return _host(root, repo)


def _host(
    root: Path, repo: Path, *, host_profile: str = "write", **overrides: object
) -> WorkspaceHostConfiguration:
    retrieval: dict[str, object] = {
        "schema_version": 2,
        "source_roots": {"repo": str(repo)},
        "index_root": str(root / "index"),
        "corpus_ref": "corpus:code",
        "retrieval_profile_ref": PROFILE,
        "max_source_bytes": 1024 * 1024,
        "model_config_path": None,
        "model_config_sha256": None,
        "model_binding": None,
        "max_units": 10_000,
        "max_vector_bytes": 0,
        "code_policy_ref": "code-symbols-v1",
    }
    retrieval.update(overrides)
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": host_profile,
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.get_evidence", "capicove.search"],
        "trace_store": None,
        "capabilities": {"retrieval": retrieval},
        "network": {"enabled": False, "source_hosts": []},
    }
    parsed = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(parsed, WorkspaceHostConfiguration)
    return parsed


def _repo(host: WorkspaceHostConfiguration) -> Path:
    retrieval = cast(dict[str, Any], host.capabilities["retrieval"])
    return Path(retrieval["source_roots"]["repo"])


def _index(host: WorkspaceHostConfiguration) -> Path:
    return Path(cast(dict[str, Any], host.capabilities["retrieval"])["index_root"])


def _context() -> InvocationContext:
    return InvocationContext(
        request_id="r",
        trace_id="t",
        workspace_id="workspace",
        actor_id="owner",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(seconds=60),
        permissions=frozenset({"evidence:read", "evidence:search"}),
    )


def _search(service: Any, query: str, top_k: int = 5, **filters: Any) -> Any:
    return service.search(
        SearchRequest(
            TextQuery(query),
            frozenset({TargetType.TEXT_CHUNK}),
            top_k=top_k,
            retrieval_profile_ref=PROFILE,
            filters=SearchFilters(**filters),
        )
    )


def _built(host: WorkspaceHostConfiguration) -> Any:
    assert GIT
    report = runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host)
    return report, runtime.open_code_retrieval(host)


def test_repository_discovery_applies_policy_and_reports_exclusions(
    host: WorkspaceHostConfiguration,
) -> None:
    report, _ = _built(host)
    assert sorted(report["imported"]) == [
        "repo:docs/guide.md",
        "repo:src/app/client.py",
        "repo:src/app/textFusion.py",
        "repo:src/app/views.ts",
    ]
    assert report["excluded"] == {
        "binary_or_unsupported": 1,
        "hidden": 1,
        "ignored": 1,
        "lockfile": 1,
        "secret_detected": 1,
        "symlink": 1,
    }
    assert len(report["receipt"]["git_sha256"]) == 64
    assert report["receipt"]["policy"] == "code-symbols-v1"


def test_exact_definition_wins_top1_over_uses_supported_by_both_lanes(
    host: WorkspaceHostConfiguration,
) -> None:
    _, service = _built(host)
    result = _search(service, "fetch_user_profile", top_k=1)
    assert result.status is SearchStatus.SUCCEEDED
    evidence = service.read_evidence(_context(), result.hits[0].evidence_id, ("metadata",))
    metadata = evidence["metadata"]
    assert metadata["relative_path"] == "src/app/client.py"
    assert {
        "name": "fetch_user_profile",
        "qualified_name": "HttpClient.fetch_user_profile",
        "kind": "function",
        "line": 5,
        "method": "lexical-heuristic-v1",
    } in metadata["definitions"]
    assert metadata["definition_method"] == "lexical-heuristic-v1"


def test_qualified_and_camel_queries_and_split_matches(host: WorkspaceHostConfiguration) -> None:
    _, service = _built(host)
    top = _search(service, "HttpClient", top_k=1)
    meta = service.read_evidence(_context(), top.hits[0].evidence_id, ("metadata",))["metadata"]
    assert meta["relative_path"] == "src/app/client.py" and meta["definition_focus_line"] == 4
    split = _search(service, "user profile fetch")
    paths = {
        service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"][
            "relative_path"
        ]
        for hit in split.hits
    }
    assert {"src/app/client.py", "src/app/views.ts"} <= paths


def test_path_only_query_finds_file_whose_body_lacks_the_term(
    host: WorkspaceHostConfiguration,
) -> None:
    _, service = _built(host)
    result = _search(service, "textFusion.py", top_k=1)
    meta = service.read_evidence(_context(), result.hits[0].evidence_id, ("metadata",))["metadata"]
    assert meta["relative_path"] == "src/app/textFusion.py"
    assert "textFusion" not in meta["text"]
    assert all(str(host.workspace_root) not in json.dumps(hit.snippet) for hit in result.hits)


def test_evidence_text_lines_and_hash_match_the_source(host: WorkspaceHostConfiguration) -> None:
    _, service = _built(host)
    hit = _search(service, "retry_with_backoff", top_k=1).hits[0]
    meta = service.read_evidence(_context(), hit.evidence_id, ("metadata", "lineage"))["metadata"]
    source = (_repo(host) / "src/app/client.py").read_text()
    assert meta["text"] == source[meta["start_char"] : meta["end_char"]]
    assert meta["start_line"] == source.count("\n", 0, meta["start_char"]) + 1
    assert meta["language"] == "python"
    assert hit.text_locator is not None and hit.text_locator.origin is TextOrigin.DOCUMENT
    assert {
        "name": "retry_with_backoff",
        "qualified_name": "retry_with_backoff",
        "kind": "function",
        "line": 9,
        "method": "lexical-heuristic-v1",
    } in meta["definitions"]


def test_filters_apply_before_ranking(host: WorkspaceHostConfiguration) -> None:
    _, service = _built(host)
    only_ts = _search(service, "fetch_user_profile", media_types=frozenset({"text/x-typescript"}))
    assert only_ts.hits
    for hit in only_ts.hits:
        meta = service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"]
        assert meta["relative_path"].endswith(".ts")
    assert _search(service, "fetch_user_profile", time_range=TemporalRange(0, 1)).hits == ()
    assert (
        _search(service, "fetch_user_profile", corpus_refs=frozenset({"corpus:notes"})).hits == ()
    )


def test_unknown_and_no_match_queries_succeed_empty(host: WorkspaceHostConfiguration) -> None:
    _, service = _built(host)
    assert _search(service, "zzqxv_nonexistent_symbol").hits == ()


def test_rebuild_is_idempotent_and_edits_create_new_generations(
    host: WorkspaceHostConfiguration,
) -> None:
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    first = runtime.build_index(host)
    again = runtime.build_index(host)
    assert again["generation_id"] == first["generation_id"]
    repo = _repo(host)
    (repo / "src/app/textFusion.py").write_text("def merge(a):\n    return a\n", encoding="utf-8")
    runtime.import_repository(host, "repo", git=GIT)
    changed = runtime.build_index(host)
    assert changed["generation_id"] != first["generation_id"]
    service = runtime.open_code_retrieval(host)
    assert _search(service, "combine").hits == ()
    assert _search(service, "merge", top_k=1).hits


def test_old_evidence_fails_after_reactivation_and_tamper_fails_closed(
    host: WorkspaceHostConfiguration,
) -> None:
    assert GIT
    _, service = _built(host)
    hit = _search(service, "retry_with_backoff", top_k=1).hits[0]
    repo = _repo(host)
    (repo / "src/app/client.py").write_text("def other():\n    pass\n", encoding="utf-8")
    runtime.import_repository(host, "repo", git=GIT)
    active = runtime.build_index(host)
    with pytest.raises(LookupError):
        service.read_evidence(_context(), hit.evidence_id, ("metadata",))
    assert _search(service, "other", top_k=1).hits
    index_root = _index(host)
    database = index_root / "generations" / active["generation_id"] / "units.sqlite3"
    database.chmod(0o600)
    with database.open("r+b") as stream:
        stream.seek(300)
        stream.write(b"\xde\xad")
    assert _search(service, "other").status is SearchStatus.FAILED


def test_single_file_import_uses_the_same_policy(host: WorkspaceHostConfiguration) -> None:
    with pytest.raises(runtime.CodeRetrievalError, match="secret_detected"):
        runtime.import_file(host, "repo", "keys.py")
    with pytest.raises(runtime.CodeRetrievalError, match="lockfile"):
        runtime.import_file(host, "repo", "package-lock.json")
    assert runtime.import_file(host, "repo", "src/app/client.py")["imported"] is True


def test_read_profile_and_bad_configuration_are_rejected(tmp_path: Path) -> None:
    root = (tmp_path / "w").resolve()
    root.mkdir()
    repo = _repository(root)
    assert GIT
    with pytest.raises(PermissionError):
        runtime.import_repository(_host(root, repo, host_profile="read"), "repo", git=GIT)
    assert not (root / "index").exists()
    for override in (
        {"max_vector_bytes": 1},
        {"model_binding": "unified"},
        {"code_policy_ref": "code-hybrid-v1"},
        {"retrieval_profile_ref": "local-default"},
        {"max_units": 0},
        {"max_units": True},
    ):
        with pytest.raises((ValueError, runtime.CodeRetrievalError)):
            runtime.code_settings(_host(root, repo, **override))


def test_repository_hooks_and_fsmonitor_never_execute(tmp_path: Path) -> None:
    root = (tmp_path / "w").resolve()
    root.mkdir()
    repo = _repository(root)
    marker = tmp_path / "executed"
    hook = repo / ".git" / "hooks" / "fsmonitor-capicove"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o700)
    _git(repo, "config", "core.fsmonitor", str(hook))
    _git(repo, "config", "core.hooksPath", str(hook.parent))
    assert GIT
    runtime.import_repository(_host(root, repo), "repo", git=GIT)
    assert not marker.exists()


def test_unit_budget_overflow_fails_without_changing_activation(tmp_path: Path) -> None:
    root = (tmp_path / "w").resolve()
    root.mkdir()
    repo = _repository(root)
    assert GIT
    roomy = _host(root, repo)
    runtime.import_repository(roomy, "repo", git=GIT)
    active = runtime.build_index(roomy)
    tight = _host(root, repo, max_units=1)
    with pytest.raises(runtime.CodeRetrievalError, match="max_units"):
        runtime.build_index(tight)
    pointer = json.loads((root / "index" / "code-active.json").read_text())
    assert pointer["generation_id"] == active["generation_id"]


def test_escaped_unicode_evidence_stays_within_tool_section_limit(tmp_path: Path) -> None:
    from capicove.tools.evidence import EvidenceToolHandler

    root = (tmp_path / "w").resolve()
    root.mkdir()
    repo = _repository(root)
    heavy = "".join(f'PATTERN_{i} = "\\\\d+\\\\s*小宁\\t\\\\"\\\\\\\\"\n' for i in range(400))
    (repo / "src" / "app" / "patterns.py").write_text(heavy, encoding="utf-8")
    _git(repo, "add", "src/app/patterns.py")
    assert GIT
    host = _host(root, repo)
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host)
    service = runtime.open_code_retrieval(host)
    handler = EvidenceToolHandler(service.reader)
    hits = _search(service, "PATTERN_10 PATTERN_390", top_k=20).hits
    assert hits
    for hit in hits:
        result = handler(_context(), {"evidence_id": hit.evidence_id, "sections": ["metadata"]})
        assert result.error is None, hit.id
        assert result.result is not None
        metadata = cast(dict[str, Any], result.result["metadata"])
        text = metadata["text"]
        assert len(json.dumps(metadata, ensure_ascii=False).encode()) <= 4096
        assert text in heavy


def test_qualified_method_definition_and_exact_path_classes(
    host: WorkspaceHostConfiguration,
) -> None:
    _, service = _built(host)
    hit = _search(service, "HttpClient.fetch_user_profile", top_k=1).hits[0]
    meta = service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"]
    assert meta["relative_path"] == "src/app/client.py"
    assert {
        "name": "fetch_user_profile",
        "qualified_name": "HttpClient.fetch_user_profile",
        "kind": "function",
        "line": 5,
        "method": "lexical-heuristic-v1",
    } in meta["definitions"]
    exact = _search(service, "views.ts", top_k=1).hits[0]
    assert (
        service.read_evidence(_context(), exact.evidence_id, ("metadata",))["metadata"][
            "relative_path"
        ]
        == "src/app/views.ts"
    )


def test_literal_highlight_markup_in_source_is_neutralized_only_in_snippets(tmp_path: Path) -> None:
    root = (tmp_path / "w").resolve()
    root.mkdir()
    repo = _repository(root)
    (repo / "src" / "app" / "markup.py").write_text(
        "def strip_markup(text):\n    return text.replace('<em>', '').replace('</EM>', '')\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/app/markup.py")
    assert GIT
    host = _host(root, repo)
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host)
    service = runtime.open_code_retrieval(host)
    result = _search(service, "strip_markup", top_k=1)
    assert result.status is SearchStatus.SUCCEEDED
    hit = result.hits[0]
    assert hit.snippet is not None and "<em>" not in hit.snippet.casefold()
    meta = service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"]
    assert "'<em>'" in meta["text"] and "'</EM>'" in meta["text"]
