"""S4 ``code-relations-v1``: one-hop expansion, filters, budgets and provenance (RFC-0033)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.retrieval import (
    SearchFilters,
    SearchRequest,
    TargetType,
    TextQuery,
)
from capicove.runtime import code_retrieval as runtime
from tests.integration.retrieval.test_code_hybrid import _hybrid
from tests.integration.retrieval.test_code_lexical import _context, _git


GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(os.name != "posix" or GIT is None, reason="requires POSIX and Git")
pytest.importorskip("tree_sitter", reason="S4 requires the optional code-ast extra")


def _relations_host(tmp_path: Path, **overrides: object) -> tuple[Any, Any]:
    host, models = _hybrid(tmp_path, code_policy_ref="code-relations-v1", **overrides)
    repo = Path(host.capabilities["retrieval"]["source_roots"]["repo"])
    (repo / "src" / "app" / "orchestrator.py").write_text(
        "from src.app.resilience import call_again_later\n"
        "\n"
        "def nightly_job(task):\n"
        "    return call_again_later(task)\n",
        encoding="utf-8",
    )
    # Fillers outrank resilience.py densely for this query, so only expansion lifts it.
    for index in range(6):
        (repo / "src" / "app" / f"filler_{index}.py").write_text(
            f"def filler_{index}():\n    return {index}\n", encoding="utf-8"
        )
    _git(repo, "add", "src")
    return host, models


def _metadata(service: Any, hit: Any) -> dict[str, Any]:
    return dict(service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"])


def _search(service: Any, query: str, top_k: int = 5, **filters: Any) -> Any:
    return service.search(
        SearchRequest(
            TextQuery(query),
            frozenset({TargetType.TEXT_CHUNK}),
            top_k=top_k,
            retrieval_profile_ref="local-default",
            filters=SearchFilters(**filters),
        )
    )


def test_one_hop_expansion_lifts_called_definition_after_seeds(tmp_path: Path) -> None:
    host, models = _relations_host(tmp_path)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    built = runtime.build_index(host, models=models)
    assert built["relations"]["resolved"] >= 1
    service = runtime.open_code_retrieval(host, models=models)
    try:
        result = _search(service, "nightly_job", top_k=6)
        paths = [_metadata(service, hit)["relative_path"] for hit in result.hits]
        assert paths[0] == "src/app/orchestrator.py"
        assert paths[5] == "src/app/resilience.py"
        seed = _metadata(service, result.hits[0])
        assert any(
            edge["kind"] == "calls"
            and edge["target"] == "src/app/resilience.py:call_again_later"
            and edge["status"] == "resolved"
            for edge in seed["relations"]
        )
    finally:
        service.close()
    plain, _ = _hybrid(tmp_path)
    runtime.build_index(plain, models=models)
    baseline = runtime.open_code_retrieval(plain, models=models)
    try:
        hits = _search(baseline, "nightly_job", top_k=6).hits
        assert "src/app/resilience.py" not in [
            _metadata(baseline, hit)["relative_path"] for hit in hits
        ]
    finally:
        baseline.close()


def test_filters_apply_to_expansion_targets_and_topk_is_respected(tmp_path: Path) -> None:
    host, models = _relations_host(tmp_path)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host, models=models)
    service = runtime.open_code_retrieval(host, models=models)
    try:
        only_ts = _search(service, "nightly_job", media_types=frozenset({"text/x-typescript"}))
        assert all(_metadata(service, hit)["relative_path"].endswith(".ts") for hit in only_ts.hits)
        one = _search(service, "nightly_job", top_k=1)
        assert len(one.hits) == 1
        assert _metadata(service, one.hits[0])["relative_path"] == "src/app/orchestrator.py"
        result = _search(service, "nightly_job", top_k=10)
        assert sum(len(hit.snippet or "") for hit in result.hits) <= 16 * 1024
    finally:
        service.close()


def test_relations_policy_requires_hybrid_profile_and_identity_changes(tmp_path: Path) -> None:
    host, models = _relations_host(tmp_path)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    with_relations = runtime.build_index(host, models=models)["generation_id"]
    hybrid, _ = _hybrid(tmp_path)
    without = runtime.build_index(hybrid, models=models)["generation_id"]
    assert with_relations != without
    from tests.integration.retrieval.test_code_lexical import _host

    lexical_repo = Path(host.capabilities["retrieval"]["source_roots"]["repo"])
    with pytest.raises(runtime.CodeRetrievalError):
        runtime.code_settings(
            _host(host.workspace_root, lexical_repo, code_policy_ref="code-relations-v1")
        )


def test_edits_recompute_relations_and_unresolved_targets_add_no_hits(tmp_path: Path) -> None:
    host, models = _relations_host(tmp_path)
    repo = Path(host.capabilities["retrieval"]["source_roots"]["repo"])
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host, models=models)
    (repo / "src" / "app" / "orchestrator.py").write_text(
        "def nightly_job(task):\n    return getattr(task, 'run')()\n", encoding="utf-8"
    )
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host, models=models)
    service = runtime.open_code_retrieval(host, models=models)
    try:
        result = _search(service, "nightly_job", top_k=2)
        seed = _metadata(service, result.hits[0])
        assert seed["relative_path"] == "src/app/orchestrator.py"
        assert all(
            edge["status"] != "resolved" or edge["kind"] != "calls" for edge in seed["relations"]
        )
    finally:
        service.close()
