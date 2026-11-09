"""Every saved offline baseline reproduces byte for byte through the harness.

Baselines live in ``docs/development/eval/baselines``. Default-profile runs prove
that moving hardcoded settings into profiles preserved rankings; fake-reranker
and GraphRAG runs pin their deterministic mechanics. Fake-model numbers are not
model quality (real-model evaluation: not run). The frozen code-set runs index
the repository commit and take minutes, so they run only when
``CAPICOVE_EVAL_CODE_BASELINES=1``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest

from capicove.evaluation.harness import (
    RunOptions,
    load_dataset,
    run_evaluation,
    score_run,
    validate_run,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "docs" / "development" / "eval"
BASELINES = sorted((EVAL / "baselines").glob("*.run.json"))
DATASETS = {
    "personal-notes-eval-v1": EVAL / "personal-notes-eval-v1.json",
    "unified-mixed-eval-v1": EVAL / "unified-mixed-eval-v1.json",
    "image-synthetic-eval-v1": EVAL / "image-synthetic-eval-v1.json",
    "multihop-graph-eval-v1": EVAL / "multihop-graph-eval-v1.json",
    "video-retrieval-fixtures-v1": ROOT / "tests" / "evaluation" / "video_retrieval" / "fixtures",
    "code-search-eval-v1": ROOT / "docs" / "development" / "code-search-eval-v1.json",
    "code-search-eval-heldout-v1": ROOT
    / "docs"
    / "development"
    / "code-search-eval-heldout-v1.json",
}
GIT = shutil.which("git")


def rerun(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    saved = json.loads(path.read_text())
    validate_run(saved)
    dataset = load_dataset(DATASETS[saved["dataset"]["id"]])
    assert dataset.sha256 == saved["dataset"]["sha256"]
    options = RunOptions(
        k=saved["options"]["k"],
        reranker=saved["options"]["reranker"],
        fake_dense=saved["options"]["fake_dense"],
        corpus_repo=ROOT if saved["stack"] == "code" else None,
        git=GIT if saved["stack"] == "code" else None,
    )
    return saved, run_evaluation(saved["stack"], dataset, options)


def test_every_stack_has_a_saved_baseline() -> None:
    stacks = {json.loads(path.read_text())["stack"] for path in BASELINES}
    assert stacks == {
        "offline-text",
        "kb",
        "unified",
        "code",
        "image",
        "video",
        "graphrag-local",
        "graphrag-global",
    }
    for path in BASELINES:
        saved = json.loads(path.read_text())
        assert saved["model"]["real_model_validation"] == "not_run"
        assert saved["model"]["kind"] in {"none", "deterministic-fake"}


@pytest.mark.evaluation
@pytest.mark.parametrize(
    "path",
    [path for path in BASELINES if not path.name.startswith("code--")],
    ids=lambda path: path.name.removesuffix(".run.json"),
)
def test_saved_baseline_reproduces_byte_for_byte(path: Path) -> None:
    saved, fresh = rerun(path)
    assert fresh["results"] == saved["results"]
    assert fresh["results_sha256"] == saved["results_sha256"]
    assert fresh["profile"] == saved["profile"]
    assert score_run(fresh)["metrics"] == score_run(saved)["metrics"]


def test_global_graphrag_fixture_ranking_is_independent_of_python_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_dataset(DATASETS["multihop-graph-eval-v1"])
    expected = run_evaluation("graphrag-global", dataset, RunOptions(k=5))
    with monkeypatch.context() as runtime:
        runtime.setattr("platform.python_version", lambda: "3.11.999")
        actual = run_evaluation("graphrag-global", dataset, RunOptions(k=5))
    assert actual["results_sha256"] == expected["results_sha256"]


@pytest.mark.evaluation
@pytest.mark.skipif(
    os.environ.get("CAPICOVE_EVAL_CODE_BASELINES") != "1" or GIT is None,
    reason="frozen code-set baselines index the repository (opt in)",
)
@pytest.mark.parametrize(
    "path",
    [path for path in BASELINES if path.name.startswith("code--")],
    ids=lambda path: path.name.removesuffix(".run.json"),
)
def test_saved_code_baseline_reproduces_byte_for_byte(path: Path) -> None:
    saved, fresh = rerun(path)
    assert fresh["results_sha256"] == saved["results_sha256"]


@pytest.mark.skipif(GIT is None, reason="requires git")
def test_code_adapter_indexes_a_commit_export(tmp_path: Path) -> None:
    assert GIT is not None
    repository = tmp_path / "repo"
    (repository / "src" / "pkg").mkdir(parents=True)
    (repository / "src" / "pkg" / "rank.py").write_text(
        "def blend_rankings(left, right):\n    return left + right\n"
    )
    (repository / "src" / "pkg" / "other.py").write_text("VALUE = 1\n")
    (repository / "docs").mkdir()
    (repository / "docs" / "ignored.md").write_text("blend_rankings outside the corpus\n")
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}
    for arguments in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@invalid", "commit", "-qm", "c"],
    ):
        subprocess.run([GIT, "-C", str(repository), *arguments], check=True, env=environment)  # nosec B603
    commit = subprocess.run(  # nosec B603
        [GIT, "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        env=environment,
    ).stdout.decode()[:12]
    dataset_path = tmp_path / "code.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema": "capicove.code-search-eval.v1",
                "corpus": f"fixture at commit {commit}",
                "queries": [
                    {
                        "id": "q1",
                        "category": "identifier",
                        "query": "blend_rankings",
                        "relevant": [{"path": "src/pkg/rank.py", "symbol": "blend_rankings"}],
                    },
                    {"id": "q2", "category": "no_answer", "query": "zzqx", "relevant": []},
                ],
            }
        )
    )
    record = run_evaluation(
        "code",
        load_dataset(dataset_path),
        RunOptions(k=5, corpus_repo=repository, git=GIT),
    )
    assert record["index"]["selected_files"] == 2
    assert record["index"]["repository_commit"].startswith(commit)
    first = record["results"][0]["ranked"][0]
    assert first["key"].startswith("src/pkg/rank.py@")
    assert first["labels"] == ["src/pkg/rank.py#blend_rankings"]
    assert score_run(record)["metrics"]["mrr"] == 1.0
    from capicove.evaluation.harness import EvaluationError

    with pytest.raises(EvaluationError, match="corpus-repo"):
        run_evaluation("code", load_dataset(dataset_path), RunOptions())
    with pytest.raises(EvaluationError, match="default lexical"):
        run_evaluation(
            "code",
            load_dataset(dataset_path),
            RunOptions(reranker="deterministic-fake", corpus_repo=repository, git=GIT),
        )
    bad = json.loads(dataset_path.read_text())
    bad["corpus"] = "fixture at commit 0000000"
    dataset_path.write_text(json.dumps(bad))
    with pytest.raises(EvaluationError, match="git"):
        run_evaluation(
            "code", load_dataset(dataset_path), RunOptions(corpus_repo=repository, git=GIT)
        )


@pytest.mark.parametrize(
    ("stack", "dataset", "options"),
    [
        ("video", "video-retrieval-fixtures-v1", RunOptions(reranker="deterministic-fake")),
        ("image", "image-synthetic-eval-v1", RunOptions(fake_dense=True)),
        ("unified", "unified-mixed-eval-v1", RunOptions(fake_dense=True)),
        ("graphrag-local", "multihop-graph-eval-v1", RunOptions(fake_dense=True)),
    ],
)
def test_adapters_reject_unsupported_options(stack: str, dataset: str, options: RunOptions) -> None:
    from capicove.evaluation.harness import EvaluationError

    with pytest.raises(EvaluationError):
        run_evaluation(stack, load_dataset(DATASETS[dataset]), options)


def test_profile_files_drive_text_stacks(tmp_path: Path) -> None:
    from dataclasses import replace

    from capicove.retrieval.pipeline_profiles import RerankPolicy, default_knowledge_profile

    profile = replace(
        default_knowledge_profile(),
        profile_id="kb-degrade-v1",
        rerank=RerankPolicy(enabled=True, on_unavailable="degrade"),
    )
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(profile.to_mapping()))
    record = run_evaluation(
        "kb", load_dataset(DATASETS["personal-notes-eval-v1"]), RunOptions(profile_path=path)
    )
    assert record["profile"] == profile.identity()
    statuses = {
        (row["diagnostics"]["rerank"]["status"], row["diagnostics"]["rerank"]["reason"])
        for row in record["results"]
    }
    # Queries without lexical candidates have nothing to rerank and say so.
    assert statuses == {("degraded", "reranker_unavailable"), ("no_candidates", None)}
    baseline = json.loads((EVAL / "baselines" / "kb--personal-notes-eval-v1.run.json").read_text())
    assert [[item["key"] for item in row["ranked"]] for row in record["results"]] == [
        [item["key"] for item in row["ranked"]] for row in baseline["results"]
    ]
    failing = replace(profile, profile_id="kb-fail-v1", rerank=RerankPolicy(enabled=True))
    path.write_text(json.dumps(failing.to_mapping()))
    failed = run_evaluation(
        "kb", load_dataset(DATASETS["personal-notes-eval-v1"]), RunOptions(profile_path=path)
    )
    assert failed["results"][0]["diagnostics"]["error"] == "reranker_unavailable"
    # Only queries with no candidates (nothing to rerank) succeed under "fail".
    assert all(row["status"] == "failed" or not row["ranked"] for row in failed["results"])
    assert sum(row["status"] == "failed" for row in failed["results"]) >= 18


def test_bge_reranker_without_local_assets_fails_explicitly(tmp_path: Path) -> None:
    from capicove.evaluation.harness import EvaluationError

    with pytest.raises(EvaluationError, match="requires --reranker-model"):
        RunOptions(reranker="bge")
    with pytest.raises(EvaluationError, match="requires --reranker-model"):
        RunOptions(reranker_model=tmp_path)
    options = RunOptions(reranker="bge", reranker_model=tmp_path / "missing")
    with pytest.raises(EvaluationError, match="BGE reranker unavailable"):
        run_evaluation("kb", load_dataset(DATASETS["personal-notes-eval-v1"]), options)


def test_local_model_runs_are_marked_measured_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from capicove.evaluation.adapters import _profiles
    from capicove.evaluation.fakes import DeterministicFakeReranker

    def fake_bge(profile: Any, options: RunOptions) -> Any:
        return profile, DeterministicFakeReranker(), {"kind": "local-model", "components": []}

    monkeypatch.setattr(_profiles, "_bge", fake_bge)
    from dataclasses import replace

    from capicove.retrieval.pipeline_profiles import RerankPolicy, default_knowledge_profile

    profile = replace(
        default_knowledge_profile(),
        profile_id="kb-x-v1",
        rerank=RerankPolicy(enabled=True, model_id="capicove/deterministic-fake-reranker"),
    )
    path = Path(__file__).parent / "_unused"
    options = RunOptions(reranker="bge", reranker_model=path)
    monkeypatch.setattr(
        _profiles, "resolve_profile", lambda stack, opts: (profile, *fake_bge(profile, opts)[1:])
    )
    from capicove.evaluation.adapters import knowledge_base

    monkeypatch.setattr(knowledge_base, "resolve_profile", _profiles.resolve_profile)
    record = run_evaluation("kb", load_dataset(DATASETS["personal-notes-eval-v1"]), options)
    assert record["model"]["real_model_validation"] == "measured_unaccepted"
