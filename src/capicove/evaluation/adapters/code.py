"""Code adapter: the frozen code-search sets over the repository at their commit.

The dataset names a commit. Its ``src/`` and ``scripts/`` trees are exported from
the local ``corpus_repo`` object store into the run's scratch directory (as in the
recorded S1 measurement), imported and indexed with the lexical
``code-symbols-v1`` policy, which needs no model. Labels match by relative path and optional symbol exactly as
``scripts/evaluate_code_search.py`` does. The pool is the deepest public result
list (``POOL_DEPTH``).
"""

from __future__ import annotations

import subprocess  # nosec B404
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capicove.evaluation.adapters._common import host_config, invocation_context, union_labels
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
)


# Each Evidence read re-verifies the whole generation (about 0.4 s on the reference
# host), so the pool is bounded to keep a 30-query run near five minutes.
POOL_DEPTH = 30
POLICY = "code-symbols-v1"
# The 2026-10-23 S1 record indexed these trees of the dataset commit.
CORPUS_PATHS = ("src", "scripts")


def _git(git: str, arguments: list[str], home: Path) -> str:
    completed = subprocess.run(  # nosec B603
        [
            git,
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.file.allow=always",
            *arguments,
        ],
        capture_output=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        },
        timeout=600,
    )
    if completed.returncode != 0:
        raise EvaluationError("git could not prepare the frozen code corpus")
    return completed.stdout.decode("utf-8", "replace").strip()


def checkout_commit(git: str, repository: Path, commit: str, destination: Path) -> str:
    """Export ``CORPUS_PATHS`` of one commit into a fresh scratch repository.

    ``git archive`` reads the local object store only; the export is staged in a
    new scratch repository so the code importer's Git enumeration applies. The
    source repository is never modified. Returns the full commit ID.
    """
    import io
    import tarfile

    home = destination.parent / "git-home"
    home.mkdir(mode=0o700, exist_ok=True)
    resolved = _git(
        git, ["-C", str(repository), "rev-parse", "--verify", f"{commit}^{{commit}}"], home
    )
    if not resolved.startswith(commit):
        raise EvaluationError("resolved commit does not match the dataset commit")
    present = set(
        _git(git, ["-C", str(repository), "ls-tree", "--name-only", resolved], home).splitlines()
    )
    paths = [path for path in CORPUS_PATHS if path in present]
    if not paths:
        raise EvaluationError("the dataset commit contains none of the corpus paths")
    archive = subprocess.run(  # nosec B603
        [git, "-C", str(repository), "archive", "--format=tar", resolved, *paths],
        capture_output=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"},
        timeout=600,
    )
    if archive.returncode != 0:
        raise EvaluationError("git could not export the frozen code corpus")
    destination.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(destination, filter="data")  # nosec B202
    _git(git, ["-C", str(destination), "init", "--quiet"], home)
    _git(git, ["-C", str(destination), "add", "--all"], home)
    return resolved


def label_matches(metadata: Mapping[str, Any], label: Mapping[str, Any]) -> bool:
    """Match by relative path, then by an optional definition or textual symbol."""
    if metadata["relative_path"] != label["path"]:
        return False
    symbol = label.get("symbol")
    if symbol is None:
        return True
    return any(item["name"] == symbol for item in metadata.get("definitions", ())) or (
        symbol in metadata["text"]
    )


@dataclass(slots=True)
class _PreparedCode:
    runtime: Any
    identity: Mapping[str, Any]
    _evidence: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        from capicove.contracts.retrieval import SearchRequest, TargetType, TextQuery

        if query.text is None:
            raise EvaluationError("code queries require text")
        started = time.perf_counter_ns()
        result = self.runtime.search(
            SearchRequest(
                TextQuery(query.text),
                frozenset({TargetType.TEXT_CHUNK}),
                top_k=POOL_DEPTH,
                retrieval_profile_ref="local-default",
            )
        )
        elapsed = time.perf_counter_ns() - started
        if result.status.value != "succeeded":
            code = None if result.failure_code is None else result.failure_code.value
            return QueryOutcome((), 0, (), status="failed", diagnostics={"error": code})
        items = []
        for hit in result.hits:
            metadata = self._metadata(hit.evidence_id)
            locator = hit.text_locator
            span = (
                ""
                if locator is None
                else (f"@{locator.character_range.start_char}-{locator.character_range.end_char}")
            )
            labels = tuple(
                label["label"]
                for label in query.payload["labels"]
                if label_matches(metadata, label)
            )
            items.append(
                RankedItem(f"{metadata['relative_path']}{span}", labels, len(metadata["text"]))
            )
        return QueryOutcome(
            tuple(items[:k]),
            len(items),
            union_labels(item.labels for item in items),
            search_ns=elapsed,
        )

    def _metadata(self, evidence_id: str) -> Mapping[str, Any]:
        cached = self._evidence.get(evidence_id)
        if cached is None:
            cached = dict(
                self.runtime.read_evidence(invocation_context(), evidence_id, ("metadata",))[
                    "metadata"
                ]
            )
            self._evidence[evidence_id] = cached
        return cached

    def close(self) -> None:
        self.runtime.close()


class CodeAdapter:
    """Index the dataset's repository commit with the lexical code policy."""

    stack = "code"
    adapter_version = "1"

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> _PreparedCode:
        from capicove.runtime import code_retrieval

        if options.reranker != "none" or options.profile_path is not None or options.fake_dense:
            raise EvaluationError("the code adapter supports only its default lexical policy")
        if options.corpus_repo is None or options.git is None:
            raise EvaluationError("code evaluation requires --corpus-repo and an absolute --git")
        repository = workdir / "repository"
        commit = checkout_commit(
            options.git,
            Path(options.corpus_repo).resolve(),
            dataset.corpus["repository_commit"],
            repository,
        )
        workspace = workdir / "workspace"
        workspace.mkdir(mode=0o700)
        config = host_config(
            workspace,
            {
                "retrieval": {
                    "schema_version": 2,
                    "source_roots": {"repo": str(repository.resolve())},
                    "index_root": str(workspace / "index"),
                    "corpus_ref": "corpus:evaluation",
                    "retrieval_profile_ref": "local-code-lexical-v1",
                    "max_source_bytes": 1024 * 1024,
                    "model_config_path": None,
                    "model_config_sha256": None,
                    "model_binding": None,
                    "max_units": 10_000,
                    "max_vector_bytes": 0,
                    "code_policy_ref": POLICY,
                }
            },
            ["capicove.get_evidence", "capicove.search"],
        )
        imported = code_retrieval.import_repository(config, "repo", git=options.git)
        built = code_retrieval.build_index(config)
        runtime = code_retrieval.open_code_retrieval(config)
        return _PreparedCode(
            runtime,
            {
                "profile": {
                    "id": "local-code-lexical-v1",
                    "version": 1,
                    "fingerprint": built.get("policy_sha256") or _policy_fingerprint(),
                    "code_policy_ref": POLICY,
                },
                "index": {
                    "kind": "local-code-generation",
                    "repository_commit": commit,
                    "corpus_paths": list(CORPUS_PATHS),
                    "dataset_corpus": dataset.corpus["description"],
                    "generation_id": built["generation_id"],
                    "units": built["units"],
                    "selected_files": imported["selected"],
                    "selection_sha256": imported["receipt"]["selection_sha256"],
                },
                "model": {"kind": "none", "components": []},
            },
        )


def _policy_fingerprint() -> str:
    from capicove.evaluation.adapters._common import fingerprint

    return fingerprint({"policy": POLICY, "profile": "local-code-lexical-v1"})


def adapter(stack: str) -> CodeAdapter:
    """Return the code stack adapter."""
    return CodeAdapter()
