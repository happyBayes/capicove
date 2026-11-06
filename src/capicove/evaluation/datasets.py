"""Strict loaders for authored and existing frozen retrieval evaluation sets.

Supported inputs:

* ``capicove.retrieval-eval.v1`` JSON: authored corpora and graded labels.
* ``capicove.code-search-eval.v1`` JSON: the existing code-search sets, whose
  labels name a relative path and optional symbol, never a chunk ID.
* The existing frozen video fixture directory (``corpus.jsonl``,
  ``queries.jsonl``, ``judgments.jsonl``) with its pinned SHA-256 digests.

Dataset identity is the SHA-256 of the exact source bytes (for a directory, of
the canonical map of file digests), so a run cannot silently use edited data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from capicove.evaluation.harness import (
    EVAL_DATASET_SCHEMA,
    EvalDataset,
    EvalQuery,
    EvaluationError,
    read_json,
    sha256_json,
)


CODE_SEARCH_SCHEMA = "capicove.code-search-eval.v1"
VIDEO_FIXTURE_SCHEMA = "capicove.video-retrieval-fixture.v1"
# Frozen authored-fixture SHA-256 values (tests/evaluation/video_retrieval/fixtures).
VIDEO_FIXTURE_DIGESTS = {
    "corpus": "333ddbc56135a31def7b9bcbbd2ecf29d4daf082b0b61ea12c76fd848ff038fe",  # pragma: allowlist secret
    "queries": "5a3ec9eac558a10af59bb687fe82206f495caa68e89d8c4f0cf4ae8b671de697",  # pragma: allowlist secret
    "judgments": "bc16f26620273bb97a42d9cfc8820fbce6913dd04a7c4740e4ddadc977e7d381",  # pragma: allowlist secret
}
GENERIC_STACKS = frozenset(
    {"offline-text", "kb", "unified", "image", "graphrag-local", "graphrag-global"}
)
_GENERIC_FIELDS = {"schema", "dataset_id", "stacks", "frozen_at", "provenance", "corpus", "queries"}
_QUERY_FIELDS = {"id", "category", "text", "relevant"}
_LABEL_FIELDS = {"label", "document", "contains", "grade"}
_MAX_DATASET_BYTES = 16 * 1024 * 1024


def load_dataset(path: Path) -> EvalDataset:
    """Load one supported frozen dataset from a file or fixture directory."""
    path = Path(path)
    if path.is_dir():
        return _load_video_fixture(path)
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > _MAX_DATASET_BYTES:
        raise EvaluationError("dataset exceeds the size limit")
    value = read_json(path, max_bytes=_MAX_DATASET_BYTES)
    if not isinstance(value, dict):
        raise EvaluationError("dataset must be a JSON object")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    schema = value.get("schema")
    if schema == EVAL_DATASET_SCHEMA:
        return _load_generic(value, digest, path)
    if schema == CODE_SEARCH_SCHEMA:
        return _load_code_search(value, digest, path)
    raise EvaluationError("unsupported dataset schema")


def _load_generic(value: dict[str, Any], digest: str, path: Path) -> EvalDataset:
    if set(value) != _GENERIC_FIELDS:
        raise EvaluationError("generic dataset fields must match the closed schema")
    stacks = value["stacks"]
    if (
        not isinstance(stacks, list)
        or not stacks
        or len(set(stacks)) != len(stacks)
        or not set(stacks) <= GENERIC_STACKS
    ):
        raise EvaluationError("dataset stacks must name supported generic stacks")
    corpus = value["corpus"]
    if not isinstance(corpus, dict):
        raise EvaluationError("dataset corpus must be an object")
    queries = []
    for row in value["queries"] if isinstance(value["queries"], list) else ():
        if not isinstance(row, dict) or set(row) - {"payload"} != _QUERY_FIELDS:
            raise EvaluationError("dataset query fields must match the closed schema")
        judgments: dict[str, int] = {}
        labels: list[dict[str, Any]] = []
        for label in row["relevant"] if isinstance(row["relevant"], list) else [None]:
            if not isinstance(label, dict) or set(label) != _LABEL_FIELDS:
                raise EvaluationError("dataset labels must match the closed schema")
            if label["label"] in judgments:
                raise EvaluationError("dataset labels must be unique within a query")
            if not isinstance(label["document"], str) or not label["document"]:
                raise EvaluationError("dataset labels must name a document")
            if label["contains"] is not None and (
                not isinstance(label["contains"], str) or not label["contains"].strip()
            ):
                raise EvaluationError("label contains must be null or nonblank text")
            judgments[label["label"]] = label["grade"]
            labels.append(dict(label))
        payload = dict(row.get("payload", {}))
        payload["labels"] = labels
        queries.append(EvalQuery(row["id"], row["category"], row["text"], judgments, payload))
    return EvalDataset(
        value["dataset_id"],
        EVAL_DATASET_SCHEMA,
        tuple(stacks),
        digest,
        tuple(queries),
        corpus,
        path,
    )


def _load_code_search(value: dict[str, Any], digest: str, path: Path) -> EvalDataset:
    corpus_text = value.get("corpus")
    # "... at commit X" or "... (N files ... at X)": the export source commit.
    match = re.search(
        r"\b(?:commit|at) ([0-9a-f]{7,40})\b", corpus_text if isinstance(corpus_text, str) else ""
    )
    if match is None or not isinstance(value.get("queries"), list):
        raise EvaluationError("code-search dataset must name its corpus commit and queries")
    queries = []
    for row in value["queries"]:
        if not isinstance(row, dict):
            raise EvaluationError("code-search queries must be objects")
        labels = []
        judgments: dict[str, int] = {}
        for label in row.get("relevant", []):
            symbol = label.get("symbol")
            identity = label["path"] if symbol is None else f"{label['path']}#{symbol}"
            judgments[identity] = 1
            labels.append({"label": identity, "path": label["path"], "symbol": symbol})
        context = []
        for label in row.get("context", []):
            symbol = label.get("symbol")
            context.append({"path": label["path"], "symbol": symbol})
        queries.append(
            EvalQuery(
                row["id"],
                row["category"],
                row["query"],
                judgments,
                {"labels": labels, "context": context, "definition": row.get("definition")},
            )
        )
    name = path.name.removesuffix(".json")
    return EvalDataset(
        name,
        CODE_SEARCH_SCHEMA,
        ("code",),
        digest,
        tuple(queries),
        {"repository_commit": match.group(1), "description": corpus_text},
        path,
    )


def _load_video_fixture(root: Path) -> EvalDataset:
    rows: dict[str, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for name, expected in VIDEO_FIXTURE_DIGESTS.items():
        raw = (root / f"{name}.jsonl").read_bytes()
        digests[name] = hashlib.sha256(raw).hexdigest()
        if digests[name] != expected:
            raise EvaluationError("frozen video fixture digest mismatch")
        rows[name] = [json.loads(line) for line in raw.splitlines()]
    expected_by_id = {row["id"]: row["expected"] for row in rows["judgments"]}
    queries = []
    for row in rows["queries"]:
        expected = expected_by_id[row["id"]]
        # Authored order is the ideal ranking; earlier units carry higher grades.
        judgments = {unit: max(1, 3 - index) for index, unit in enumerate(expected)}
        queries.append(
            EvalQuery(
                row["id"],
                row["scenario"],
                None,
                judgments,
                {"query": row, "expected": list(expected)},
            )
        )
    return EvalDataset(
        "video-retrieval-fixtures-v1",
        VIDEO_FIXTURE_SCHEMA,
        ("video",),
        sha256_json(digests),
        tuple(queries),
        {"units": rows["corpus"], "file_sha256": digests},
        root,
    )


__all__ = ["CODE_SEARCH_SCHEMA", "VIDEO_FIXTURE_SCHEMA", "load_dataset"]
