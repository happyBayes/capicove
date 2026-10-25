"""S3 ``local-code-hybrid-v1``: persisted vectors, dense recall, fusion and degradation.

A handwritten concept encoder is the independent oracle: documents and queries that
mention the same concept word share one axis, so dense neighbours are known in
advance and differ from lexical matches. It tests orchestration, not model quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    SearchRequest,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TextQuery,
    VectorNormalization,
)
from capicove.model.configuration import ModelCapability, ModelConfiguration, ModelProfile
from capicove.model.factory import ModelFactory
from capicove.model.registry import ProviderDefinition, ProviderRegistry
from capicove.retrieval.models import TextEmbedding
from capicove.runtime import code_retrieval as runtime
from tests.integration.retrieval.test_code_lexical import _context, _git, _host, _repository


GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(os.name != "posix" or GIT is None, reason="requires POSIX and Git")
HYBRID = "local-code-hybrid-v1"
SPACE = EmbeddingSpaceRef(
    space_id="test-code:fixture",
    provider_id="test-code",
    model_revision="fixture-1",
    preprocessing_version="fixture-1",
    dimension=4,
    normalization=VectorNormalization.L2,
    distance_metric=DistanceMetric.COSINE,
    query_modalities=frozenset({QueryModality.TEXT}),
    target_modalities=frozenset({TargetType.TEXT_CHUNK}),
)
# Concept axes: retry/backoff, rendering/view, persistence/cache, other.
_AXES = (
    ("resilient", "flaky", "robust"),
    ("render", "view", "display"),
    ("cache", "persist", "memory"),
)


def _vector(text: str) -> tuple[float, ...]:
    lowered = text.lower()
    raw = [float(any(word in lowered for word in axis)) for axis in _AXES] + [0.05]
    norm = sum(value * value for value in raw) ** 0.5
    return tuple(value / norm for value in raw)


class ConceptEncoder:
    space = SPACE
    max_batch = 4
    calls: ClassVar[list[tuple[str, int]]] = []
    documents: ClassVar[list[str]] = []
    fail: ClassVar[str | None] = None

    def encode(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]:
        if ConceptEncoder.fail == role:
            from capicove.model.multimodal_embedding import MultimodalEmbeddingError

            raise MultimodalEmbeddingError("unavailable", "fixture encoder stopped")
        assert space == SPACE and role in {"query", "document"}
        self.calls.append((role, len(texts)))
        if role == "document":
            self.documents.extend(texts)
        return tuple(TextEmbedding(SPACE, _vector(text)) for text in texts)

    def close(self) -> None:
        return None


def _models(root: Path) -> ModelFactory:
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "test-code", "1", ModelCapability.TEXT_EMBEDDING, lambda _: ConceptEncoder()
        )
    )
    profile = ModelProfile("code", "test-code", ModelCapability.TEXT_EMBEDDING, root, "0" * 64)
    return ModelFactory(ModelConfiguration({"code": profile}, {"code": "code"}), registry=registry)


def _hybrid(tmp_path: Path, **overrides: object) -> tuple[Any, ModelFactory]:
    root = (tmp_path / "w").resolve()
    root.mkdir(exist_ok=True)
    repo = root / "repo" if (root / "repo").exists() else _repository(root)
    (repo / "src" / "app" / "resilience.py").write_text(
        "def call_again_later(operation):\n    '''Resilient helper with exponential backoff.'''\n"
        "    return operation()\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/app/resilience.py")
    config = tmp_path / "models.json"
    config.write_text(json.dumps({"schema_version": 1, "profiles": {}, "bindings": {}}))
    values: dict[str, Any] = {
        "retrieval_profile_ref": HYBRID,
        "code_policy_ref": "code-hybrid-v1",
        "model_config_path": str(config),
        "model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "model_binding": "code",
        "max_vector_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return _host(root, repo, **values), _models(tmp_path)


def _search(service: Any, query: str, top_k: int = 5) -> Any:
    return service.search(
        SearchRequest(
            TextQuery(query),
            frozenset({TargetType.TEXT_CHUNK}),
            top_k=top_k,
            retrieval_profile_ref=HYBRID,
        )
    )


def _paths(service: Any, result: Any) -> list[str]:
    return [
        service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"][
            "relative_path"
        ]
        for hit in result.hits
    ]


def test_vectors_persist_and_dense_lane_finds_semantic_match_absent_from_lexical(
    tmp_path: Path,
) -> None:
    host, models = _hybrid(tmp_path)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    built = runtime.build_index(host, models=models)
    assert built["vectors"] == built["units"] > 0
    service = runtime.open_code_retrieval(host, models=models)
    try:
        # No query word occurs in any document, so only the dense lane can match.
        dense_only = _search(service, "robust against flaky calls", top_k=1)
        assert dense_only.status is SearchStatus.SUCCEEDED
        assert _paths(service, dense_only) == ["src/app/resilience.py"]
        lexical = _search(service, "fetch_user_profile", top_k=1)
        assert _paths(service, lexical) == ["src/app/client.py"]
    finally:
        service.close()


def test_documents_carry_path_context_and_identity_changes_with_space(tmp_path: Path) -> None:
    host, models = _hybrid(tmp_path)
    assert GIT
    ConceptEncoder.calls.clear()
    runtime.import_repository(host, "repo", git=GIT)
    first = runtime.build_index(host, models=models)
    documents = sum(count for role, count in ConceptEncoder.calls if role == "document")
    assert documents == first["vectors"]
    assert any(text.startswith("src/app/client.py\n") for text in ConceptEncoder.documents)
    ConceptEncoder.calls.clear()
    again = runtime.build_index(host, models=models)
    assert again["generation_id"] == first["generation_id"]
    assert sum(count for role, count in ConceptEncoder.calls if role == "document") == 0
    lexical_host = _host(
        host.workspace_root, Path(host.capabilities["retrieval"]["source_roots"]["repo"])
    )
    assert runtime.build_index(lexical_host)["generation_id"] != first["generation_id"]


def test_encoder_unavailability_degrades_to_lexical_with_warning(tmp_path: Path) -> None:
    host, models = _hybrid(tmp_path)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    runtime.build_index(host, models=models)
    service = runtime.open_code_retrieval(host, models=models)
    ConceptEncoder.fail = "query"
    try:
        result = _search(service, "fetch_user_profile", top_k=1)
        assert result.status is SearchStatus.SUCCEEDED and result.degraded
        assert SearchWarningCode.TEXT_DENSE_EMBEDDING_UNAVAILABLE in result.warning_codes
        assert _paths(service, result) == ["src/app/client.py"]
    finally:
        ConceptEncoder.fail = None
        service.close()


def test_vector_budget_and_model_pin_fail_closed(tmp_path: Path) -> None:
    host, models = _hybrid(tmp_path, max_vector_bytes=16)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    with pytest.raises(runtime.CodeRetrievalError, match="max_vector_bytes"):
        runtime.build_index(host, models=models)
    for override in (
        {"model_config_sha256": "0" * 64},
        {"model_binding": None},
        {"max_vector_bytes": 0},
        {"code_policy_ref": "code-symbols-v1"},
    ):
        bad, _ = _hybrid(tmp_path, **override)
        with pytest.raises(runtime.CodeRetrievalError):
            runtime.code_settings(bad)


def test_corrupt_vectors_fail_instead_of_degrading(tmp_path: Path) -> None:
    import sqlite3

    host, models = _hybrid(tmp_path)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    built = runtime.build_index(host, models=models)
    database = (
        Path(host.capabilities["retrieval"]["index_root"])
        / "generations"
        / built["generation_id"]
        / "units.sqlite3"
    )
    database.chmod(0o600)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE units SET vector = zeroblob(16)")
    service = runtime.open_code_retrieval(host, models=models)
    try:
        assert _search(service, "robust against flaky calls").status is SearchStatus.FAILED
    finally:
        service.close()
