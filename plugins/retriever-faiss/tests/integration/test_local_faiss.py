"""Opt-in integration test against an explicitly provisioned native FAISS."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import pytest
from capicove_retriever_faiss.adapter import FaissVisualRetriever
from capicove_retriever_faiss.config import FaissConfiguration
from integration_helpers import entry, manifest

from capicove.contracts import SearchFilters
from capicove.retrieval.video_models import VisualIndexEntry


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    importlib.util.find_spec("faiss") is None or os.environ.get("CAPICOVE_TEST_REAL_FAISS") != "1",
    reason="requires native FAISS and CAPICOVE_TEST_REAL_FAISS=1",
)
def test_native_faiss_build_validate_reopen_search(tmp_path: Path) -> None:
    entries = (
        entry("unit-a", (1.0, 0.0)),
        entry("unit-b", (0.0, 1.0)),
    )
    generation = manifest(entries)
    configuration = FaissConfiguration(index_root=tmp_path.resolve(), threads=1)
    backend = FaissVisualRetriever(configuration)
    backend.create(generation)
    backend.add(index_generation_id=generation.generation_id, entries=entries)
    report = backend.validate(generation)
    pin = backend.activate(generation, report)
    backend.close()

    candidates = FaissVisualRetriever(configuration).retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=generation.embedding_space,
        filters=SearchFilters(),
        generation=pin,
        top_k=1,
    )

    assert report.valid
    assert [candidate.unit_id for candidate in candidates] == ["unit-a"]


@pytest.mark.skipif(
    importlib.util.find_spec("faiss") is None or os.environ.get("CAPICOVE_TEST_REAL_FAISS") != "1",
    reason="requires native FAISS and CAPICOVE_TEST_REAL_FAISS=1",
)
def test_native_ann_matches_exact_on_frozen_synthetic_vectors(tmp_path: Path) -> None:
    """Exercise native ANN numerics; this is not semantic or scale qualification."""
    import numpy as np

    rng = np.random.default_rng(20260915)
    vectors = rng.normal(size=(512, 32)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    template = entry("template", (1.0, 0.0))
    space = replace(template.unit.embedding_spaces[0], dimension=32)
    entries = tuple(
        VisualIndexEntry(
            replace(template.unit, unit_id=f"unit-{i:04d}", embedding_spaces=(space,)),
            tuple(float(x) for x in row),
        )
        for i, row in enumerate(vectors)
    )
    generation = replace(manifest(entries), embedding_space=space)
    configuration = FaissConfiguration(index_root=tmp_path.resolve(), threads=1)
    backend = FaissVisualRetriever(configuration)
    backend.create(generation)
    backend.add(index_generation_id=generation.generation_id, entries=entries)
    pin = backend.activate(generation, backend.validate(generation))
    backend.close()
    reopened = FaissVisualRetriever(configuration)
    try:
        for row in vectors[::32]:
            expected = {f"unit-{int(i):04d}" for i in np.argsort(-(vectors @ row))[:20]}
            found = reopened.retrieve(
                query_vector=tuple(float(x) for x in row),
                embedding_space=space,
                filters=SearchFilters(),
                generation=pin,
                top_k=20,
            )
            assert len(expected & {candidate.unit_id for candidate in found}) / 20 >= 0.95
    finally:
        reopened.close()
