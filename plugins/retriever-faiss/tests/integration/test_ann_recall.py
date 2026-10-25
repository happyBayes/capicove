"""Opt-in, deterministic 10k-vector native ANN recall qualification."""

import os
from pathlib import Path

import numpy as np
import pytest
from capicove_retriever_faiss.adapter import FaissVisualRetriever
from capicove_retriever_faiss.config import FaissConfiguration
from integration_helpers import entry, manifest

from capicove.contracts import SearchFilters


@pytest.mark.skipif(
    os.environ.get("CAPICOVE_TEST_REAL_FAISS") != "1", reason="native FAISS opt-in required"
)
def test_ten_thousand_vectors_recall_at_twenty(tmp_path: Path) -> None:
    import faiss

    rng = np.random.default_rng(20260915)
    vectors = rng.standard_normal((10_000, 64), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.standard_normal((100, 64), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    entries = tuple(entry(f"unit-{i}", tuple(map(float, row))) for i, row in enumerate(vectors))
    generation = manifest(entries)
    backend = FaissVisualRetriever(FaissConfiguration(tmp_path.resolve(), threads=2))
    backend.create(generation)
    backend.add_stream(index_generation_id=generation.generation_id, entries=iter(entries))
    report = backend.validate(generation)
    assert report.valid, report.issue_codes
    pin = backend.activate(generation, report)
    exact = faiss.IndexFlatIP(64)
    exact.add(vectors)
    _, expected = exact.search(queries, 20)
    recovered = 0
    try:
        for query, gold in zip(queries, expected, strict=True):
            candidates = backend.retrieve(
                query_vector=tuple(map(float, query)),
                embedding_space=generation.embedding_space,
                filters=SearchFilters(),
                generation=pin,
                top_k=20,
            )
            recovered += len({item.unit_id for item in candidates} & {f"unit-{i}" for i in gold})
    finally:
        backend.close()
    recall = recovered / 2000
    print(f"synthetic-native-ann count=10000 dimension=64 queries=100 recall_at_20={recall:.6f}")
    assert recall >= 0.95
