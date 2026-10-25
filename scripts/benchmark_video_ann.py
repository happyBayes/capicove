#!/usr/bin/env python3
"""Explicit local synthetic-vector FAISS qualification; never semantic evidence.

Run with the FAISS plugin on PYTHONPATH and a fresh temporary output directory.
Raw vectors, indexes and reports are generated only in that directory. The full
corpus is memory mapped; no million-row Python vector collection is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Iterator
from pathlib import Path

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.video_models import (
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    searchable_units_manifest_sha256,
    visual_entries_content_sha256,
)


def main() -> int:
    """Build, reopen and query a frozen seeded corpus through the real plugin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if not 20 <= args.count <= 1_000_000 or not 2 <= args.dimension <= 2048:
        parser.error("count must be 20..1000000 and dimension 2..2048")
    if not 1 <= args.queries <= 10_000:
        parser.error("queries must be 1..10000")
    if not 1 <= args.threads <= 64 or not 1 <= args.ef_search <= 4096:
        parser.error("threads must be 1..64 and ef-search 1..4096")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must be an unsigned 64-bit integer")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    import faiss
    import numpy as np
    import psutil
    from capicove_retriever_faiss.adapter import FaissVisualRetriever
    from capicove_retriever_faiss.config import FaissConfiguration

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    vectors = np.lib.format.open_memmap(
        output / "vectors.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.count, args.dimension),
    )
    vector_digest = hashlib.sha256()
    for offset in range(0, args.count, 512):
        size = min(512, args.count - offset)
        batch = rng.standard_normal((size, args.dimension), dtype=np.float32)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        vectors[offset : offset + size] = batch
        vector_digest.update(batch.tobytes())
    vectors.flush()
    space = EmbeddingSpaceRef(
        f"synthetic-v1-{args.dimension}",
        "authored-synthetic",
        "seeded-normal-v1",
        "l2-v1",
        args.dimension,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME}),
    )
    fingerprint = ProcessingFingerprint.from_manifest(
        {"seed": args.seed, "dimension": args.dimension, "vector_sha256": vector_digest.hexdigest()}
    )

    def units() -> Iterator[SearchableUnit]:
        for ordinal in range(args.count):
            yield SearchableUnit(
                f"unit-{ordinal}",
                TargetType.KEYFRAME,
                vector_digest.hexdigest(),
                "synthetic-artifact",
                f"evidence:{ordinal}",
                (f"representation:{ordinal}",),
                "synthetic-build",
                "v1",
                fingerprint,
                embedding_spaces=(space,),
                timestamp_ms=ordinal,
            )

    def entries() -> Iterator[VisualIndexEntry]:
        for ordinal, unit in enumerate(units()):
            yield VisualIndexEntry(unit, tuple(map(float, vectors[ordinal])))

    manifest = VisualIndexGenerationManifest(
        "synthetic-keyframes",
        "benchmark",
        ("synthetic-vectors",),
        TargetType.KEYFRAME,
        searchable_units_manifest_sha256(units()),
        visual_entries_content_sha256(entries()),
        space,
        "capicove.retriever.faiss",
        "faiss-hnsw-flat-ip-v1",
        fingerprint,
        args.count,
    )
    print("building immutable native index", flush=True)
    config = FaissConfiguration(output / "indexes", threads=args.threads, ef_search=args.ef_search)
    backend = FaissVisualRetriever(config)
    backend.create(manifest)
    backend.add_stream(index_generation_id=manifest.generation_id, entries=entries())
    validation = backend.validate(manifest)
    if not validation.valid:
        raise RuntimeError(f"generation validation failed: {validation.issue_codes}")
    pin = backend.activate(manifest, validation)
    backend.close()
    backend = FaissVisualRetriever(config)
    build_seconds = time.perf_counter() - started

    # Full exact recall is measured only on <=10k corpora. A large run separately
    # reports known-neighbor recovery, NOT mislabeled subset/global ANN recall.
    exact = faiss.IndexFlatIP(args.dimension) if args.count <= 10_000 else None
    if exact is not None:
        exact.add(vectors)
    queries = rng.standard_normal((args.queries, args.dimension), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    expected = exact.search(queries, 20)[1] if exact is not None else None
    if expected is None:
        known = rng.integers(0, args.count, size=args.queries)
        queries = np.asarray(vectors[known]).copy()
    else:
        known = None
    process = psutil.Process()
    samples: list[float] = []
    recall_hits = 0
    known_hits = 0
    rss_samples: list[int] = []
    try:
        for i in range(100 + args.queries):
            query_number = i % args.queries
            start = time.perf_counter_ns()
            result = backend.retrieve(
                query_vector=tuple(map(float, queries[query_number])),
                embedding_space=space,
                filters=SearchFilters(),
                generation=pin,
                top_k=20,
            )
            elapsed = (time.perf_counter_ns() - start) / 1_000_000
            if i < 100:
                continue
            samples.append(elapsed)
            rss_samples.append(process.memory_info().rss)
            found = {item.unit_id for item in result}
            if expected is not None:
                recall_hits += len(found & {f"unit-{n}" for n in expected[query_number]})
            elif known is not None:
                known_hits += f"unit-{known[query_number]}" in found
    finally:
        backend.close()
    index_bytes = sum(p.stat().st_size for p in (output / "indexes").rglob("*") if p.is_file())
    report = {
        "schema_version": 1,
        "evidence_kind": "synthetic_vectors_real_native_index",
        "semantic_quality_measured": False,
        "production_profile_qualified": False,
        "count": args.count,
        "dimension": args.dimension,
        "seed": args.seed,
        "query_count": args.queries,
        "warmup_count": 100,
        "ef_search": args.ef_search,
        "threads": args.threads,
        "faiss_version": faiss.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "vector_sha256": vector_digest.hexdigest(),
        "manifest_sha256": manifest.manifest_sha256,
        "build_seconds": build_seconds,
        "index_disk_bytes": index_bytes,
        "plugin_retrieve_p95_ms": float(np.percentile(samples, 95)),
        "rss_max_bytes": max(rss_samples),
        "steady_rss_growth_fraction": max(0, max(rss_samples) / rss_samples[0] - 1),
        "full_corpus_recall_at_20": recall_hits / (args.queries * 20)
        if expected is not None
        else None,
        "known_neighbor_recovery_at_20": known_hits / args.queries if known is not None else None,
        "client_http_latency_measured": False,
    }
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if expected is None or recall_hits / (args.queries * 20) >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
