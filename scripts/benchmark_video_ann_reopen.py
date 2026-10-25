#!/usr/bin/env python3
"""Compare startup search budgets on one immutable million-vector index.

The known-neighbor probe is explicitly distinct from full random-query recall.
No graph rebuild, threshold reduction or mutation of provider internals occurs.
"""

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--index-subdirectory", default="indexes/synthetic-keyframes")
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--unit-prefix", default="unit-")
    args = parser.parse_args()
    if any(not 1 <= value <= 4096 for value in args.ef_search):
        parser.error("ef-search must be 1..4096")
    root = args.root.resolve()
    physical = (root / args.index_subdirectory).resolve()
    if not physical.is_relative_to(root) or not args.unit_prefix.strip():
        parser.error("index must remain inside the explicit root and unit prefix must be nonempty")
    import numpy as np
    import psutil
    from capicove_retriever_faiss.adapter import FaissVisualRetriever
    from capicove_retriever_faiss.codec import decode_visual_manifest
    from capicove_retriever_faiss.config import FaissConfiguration

    from capicove.contracts import SearchFilters
    from capicove.retrieval.video_models import PinnedVisualIndexGeneration

    manifest = decode_visual_manifest(json.loads((physical / "build-manifest.json").read_text()))
    vectors = np.load(args.vectors or root / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.shape != (manifest.expected_unit_count, manifest.embedding_space.dimension):
        raise ValueError("reference vector shape mismatch")
    known = np.random.default_rng(20260916).integers(0, len(vectors), size=1000)
    queries = np.asarray(vectors[known]).copy()
    pin = PinnedVisualIndexGeneration(
        manifest.generation_id,
        manifest.manifest_sha256,
        str(physical),
        manifest.target_type,
        manifest.embedding_space,
    )
    records = []
    process = psutil.Process()
    for ef in args.ef_search:
        print(f"opening frozen index with efSearch={ef}", flush=True)
        backend = FaissVisualRetriever(FaissConfiguration(physical.parent, threads=4, ef_search=ef))
        found_count = 0
        latencies = []
        rss = []
        try:
            for i in range(1100):
                ordinal = i % 1000
                started = time.perf_counter_ns()
                result = backend.retrieve(
                    query_vector=tuple(map(float, queries[ordinal])),
                    embedding_space=manifest.embedding_space,
                    generation=pin,
                    filters=SearchFilters(),
                    top_k=20,
                )
                if i >= 100:
                    latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                    found_count += f"{args.unit_prefix}{known[ordinal]}" in {
                        item.unit_id for item in result
                    }
                    rss.append(process.memory_info().rss)
        finally:
            backend.close()
        record = {
            "ef_search": ef,
            "known_neighbor_recovery_at_20": found_count / 1000,
            "p95_ms": float(np.percentile(latencies, 95)),
            "rss_max_bytes": max(rss),
            "steady_rss_growth_fraction": max(0, max(rss) / rss[0] - 1),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    with args.report.open("x") as stream:
        json.dump(
            {
                "manifest_sha256": manifest.manifest_sha256,
                "query_seed": 20260916,
                "query_count": 1000,
                "unit_prefix": args.unit_prefix,
                "dimension": manifest.embedding_space.dimension,
                "keyframe_count": manifest.expected_unit_count,
                "warmup_count": 100,
                "full_random_recall_measured": False,
                "runs": records,
            },
            stream,
            indent=2,
            allow_nan=False,
        )


if __name__ == "__main__":
    main()
