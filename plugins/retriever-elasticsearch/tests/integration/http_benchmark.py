"""Opt-in real HTTP measurement over a declared, frozen serving corpus."""

import gc
import hashlib
import importlib.util
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path
from threading import Thread

import uvicorn

from capicove.api.search import create_search_app
from capicove.retrieval.video_search import VideoSearchService


def measure_local_http(
    service: VideoSearchService,
    output: Path,
    *,
    keyframe_count: int,
    generation_manifest_sha256: str,
    profile_ref: str = "video-local-exact",
) -> None:
    """Measure the real public endpoint then stop the isolated server."""
    path = Path(__file__).parents[4] / "scripts" / "benchmark_video_search.py"
    spec = importlib.util.spec_from_file_location("capicove_video_http_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark module unavailable")
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        server = uvicorn.Server(
            uvicorn.Config(
                create_search_app(service),
                host="127.0.0.1",
                log_level="error",
                access_log=False,
            )
        )
        worker = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not server.started:
                raise RuntimeError("local search HTTP startup failed")
            client = benchmark.UrllibHttpClient(
                f"http://127.0.0.1:{listener.getsockname()[1]}/v1/search"
            )
            queries = tuple(
                benchmark.BenchmarkQuery(
                    query_id=f"authored-runtime-{i}",
                    payload={
                        "query": {"type": "text", "text": text},
                        "targets": ["video_segment"],
                        "top_k": 20,
                        "retrieval_profile_ref": profile_ref,
                    },
                )
                for i, text in enumerate(("authored", "different scene", "authored text"))
            )
            psutil = importlib.import_module("psutil")

            process = psutil.Process()
            model_pid = os.environ.get("CAPICOVE_TEST_VIDEO_MODEL_PID")
            model = psutil.Process(int(model_pid)) if model_pid else None
            gc.collect()
            started = time.perf_counter_ns()
            with socket.create_connection(listener.getsockname(), timeout=1):
                network_rtt_ms = (time.perf_counter_ns() - started) / 1_000_000
            runs = []
            rss_samples = []
            baselines = []

            def warmed_baseline() -> None:
                gc.collect()
                baselines.append(process.memory_info().rss)

            for concurrency in (1, 4):
                runs.append(
                    benchmark.measure_requests(
                        client,
                        queries,
                        concurrency=concurrency,
                        warmup_requests=100,
                        measured_requests=1000,
                        timeout_ms=30_000,
                        after_warmup=warmed_baseline,
                    )
                )
                gc.collect()
                rss_samples.append(process.memory_info().rss)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_bytes = peak if sys.platform == "darwin" else peak * 1024
            model_rss = model.memory_info().rss if model else None
            report = {
                "scope": "authored corpus; real model, FAISS, SQLite, Elasticsearch, HTTP",
                "keyframe_count": keyframe_count,
                "generation_manifest_sha256": generation_manifest_sha256,
                "profile_ref": profile_ref,
                "query_set_sha256": hashlib.sha256(
                    json.dumps(
                        [dict(query.payload) for query in queries],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "warmup_requests_per_run": 100,
                "measured_requests_per_run": 1000,
                "network_rtt_ms": network_rtt_ms,
                "search_peak_rss_bytes_including_build": peak_bytes,
                "model_process_rss_bytes": model_rss,
                "search_lifetime_peak_plus_model_observed_rss_bytes": peak_bytes + model_rss
                if model_rss
                else None,
                "post_warmup_rss_bytes": baselines,
                "steady_rss_growth_fraction": max(
                    0,
                    *(
                        after / before - 1
                        for before, after in zip(baselines, rss_samples, strict=True)
                    ),
                ),
                "million_vector_profile_qualified": False,
                "semantic_quality_measured": False,
                "runs": [run.to_dict() for run in runs],
            }
            with output.open("x") as stream:
                json.dump(report, stream, allow_nan=False, indent=2)
            for run in runs:
                print(
                    f"HTTP c={run.concurrency} p95={run.p95_ms:.3f}ms errors={run.failed_requests} degraded={run.degraded_requests} bytes={run.maximum_response_bytes}"
                )
                assert run.failed_requests == 0
                assert run.degraded_requests == 0
                assert run.maximum_response_bytes <= 16_384
                assert run.p95_ms < 500
            assert network_rtt_ms <= 20
            if keyframe_count == 1_000_000:
                assert model_rss is not None, "capacity measurement requires model-process identity"
                assert peak_bytes + model_rss <= 16 * 1024**3
                assert all(
                    after / before - 1 <= 0.05
                    for before, after in zip(baselines, rss_samples, strict=True)
                )
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            if worker.is_alive():
                raise RuntimeError("local search HTTP server did not stop")
