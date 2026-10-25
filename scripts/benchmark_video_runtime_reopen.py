#!/usr/bin/env python3
"""Cold-process qualification of an explicitly provisioned local test runtime.

Reuses immutable visual/text members from the authored integration fixture.
Registers a separate ANN profile set in its test catalog, never rebuilds indexes.
Raw reports stay local. This is runtime engineering evidence, not model quality.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import BoundedSemaphore, Event, Thread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-pid", type=int, required=True)
    parser.add_argument("--text-pid", type=int, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.model_pid <= 0 or args.text_pid <= 0:
        parser.error("explicit positive local service PIDs required")
    if len(args.model_manifest_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.model_manifest_sha256
    ):
        parser.error("verified model manifest SHA-256 required")
    if not (args.root / "sets.sqlite3").is_file() or args.report.exists():
        parser.error("existing authored runtime and fresh report path required")

    if sys.platform not in {"linux", "darwin"}:
        parser.error("runtime memory qualification requires Linux or macOS")
    import resource

    import psutil
    from capicove_retriever_elasticsearch.adapter import ElasticsearchTextRetriever
    from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration
    from capicove_retriever_faiss.adapter import FaissVisualRetriever
    from capicove_retriever_faiss.config import FaissConfiguration

    from capicove.model import load_packaged_manifest
    from capicove.retrieval.video_generations import VideoTextMemberValidator, open_video_runtime
    from capicove.retrieval.video_http_encoding import LocalSiglip2QueryEncoder
    from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry
    from capicove.retrieval.video_search import VideoSearchService

    # Start sampling before cold native opens. IDs are explicit; no process discovery.
    processes = [psutil.Process(), psutil.Process(args.model_pid), psutil.Process(args.text_pid)]
    stop = Event()
    samples = []
    sampling_errors = []

    def sample() -> None:
        while not stop.is_set():
            try:
                samples.append(tuple(process.memory_info().rss for process in processes))
            except psutil.Error as error:
                sampling_errors.append(type(error).__name__)
                return
            stop.wait(0.1)

    sampler = Thread(target=sample, daemon=True)
    sampler.start()
    runtime = None
    started = time.monotonic()
    try:
        text = ElasticsearchTextRetriever(
            ElasticsearchConfiguration(
                hosts=(os.environ["CAPICOVE_VIDEO_ES_ENDPOINT"],),
                ca_certs=os.environ["CAPICOVE_VIDEO_ES_CA"],
                index_prefix="capicove-video-text",
                active_alias="capicove-video-text-active",
            )
        )
        runtime = open_video_runtime(
            generation_database_path=args.root / "sets.sqlite3",
            visual_backend=FaissVisualRetriever(
                FaissConfiguration(args.root / "visual", threads=4, ef_search=1024)
            ),
            validate_text_members=VideoTextMemberValidator(
                asr=text.generation_lifecycle, ocr=text.generation_lifecycle
            ),
        )
        original = runtime.generation_sets.activate("full-set")
        if original.manifest.keyframe_visual.expected_unit_count != 1_000_000:
            raise ValueError("this profile requires the full million-keyframe fixture")
        manifest = replace(
            original.manifest,
            set_id="full-set-ann-qualified-candidate",
            profile_ref="video-local-ann",
        )
        pin = replace(
            original.pin,
            set_id=manifest.set_id,
            profile_ref=manifest.profile_ref,
            manifest_sha256=manifest.manifest_sha256,
        )
        runtime.generation_sets.register(manifest, pin)
        runtime.generation_sets.activate(manifest.set_id)
        transition_samples = []
        with runtime.generation_sets.lease_active(
            workspace_ref="qualification", profile_ref="video-local-ann"
        ) as old_pin:
            successor = replace(manifest, set_id="full-set-ann-reopen-successor")
            successor_pin = replace(
                pin, set_id=successor.set_id, manifest_sha256=successor.manifest_sha256
            )
            runtime.generation_sets.register(successor, successor_pin)
            transition_start = len(samples)
            runtime.generation_sets.activate(successor.set_id)
            if runtime.generation_sets.can_evict(old_pin.set_id):
                raise RuntimeError("active query lease was not retained across activation")
            transition_samples = samples[transition_start:]
            manifest, pin = successor, successor_pin
        if not runtime.generation_sets.can_evict(old_pin.set_id):
            raise RuntimeError("released superseded generation remains leased")
        model = next(
            item
            for item in load_packaged_manifest().models
            if item.key == "siglip2-base-patch16-224"
        )
        encoder = LocalSiglip2QueryEncoder(
            endpoint=os.environ["CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"],
            model_id=model.repository,
            embedding_space=pin.keyframe_visual.embedding_space,
        )
        path = (
            Path(__file__).parents[1]
            / "plugins/retriever-elasticsearch/tests/integration/http_benchmark.py"
        )
        spec = importlib.util.spec_from_file_location("runtime_http_benchmark", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("HTTP benchmark helper unavailable")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        http_report = args.report.with_suffix(".http.json")
        os.environ["CAPICOVE_TEST_VIDEO_MODEL_PID"] = str(args.model_pid)
        with ThreadPoolExecutor(max_workers=4) as executor:
            service = VideoSearchService(
                workspace_ref="qualification",
                generation_sets=runtime.generation_sets,
                profiles=StaticVideoRetrievalProfileRegistry(),
                catalog_resolver=runtime.catalog_resolver,
                visual_retriever=runtime.visual_backend,
                query_encoder=encoder,
                executor=executor,
                concurrency=BoundedSemaphore(4),
                text_backends={"asr_lexical": text, "ocr_lexical": text},
            )
            helper.measure_local_http(
                service,
                http_report,
                keyframe_count=1_000_000,
                generation_manifest_sha256=manifest.manifest_sha256,
                profile_ref="video-local-ann",
            )
        gc.collect()
        opened_memory_upper_bound = processes[0].memory_info().rss
        stop.set()
        sampler.join(timeout=2)
        if sampling_errors or not samples or sampler.is_alive():
            raise RuntimeError("incomplete process-memory sampling")
        peaks = [max(row[i] for row in samples) for i in range(3)]
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        lifetime_peak = peak if sys.platform == "darwin" else peak * 1024
        conservative_total = max(peaks[0], lifetime_peak) + peaks[1] + peaks[2]
        report = {
            "schema_version": 1,
            "scope": "authored million-unit full runtime, cold process reopen",
            "generation_manifest_sha256": manifest.manifest_sha256,
            "profile_ref": manifest.profile_ref,
            "model_repository": model.repository,
            "model_revision": model.revision,
            "declared_verified_model_manifest_sha256": args.model_manifest_sha256,
            "synthetic_keyframe_vectors": True,
            "embedding_dimension": manifest.keyframe_visual.embedding_space.dimension,
            "keyframe_count": manifest.keyframe_visual.expected_unit_count,
            "native_threads": 4,
            "ef_search": 1024,
            "preprocessing_version": model.preprocess_revision,
            "keyframe_index_manifest_sha256": manifest.keyframe_visual.manifest_sha256,
            "segment_index_manifest_sha256": manifest.segment_visual.manifest_sha256,
            "unit_manifest_sha256": manifest.unit_manifest_sha256,
            "host": platform.platform(),
            "machine": platform.machine(),
            "host_ram_bytes": psutil.virtual_memory().total,
            "storage": "local-ssd",
            "sampling_interval_ms": 100,
            "sample_count": len(samples),
            "sampled_peak_rss_bytes_by_role": dict(
                zip(("search", "model", "text"), peaks, strict=True)
            ),
            "search_lifetime_peak_rss_bytes": lifetime_peak,
            "sum_of_individual_peaks_bytes": conservative_total,
            "opened_visual_memory_upper_bound_bytes": opened_memory_upper_bound,
            "leased_transition_peak_combined_rss_bytes": max(map(sum, transition_samples))
            if transition_samples
            else None,
            "leased_transition_shares_immutable_physical_members": True,
            "elapsed_seconds": time.monotonic() - started,
            "http_report_sha256": hashlib.sha256(http_report.read_bytes()).hexdigest(),
            "semantic_quality_measured": False,
            "human_review": "pending",
        }
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        if opened_memory_upper_bound > 8 * 1024**3 or conservative_total > 16 * 1024**3:
            raise RuntimeError("process memory qualification failed")
        print(json.dumps(report), flush=True)
    finally:
        stop.set()
        sampler.join(timeout=2)
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
