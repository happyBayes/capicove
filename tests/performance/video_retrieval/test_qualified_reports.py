"""Opt-in gates over raw reports produced by the native benchmark commands.

This validates a completed run; it never fabricates reports or downloads assets.
The procedure documents the commands that produce these local artifacts.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest


pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        os.environ.get("CAPICOVE_VIDEO_QUALIFICATION") != "1",
        reason="explicit completed native qualification reports required",
    ),
]


def report(variable: str) -> dict[str, Any]:
    path = Path(os.environ[variable])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("absolute regular report path required")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("report object required")
    return value


def test_native_ann_recall_and_representative_subset() -> None:
    scale = report("CAPICOVE_VIDEO_ANN_SCALE_REPORT")
    reopened = report("CAPICOVE_VIDEO_ANN_REOPEN_REPORT")
    subset = report("CAPICOVE_VIDEO_ANN_SUBSET_REPORT")
    runtime = report("CAPICOVE_VIDEO_RUNTIME_REPORT")
    assert scale["count"] == 1_000_000
    assert scale["dimension"] == 768
    assert reopened["query_count"] >= 1000
    assert reopened["manifest_sha256"] == runtime["keyframe_index_manifest_sha256"]
    assert reopened["dimension"] == scale["dimension"]
    assert reopened["keyframe_count"] == scale["count"]
    assert all(run["known_neighbor_recovery_at_20"] >= 0.95 for run in reopened["runs"])
    assert subset["count"] == 10_000
    assert subset["dimension"] == scale["dimension"]
    assert subset["full_corpus_recall_at_20"] >= 0.95
    scale_path = Path(os.environ["CAPICOVE_VIDEO_ANN_SCALE_REPORT"]).parent / "vectors.npy"
    subset_path = Path(os.environ["CAPICOVE_VIDEO_ANN_SUBSET_REPORT"]).parent / "vectors.npy"
    large = np.load(scale_path, mmap_mode="r", allow_pickle=False)
    small = np.load(subset_path, mmap_mode="r", allow_pickle=False)
    assert np.array_equal(large[: len(small)], small)


def test_full_serving_memory_and_lease_transition() -> None:
    measured = report("CAPICOVE_VIDEO_RUNTIME_REPORT")
    assert measured["keyframe_count"] == 1_000_000
    assert measured["profile_ref"] == "video-local-ann"
    assert measured["opened_visual_memory_upper_bound_bytes"] <= 8 * 1024**3
    assert measured["sum_of_individual_peaks_bytes"] <= 16 * 1024**3
    assert measured["sample_count"] > 100
    assert 0 < measured["leased_transition_peak_combined_rss_bytes"] <= 16 * 1024**3
    assert measured["human_review"] == "pending"


def test_full_http_latency_is_bound_to_measured_generation() -> None:
    runtime = report("CAPICOVE_VIDEO_RUNTIME_REPORT")
    http_path = Path(os.environ["CAPICOVE_VIDEO_RUNTIME_REPORT"]).with_suffix(".http.json")
    assert hashlib.sha256(http_path.read_bytes()).hexdigest() == runtime["http_report_sha256"]
    measured = json.loads(http_path.read_text())
    assert measured["generation_manifest_sha256"] == runtime["generation_manifest_sha256"]
    assert measured["keyframe_count"] == 1_000_000
    assert measured["profile_ref"] == "video-local-ann"
    assert measured["network_rtt_ms"] <= 20
    assert measured["steady_rss_growth_fraction"] <= 0.05
    assert {run["concurrency"] for run in measured["runs"]} == {1, 4}
    for run in measured["runs"]:
        assert run["warmup_requests"] >= 100
        assert run["measured_requests"] >= 1000
        assert run["failed_requests"] == run["degraded_requests"] == 0
        assert run["maximum_response_bytes"] <= 16384
        assert run["p95_ms"] < 500
