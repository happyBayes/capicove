"""RFC-0016 amendment 1: batch video operations through the host video handlers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from capicove.mcp.bindings import invoke_binding
from capicove.runtime.mcp_host import compose_mcp_host
from tests.unit.runtime.test_batch_mcp import BatchSpace, run_worker_until
from tests.unit.runtime.test_video_mcp import FFMPEG, FFPROBE, _render


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="host media handoff is POSIX-only (ADR-0026)"),
    pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs local FFmpeg"),
]
SPLIT_FP = "1" * 64
FILTER_FP = "2" * 64


@pytest.fixture
def space(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory) -> BatchSpace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    _render(clips / "moving.mp4", "testsrc=duration=4:size=64x48:rate=10")
    _render(clips / "still.mp4", "color=c=gray:duration=4:size=64x48:rate=10")
    result = BatchSpace(root, clips, root / "host.json")
    write(result)
    return result


def write(space: BatchSpace, **video_overrides: object) -> None:
    video: dict[str, object] = {
        "schema_version": 1,
        "read_roots": [str(space.images)],
        "staging_root": str(space.root / "staging"),
        "backend": "ffmpeg",
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "max_input_bytes": 64 * 1024 * 1024,
    }
    video.update(video_overrides)
    batch = space.batch(
        input_roots={"clips": str(space.images)},
        operations=["video_filter", "video_split"],
        provider_fingerprints={"video_filter": FILTER_FP, "video_split": SPLIT_FP},
    )
    body = space.body(batch)
    body["capabilities"] = {"batch": batch, "video": video}
    space.config.write_text(json.dumps(body))
    space.config.chmod(0o600)


def call(space: BatchSpace, tool: str, arguments: Mapping[str, object]) -> dict[str, Any]:
    with compose_mcp_host(space.load()) as composed:
        bound = {item.spec.tool_id: item for item in composed}
        return invoke_binding(bound[f"capicove.batch.{tool}"], arguments)


def queue(space: BatchSpace, operation: str, parameters: dict[str, object], fp: str) -> str:
    selectors: dict[str, object] = {
        "input_root": "clips",
        "operation": operation,
        "parameters": parameters,
        "provider_fingerprint": fp,
        "output_root": "results",
    }
    planned = call(space, "plan", selectors)
    assert planned["status"] == "succeeded", planned
    queued = call(
        space,
        "enqueue",
        {**selectors, "plan_id": planned["result"]["plan_id"], "idempotency_key": operation},
    )
    assert queued["status"] == "succeeded", queued
    return str(queued["result"]["job"]["job_id"])


def finished(space: BatchSpace, job_id: str) -> bool:
    state = call(space, "status", {"job_id": job_id})["result"]["job"]["state"]
    return state in {"succeeded", "failed"}


def test_batch_split_writes_per_item_outputs_with_pinned_ffmpeg(space: BatchSpace) -> None:
    job_id = queue(
        space,
        "video_split",
        {"strategy": "duration", "config": {"split_duration": 2}},
        SPLIT_FP,
    )
    sources = {path.name: path.read_bytes() for path in space.images.iterdir()}

    run_worker_until(space, lambda: finished(space, job_id), timeout=120)

    status = call(space, "status", {"job_id": job_id})["result"]
    assert status["job"]["state"] == "succeeded", status
    results = space.root / "results"
    for item in status["items"]:
        segments = item["result"]["segments"]
        assert len(segments) == 2
        for segment in segments:
            path = Path(segment["path"])
            assert path.parent.parent == results
            assert path.parent.stat().st_mode & 0o077 == 0
            assert hashlib.sha256(path.read_bytes()).hexdigest() == segment["sha256"]
        assert item["result"]["original"]["path"].startswith(str(space.images))
    assert str(space.root / "staging") not in json.dumps(status)
    assert {path.name: path.read_bytes() for path in space.images.iterdir()} == sources


def test_batch_filter_uses_host_criteria(space: BatchSpace) -> None:
    job_id = queue(
        space,
        "video_filter",
        {"criteria": [{"criterion_id": "long", "type": "metadata", "min_duration_seconds": 1}]},
        FILTER_FP,
    )

    run_worker_until(space, lambda: finished(space, job_id), timeout=120)

    status = call(space, "status", {"job_id": job_id})["result"]
    assert status["job"]["state"] == "succeeded", [i["failure_code"] for i in status["items"]]
    assert [item["state"] for item in status["items"]] == ["succeeded", "succeeded"]


@pytest.mark.parametrize(
    ("operation", "parameters", "fp"),
    [
        ("video_filter", {"criteria": [{"type": "clip_similarity"}]}, FILTER_FP),
        ("video_split", {"strategy": "duration", "config": {}, "backend": "av"}, SPLIT_FP),
        ("video_split", {"strategy": "duration"}, SPLIT_FP),
        (
            "video_filter",
            {"criteria": [{"type": "metadata", "min_duration_seconds": 1}]},
            FILTER_FP,
        ),
        ("video_split", {"strategy": "fastest", "config": {}}, SPLIT_FP),
    ],
)
def test_batch_video_parameters_are_closed(
    space: BatchSpace, operation: str, parameters: dict[str, object], fp: str
) -> None:
    result = call(
        space,
        "plan",
        {
            "input_root": "clips",
            "operation": operation,
            "parameters": parameters,
            "provider_fingerprint": fp,
            "output_root": "results",
        },
    )
    assert result["error"]["code"] == "invalid_input"


def test_split_plan_requires_an_output_root(space: BatchSpace) -> None:
    result = call(
        space,
        "plan",
        {
            "input_root": "clips",
            "operation": "video_split",
            "parameters": {"strategy": "duration", "config": {"split_duration": 2}},
            "provider_fingerprint": SPLIT_FP,
        },
    )
    assert result["error"]["code"] == "invalid_input"


def test_staging_inside_batch_roots_is_rejected(space: BatchSpace) -> None:
    write(space, staging_root=str(space.root / "results" / "staging"))
    with pytest.raises(ValueError, match="disjoint"):
        with compose_mcp_host(space.load()):
            pass


def test_replaced_ffmpeg_fails_the_item_without_running_it(
    space: BatchSpace, tmp_path: Path
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    assert FFMPEG is not None
    shutil.copyfile(FFMPEG, tools / "ffmpeg")
    (tools / "ffmpeg").chmod(0o755)
    write(space, ffmpeg=str(tools / "ffmpeg"))
    job_id = queue(
        space,
        "video_split",
        {"strategy": "duration", "config": {"split_duration": 2}},
        SPLIT_FP,
    )
    from capicove.runtime.batch_mcp import run_worker, worker_runtime

    runtime = worker_runtime(space.load())
    (tools / "ffmpeg").write_text("#!/bin/sh\ntouch " + str(tmp_path / "ran") + "\n")
    from threading import Event

    run_worker(runtime, Event(), until_idle=True, report=lambda line: None)

    status = call(space, "status", {"job_id": job_id})["result"]
    assert status["job"]["state"] == "failed"
    assert {item["failure_code"] for item in status["items"]} == {"unavailable"}
    assert not (tmp_path / "ran").exists()
