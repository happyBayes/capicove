"""Opt-in real EmbeddingGemma 2 Q8_0 llama.cpp replay of the audio-local loop.

``CAPICOVE_EMBEDDINGGEMMA2_MODEL_CONFIG`` selects the model configuration written by
``scripts/embeddinggemma_cpu.py receipt``. Inputs are synthetic signals, so this
qualifies real local inference through build, search, evidence and reopen; it is
not semantic retrieval quality, which needs labeled representative media.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import struct
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import (
    AudioClipQuery,
    DetailLevel,
    FullHit,
    LocalPathSource,
    SearchRequest,
    SearchStatus,
    TargetType,
    TemporalRange,
    TextQuery,
)
from capicove.runtime import unified_retrieval as runtime
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config


pytestmark = pytest.mark.model_small


def _write(path: Path, samples: list[float], rate: int = 16_000) -> bytes:
    pcm = struct.pack(f"<{len(samples)}h", *(int(max(-1, min(1, s)) * 20000) for s in samples))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(pcm)
    return pcm


@pytest.fixture(scope="module")
def model_config() -> Path:
    selected = os.environ.get("CAPICOVE_EMBEDDINGGEMMA2_MODEL_CONFIG")
    if not selected:
        pytest.skip("Set CAPICOVE_EMBEDDINGGEMMA2_MODEL_CONFIG to replay the real model")
    path = Path(selected)
    assert path.is_absolute() and path.is_file()
    return path


def test_real_model_build_search_evidence_and_reopen(model_config: Path, tmp_path: Path) -> None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    root = (tmp_path / "workspace").resolve()
    media = root / "media"
    media.mkdir(parents=True)
    rate = 16_000
    tone = [0.5 * math.sin(2 * math.pi * 440 * i / rate) for i in range(31 * rate)]
    generator = random.Random(7)
    noise = [generator.uniform(-0.6, 0.6) for _ in range(6 * rate)]
    tone_pcm = _write(media / "tone.wav", tone)
    _write(media / "noise.wav", noise)
    with wave.open(str(media / "query.wav"), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(tone_pcm[400_000 * 2 : 496_000 * 2])
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {
            "multimodal": {
                "schema_version": 1,
                "profile": "audio-local-v1",
                "source_roots": {"media": str(media)},
                "index_root": str(root / "index"),
                "staging_root": str(root / "staging"),
                "corpus_ref": "corpus:synthetic",
                "model_config": str(model_config),
                "model_binding": "unified",
                "ffmpeg": ffmpeg,
                "ffprobe": ffprobe,
                "max_source_bytes": 64 * 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    host = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(host, WorkspaceHostConfiguration)
    for name in ("tone.wav", "noise.wav"):
        runtime.import_media(host, "media", name)
    built = runtime.build_generation(host)
    assert built["units"] == 3
    runtime.activate_generation(host, built["generation_id"])
    rebuilt = runtime.build_generation(host)
    assert rebuilt == {**built, "reused": True}

    service = runtime.open_unified_retrieval(host)
    try:
        _replay(service, media)
    finally:
        service.close()
    reopened_service = runtime.open_unified_retrieval(host)
    try:
        reopened = reopened_service.search(_clip_request(media))
        assert reopened.hits[0].time_range == TemporalRange(25_000, 31_000)
        assert len(reopened.hits) == 3
    finally:
        reopened_service.close()


def _clip_request(media: Path, detail: DetailLevel = DetailLevel.COMPACT) -> SearchRequest:
    return SearchRequest(
        AudioClipQuery(LocalPathSource(str(media / "query.wav"))),
        frozenset({TargetType.AUDIO_SEGMENT}),
        top_k=3,
        detail_level=detail,
        retrieval_profile_ref="audio-local-v1",
    )


def _replay(service: runtime.UnifiedRetrievalRuntime, media: Path) -> None:
    clip = service.search(_clip_request(media, DetailLevel.FULL))
    assert clip.status is SearchStatus.SUCCEEDED
    first = clip.hits[0]
    assert isinstance(first, FullHit)
    assert first.time_range == TemporalRange(25_000, 31_000)
    assert first.lane_scores[0].raw_score > 0.999
    text = service.search(
        SearchRequest(
            TextQuery("white noise static hiss"),
            frozenset({TargetType.AUDIO_SEGMENT}),
            top_k=3,
            detail_level=DetailLevel.FULL,
            retrieval_profile_ref="audio-local-v1",
        )
    )
    assert text.status is SearchStatus.SUCCEEDED and len(text.hits) == 3
    assert all(isinstance(h, FullHit) for h in text.hits)
    assert all(
        math.isfinite(h.lane_scores[0].raw_score) for h in text.hits if isinstance(h, FullHit)
    )
    context = InvocationContext(
        request_id="r",
        trace_id="t",
        workspace_id="workspace",
        actor_id="owner",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(seconds=60),
        permissions=frozenset({"evidence:read"}),
    )
    evidence = service.read_evidence(context, first.evidence_id, ("metadata", "lineage"))
    assert evidence["metadata"]["source_ref"] == "media:tone.wav"
    assert evidence["lineage"]["space_id"].startswith("embeddinggemma2-llamacpp:")


def test_real_model_unified_text_image_audio_loop(model_config: Path, tmp_path: Path) -> None:
    import io

    from PIL import Image, ImageDraw

    from capicove.contracts.retrieval import ImageQuery

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    root = (tmp_path / "workspace").resolve()
    media = root / "media"
    media.mkdir(parents=True)
    (media / "notes.md").write_text(
        "Quarterly invoice QX-90417 for the harbour crane is overdue.\n", encoding="utf-8"
    )
    for name, color in (("red.png", (220, 20, 20)), ("blue.png", (20, 40, 220))):
        image = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(image).ellipse((40, 40, 184, 184), fill=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        (media / name).write_bytes(buffer.getvalue())
    rate = 16_000
    _write(
        media / "tone.wav", [0.5 * math.sin(2 * math.pi * 440 * i / rate) for i in range(2 * rate)]
    )
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {
            "multimodal": {
                "schema_version": 1,
                "profile": "unified-local-v1",
                "source_roots": {"media": str(media)},
                "index_root": str(root / "index"),
                "staging_root": str(root / "staging"),
                "corpus_ref": "corpus:synthetic-mixed",
                "model_config": str(model_config),
                "model_binding": "unified",
                "ffmpeg": ffmpeg,
                "ffprobe": ffprobe,
                "max_source_bytes": 64 * 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    host = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(host, WorkspaceHostConfiguration)
    for name in ("notes.md", "red.png", "blue.png", "tone.wav"):
        runtime.import_media(host, "media", name)
    built = runtime.build_generation(host)
    assert built["units"] == 4
    runtime.activate_generation(host, built["generation_id"])
    targets = frozenset({TargetType.TEXT_CHUNK, TargetType.IMAGE, TargetType.AUDIO_SEGMENT})
    service = runtime.open_unified_retrieval(host)
    try:
        exact = service.search(
            SearchRequest(
                ImageQuery(LocalPathSource(str(media / "red.png"))),
                targets,
                top_k=4,
                detail_level=DetailLevel.FULL,
                retrieval_profile_ref="unified-local-v1",
            )
        )
        assert exact.status is SearchStatus.SUCCEEDED
        top = exact.hits[0]
        assert isinstance(top, FullHit) and top.target_type is TargetType.IMAGE
        assert top.lane_scores[0].raw_score > 0.99
        identifier = service.search(
            SearchRequest(
                TextQuery("QX-90417"),
                targets,
                top_k=4,
                detail_level=DetailLevel.FULL,
                retrieval_profile_ref="unified-local-v1",
            )
        )
        first = identifier.hits[0]
        assert isinstance(first, FullHit) and first.target_type is TargetType.TEXT_CHUNK
        assert {lane.lane_id for lane in first.lane_scores} == {"shared_dense", "lexical"}
        assert {hit.target_type for hit in identifier.hits} == targets
    finally:
        service.close()


def test_real_model_video_keyframes_and_soundtrack(model_config: Path, tmp_path: Path) -> None:
    """Image queries find the matching interval keyframe of a real two-scene video."""
    import io
    import subprocess

    from PIL import Image

    from capicove.contracts.retrieval import ImageQuery

    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    root = (tmp_path / "workspace").resolve()
    media = root / "media"
    media.mkdir(parents=True)
    # Scene 1 (0-10 s) is solid red, scene 2 (10-20 s) solid blue, with a tone track.
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=224x224:d=10:r=5,format=yuv420p[a];"
            "color=c=blue:s=224x224:d=10:r=5,format=yuv420p[b];[a][b]concat=n=2:v=1:a=0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=20",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(media / "scenes.mp4"),
        ],
        check=True,
        timeout=120,
    )
    for name, color in (("red.png", (255, 0, 0)), ("blue.png", (0, 0, 255))):
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), color).save(buffer, format="PNG")
        (media / name).write_bytes(buffer.getvalue())
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {
            "multimodal": {
                "schema_version": 1,
                "profile": "unified-local-v1",
                "source_roots": {"media": str(media)},
                "index_root": str(root / "index"),
                "staging_root": str(root / "staging"),
                "corpus_ref": "corpus:synthetic-video",
                "model_config": str(model_config),
                "model_binding": "unified",
                "ffmpeg": ffmpeg,
                "ffprobe": ffprobe,
                "max_source_bytes": 64 * 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    host = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(host, WorkspaceHostConfiguration)
    imported = runtime.import_media(host, "media", "scenes.mp4")
    assert imported["modality"] == "video" and imported["audio_stream_index"] == 0
    built = runtime.build_generation(host)
    assert built["units"] == 3  # keyframes at 0 s and 10 s plus one soundtrack window
    runtime.activate_generation(host, built["generation_id"])
    targets = frozenset({TargetType.KEYFRAME, TargetType.AUDIO_SEGMENT})
    service = runtime.open_unified_retrieval(host)
    try:
        for name, expected in (("red.png", 0), ("blue.png", 10_000)):
            result = service.search(
                SearchRequest(
                    ImageQuery(LocalPathSource(str(media / name))),
                    targets,
                    top_k=3,
                    detail_level=DetailLevel.FULL,
                    retrieval_profile_ref="unified-local-v1",
                )
            )
            assert result.status is SearchStatus.SUCCEEDED
            top = result.hits[0]
            assert isinstance(top, FullHit) and top.target_type is TargetType.KEYFRAME
            assert top.timestamp_ms == expected, name
    finally:
        service.close()
