"""Unified-local video intake: interval keyframes and soundtrack windows (RFC-0032).

Clips are generated with the real FFmpeg; the fixture encoder seeds image vectors by
the frame bytes' SHA-256, so ``match:<frame sha256>`` is an independent dense oracle.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from capicove.contracts.retrieval import (
    DetailLevel,
    FullHit,
    MatchKind,
    SearchFilters,
    TargetType,
    TemporalRange,
    TextQuery,
)
from capicove.model.factory import ModelFactory
from capicove.runtime import unified_retrieval as runtime
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from tests.integration.retrieval.test_unified_search import (
    FFMPEG,
    _context,
    _models,
    _request,
    host,
)


__all__ = ["host"]
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="requires FFmpeg")
VIDEO_TARGETS = frozenset({TargetType.KEYFRAME, TargetType.AUDIO_SEGMENT})


def _clip(path: Path, *, seconds: int = 25, audio_streams: int = 1) -> Path:
    arguments = [
        str(FFMPEG),
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={seconds}:size=160x120:rate=5",
    ]
    for index in range(audio_streams):
        arguments += ["-f", "lavfi", "-i", f"sine=frequency={440 + 220 * index}:duration={seconds}"]
    arguments += ["-map", "0:v"]
    for index in range(audio_streams):
        arguments += ["-map", f"{index + 1}:a"]
    arguments += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(arguments, check=True, timeout=120)
    return path


def _build(host: WorkspaceHostConfiguration, tmp_path: Path) -> ModelFactory:
    models = _models(tmp_path)
    built = runtime.build_generation(host, models=models)
    runtime.activate_generation(host, built["generation_id"])
    return models


def test_video_yields_timestamped_keyframes_and_soundtrack_windows(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    media = host.workspace_root / "media"
    _clip(media / "clip.mp4")
    imported = runtime.import_media(host, "media", "clip.mp4")
    assert imported["modality"] == "video" and imported["audio_stream_index"] == 0
    assert imported["duration_ms"] == 25_000
    assert runtime.import_media(host, "media", "clip.mp4")["imported"] is False
    models = _build(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(_request(TextQuery("anything"), targets=VIDEO_TARGETS, top_k=20))
    keyframes = sorted(
        hit.timestamp_ms
        for hit in result.hits
        if hit.target_type is TargetType.KEYFRAME and hit.timestamp_ms is not None
    )
    audio = [hit for hit in result.hits if hit.target_type is TargetType.AUDIO_SEGMENT]
    assert keyframes == [0, 10_000, 20_000]
    # One 30-second window covers the decoded soundtrack (AAC adds a few ms of padding).
    assert len(audio) == 1 and audio[0].time_range is not None
    assert audio[0].time_range.start_ms == 0 and 25_000 <= audio[0].time_range.end_ms < 25_100
    keyframe = next(hit for hit in result.hits if hit.timestamp_ms == 10_000)
    evidence = service.read_evidence(_context(), keyframe.evidence_id, ("metadata", "lineage"))
    metadata = evidence["metadata"]
    assert metadata["kind"] == "keyframe" and metadata["timestamp_ms"] == 10_000
    assert metadata["media_type"] == "video/mp4"
    assert metadata["keyframe_policy"] == runtime.KEYFRAME_POLICY
    # Dense oracle: the frame's own seed retrieves exactly that keyframe.
    exact = service.search(
        _request(
            TextQuery(f"match:{metadata['frame_sha256']}"),
            targets=VIDEO_TARGETS,
            top_k=1,
            detail_level=DetailLevel.FULL,
        )
    )
    first = exact.hits[0]
    assert isinstance(first, FullHit) and first.timestamp_ms == 10_000
    assert MatchKind.VISUAL in first.match_kinds
    soundtrack = service.read_evidence(_context(), audio[0].evidence_id, ("metadata",))
    assert soundtrack["metadata"]["audio_stream_index"] == 0
    assert soundtrack["metadata"]["media_type"] == "video/mp4"
    service.close()


def test_time_filters_apply_to_keyframes_and_windows_before_ranking(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    media = host.workspace_root / "media"
    _clip(media / "clip.mp4")
    runtime.import_media(host, "media", "clip.mp4")
    models = _build(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(
        _request(
            TextQuery("anything"),
            targets=VIDEO_TARGETS,
            top_k=20,
            filters=SearchFilters(time_range=TemporalRange(5_000, 15_000)),
        )
    )
    assert [hit.timestamp_ms for hit in result.hits if hit.target_type is TargetType.KEYFRAME] == [
        10_000
    ]
    assert [hit.target_type for hit in result.hits].count(TargetType.AUDIO_SEGMENT) == 1
    service.close()


def test_silent_and_multi_track_videos_select_soundtracks_explicitly(
    host: WorkspaceHostConfiguration, tmp_path: Path
) -> None:
    media = host.workspace_root / "media"
    _clip(media / "silent.mp4", seconds=12, audio_streams=0)
    _clip(media / "dual.mkv", seconds=12, audio_streams=2)
    silent = runtime.import_media(host, "media", "silent.mp4")
    assert silent["audio_stream_index"] is None
    with pytest.raises(runtime.UnifiedRetrievalError, match="several audio streams"):
        runtime.import_media(host, "media", "dual.mkv")
    with pytest.raises(runtime.UnifiedRetrievalError, match="does not exist"):
        runtime.import_media(host, "media", "dual.mkv", audio_stream_index=2)
    dual = runtime.import_media(host, "media", "dual.mkv", audio_stream_index=1)
    assert dual["audio_stream_index"] == 1
    models = _build(host, tmp_path)
    service = runtime.open_unified_retrieval(host, models=models)
    result = service.search(_request(TextQuery("anything"), targets=VIDEO_TARGETS, top_k=20))
    by_source: dict[str, list[TargetType]] = {}
    for hit in result.hits:
        source = service.read_evidence(_context(), hit.evidence_id, ("metadata",))["metadata"][
            "source_ref"
        ]
        by_source.setdefault(source, []).append(hit.target_type)
    assert sorted(by_source["media:silent.mp4"]) == [TargetType.KEYFRAME, TargetType.KEYFRAME]
    assert by_source["media:dual.mkv"].count(TargetType.AUDIO_SEGMENT) == 1
    service.close()


def test_mismatched_or_oversized_videos_are_rejected_without_inventory_writes(
    host: WorkspaceHostConfiguration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = host.workspace_root / "media"
    (media / "fake.mp4").write_bytes(b"\x1a\x45\xdf\xa3 not an mp4 container")
    with pytest.raises(runtime.UnifiedRetrievalError, match="does not match"):
        runtime.import_media(host, "media", "fake.mp4")
    _clip(media / "long.mp4", seconds=25)
    monkeypatch.setattr(runtime, "MAX_KEYFRAMES_PER_VIDEO", 2)
    with pytest.raises(runtime.UnifiedRetrievalError, match="keyframe budget"):
        runtime.import_media(host, "media", "long.mp4")
    inventory = host.workspace_root / "index" / "inventory.sqlite3"
    if inventory.exists():
        import sqlite3

        with sqlite3.connect(inventory) as connection:
            assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
