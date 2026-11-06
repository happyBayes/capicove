"""Independent synthetic timeline oracles for the local media quality Tool."""

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from capicove.contracts import InvocationContext, ToolStatus
from capicove.tools.video_quality import VIDEO_QUALITY_SPEC, build_video_quality_tool_handler


pytestmark = pytest.mark.integration


@pytest.fixture
def media(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("local FFmpeg and FFprobe required")
    path = tmp_path / "fixture.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=f=1000:r=48000:d=2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=f=1000:r=48000:d=2",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];[3:a][4:a][5:a]concat=n=3:v=0:a=1[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return path


def context(permissions: frozenset[str] | None = None) -> InvocationContext:
    return InvocationContext(
        request_id="quality-request",
        trace_id="quality-trace",
        workspace_id="test",
        actor_id="tester",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(seconds=30),
        permissions=VIDEO_QUALITY_SPEC.required_permissions if permissions is None else permissions,
    )


def test_full_inspection_has_independent_intervals_and_coverage(media: Path) -> None:
    before = media.read_bytes()
    handler = build_video_quality_tool_handler("test", (media.parent,))
    result = handler(
        context(), {"path": str(media), "detector_options": {"freeze_min_seconds": 1.0}}
    )
    assert result.status is ToolStatus.SUCCEEDED, result
    report: dict[str, Any] = dict(result.result or {})
    assert report["verdict"] == "review"
    checks = {c["check_id"]: c for c in report["checks"]}
    assert checks["video_format"]["metrics"]["display_width"]["value"] == 160
    assert checks["decode"]["metrics"]["video_frames"]["value"] == 60
    for name in ("black", "silence"):
        event = checks[name]["events"][0]
        assert event["start_seconds"] == pytest.approx(2, abs=0.03)
        assert event["end_seconds"] == pytest.approx(4, abs=0.03)
        assert checks[name]["metrics"]["fraction"]["value"] == pytest.approx(1 / 3, abs=0.01)
    assert checks["freeze"]["events"][-1]["end_seconds"] == pytest.approx(6, abs=0.11)
    assert checks["freeze"]["events"][-1]["closed_by_eof"] is True
    assert checks["loudness"]["metrics"]["true_peak_dbtp"]["value"] == pytest.approx(
        -18.06, abs=0.2
    )
    assert media.read_bytes() == before
    assert report["source"]["sha256"] in result.evidence_refs[0]
    json.dumps(report, allow_nan=False)


def test_policy_violation_is_completed_inspection(media: Path) -> None:
    handler = build_video_quality_tool_handler("test", (media.parent,))
    result = handler(
        context(),
        {"path": str(media), "checks": ["video_format"], "policy": {"min_display_width": 1280}},
    )
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["verdict"] == "fail"
    assert cast(dict[str, Any], result.result["coverage"])["decoded"] is False


def test_no_audio_is_not_applicable_but_required_audio_fails(media: Path, tmp_path: Path) -> None:
    silent = tmp_path / "no-audio.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(media), "-an", "-c:v", "copy", str(silent)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    result = handler(context(), {"path": str(silent), "policy": {"require_audio": True}})
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["verdict"] == "fail"
    checks = {c["check_id"]: c for c in cast(list[dict[str, Any]], result.result["checks"])}
    assert checks["loudness"]["status"] == "not_applicable"


def test_offline_manifest_and_symlink_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "video.mp4"
    manifest.write_text("#EXTM3U\nhttp://127.0.0.1:9999/private.ts\n")
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    assert handler(context(), {"path": str(manifest)}).status is ToolStatus.REJECTED
    link = tmp_path / "link.mp4"
    link.symlink_to(manifest)
    assert handler(context(), {"path": str(link)}).status is ToolStatus.REJECTED


def test_permission_rejection_precedes_source_access(tmp_path: Path) -> None:
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    result = handler(context(frozenset()), {"path": str(tmp_path / "missing.mp4")})
    assert result.status is ToolStatus.REJECTED
    assert result.error is not None
    assert result.error.code.value == "permission_denied"


def test_trailing_black_is_closed_at_observed_eof(tmp_path: Path) -> None:
    path = tmp_path / "black.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=10:d=2.5",
            "-c:v",
            "ffv1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    result = handler(context(), {"path": str(path), "checks": ["black"]})
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    check = next(
        c for c in cast(list[dict[str, Any]], result.result["checks"]) if c["check_id"] == "black"
    )
    assert check["events"][0]["end_seconds"] == pytest.approx(2.5, abs=0.01)
    assert check["events"][0]["closed_by_eof"]
    assert check["metrics"]["fraction"]["value"] == pytest.approx(1)


def test_corrupt_media_never_reports_pass(media: Path, tmp_path: Path) -> None:
    broken = tmp_path / "truncated.mkv"
    data = media.read_bytes()
    broken.write_bytes(data[: len(data) // 2])
    result = build_video_quality_tool_handler("test", (tmp_path,))(context(), {"path": str(broken)})
    assert result.status is ToolStatus.FAILED
    if result.result:
        assert result.result["verdict"] == "incomplete"


def test_missing_ffmpeg_keeps_completed_probe_diagnostics(media: Path) -> None:
    handler = build_video_quality_tool_handler(
        "test", (media.parent,), ffmpeg="missing-capicove-ffmpeg"
    )
    result = handler(context(), {"path": str(media)})
    assert result.status is ToolStatus.FAILED
    assert result.result is not None
    checks = cast(list[dict[str, Any]], result.result["checks"])
    assert next(c for c in checks if c["check_id"] == "video_format")["status"] == "completed"
    assert result.result["verdict"] == "incomplete"


def test_ambiguous_tracks_require_explicit_selection(media: Path, tmp_path: Path) -> None:
    multiple = tmp_path / "multiple.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-map",
            "0:a:0",
            "-c",
            "copy",
            str(multiple),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    assert handler(context(), {"path": str(multiple)}).status is ToolStatus.REJECTED
    selected = handler(context(), {"path": str(multiple), "audio_stream_index": 2})
    assert selected.status is ToolStatus.SUCCEEDED
    assert selected.result is not None
    assert cast(dict[str, Any], selected.result["selection"])["audio_index"] == 2
    assert (
        handler(context(), {"path": str(multiple), "audio_stream_index": 0}).status
        is ToolStatus.REJECTED
    )


def test_observed_timing_preserves_audio_start_offset(media: Path, tmp_path: Path) -> None:
    shifted = tmp_path / "offset.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media),
            "-itsoffset",
            "0.5",
            "-i",
            str(media),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            str(shifted),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    result = build_video_quality_tool_handler("test", (tmp_path,))(
        context(), {"path": str(shifted), "checks": ["av_timing"]}
    )
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    check = next(
        c
        for c in cast(list[dict[str, Any]], result.result["checks"])
        if c["check_id"] == "av_timing"
    )
    assert check["metrics"]["start_delta_seconds"]["value"] == pytest.approx(0.5, abs=0.001)
    assert check["metrics"]["end_delta_seconds"]["value"] == pytest.approx(0.5, abs=0.001)


def test_silent_track_produces_finite_json_and_explicit_unknowns(tmp_path: Path) -> None:
    path = tmp_path / "silent.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=10:d=3",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=3",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    result = handler(context(), {"path": str(path), "checks": ["loudness", "silence"]})
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    check = next(
        c
        for c in cast(list[dict[str, Any]], result.result["checks"])
        if c["check_id"] == "loudness"
    )
    assert check["metrics"]["integrated_lufs"]["value"] is None
    assert check["metrics"]["integrated_lufs"]["reason"]
    json.dumps(dict(result.result), allow_nan=False)
    required = handler(
        context(),
        {"path": str(path), "checks": ["loudness"], "policy": {"min_integrated_lufs": -24}},
    )
    assert required.status is ToolStatus.FAILED


def test_report_matches_published_json_schema(media: Path) -> None:
    import importlib

    Draft202012Validator = importlib.import_module("jsonschema").Draft202012Validator

    from capicove.tools.video_quality import VIDEO_QUALITY_OUTPUT_SCHEMA

    result = build_video_quality_tool_handler("test", (media.parent,))(
        context(), {"path": str(media)}
    )
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    Draft202012Validator(VIDEO_QUALITY_OUTPUT_SCHEMA).validate(dict(result.result))


def test_execution_limits_are_part_of_configuration_identity(media: Path) -> None:
    from capicove.processing.video.quality import QualityLimits

    results = []
    for events in (1, 2):
        handler = build_video_quality_tool_handler(
            "test", (media.parent,), limits=QualityLimits(max_events=events)
        )
        result = handler(context(), {"path": str(media), "checks": ["streams"]})
        assert result.result is not None
        results.append(cast(dict[str, Any], result.result["configuration"])["sha256"])
    assert results[0] != results[1]


def test_per_channel_silence_differs_from_all_channel_silence(tmp_path: Path) -> None:
    path = tmp_path / "one-quiet-channel.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=10:d=3",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.1*sin(2*PI*1000*t)|0:s=48000:d=3",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    handler = build_video_quality_tool_handler("test", (tmp_path,))
    counts = []
    for per_channel in (False, True):
        result = handler(
            context(),
            {
                "path": str(path),
                "checks": ["silence"],
                "detector_options": {"silence_per_channel": per_channel},
            },
        )
        assert result.status is ToolStatus.SUCCEEDED
        assert result.result is not None
        check = next(
            c
            for c in cast(list[dict[str, Any]], result.result["checks"])
            if c["check_id"] == "silence"
        )
        counts.append(check["metrics"]["event_count"]["value"])
        if per_channel:
            assert check["events"][0]["channel"] is not None
            assert check["events"][0]["duration_seconds"] == pytest.approx(3, abs=0.01)
    assert counts == [0, 1]
