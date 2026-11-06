"""Policy and schema regressions independent of media providers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.processing.video.quality import QualityLimits, QualityRequest
from capicove.processing.video.quality._adapters.ffmpeg_quality import ScanParser, run_process
from capicove.processing.video.quality.video_quality_contracts import QualityError
from capicove.processing.video.quality.video_quality_rules import (
    evaluate_policy,
    metric,
    union_duration,
)


@pytest.mark.parametrize(
    "options",
    [
        {"checks": []},
        {"checks": "decode"},
        {"checks": ["unknown"]},
        {"checks": ["decode", "decode"]},
        {"video_stream_index": True},
        {"audio_stream_index": -1},
        {"ffmpeg": "evil"},
        {"detector_options": {"freeze_noise": float("nan")}},
        {"detector_options": {"black_picture_ratio": 2}},
        {"detector_options": {"silence_per_channel": 1}},
        {"detector_options": {"silence_min_seconds": 0}},
        {"detector_options": {"filter": "movie=http://example.invalid"}},
        {"policy": {"score": 10}},
        {"policy": {"max_black_fraction": 2}},
        {"policy": {"min_frame_rate": 30, "max_frame_rate": 10}},
        {"policy": {"allowed_video_codecs": "h264"}},
        {"policy": {"require_audio": "true"}},
    ],
)
def test_invalid_requests_do_not_become_subprocess_arguments(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        QualityRequest.from_mapping({"path": "/authorized/video.mp4", **options})


def test_options_are_detached_and_immutable() -> None:
    policy: dict[str, Any] = {"allowed_video_codecs": ["h264"]}
    request = QualityRequest(Path("/a.mp4"), policy=policy)
    policy["allowed_video_codecs"].append("evil")
    assert request.policy["allowed_video_codecs"] == ("h264",)
    with pytest.raises(TypeError):
        request.detector_options["freeze_noise"] = 1  # type: ignore[index]


def test_union_counts_overlap_once() -> None:
    assert union_duration([(2, 4), (1, 3), (6, 7)]) == 4


def test_missing_required_metric_cannot_pass() -> None:
    checks = [
        {
            "check_id": "video_format",
            "status": "completed",
            "verdict": "observed",
            "metrics": {"display_width": metric(None)},
            "findings": [],
        }
    ]
    assert evaluate_policy(checks, {"min_display_width": 1280}) == "incomplete"


def test_inclusive_policy_boundary_passes() -> None:
    checks = [
        {
            "check_id": "video_format",
            "status": "completed",
            "verdict": "observed",
            "metrics": {"display_width": metric(1280)},
            "findings": [],
        }
    ]
    assert evaluate_policy(checks, {"min_display_width": 1280}) == "pass"


def test_timeout_reaps_a_real_child() -> None:
    import sys

    with pytest.raises(QualityError, match="deadline"):
        run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            datetime.now(UTC) + timedelta(seconds=0.1),
            QualityLimits(),
        )


def test_parser_requires_verified_eof_to_close_pending_events() -> None:
    parser = ScanParser(QualityLimits(), {"video_index": 0, "audio_index": None})
    parser.frame_line("frame|stream_index=0|pts_time=0|duration_time=0.1", "video")
    parser.line("[Parsed_freezedetect_1 @ 0x2] lavfi.freezedetect.freeze_start: 0")
    result = parser.finish(False)
    assert not result["complete"]
    assert result["pending"] == ["freeze"]
    assert result["events"]["freeze"] == []


def test_parser_event_retention_keeps_exact_totals() -> None:
    parser = ScanParser(QualityLimits(max_events=1), {"video_index": 0, "audio_index": None})
    for start, end in [(1, 3), (4, 6)]:
        parser.line(f"[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: {start}")
        parser.line(f"[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_end: {end}")
    result = parser.finish(False)
    assert len(result["events"]["freeze"]) == 1
    assert result["totals"]["freeze"]["count"] == 2
    assert result["totals"]["freeze"]["union_seconds"] == 4
    assert result["totals"]["freeze"]["events_truncated"]


def test_decode_error_cannot_be_hidden_by_clean_exit() -> None:
    parser = ScanParser(QualityLimits(), {"video_index": 0, "audio_index": None})
    parser.line("[matroska,webm @ 0x1] File ended prematurely")
    parser.progress("progress=end")
    assert not parser.finish(True)["complete"]


def test_regressing_timestamps_are_incomplete_not_a_sync_measurement() -> None:
    parser = ScanParser(QualityLimits(), {"video_index": 0, "audio_index": None})
    parser.frame_line("frame|stream_index=0|pts_time=1|duration_time=0.1", "video")
    with pytest.raises(QualityError, match="monotonic"):
        parser.frame_line("frame|stream_index=0|pts_time=0|duration_time=0.1", "video")


def test_direct_report_is_deeply_immutable() -> None:
    from capicove.processing.video.quality import QualityInspection

    source: dict[str, Any] = {"checks": [{"verdict": "observed"}]}
    result = QualityInspection(source)
    source["checks"][0]["verdict"] = "fail"
    assert result.to_dict()["checks"][0]["verdict"] == "observed"
    with pytest.raises(TypeError):
        result.report["checks"][0]["verdict"] = "fail"


def test_mov_external_reference_is_rejected_before_provider_use(tmp_path: Path) -> None:
    from capicove.processing.video.quality._adapters.local_media import LocalQualityIntake

    def box(kind: bytes, body: bytes) -> bytes:
        return (len(body) + 8).to_bytes(4, "big") + kind + body

    reference = box(b"url ", b"\x00\x00\x00\x00" + b"file:///private/secret\0")
    dref = box(b"dref", b"\x00\x00\x00\x00\x00\x00\x00\x01" + reference)
    path = tmp_path / "external.mov"
    path.write_bytes(box(b"moov", box(b"trak", box(b"mdia", box(b"minf", box(b"dinf", dref))))))
    with pytest.raises(QualityError, match="External media"):
        with LocalQualityIntake((tmp_path,), QualityLimits()).stage(
            path, datetime.now(UTC) + timedelta(seconds=5)
        ):
            raise AssertionError("external references must never reach a provider")


def test_human_logs_cannot_change_structured_frame_timestamps() -> None:
    parser = ScanParser(QualityLimits(), {"video_index": 0, "audio_index": 1})
    parser.frame_line("frame|stream_index=0|pts_time=0.1|duration_time=0.1", "video")
    parser.frame_line(
        "frame|stream_index=1|pts_time=0.6|duration_time=0.02|nb_samples=960", "audio"
    )
    parser.line(
        "[Parsed_showinfo_0 @ 0x1] n: 0 pts_time:10 duration_time:0.1 "
        "[Parsed_ashowinfo_0 @ 0x2] n:0 pts_time:20 rate:48000 nb_samples:1024"
    )
    result = parser.finish(False)
    assert result["video"]["start"] == pytest.approx(0.1)
    assert result["audio"]["start"] == pytest.approx(0.6)
    assert result["video"]["frames"] == 1
    assert result["audio"]["samples"] == 960


def test_decoded_audio_duration_takes_precedence_over_stream_header_rate() -> None:
    parser = ScanParser(
        QualityLimits(), {"video_index": 0, "audio_index": 1, "audio_sample_rate": 48000}
    )
    parser.frame_line(
        "frame|stream_index=1|pts_time=0|duration_time=0.023220|nb_samples=1024", "audio"
    )
    assert parser.finish(False)["audio"]["end"] == pytest.approx(0.023220)
