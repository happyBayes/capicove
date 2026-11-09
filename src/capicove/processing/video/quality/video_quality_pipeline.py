"""Evidence-preserving orchestration for video media quality inspection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from fractions import Fraction
from typing import Any

from capicove.processing.video.quality.video_quality_contracts import (
    AUDIO_CHECKS,
    CHECKS,
    TEMPORAL_CHECKS,
    QualityError,
    QualityInspection,
    QualityLimits,
    QualityRequest,
)
from capicove.processing.video.quality.video_quality_ports import (
    QualityBackend,
    QualityIntake,
    QualitySource,
)
from capicove.processing.video.quality.video_quality_rules import evaluate_policy, metric


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> float | None:
    try:
        return _finite(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None


def _select(streams: list[dict[str, Any]], kind: str, index: int | None) -> dict[str, Any] | None:
    eligible = [
        s
        for s in streams
        if s.get("codec_type") == kind
        and not (kind == "video" and s.get("disposition", {}).get("attached_pic"))
    ]
    if index is not None:
        eligible = [s for s in eligible if s.get("index") == index]
        if not eligible:
            raise QualityError(
                "invalid_input", "The selected stream index has the wrong type or is unavailable."
            )
    if len(eligible) > 1:
        raise QualityError(
            "invalid_input", "Multiple eligible streams require an explicit stream index."
        )
    if not eligible and kind == "video":
        raise QualityError("invalid_input", "Video quality inspection requires a video stream.")
    return eligible[0] if eligible else None


def _blank_check(name: str, requested: bool) -> dict[str, Any]:
    return {
        "check_id": name,
        "status": "completed" if requested else "not_run",
        "verdict": "observed" if requested else "unknown",
        "metrics": {},
        "events": [],
        "findings": [],
        "unavailable_reason": None,
    }


def _measurements(
    values: Mapping[str, Any], units: Mapping[str, str] | None = None
) -> dict[str, Any]:
    return {
        key: metric(value, (units or {}).get(key), "not_declared" if value is None else None)
        for key, value in values.items()
    }


class VideoQualityPipeline:
    """Keep adapters behind capability-owned ports and policy separate from facts."""

    def __init__(
        self, intake: QualityIntake, backend: QualityBackend, limits: QualityLimits | None = None
    ) -> None:
        self._intake, self._backend, self._limits = intake, backend, limits or QualityLimits()

    def inspect(self, request: QualityRequest, deadline: datetime) -> QualityInspection:
        """Inspect one authorized file, preserving completed diagnostics on failure."""
        report: dict[str, Any] | None = None
        try:
            with self._intake.stage(request.path, deadline) as source:
                report = self._inspect_source(source, request, deadline)
        except QualityError as error:
            if report is None:
                raise
            report["verdict"] = "incomplete"
            report["failures"].append({"code": error.code, "message": str(error)})
        assert report is not None
        failure = report["failures"][0]["code"] if report["failures"] else None
        return QualityInspection(report, failure)

    def _inspect_source(
        self, source: QualitySource, request: QualityRequest, deadline: datetime
    ) -> dict[str, Any]:
        probe = self._backend.probe(source, deadline)
        streams = probe["streams"]
        if any(not isinstance(s, dict) or type(s.get("index")) is not int for s in streams):
            raise QualityError("parse_failed", "The media stream inventory has invalid indexes.")
        video = _select(streams, "video", request.video_stream_index)
        assert video is not None
        audio = _select(streams, "audio", request.audio_stream_index)
        selection = {
            "video_index": video["index"],
            "audio_index": audio["index"] if audio else None,
            "tracks": [
                {
                    key: s.get(key)
                    for key in (
                        "index",
                        "codec_type",
                        "codec_name",
                        "disposition",
                        "start_time",
                        "duration",
                    )
                }
                for s in streams
            ],
        }
        config = {
            "checks": list(request.checks),
            "detector_options": dict(request.detector_options),
            "policy": dict(request.policy),
            "limits": asdict(self._limits),
        }
        config["sha256"] = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        checks = [_blank_check(name, name in request.checks) for name in CHECKS]
        lookup = {c["check_id"]: c for c in checks}
        metadata = self._metadata(video, audio, probe)
        for name, values in metadata.items():
            if name in request.checks:
                lookup[name]["metrics"] = values
        for name in AUDIO_CHECKS & set(request.checks):
            if audio is None:
                lookup[name].update(
                    status="not_applicable", verdict="unknown", unavailable_reason="no_audio_stream"
                )
        report: dict[str, Any] = {
            "schema_version": 1,
            "source": {
                "path": str(source.original_path),
                "sha256": source.sha256,
                "size_bytes": source.size_bytes,
            },
            "selection": selection,
            "configuration": config,
            "coverage": {
                "decoded": False,
                "origin_seconds": None,
                "video": None,
                "audio": None,
                "passes": [],
            },
            "checks": checks,
            "verdict": "observed",
            "warnings": [],
            "failures": [],
            "lineage": {
                "parser_version": 1,
                "ffprobe_version": probe.get("program_version", {}).get("version"),
                "ffprobe_sha256": probe.get("ffprobe_sha256"),
                "ffmpeg_version": None,
                "ffmpeg_sha256": None,
            },
        }
        if set(request.checks) & TEMPORAL_CHECKS:
            declared = _finite(probe.get("format", {}).get("duration"))
            try:
                if declared is not None and declared > self._limits.max_duration_seconds:
                    raise QualityError(
                        "resource_limit", "Declared media duration exceeds the composition budget."
                    )
                scan_selection = {
                    **selection,
                    "audio_sample_rate": _finite(audio.get("sample_rate")) if audio else None,
                }
                scan = self._backend.scan(source, scan_selection, request, self._limits, deadline)
                self._apply_scan(report, request, scan)
            except QualityError as error:
                report["failures"].append({"code": error.code, "message": str(error)})
                for name in set(request.checks) & TEMPORAL_CHECKS:
                    if lookup[name]["status"] != "not_applicable":
                        lookup[name].update(
                            status="failed", verdict="unknown", unavailable_reason=error.code
                        )
        report["verdict"] = evaluate_policy(checks, request.policy)
        if report["verdict"] == "incomplete" and not report["failures"]:
            report["failures"].append(
                {
                    "code": "measurement_unavailable",
                    "message": "A required measurement could not be established.",
                }
            )
        if report["failures"]:
            report["verdict"] = "incomplete"
        return report

    @staticmethod
    def _metadata(
        video: dict[str, Any], audio: dict[str, Any] | None, probe: dict[str, Any]
    ) -> dict[str, Any]:
        rotation = _finite(video.get("tags", {}).get("rotate")) or 0.0
        for side in video.get("side_data_list", []):
            if "rotation" in side:
                rotation = _finite(side["rotation"]) or 0.0
        width, height = video.get("width"), video.get("height")
        display_width, display_height = (
            (height, width) if abs(rotation) % 180 == 90 else (width, height)
        )
        nominal, average = _rate(video.get("r_frame_rate")), _rate(video.get("avg_frame_rate"))
        values = {
            "streams": _measurements(
                {
                    "container": probe.get("format", {}).get("format_name"),
                    "has_audio": audio is not None,
                    "stream_count": len(probe["streams"]),
                    "subtitle_stream_count": sum(
                        s.get("codec_type") == "subtitle" for s in probe["streams"]
                    ),
                }
            ),
            "video_format": _measurements(
                {
                    "codec": video.get("codec_name"),
                    "encoded_width": width,
                    "encoded_height": height,
                    "display_width": display_width,
                    "display_height": display_height,
                    "rotation_degrees": rotation,
                    "sample_aspect_ratio": video.get("sample_aspect_ratio"),
                    "display_aspect_ratio": video.get("display_aspect_ratio"),
                    "pixel_format": video.get("pix_fmt"),
                    "nominal_frame_rate": nominal,
                    "average_frame_rate": average,
                    "nominal_frame_rate_rational": video.get("r_frame_rate"),
                    "average_frame_rate_rational": video.get("avg_frame_rate"),
                    "rate_disagreement": nominal != average
                    if nominal is not None and average is not None
                    else None,
                    "declared_bitrate_bps": _finite(video.get("bit_rate")),
                    "declared_frame_count": _finite(video.get("nb_frames")),
                },
                {
                    "encoded_width": "px",
                    "encoded_height": "px",
                    "display_width": "px",
                    "display_height": "px",
                    "rotation_degrees": "degrees",
                    "nominal_frame_rate": "fps",
                    "average_frame_rate": "fps",
                    "declared_bitrate_bps": "bit/s",
                },
            ),
            "audio_format": _measurements(
                {
                    "codec": audio.get("codec_name"),
                    "sample_rate_hz": _finite(audio.get("sample_rate")),
                    "channels": audio.get("channels"),
                    "channel_layout": audio.get("channel_layout"),
                    "declared_bitrate_bps": _finite(audio.get("bit_rate")),
                },
                {"sample_rate_hz": "Hz", "declared_bitrate_bps": "bit/s"},
            )
            if audio
            else {},
        }
        return values

    @staticmethod
    def _apply_scan(report: dict[str, Any], request: QualityRequest, scan: dict[str, Any]) -> None:
        checks = {c["check_id"]: c for c in report["checks"]}
        video, audio = scan["video"], scan["audio"]
        origin = scan["origin_seconds"]
        complete = (
            scan["complete"]
            and video["frames"] > 0
            and (report["selection"]["audio_index"] is None or audio["samples"] > 0)
        )
        coverage = {"decoded": True, "origin_seconds": origin, "passes": scan["passes"]}
        for name, track in (("video", video), ("audio", audio)):
            if name == "audio" and report["selection"]["audio_index"] is None:
                coverage[name] = None
                continue
            coverage[name] = {
                "start_seconds": track["start"] - origin
                if track["start"] is not None and origin is not None
                else None,
                "end_seconds": track["end"] - origin
                if track["end"] is not None and origin is not None
                else None,
                "decoded_duration_seconds": track["end"] - track["start"]
                if track["end"] is not None
                else None,
                "eof": complete,
                "frames": track.get("frames"),
                "samples": track.get("samples"),
            }
        report["coverage"] = coverage
        report["lineage"].update({k: scan.get(k) for k in ("ffmpeg_version", "ffmpeg_sha256")})
        for name in set(request.checks) & TEMPORAL_CHECKS:
            check = checks[name]
            if check["status"] == "not_applicable":
                continue
            if not complete:
                check.update(
                    status="failed",
                    verdict="unknown",
                    unavailable_reason=scan.get("failure_code") or "decode_failed",
                )
        if not complete:
            report["failures"].append(
                {
                    "code": scan.get("failure_code") or "decode_failed",
                    "message": "Selected-stream decoding did not complete without errors.",
                }
            )
        if "decode" in request.checks:
            checks["decode"]["metrics"] = _measurements(
                {
                    "complete": complete,
                    "video_frames": video["frames"],
                    "audio_samples": audio["samples"],
                    "decode_errors": scan["decode_errors"],
                }
            )
        for name in ("black", "freeze", "silence"):
            if name not in request.checks or checks[name]["status"] == "not_applicable":
                continue
            check = checks[name]
            totals = scan["totals"][name]
            track = audio if name == "silence" else video
            duration = track["end"] - track["start"] if track["end"] is not None else None
            check["metrics"] = _measurements(
                {
                    "event_count": totals["count"],
                    "union_seconds": totals["union_seconds"],
                    "longest_seconds": totals["longest"],
                    "coverage_seconds": duration,
                    "fraction": min(1.0, totals["union_seconds"] / duration) if duration else None,
                    "events_truncated": totals["events_truncated"],
                },
                {
                    "union_seconds": "s",
                    "longest_seconds": "s",
                    "coverage_seconds": "s",
                    "fraction": "ratio",
                },
            )
            check["events"] = scan["events"][name]
            if duration is None or duration <= 0:
                check.update(
                    status="failed",
                    verdict="unknown",
                    unavailable_reason="coverage_duration_unavailable",
                )
            if name in scan["pending"]:
                check.update(
                    status="failed", verdict="unknown", unavailable_reason="unclosed_interval"
                )
            if totals["count"] and check["status"] == "completed":
                check["verdict"] = "review"
                check["findings"].append(
                    {
                        "code": f"{name}_intervals",
                        "severity": "warning",
                        "message": "Detector intervals require contextual review; intentional teaching layouts or pauses may trigger them.",
                    }
                )
        if "loudness" in request.checks and report["selection"]["audio_index"] is not None:
            check = checks["loudness"]
            loudness = scan["loudness"]
            if set(loudness) != {"integrated_lufs", "range_lu", "true_peak_dbtp"}:
                check.update(
                    status="failed",
                    verdict="unknown",
                    unavailable_reason="loudness_summary_unavailable",
                )
            check["metrics"] = {}
            below_gate = (
                loudness.get("integrated_lufs") is None
                or loudness.get("integrated_lufs", -70) <= -70
            )
            for key, unit in (
                ("integrated_lufs", "LUFS"),
                ("range_lu", "LU"),
                ("true_peak_dbtp", "dBTP"),
            ):
                value = loudness.get(key)
                reason = None
                if value is None or (below_gate and key != "true_peak_dbtp"):
                    value = None
                    reason = "below_gate_or_unmeasurable"
                check["metrics"][key] = metric(value, unit, reason)
        if "av_timing" in request.checks and report["selection"]["audio_index"] is not None:
            values = {
                "start_delta_seconds": audio["start"] - video["start"]
                if audio["start"] is not None and video["start"] is not None
                else None,
                "end_delta_seconds": audio["end"] - video["end"]
                if audio["end"] is not None and video["end"] is not None
                else None,
            }
            tracks = {s["index"]: s for s in report["selection"]["tracks"]}
            v, a = (
                tracks[report["selection"]["video_index"]],
                tracks[report["selection"]["audio_index"]],
            )
            vs, ast = _finite(v.get("start_time")), _finite(a.get("start_time"))
            vd, ad = _finite(v.get("duration")), _finite(a.get("duration"))
            values["declared_start_delta_seconds"] = (
                ast - vs if ast is not None and vs is not None else None
            )
            values["declared_end_delta_seconds"] = (
                ast + ad - vs - vd
                if ast is not None and ad is not None and vs is not None and vd is not None
                else None
            )
            checks["av_timing"]["metrics"] = _measurements(values, dict.fromkeys(values, "s"))
