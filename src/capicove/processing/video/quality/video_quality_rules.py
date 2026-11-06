"""Pure interval and explicit delivery-policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def union_duration(intervals: Sequence[tuple[float, float]]) -> float:
    """Measure an interval union without counting overlap twice."""
    total = 0.0
    end: float | None = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("interval ends before its start")
        total += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def metric(value: Any, unit: str | None = None, reason: str | None = None) -> dict[str, Any]:
    """Label a measurement and preserve explicit unknown values."""
    return {"value": value, "unit": unit, "reason": reason}


def evaluate_policy(checks: list[dict[str, Any]], policy: Mapping[str, Any]) -> str:
    """Apply inclusive policy bounds, independently of detector execution."""
    lookup = {c["check_id"]: c for c in checks}
    rules: list[tuple[str, str, str, str]] = [
        ("require_audio", "streams", "has_audio", "required"),
        ("allowed_video_codecs", "video_format", "codec", "allowed"),
        ("allowed_audio_codecs", "audio_format", "codec", "allowed"),
        ("min_display_width", "video_format", "display_width", "min"),
        ("min_display_height", "video_format", "display_height", "min"),
        ("min_frame_rate", "video_format", "average_frame_rate", "min"),
        ("max_frame_rate", "video_format", "average_frame_rate", "max"),
        ("min_sample_rate_hz", "audio_format", "sample_rate_hz", "min"),
        ("max_sample_rate_hz", "audio_format", "sample_rate_hz", "max"),
        ("min_channels", "audio_format", "channels", "min"),
        ("max_channels", "audio_format", "channels", "max"),
        ("min_integrated_lufs", "loudness", "integrated_lufs", "min"),
        ("max_integrated_lufs", "loudness", "integrated_lufs", "max"),
        ("max_true_peak_dbtp", "loudness", "true_peak_dbtp", "max"),
        ("max_abs_start_delta_seconds", "av_timing", "start_delta_seconds", "abs"),
        ("max_abs_end_delta_seconds", "av_timing", "end_delta_seconds", "abs"),
    ]
    for name in ("black", "freeze", "silence"):
        rules.extend(
            [
                (f"max_{name}_duration_seconds", name, "longest_seconds", "max"),
                (f"max_{name}_fraction", name, "fraction", "max"),
            ]
        )
    applicable: set[str] = set()
    for key, check_id, measured, comparison in rules:
        if key not in policy or (key == "require_audio" and not policy[key]):
            continue
        check = lookup[check_id]
        applicable.add(check_id)
        if check["status"] == "not_applicable" and check_id in {
            "audio_format",
            "loudness",
            "av_timing",
            "silence",
        }:
            continue
        value = check["metrics"].get(measured, {}).get("value")
        if value is None:
            check["status"] = "failed"
            check["verdict"] = "unknown"
            check["unavailable_reason"] = "required_metric_unavailable"
            continue
        target = policy[key]
        if comparison == "required":
            passed = bool(value)
        elif comparison == "allowed":
            passed = value in target
        elif comparison == "min":
            passed = value >= target
        elif comparison == "abs":
            passed = abs(value) <= target
        else:
            passed = value <= target
        if not passed:
            check["findings"].append(
                {
                    "code": key,
                    "severity": "error",
                    "message": f"Measured {measured} violates {key}.",
                }
            )
            check["verdict"] = "fail"
    for check in checks:
        if (
            check["check_id"] in applicable
            and check["status"] == "completed"
            and check["verdict"] == "observed"
        ):
            check["verdict"] = "pass"
    if any(c["status"] == "failed" for c in checks):
        return "incomplete"
    if any(c["verdict"] == "fail" for c in checks):
        return "fail"
    if any(c["verdict"] == "review" for c in checks):
        return "review"
    return "pass" if policy else "observed"
