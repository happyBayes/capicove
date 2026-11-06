"""Provider-independent contracts for deterministic video quality inspection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


CHECKS = (
    "streams",
    "video_format",
    "audio_format",
    "decode",
    "black",
    "freeze",
    "silence",
    "loudness",
    "av_timing",
)
TEMPORAL_CHECKS = frozenset({"decode", "black", "freeze", "silence", "loudness", "av_timing"})
AUDIO_CHECKS = frozenset({"audio_format", "silence", "loudness", "av_timing"})
DETECTOR_DEFAULTS = {
    "black_min_seconds": 2.0,
    "black_picture_ratio": 0.98,
    "black_pixel_threshold": 0.10,
    "freeze_min_seconds": 2.0,
    "freeze_noise": 0.001,
    "silence_min_seconds": 2.0,
    "silence_threshold_dbfs": -50.0,
    "silence_per_channel": False,
}
POLICY_FIELDS = {
    "require_audio",
    "min_display_width",
    "min_display_height",
    "allowed_video_codecs",
    "allowed_audio_codecs",
    "min_frame_rate",
    "max_frame_rate",
    "min_sample_rate_hz",
    "max_sample_rate_hz",
    "min_channels",
    "max_channels",
    "min_integrated_lufs",
    "max_integrated_lufs",
    "max_true_peak_dbtp",
    "max_abs_start_delta_seconds",
    "max_abs_end_delta_seconds",
    "max_black_duration_seconds",
    "max_black_fraction",
    "max_freeze_duration_seconds",
    "max_freeze_fraction",
    "max_silence_duration_seconds",
    "max_silence_fraction",
}


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    return float(value)


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class QualityLimits:
    """Composition-owned limits that a Tool request cannot enlarge."""

    max_input_bytes: int = 4 * 1024**3
    max_duration_seconds: float = 7200.0
    max_events: int = 1000
    max_records: int = 2_000_000
    max_diagnostic_bytes: int = 65536
    max_probe_bytes: int = 1024 * 1024
    threads: int = 2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if _number(value, name) <= 0:
                raise ValueError(f"{name} must be positive")
            if name != "max_duration_seconds" and type(value) is not int:
                raise ValueError(f"{name} must be an integer")


@dataclass(frozen=True, slots=True)
class QualityRequest:
    """Validated, immutable options for one authorized local file."""

    path: Path
    checks: tuple[str, ...] = CHECKS
    video_stream_index: int | None = None
    audio_stream_index: int | None = None
    detector_options: Mapping[str, Any] = field(default_factory=dict)
    policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("path must be an absolute local path")
        if (
            not self.checks
            or len(set(self.checks)) != len(self.checks)
            or set(self.checks) - set(CHECKS)
        ):
            raise ValueError("checks must be nonempty, unique and supported")
        for index in (self.video_stream_index, self.audio_stream_index):
            if index is not None and (type(index) is not int or index < 0):
                raise ValueError("stream indexes must be non-negative integers")
        options = dict(DETECTOR_DEFAULTS)
        supplied = _object(self.detector_options, "detector_options")
        if set(supplied) - set(options):
            raise ValueError("unknown detector option")
        options.update(supplied)
        for name, value in options.items():
            if name == "silence_per_channel":
                if type(value) is not bool:
                    raise ValueError("silence_per_channel must be boolean")
                continue
            number = _number(value, name)
            if name.endswith("min_seconds") and not 0 < number <= 7200:
                raise ValueError(f"{name} must be in (0, 7200]")
            if (
                name in {"black_picture_ratio", "black_pixel_threshold", "freeze_noise"}
                and not 0 <= number <= 1
            ):
                raise ValueError(f"{name} must be in [0, 1]")
            if name == "silence_threshold_dbfs" and not -120 <= number <= 0:
                raise ValueError("silence_threshold_dbfs must be in [-120, 0]")
        policy = _object(self.policy, "policy")
        if set(policy) - POLICY_FIELDS:
            raise ValueError("unknown quality policy field")
        for name, value in policy.items():
            if name == "require_audio":
                if type(value) is not bool:
                    raise ValueError("require_audio must be boolean")
            elif name.startswith("allowed_"):
                if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
                    raise ValueError(f"{name} must be a nonempty codec array")
                if any(
                    not isinstance(v, str)
                    or not v
                    or not v.isascii()
                    or not v.replace("_", "").isalnum()
                    for v in value
                ):
                    raise ValueError("invalid codec name")
                policy[name] = tuple(value)
            else:
                number = _number(value, name)
                if "lufs" not in name and "dbtp" not in name and number < 0:
                    raise ValueError(f"{name} must be non-negative")
                if name.endswith("fraction") and number > 1:
                    raise ValueError(f"{name} must be in [0, 1]")
                if (
                    any(s in name for s in ("width", "height", "channels", "sample_rate"))
                    and type(value) is not int
                ):
                    raise ValueError(f"{name} must be an integer")
        for name, value in policy.items():
            maximum = "max_" + name[4:]
            if name.startswith("min_") and maximum in policy and value > policy[maximum]:
                raise ValueError(f"{name} exceeds {maximum}")
        object.__setattr__(self, "checks", tuple(c for c in CHECKS if c in self.checks))
        object.__setattr__(self, "detector_options", MappingProxyType(options))
        object.__setattr__(self, "policy", MappingProxyType(policy))

    @classmethod
    def from_mapping(cls, arguments: Mapping[str, object]) -> QualityRequest:
        """Reject undeclared authority and parse the canonical input."""
        allowed = {
            "path",
            "checks",
            "video_stream_index",
            "audio_stream_index",
            "detector_options",
            "policy",
        }
        if set(arguments) - allowed or not isinstance(arguments.get("path"), str):
            raise ValueError("quality input contains unknown or missing fields")
        values = dict(arguments)
        values["path"] = Path(str(values["path"]))
        checks = values.get("checks", CHECKS)
        if (
            isinstance(checks, (str, bytes))
            or not isinstance(checks, Sequence)
            or any(not isinstance(c, str) for c in checks)
        ):
            raise ValueError("checks must be an array of check IDs")
        values["checks"] = tuple(checks)
        return cls(**values)  # type: ignore[arg-type]


class QualityError(ValueError):
    """Carry a stable, path-free inspection failure category."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class QualityInspection:
    """Completed or incomplete canonical report, without transport dependencies."""

    report: Mapping[str, Any]
    failure_code: str | None = None
    _encoded: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Serialize a private copy at construction; callers receive fresh JSON values.
        import json

        encoded = json.dumps(dict(self.report), allow_nan=False)
        object.__setattr__(self, "_encoded", encoded)

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return MappingProxyType({k: freeze(v) for k, v in value.items()})
            if isinstance(value, list):
                return tuple(freeze(v) for v in value)
            return value

        object.__setattr__(self, "report", freeze(json.loads(encoded)))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, finite JSON projection."""
        import json

        return dict(json.loads(self._encoded))
