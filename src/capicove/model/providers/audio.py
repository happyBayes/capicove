"""Explicit offline Whisper.cpp registration with pinned model hashes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from capicove.model.configuration import ModelProfile


_REQUIRED = {
    "model",
    "model_sha256",
    "vad_model",
    "vad_model_sha256",
    "whisper_cli",
    "whisper_vad_cli",
    "ffmpeg",
    "ffprobe",
    "backend_version",
}


def validate_audio(profile: ModelProfile) -> None:
    """Reject unknown settings, implicit executables and unbounded decode budgets."""
    options = profile.options
    if (
        profile.device != "cpu"
        or not _REQUIRED <= set(options)
        or set(options) - _REQUIRED - {"max_duration_seconds"}
    ):
        raise ValueError("invalid audio options")
    for key in _REQUIRED:
        value = options[key]
        if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
            raise ValueError("invalid audio option")
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid audio digest")
        if key in {"model", "vad_model"} and (Path(value).name != value or value in {".", ".."}):
            raise ValueError("audio assets must be direct children")
        if (
            key in {"whisper_cli", "whisper_vad_cli", "ffmpeg", "ffprobe"}
            and not Path(value).is_absolute()
        ):
            raise ValueError("explicit executable path required")
    duration = options.get("max_duration_seconds", 1800)
    if type(duration) is not int or not 1 <= duration <= 7200:
        raise ValueError("invalid duration limit")
    inventory = sorted(
        (str(options[k]), str(options[k + "_sha256"])) for k in ("model", "vad_model")
    )
    if (
        len({name for name, _ in inventory}) != 2
        or hashlib.sha256(
            json.dumps(inventory, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        != profile.manifest_sha256
    ):
        raise ValueError("audio manifest digest mismatch")


def create_audio(profile: ModelProfile) -> object:
    """Verify assets locally; importing the registry never loads or downloads weights."""
    from capicove.processing.audio.local_runtime import LocalAudioProvider

    validate_audio(profile)
    options = profile.options
    for key in ("model", "vad_model"):
        path = profile.root / str(options[key])
        if path.resolve() != path:
            raise ValueError("unsafe audio model path")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError("unsafe audio model file")
            if hashlib.file_digest(stream, "sha256").hexdigest() != options[key + "_sha256"]:
                raise ValueError("audio model digest mismatch")
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError("audio model changed")
    for key in ("whisper_cli", "whisper_vad_cli", "ffmpeg", "ffprobe"):
        path = Path(str(options[key]))
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("audio executable unavailable")
    return LocalAudioProvider(
        model=profile.root / str(options["model"]),
        vad_model=profile.root / str(options["vad_model"]),
        whisper_cli=str(options["whisper_cli"]),
        whisper_vad_cli=str(options["whisper_vad_cli"]),
        ffmpeg=str(options["ffmpeg"]),
        ffprobe=str(options["ffprobe"]),
        backend_version=str(options["backend_version"]),
        max_duration_seconds=int(options.get("max_duration_seconds", 1800)),
        expected_model_sha256=str(options["model_sha256"]),
        expected_vad_sha256=str(options["vad_model_sha256"]),
    )
