"""Register local speech through the existing model factory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from capicove.model.configuration import ModelProfile


_REQUIRED = {
    "executable",
    "executable_sha256",
    "model",
    "model_sha256",
    "mmproj",
    "mmproj_sha256",
    "backend_version",
}


def validate_speech(profile: ModelProfile) -> None:
    """Reject implicit asset paths and unbounded native flags before loading."""
    options = profile.options
    if not _REQUIRED <= set(options) or set(options) - _REQUIRED - {
        "threads",
        "max_frames",
        "timeout_seconds",
    }:
        raise ValueError("invalid synthesis options")
    if profile.device not in {"cpu", "mps"}:
        raise ValueError("speech device must be cpu or mps")
    for key in _REQUIRED:
        value = options[key]
        if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
            raise ValueError("invalid synthesis option")
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid synthesis digest")
    if not Path(str(options["executable"])).is_absolute():
        raise ValueError("explicit synthesis executable required")
    for key in ("model", "mmproj"):
        name = str(options[key])
        if Path(name).name != name or not name.endswith(".gguf") or "/" in name or "\\" in name:
            raise ValueError("speech assets must be GGUF direct children")
    for key, default, maximum in (
        ("threads", 2, 16),
        ("max_frames", 500, 750),
        ("timeout_seconds", 240, 600),
    ):
        value = options.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("invalid synthesis budget")
    inventory = sorted((str(options[k]), str(options[k + "_sha256"])) for k in ("model", "mmproj"))
    digest = hashlib.sha256(
        json.dumps(inventory, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != profile.manifest_sha256:
        raise ValueError("speech manifest digest mismatch")


def create_speech(profile: ModelProfile) -> object:
    """Construct through the capability's public composition function."""
    from capicove.processing.audio.synthesis_runtime import create_local_speech

    validate_speech(profile)
    return create_local_speech(profile.root, profile.options, device=profile.device)
