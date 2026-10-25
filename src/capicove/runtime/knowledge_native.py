"""Verified native executable and asset handoff for knowledge media preparation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from capicove.runtime.executables import PinnedExecutables
from capicove.runtime.knowledge_config import KnowledgeConfiguration, read_configuration
from capicove.runtime.media_handoff import MediaIntake, verify_asset


@contextmanager
def native_runtime(
    config: KnowledgeConfiguration, *, models: bool
) -> Iterator[tuple[Mapping[str, str], Mapping[str, Path], dict[str, Any]]]:
    """Execute only pinned descriptors; stage hash-verified models before use."""
    if config.audio is None:
        raise ValueError("audio_runtime_not_configured")
    raw = read_configuration(config.audio)
    if set(raw) - {
        "schema_version",
        "files",
        "backend_version",
        "speech_keywords",
        "speech_provenance",
    }:
        raise ValueError("invalid_audio_configuration")
    files = raw["files"]
    names = (
        ("ffmpeg", "ffprobe", "whisper_cli", "whisper_vad_cli") if models else ("ffmpeg", "ffprobe")
    )
    paths = {name: Path(files[name]["path"]) for name in names}
    pins = PinnedExecutables(paths)
    for name, path in paths.items():
        verify_asset(path, files[name]["sha256"])
    intake = MediaIntake(
        read_roots=(),
        staging_root=config.storage_root / "native-staging",
        max_input_bytes=128 * 1024 * 1024,
        accepts=lambda path, head: True,
    )
    assets = (
        tuple((Path(files[name]["path"]), files[name]["sha256"]) for name in ("model", "vad_model"))
        if models
        else ()
    )
    with pins.open() as executables, intake.stage((), assets) as staged:
        yield (
            executables,
            dict(zip(("model", "vad_model"), (row.staged for row in staged), strict=models)),
            raw,
        )
