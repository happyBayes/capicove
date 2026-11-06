"""Public composition boundary for the optional local speech adapter."""

from collections.abc import Mapping
from pathlib import Path

from capicove.processing.audio.synthesis import SpeechProvider


def create_local_speech(
    root: Path, options: Mapping[str, str | int | float | bool], *, device: str
) -> SpeechProvider:
    """Create a native adapter only after trusted model-profile validation."""
    from capicove.processing.audio._adapters.llama_tts import LlamaSpeechProvider

    provider = LlamaSpeechProvider(
        Path(str(options["executable"])),
        root / str(options["model"]),
        root / str(options["mmproj"]),
        hashes={key: str(options[key + "_sha256"]) for key in ("executable", "model", "mmproj")},
        device=device,
        threads=int(options.get("threads", 2)),
        max_frames=int(options.get("max_frames", 500)),
        backend_version=str(options["backend_version"]),
    )

    provider.verify_assets()
    return provider
