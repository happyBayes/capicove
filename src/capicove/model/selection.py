"""Select locked image models and resolve their managed local root."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from capicove.model.manifest import ImageModelManifest, ImageModelSpec


def select_models(
    manifest: ImageModelManifest,
    *,
    default_profile: bool,
    keys: Sequence[str],
    include_restricted: bool,
) -> tuple[ImageModelSpec, ...]:
    """Return a validated selection in deterministic manifest order."""
    if not default_profile and not keys:
        raise ValueError("select --profile default or at least one --model")
    requested = set(keys)
    known = {model.key for model in manifest.models}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown image model: {', '.join(unknown)}")
    selected = tuple(
        model
        for model in manifest.models
        if (default_profile and model.default and not model.restricted) or model.key in requested
    )
    restricted = tuple(model.key for model in selected if model.restricted)
    if restricted and not include_restricted:
        joined = ", ".join(restricted)
        raise ValueError(f"restricted image model requires --include-restricted: {joined}")
    return selected


def resolve_model_home(
    destination: str | Path | None,
    environ: Mapping[str, str],
    system: str,
    home: Path,
) -> Path:
    """Resolve the model root without creating or inspecting it."""
    if destination is not None:
        return Path(destination).expanduser().resolve()
    configured = environ.get("CAPICOVE_MODEL_HOME")
    if configured:
        return Path(configured).expanduser()
    if system == "Darwin":
        return home / "Library" / "Caches" / "capicove" / "models"
    if system == "Windows":
        local_data = environ.get("LOCALAPPDATA")
        base = Path(local_data) if local_data else home / "AppData" / "Local"
        return base / "capicove" / "models"
    cache = environ.get("XDG_CACHE_HOME")
    base = Path(cache) if cache else home / ".cache"
    return base / "capicove" / "models"
