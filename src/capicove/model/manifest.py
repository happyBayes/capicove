"""Strict contracts for immutable image-model lock manifests."""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TypeAlias, cast


SettingValue: TypeAlias = str | int | float | bool

_KEY_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_TOP_LEVEL_FIELDS = frozenset({"manifest_version", "models"})
_MODEL_FIELDS = frozenset(
    {
        "adapter",
        "architecture",
        "capability",
        "default",
        "key",
        "license",
        "license_url",
        "preprocess_revision",
        "repository",
        "restricted",
        "revision",
        "settings",
    }
)
_ADAPTERS = frozenset(
    {
        "huggingface-aesthetics",
        "huggingface-blip-itm",
        "huggingface-clip-similarity",
        "huggingface-difference-caption",
        "huggingface-image-classification",
        "huggingface-region-caption",
        "huggingface-siglip2-retrieval",
        "onnx-image-embedding",
    }
)


@dataclass(frozen=True, slots=True)
class ImageModelSpec:
    """Describe one immutable, locally materialized model preset."""

    key: str
    repository: str
    revision: str
    license: str
    license_url: str
    capability: str
    adapter: str
    architecture: str
    preprocess_revision: str
    default: bool
    restricted: bool
    settings: tuple[tuple[str, SettingValue], ...] = ()


@dataclass(frozen=True, slots=True)
class ImageModelManifest:
    """Contain a validated ordered model catalog and its source digest."""

    manifest_version: int
    models: tuple[ImageModelSpec, ...]
    source_sha256: str


def load_packaged_manifest() -> ImageModelManifest:
    """Load Capicove's packaged immutable image-model catalog."""
    payload = resources.files("capicove.model").joinpath("image_models.toml").read_bytes()
    return parse_image_model_manifest(payload)


def load_manifest(path: str | Path) -> ImageModelManifest:
    """Load a complete user-supplied replacement manifest."""
    return parse_image_model_manifest(Path(path).expanduser().read_bytes())


def parse_image_model_manifest(payload: bytes) -> ImageModelManifest:
    """Parse exact image-model lock bytes already read by a trusted caller."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return _parse_manifest(payload)


def _parse_manifest(payload: bytes) -> ImageModelManifest:
    try:
        decoded = payload.decode("utf-8")
        raw = tomllib.loads(decoded)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("image-model manifest must be valid UTF-8 TOML") from error
    unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown manifest fields: {', '.join(unknown)}")
    version = raw.get("manifest_version")
    if version != 1:
        raise ValueError("image-model manifest_version must be 1")
    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("image-model manifest must contain at least one model")
    models = tuple(_model(item, index) for index, item in enumerate(raw_models))
    keys = [model.key for model in models]
    repositories = [model.repository for model in models]
    _require_unique(keys, "model key")
    _require_unique(repositories, "model repository")
    return ImageModelManifest(
        manifest_version=1,
        models=models,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _model(raw: object, index: int) -> ImageModelSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"model entry {index} must be a TOML table")
    values = cast(dict[str, object], raw)
    unknown = sorted(set(values) - _MODEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown model fields: {', '.join(unknown)}")
    required = _MODEL_FIELDS - {"settings"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"missing model fields: {', '.join(missing)}")
    key = _string(values, "key")
    repository = _string(values, "repository")
    revision = _string(values, "revision")
    adapter = _string(values, "adapter")
    if _KEY_PATTERN.fullmatch(key) is None:
        raise ValueError("model key must contain lowercase letters, digits, and hyphens")
    if _REPOSITORY_PATTERN.fullmatch(repository) is None:
        raise ValueError("model repository must use owner/name form")
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("model revision must be a lowercase 40-character Git SHA")
    if adapter not in _ADAPTERS:
        raise ValueError(f"unknown image-model adapter: {adapter}")
    settings = values.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("model settings must be a TOML table")
    normalized_settings: list[tuple[str, SettingValue]] = []
    for name, value in sorted(cast(dict[str, object], settings).items()):
        if not name.strip():
            raise ValueError("model setting names must not be blank")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("model setting values must be scalar")
        normalized_settings.append((name, value))
    return ImageModelSpec(
        key=key,
        repository=repository,
        revision=revision,
        license=_string(values, "license"),
        license_url=_string(values, "license_url"),
        capability=_string(values, "capability"),
        adapter=adapter,
        architecture=_string(values, "architecture"),
        preprocess_revision=_string(values, "preprocess_revision"),
        default=_boolean(values, "default"),
        restricted=_boolean(values, "restricted"),
        settings=tuple(normalized_settings),
    )


def _string(values: dict[str, object], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model {name} must be a nonblank string")
    return value.strip()


def _boolean(values: dict[str, object], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"model {name} must be a boolean")
    return value


def _require_unique(values: list[str], name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {name}: {value}")
        seen.add(value)
