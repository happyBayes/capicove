"""Strict, immutable offline model profiles (RFC-0023)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast


if TYPE_CHECKING:
    from capicove.model.http_profile import HttpModelProfile


OptionValue: TypeAlias = str | int | float | bool
_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEVICE = re.compile(r"cpu|mps|cuda(?::[0-9]+)?\Z")


class ModelCapability(StrEnum):
    """Select a typed inference port; no universal inference operation exists."""

    AUDIO_TRANSCRIPTION = "audio_transcription"
    TEXT_EMBEDDING = "text_embedding"
    TEXT_RERANKING = "text_reranking"
    IMAGE_TEXT_EMBEDDING = "image_text_embedding"
    IMAGE_OCR = "image_ocr"
    DOCUMENT_PARSER = "document_parser"
    MULTIMODAL_GENERATION = "multimodal_generation"
    MULTIMODAL_EMBEDDING = "multimodal_embedding"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Bind one provider to trusted local assets and explicit inference policy."""

    name: str
    provider: str
    capability: ModelCapability
    root: Path
    manifest_sha256: str
    device: str = "cpu"
    options: Mapping[str, OptionValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(v, str) or not _NAME.fullmatch(v) for v in (self.name, self.provider)
        ):
            raise ValueError("invalid model profile or provider name")
        if not isinstance(self.capability, ModelCapability):
            raise ValueError("invalid model capability")
        if self.capability == ModelCapability.MULTIMODAL_GENERATION:
            raise ValueError("multimodal_generation requires an explicit HTTP profile")
        if (
            not isinstance(self.root, Path)
            or not self.root.is_absolute()
            or self.root.resolve() != self.root
        ):
            raise ValueError("model root must be an absolute unlinked local path")
        if not isinstance(self.manifest_sha256, str) or not _SHA256.fullmatch(self.manifest_sha256):
            raise ValueError("model profile requires a trusted manifest SHA-256")
        if not isinstance(self.device, str) or not _DEVICE.fullmatch(self.device):
            raise ValueError("explicit model device is required")
        if not isinstance(self.options, Mapping) or len(self.options) > 32:
            raise ValueError("model options must be a bounded mapping")
        for key, value in self.options.items():
            if (
                not isinstance(key, str)
                or not _NAME.fullmatch(key)
                or type(value) not in (str, int, float, bool)
                or (isinstance(value, float) and not math.isfinite(value))
                or (isinstance(value, str) and len(value) > 1024)
            ):
                raise ValueError("model options must contain finite scalar values")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    @property
    def identity(self) -> str:
        """Hash exact assets, root and settings; aliases do not change identity."""
        body = {
            "schema": "capicove.model.profile.v1",
            "provider": self.provider,
            "capability": self.capability.value,
            "root": str(self.root),
            "manifest_sha256": self.manifest_sha256,
            "device": self.device,
            "options": dict(self.options),
        }
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """Separate inference selection from operator installation authority."""

    profiles: Mapping[str, ModelProfile | HttpModelProfile]
    bindings: Mapping[str, str]
    max_instances: int = 1

    def __post_init__(self) -> None:
        from capicove.model.http_profile import HttpModelProfile

        if type(self.max_instances) is not int or not 1 <= self.max_instances <= 16:
            raise ValueError("max_instances must be an integer from 1 to 16")
        if not isinstance(self.profiles, Mapping) or not 1 <= len(self.profiles) <= 64:
            raise ValueError("model profiles must contain 1 to 64 entries")
        if any(
            not isinstance(p, (ModelProfile, HttpModelProfile)) or name != p.name
            for name, p in self.profiles.items()
        ):
            raise ValueError("invalid named model profiles")
        if not isinstance(self.bindings, Mapping) or not 1 <= len(self.bindings) <= 128:
            raise ValueError("model bindings must contain 1 to 128 entries")
        if any(
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or not isinstance(profile, str)
            or profile not in self.profiles
            for name, profile in self.bindings.items()
        ):
            raise ValueError("model bindings must reference configured profiles")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ModelConfiguration:
        """Read v1 local profiles or v2 explicitly opted-in HTTP profiles."""
        if (
            not isinstance(raw, Mapping)
            or set(raw) - {"schema_version", "profiles", "bindings", "max_instances"}
            or not {"schema_version", "profiles", "bindings"} <= set(raw)
            or type(raw["schema_version"]) is not int
            or raw["schema_version"] not in (1, 2)
        ):
            raise ValueError("invalid model configuration schema")
        entries, bindings = raw["profiles"], raw["bindings"]
        if not isinstance(entries, Mapping) or not isinstance(bindings, Mapping):
            raise ValueError("model profiles and bindings must be mappings")
        profiles: dict[str, ModelProfile | HttpModelProfile] = {}
        for name, entry in entries.items():
            if isinstance(entry, Mapping) and entry.get("transport") == "http":
                if raw["schema_version"] != 2:
                    raise ValueError("HTTP model profiles require schema_version=2")
                profiles[name] = _http_profile(name, entry)
                continue
            if (
                not isinstance(entry, Mapping)
                or set(entry)
                - {"provider", "capability", "root", "manifest_sha256", "device", "options"}
                or not {"provider", "capability", "root", "manifest_sha256"} <= set(entry)
                or not isinstance(entry["root"], str)
            ):
                raise ValueError("invalid model profile fields")
            try:
                profiles[name] = ModelProfile(
                    name,
                    entry["provider"],
                    ModelCapability(entry["capability"]),
                    Path(entry["root"]),
                    entry["manifest_sha256"],
                    entry.get("device", "cpu"),
                    entry.get("options", {}),
                )
            except (TypeError, KeyError) as error:
                raise ValueError("invalid model profile") from error
        return cls(profiles, bindings, cast(int, raw.get("max_instances", 1)))


def _http_profile(name: str, entry: Mapping[str, object]) -> HttpModelProfile:
    from typing import Any

    from capicove.model.http_profile import HttpModelProfile

    allowed = set(HttpModelProfile.__dataclass_fields__) - {"name"}
    if set(entry) - (allowed | {"transport"}) or not {
        "provider",
        "capability",
        "endpoint",
        "model",
        "allow_network",
    } <= set(entry):
        raise ValueError("invalid HTTP model profile fields")
    values: dict[str, Any] = {key: value for key, value in entry.items() if key != "transport"}
    try:
        values["capability"] = ModelCapability(values["capability"])
        modalities = values.get("modalities", ["image"])
        if not isinstance(modalities, list) or any(not isinstance(m, str) for m in modalities):
            raise ValueError("modalities must be a list of names")
        values["modalities"] = tuple(modalities)
        return HttpModelProfile(name=name, **values)
    except (TypeError, KeyError) as error:
        raise ValueError("invalid HTTP model profile") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate model configuration field")
        result[key] = value
    return result


def load_configuration(path: Path) -> ModelConfiguration:
    """Read a bounded JSON or TOML operator file without network or model loading."""
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("model configuration must be an unlinked regular file")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("model configuration must be a regular unlinked file")
        payload = stream.read(1_048_577)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("model configuration changed during reading")
    if len(payload) > 1_048_576:
        raise ValueError("model configuration exceeds size limit")
    if path.suffix == ".toml":
        raw = tomllib.loads(payload.decode("utf-8"))
    elif path.suffix == ".json":
        raw = json.loads(payload, object_pairs_hook=_unique_object)
    else:
        raise ValueError("model configuration must be JSON or TOML")
    return ModelConfiguration.from_mapping(raw)
