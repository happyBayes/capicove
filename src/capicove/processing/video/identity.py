"""Deterministic configuration and output identities for video processing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path


def _normalize(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    return value


def configuration_hash(configuration: object) -> str:
    """Hash canonical finite JSON for a normalized configuration."""
    encoded = json.dumps(
        _normalize(configuration),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_identity(
    source_sha256: str,
    strategy: str,
    config_sha256: str,
    backend_identity: str,
    segment_index: int,
) -> str:
    """Derive a stable output identity from evidence and execution inputs."""
    material = "\n".join(
        (source_sha256, strategy, config_sha256, backend_identity, str(segment_index))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
