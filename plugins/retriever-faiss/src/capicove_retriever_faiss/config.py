"""Closed local configuration for FAISS HNSW visual retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from capicove.contracts import TargetType


_KEYS = frozenset(
    {
        "index_root",
        "maximum_keyframes",
        "maximum_segments",
        "hnsw_m",
        "ef_construction",
        "ef_search",
        "threads",
    }
)
_MAXIMUM_UNITS = 1_000_000
_MAXIMUM_EF = 4_096
_MAXIMUM_THREADS = 256


@dataclass(frozen=True, slots=True)
class FaissConfiguration:
    """Contain only reviewed, bounded FAISS startup parameters."""

    index_root: Path
    threads: int
    maximum_keyframes: int = 1_000_000
    maximum_segments: int = 1_000_000
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128

    def __post_init__(self) -> None:
        if not isinstance(self.index_root, Path) or not self.index_root.is_absolute():
            raise ValueError("index_root must be an absolute Path")
        _reject_symlink_components(self.index_root)
        _bounded_integer("maximum_keyframes", self.maximum_keyframes, 1, _MAXIMUM_UNITS)
        _bounded_integer("maximum_segments", self.maximum_segments, 1, _MAXIMUM_UNITS)
        _bounded_integer("hnsw_m", self.hnsw_m, 2, 128)
        _bounded_integer("ef_construction", self.ef_construction, 1, _MAXIMUM_EF)
        if self.ef_construction < self.hnsw_m:
            raise ValueError("ef_construction must be greater than or equal to hnsw_m")
        _bounded_integer("ef_search", self.ef_search, 1, _MAXIMUM_EF)
        _bounded_integer("threads", self.threads, 1, _MAXIMUM_THREADS)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FaissConfiguration:
        """Validate one closed startup mapping without importing FAISS."""
        if not isinstance(value, Mapping):
            raise TypeError("FAISS configuration must be a mapping")
        unknown = set(value) - _KEYS
        if unknown:
            raise ValueError(f"unsupported FAISS configuration key: {sorted(unknown)[0]}")
        if "index_root" not in value:
            raise ValueError("index_root must be provided")
        if "threads" not in value:
            raise ValueError("threads must be provided explicitly")
        root_value = value["index_root"]
        if isinstance(root_value, str):
            root = Path(root_value)
        elif isinstance(root_value, Path):
            root = root_value
        else:
            raise TypeError("index_root must be a Path or string")
        return cls(
            index_root=root,
            threads=_mapping_integer(value, "threads"),
            maximum_keyframes=_mapping_integer(value, "maximum_keyframes", 1_000_000),
            maximum_segments=_mapping_integer(value, "maximum_segments", 1_000_000),
            hnsw_m=_mapping_integer(value, "hnsw_m", 32),
            ef_construction=_mapping_integer(value, "ef_construction", 200),
            ef_search=_mapping_integer(value, "ef_search", 128),
        )

    def capacity_for(self, target_type: TargetType) -> int:
        """Return the independent capacity for one supported visual target."""
        if target_type is TargetType.KEYFRAME:
            return self.maximum_keyframes
        if target_type is TargetType.VIDEO_SEGMENT:
            return self.maximum_segments
        raise ValueError("FAISS visual indexes support only keyframes and video segments")


def _mapping_integer(value: Mapping[str, object], key: str, default: int | None = None) -> int:
    item = value.get(key, default)
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError(f"{key} must be an integer")
    return item


def _bounded_integer(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")


def _reject_symlink_components(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("index_root must not contain symlink components")
        current = current.parent


__all__ = ["FaissConfiguration"]
