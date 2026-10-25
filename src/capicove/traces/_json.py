"""Internal helpers for immutable finite JSON values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
FrozenJson: TypeAlias = JsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


def freeze_json(
    value: object,
    /,
    *,
    field_name: str,
    max_depth: int = 32,
) -> FrozenJson:
    """Validate and deeply freeze one finite JSON-compatible value."""
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    active_containers: set[int] = set()

    def freeze(current: object, *, depth: int, path: str) -> FrozenJson:
        if current is None or isinstance(current, (str, bool)):
            return current
        if isinstance(current, int):
            return current
        if isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError(f"{path} must contain only finite JSON numbers")
            return current
        if not isinstance(current, (Mapping, list, tuple)):
            raise ValueError(f"{path} must contain only JSON-compatible values")
        if depth >= max_depth:
            raise ValueError(f"{field_name} exceeds the maximum JSON depth")

        identity = id(current)
        if identity in active_containers:
            raise ValueError(f"{field_name} must not contain a cycle")
        active_containers.add(identity)
        try:
            if isinstance(current, Mapping):
                frozen: dict[str, FrozenJson] = {}
                for key, item in current.items():
                    if not isinstance(key, str):
                        raise ValueError(f"{path} must contain only string object keys")
                    frozen[key] = freeze(item, depth=depth + 1, path=f"{path}.{key}")
                return MappingProxyType(frozen)
            return tuple(
                freeze(item, depth=depth + 1, path=f"{path}[{index}]")
                for index, item in enumerate(current)
            )
        finally:
            active_containers.remove(identity)

    return freeze(value, depth=0, path=field_name)


def thaw_json(value: FrozenJson, /) -> object:
    """Return plain dictionaries and lists for JSON serialization."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value
