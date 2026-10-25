"""Private validation helpers for canonical contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import TypeVar


_TOOL_ID_PATTERN = re.compile(r"^capicove\.[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


def require_non_blank(field_name: str, value: str) -> None:
    """Require a string contract field to contain non-whitespace content."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def require_positive_int(field_name: str, value: int) -> None:
    """Require an integer contract field to be positive."""
    if value < 1:
        raise ValueError(f"{field_name} must be positive")


def require_non_negative_int(field_name: str, value: int) -> None:
    """Require an integer contract field to be non-negative."""
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


def require_positive_float(field_name: str, value: float) -> None:
    """Require a numeric contract field to be positive."""
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def require_aware_datetime(field_name: str, value: datetime) -> None:
    """Require a datetime to include a usable UTC offset."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def require_canonical_tool_id(value: str) -> None:
    """Require a stable `capicove.<name>` Tool identifier."""
    if _TOOL_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("tool_id must be a canonical capicove.<name> identifier")


def require_sha256(field_name: str, value: str) -> None:
    """Require a lowercase hexadecimal SHA-256 content hash."""
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def freeze_mapping(value: Mapping[_Key, _Value]) -> Mapping[_Key, _Value]:
    """Return a read-only defensive copy of a contract mapping."""
    return MappingProxyType(dict(value))
