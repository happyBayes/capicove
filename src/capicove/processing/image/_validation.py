"""Private validation helpers for image-processing contracts."""

from __future__ import annotations

import re


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def require_non_blank(field_name: str, value: str) -> None:
    """Require an image contract string to contain non-whitespace content."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def require_sha256(field_name: str, value: str) -> None:
    """Require a lowercase hexadecimal SHA-256 content hash."""
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")
