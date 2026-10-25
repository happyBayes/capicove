"""Database-independent result cache contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class CacheConflictError(RuntimeError):
    """Report conflicting payloads for one immutable cache key."""


class ResultCache(Protocol):
    """Store JSON-compatible results by a complete content cache key."""

    def get(self, cache_key: str) -> Mapping[str, object] | None:
        """Return a cached result or ``None`` when the key is absent."""

    def put(self, cache_key: str, result: Mapping[str, object]) -> None:
        """Store one result idempotently without overwriting conflicts."""

    def evict(self, cache_key: str) -> None:
        """Remove one result that failed operation-specific validation."""
