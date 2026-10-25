"""Process-local reuse of values decoded from re-verified content (RFC-0019).

Callers re-read and re-hash the exact bytes on every call and look up a value
only under that freshly computed content identity. A hit therefore skips pure
decoding of byte-identical input; it never skips path safety, hashing, scope,
authorization or cross-artifact validation. Nothing here is persisted.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any


_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_OPT_OUT = "CAPICOVE_GRAPH_DECODE_REUSE"


def reuse_enabled() -> bool:
    """Return False when the explicit diagnostic opt-out is set to ``0``."""
    return os.environ.get(_OPT_OUT, "1") != "0"


class VerifiedDecodeMemo:
    """Thread-safe LRU bounded by the source bytes of its retained entries.

    Values must be immutable. ``None`` is reserved to report a miss and is
    never stored.
    """

    def __init__(self, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        self._max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        """Return the stored value and mark it recently used, or None."""
        if not reuse_enabled():
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: Hashable, value: Any, size: int) -> None:
        """Retain one successfully verified value, evicting the oldest entries."""
        if value is None or not reuse_enabled() or size > self._max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._size -= previous[1]
            self._entries[key] = (value, size)
            self._size += size
            while self._size > self._max_bytes:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._size -= evicted

    def clear(self) -> None:
        """Drop every retained value."""
        with self._lock:
            self._entries.clear()
            self._size = 0


GRAPH_MANIFESTS = VerifiedDecodeMemo()
GRAPH_TABLES = VerifiedDecodeMemo()
STAGE_ARTIFACTS = VerifiedDecodeMemo()
