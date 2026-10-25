"""Tamper-evident, append-only audit chain for remote control-plane events (R3.05).

Each line carries a sequence number, the previous line's hash and the event; its
hash covers all three. Deleting, reordering or editing any line breaks the chain.
Appends hold an exclusive ``flock`` and are flushed with ``fsync`` before return.
The control store's transactional outbox is the durable source; this log is its
exported, verifiable copy.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


GENESIS = "0" * 64


def _hash(sequence: int, previous: str, event: Mapping[str, Any]) -> str:
    body = json.dumps(
        [sequence, previous, event], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(body.encode()).hexdigest()


class AuditLog:
    """Hash-chained JSON-lines audit file with owner-only permissions."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("audit log must not be a symlink")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                os.close(
                    os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                )
            except FileExistsError:
                pass

    def writable(self) -> bool:
        """Probe whether an append can currently be opened."""
        try:
            os.close(os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW))
        except OSError:
            return False
        return True

    @staticmethod
    def _tail(handle: Any) -> tuple[int, str]:
        size = os.fstat(handle.fileno()).st_size
        if size == 0:
            return 0, GENESIS
        handle.seek(max(0, size - 65536))
        last = handle.read().splitlines()[-1]
        record = json.loads(last)
        return int(record["sequence"]), str(record["hash"])

    def append(self, event: Mapping[str, Any]) -> str:
        """Append one event under an exclusive lock; raise ``OSError`` on failure."""
        descriptor = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            sequence, previous = self._tail(handle)
            sequence += 1
            digest = _hash(sequence, previous, event)
            line = json.dumps(
                {"sequence": sequence, "previous": previous, "event": dict(event), "hash": digest},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            handle.write(line.encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return digest

    def verify(self) -> int:
        """Recompute the entire chain; return the event count or raise ``ValueError``."""
        previous = GENESIS
        count = 0
        for number, line in enumerate(self.path.read_bytes().splitlines(), start=1):
            try:
                record = json.loads(line)
                valid = (
                    record["sequence"] == number
                    and record["previous"] == previous
                    and record["hash"] == _hash(number, previous, record["event"])
                )
            except (ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise ValueError(f"audit chain broken at line {number}")
            previous = record["hash"]
            count = number
        return count

    def checkpoint(self) -> str:
        """Return the verified head hash for external anchoring."""
        self.verify()
        with self.path.open("rb") as handle:
            return self._tail(handle)[1]
