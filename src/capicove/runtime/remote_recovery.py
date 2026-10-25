"""Restore gate: deletions and voids survive restoring older store backups (R3.05).

The tombstone journal lives outside the restorable control and artifact stores.
Services record a tombstone before they apply a deletion or void, and reapply
every recorded tombstone when they open a store, so restoring an older backup
cannot resurrect deleted artifacts or reusable approvals.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path


_KINDS = frozenset({"artifact", "approval"})
_IDENTIFIER = re.compile(r"^[a-z]+:[0-9a-f]{32}$")


class TombstoneJournal:
    """Owner-only append-only record of irreversible deletions and voids."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink():
            raise ValueError("tombstone journal must not be a symlink")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                os.close(
                    os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                )
            except FileExistsError:
                pass

    def record(self, kind: str, identifier: str) -> None:
        """Durably append one tombstone before the corresponding store change."""
        if kind not in _KINDS or not _IDENTIFIER.fullmatch(identifier):
            raise ValueError("invalid tombstone")
        line = json.dumps({"kind": kind, "id": identifier}, separators=(",", ":")) + "\n"
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line.encode())
            handle.flush()
            os.fsync(handle.fileno())

    def entries(self, kind: str) -> frozenset[str]:
        """Return every recorded identifier of one kind; malformed lines fail closed."""
        found = set()
        for line in self.path.read_bytes().splitlines():
            record = json.loads(line)
            if (
                not isinstance(record, dict)
                or record.get("kind") not in _KINDS
                or not _IDENTIFIER.fullmatch(str(record.get("id")))
            ):
                raise ValueError("tombstone journal is corrupt")
            if record["kind"] == kind:
                found.add(record["id"])
        return frozenset(found)
