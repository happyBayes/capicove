"""Job-owned checkpoint files with a separately shared result cache."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from capicove.batch._adapters.sqlite import SqliteBatchStore


_JOB_ID = re.compile(r"job:([0-9a-f]{24})\Z")


class JobCheckpointStores:
    """Resolve one private Batch checkpoint database per durable Job ID."""

    def __init__(self, cache_store: SqliteBatchStore) -> None:
        self._root = cache_store.database_path.with_name(f"{cache_store.database_path.stem}-jobs")

    def for_job(self, job_id: str, *, require_existing: bool = False) -> SqliteBatchStore:
        """Open a Job's checkpoint store without sharing item state across Jobs."""
        matched = _JOB_ID.fullmatch(job_id)
        if matched is None:
            raise ValueError("invalid checkpoint job ID")
        if any(parent.is_symlink() for parent in self._root.parents) or self._root.is_symlink():
            raise ValueError("checkpoint root must not contain symlinks")
        if require_existing and not self._root.exists():
            raise FileNotFoundError("checkpoint database is unavailable")
        self._root.mkdir(mode=0o700, exist_ok=True)
        if not self._root.is_dir():
            raise ValueError("checkpoint root must be a directory")
        if os.name == "posix":
            metadata = self._root.stat()
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise ValueError("checkpoint root must be owner-private")
        path = self._root / f"{matched.group(1)}.sqlite3"
        if require_existing and not path.exists():
            raise FileNotFoundError("checkpoint database is unavailable")
        if os.name == "nt":
            from capicove._windows_files import validate_windows_file_target

            validate_windows_file_target(path)
        self._prepare_file(path)
        return SqliteBatchStore(path)

    @staticmethod
    def _prepare_file(path: Path) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o077
                or (os.name == "posix" and metadata.st_uid != os.geteuid())
            ):
                raise ValueError("checkpoint database must be a private regular file") from None
        else:
            os.close(descriptor)
