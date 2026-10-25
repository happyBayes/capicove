"""Deterministic local path policy and video output storage."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path

from capicove.processing.video.ports import (
    OutputStoreError,
    PathPolicyError,
    PendingOutput,
    StoredOutput,
    WrittenClip,
)


_IDENTITY = re.compile(r"[0-9a-f]{64}")
_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,16}")
PathRoots = tuple[tuple[Path, ...], tuple[Path, ...]]


class ExplicitPathAuthorizer:
    """Authorize paths against explicitly configured resolved roots."""

    def __init__(self, policies: Mapping[str, PathRoots]) -> None:
        normalized: dict[str, PathRoots] = {}
        for policy_ref, (read_roots, write_roots) in policies.items():
            if not policy_ref.strip():
                raise ValueError("policy reference must not be blank")
            roots = read_roots + write_roots
            if any(not root.is_absolute() for root in roots):
                raise ValueError("path policy roots must be absolute")
            normalized[policy_ref] = (
                tuple(root.resolve() for root in read_roots),
                tuple(root.resolve() for root in write_roots),
            )
        self._policies = normalized

    def authorize_read(self, policy_ref: str, path: Path) -> None:
        """Authorize a resolved path under a configured read root."""
        self._authorize(policy_ref, path, write=False)

    def authorize_write(self, policy_ref: str, path: Path) -> None:
        """Authorize a resolved path under a configured write root."""
        self._authorize(policy_ref, path, write=True)

    def _authorize(self, policy_ref: str, path: Path, *, write: bool) -> None:
        try:
            roots = self._policies[policy_ref][1 if write else 0]
        except KeyError as error:
            raise PathPolicyError("execution policy is not configured") from error
        resolved = path.resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            access = "write" if write else "read"
            raise PathPolicyError(f"path is outside configured {access} roots")


class FilesystemOutputStore:
    """Store verified outputs by deterministic content-derived identity."""

    def __init__(self, root: Path) -> None:
        if not root.is_absolute():
            raise ValueError("output store root must be absolute")
        self._root = root.resolve()

    def allocate(self, identity: str, suffix: str) -> PendingOutput:
        """Allocate an exclusive partial path for one output identity."""
        self._validate_identity(identity, suffix)
        self._root.mkdir(parents=True, exist_ok=True)
        pending = self._root / f"{identity}{suffix}.partial"
        try:
            pending.touch(exist_ok=False)
        except FileExistsError as error:
            raise OutputStoreError("a partial output already exists") from error
        return PendingOutput(identity, pending)

    def finalize(self, pending: PendingOutput, written: WrittenClip) -> StoredOutput:
        """Verify and atomically finalize a backend-written output."""
        self._require_managed_partial(pending)
        if written.path.resolve() != pending.path.resolve():
            raise OutputStoreError("writer returned an unmanaged output path")
        actual_hash = _hash_file(pending.path)
        if actual_hash != written.sha256:
            raise OutputStoreError("written output hash verification failed")
        suffix = pending.path.name.removeprefix(pending.identity).removesuffix(".partial")
        final_path = self._root / f"{pending.identity}{suffix}"
        if final_path.exists():
            if _hash_file(final_path) != actual_hash:
                raise OutputStoreError("output identity collision with mismatched content")
            pending.path.unlink()
        else:
            os.replace(pending.path, final_path)
        return StoredOutput(
            identity=pending.identity,
            path=final_path,
            sha256=actual_hash,
            mime_type=written.mime_type,
        )

    def load_bytes(self, stored: StoredOutput) -> bytes:
        """Load an output only after verifying its stored content hash."""
        if not stored.path.resolve().is_relative_to(self._root):
            raise OutputStoreError("stored output path is unmanaged")
        payload = stored.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != stored.sha256:
            raise OutputStoreError("stored output hash verification failed")
        return payload

    def discard(self, pending: PendingOutput) -> None:
        """Remove one exact managed partial output if it exists."""
        self._require_managed_partial(pending)
        pending.path.unlink(missing_ok=True)

    def _require_managed_partial(self, pending: PendingOutput) -> None:
        resolved = pending.path.resolve()
        if resolved.parent != self._root or not resolved.name.endswith(".partial"):
            raise OutputStoreError("partial output path is unmanaged")
        if not resolved.name.startswith(pending.identity):
            raise OutputStoreError("partial output identity does not match its path")

    @staticmethod
    def _validate_identity(identity: str, suffix: str) -> None:
        if _IDENTITY.fullmatch(identity) is None:
            raise OutputStoreError("output identity must be a SHA-256 digest")
        if _SUFFIX.fullmatch(suffix) is None:
            raise OutputStoreError("output suffix is not permitted")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
