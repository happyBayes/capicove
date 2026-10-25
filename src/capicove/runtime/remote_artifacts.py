"""Owner-scoped resumable artifact uploads and authorized bounded reads (R3.01).

Implements the RFC-0028 upload protocol as an application service: begin reserves
aggregate application quota once per idempotency key, numbered chunks of at most
256 KiB are idempotent by digest, and commit assembles the chunks in quarantine,
verifies size, SHA-256 and declared media type, then publishes one immutable
artifact. Every call is reauthorized through the host ``authorize`` hook. Other
owners observe ``not_found`` for every foreign or absent ID, never hash existence.
No server path appears in any returned value.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from capicove.contracts.remote import ArtifactRef, AuthorityKey
from capicove.runtime._remote.artifact_store import (
    ArtifactLedger,
    private_directory,
    read_private,
    write_private,
)
from capicove.runtime.remote_recovery import TombstoneJournal


CHUNK_BYTES = 256 * 1024
_MEDIA = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$")
_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_UPLOAD = re.compile(r"^upload:[0-9a-f]{32}$")
_ARTIFACT = re.compile(r"^artifact:[0-9a-f]{32}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURES = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "application/pdf": (b"%PDF-",),
    "image/gif": (b"GIF87a", b"GIF89a"),
}


class ArtifactError(RuntimeError):
    """A typed artifact failure; ``code`` is stable and safe to return."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True, slots=True)
class UploadStatus:
    """Owner-visible upload progress: received chunk indices and digests."""

    upload_id: str
    state: str
    size: int
    sha256: str
    media_type: str
    chunks: tuple[tuple[int, str], ...]


def _canonical(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _owner(authority: AuthorityKey) -> str:
    key = authority.principal_key
    return _canonical(
        [
            key.issuer,
            key.tenant_id,
            key.application_id,
            key.subject_kind,
            key.subject,
            authority.workspace_binding_id,
            authority.grant_id,
            authority.audience,
        ]
    )


def _application(authority: AuthorityKey) -> str:
    key = authority.principal_key
    return _canonical([key.issuer, key.tenant_id, key.application_id])


def _tenant(authority: AuthorityKey) -> str:
    key = authority.principal_key
    return _canonical([key.issuer, key.tenant_id])


def _media_matches(media_type: str, head: bytes, path: Path) -> bool:
    signatures = _SIGNATURES.get(media_type)
    if signatures is not None:
        return any(head.startswith(signature) for signature in signatures)
    if media_type == "audio/wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if media_type.startswith("text/") or media_type == "application/json":
        try:
            read_private(path).decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


class ArtifactService:
    """Resumable uploads, quota accounting and reauthorized reads over one store."""

    def __init__(
        self,
        root: Path,
        *,
        max_artifact_bytes: int = 1024**3,
        application_quota_bytes: int = 10 * 1024**3,
        tenant_quota_bytes: int = 100 * 1024**3,
        upload_ttl_seconds: float = 24 * 3600,
        tombstones: TombstoneJournal | None = None,
        authorize: Callable[[AuthorityKey], None] | None = None,
        barrier: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._root = private_directory(Path(root))
        for name in ("uploads", "quarantine", "artifacts"):
            private_directory(self._root / name)
        self._ledger = ArtifactLedger(self._root / "control.sqlite3")
        self._max = max_artifact_bytes
        self._quota = application_quota_bytes
        self._tenant_quota = tenant_quota_bytes
        self._tombstones = tombstones
        self._ttl = upload_ttl_seconds
        self._authorize = authorize or (lambda _authority: None)
        self._barrier = barrier or (lambda _name: None)
        self._clock = clock
        if tombstones is not None:
            self._apply_tombstones(tombstones.entries("artifact"))

    def _apply_tombstones(self, deleted: frozenset[str]) -> None:
        """Reapply recorded deletions, e.g. after restoring an older backup."""
        with self._ledger.transaction() as connection:
            for artifact_id in sorted(deleted):
                row = connection.execute(
                    "SELECT application, size FROM artifacts WHERE artifact_id = ? AND state = 'live'",
                    (artifact_id,),
                ).fetchone()
                if row is None:
                    continue
                connection.execute(
                    "UPDATE artifacts SET state = 'deleted' WHERE artifact_id = ?", (artifact_id,)
                )
                connection.execute(
                    "UPDATE quotas SET stored = stored - ? WHERE application = ?", (row[1], row[0])
                )
        for artifact_id in deleted:
            (self._root / "artifacts" / artifact_id[9:]).unlink(missing_ok=True)

    def _admit(self, authority: AuthorityKey) -> str:
        if not isinstance(authority, AuthorityKey):
            raise ArtifactError("invalid_input", "authority is required")
        try:
            self._authorize(authority)
        except PermissionError as error:
            raise ArtifactError("permission_denied") from error
        return _owner(authority)

    def _expire(self, connection: object) -> int:
        rows = connection.execute(  # type: ignore[attr-defined]
            "SELECT upload_id, application, size FROM uploads WHERE state = 'open' AND expires <= ?",
            (self._clock(),),
        ).fetchall()
        for upload_id, application, size in rows:
            connection.execute(  # type: ignore[attr-defined]
                "UPDATE uploads SET state = 'expired' WHERE upload_id = ?", (upload_id,)
            )
            connection.execute(  # type: ignore[attr-defined]
                "UPDATE quotas SET reserved = reserved - ? WHERE application = ?",
                (size, application),
            )
        return len(rows)

    def _upload(self, connection: object, owner: str, upload_id: str) -> tuple[object, ...]:
        if not isinstance(upload_id, str) or not _UPLOAD.fullmatch(upload_id):
            raise ArtifactError("not_found")
        row = connection.execute(  # type: ignore[attr-defined]
            "SELECT upload_id, application, size, sha256, media_type, state, artifact_id"
            " FROM uploads WHERE upload_id = ? AND owner = ?",
            (upload_id, owner),
        ).fetchone()
        if row is None:
            raise ArtifactError("not_found")
        return tuple(row)

    def _status(self, connection: object, row: tuple[object, ...]) -> UploadStatus:
        chunks = connection.execute(  # type: ignore[attr-defined]
            "SELECT idx, sha256 FROM chunks WHERE upload_id = ? ORDER BY idx", (row[0],)
        ).fetchall()
        return UploadStatus(
            upload_id=str(row[0]),
            state=str(row[5]),
            size=int(row[2]),  # type: ignore[call-overload]
            sha256=str(row[3]),
            media_type=str(row[4]),
            chunks=tuple((int(index), str(digest)) for index, digest in chunks),
        )

    def begin(
        self,
        authority: AuthorityKey,
        *,
        size: int,
        sha256: str,
        media_type: str,
        idempotency_key: str,
    ) -> UploadStatus:
        """Reserve quota once per key; an identical retry returns the same upload."""
        owner = self._admit(authority)
        if type(size) is not int or size < 1:
            raise ArtifactError("invalid_input", "size must be a positive integer")
        if size > self._max:
            raise ArtifactError("too_large")
        if not isinstance(sha256, str) or not _HEX.fullmatch(sha256):
            raise ArtifactError("invalid_input", "sha256 must be lowercase hex")
        if not isinstance(media_type, str) or not _MEDIA.fullmatch(media_type):
            raise ArtifactError("invalid_input", "media type must be a bare type/subtype")
        if not isinstance(idempotency_key, str) or not _KEY.fullmatch(idempotency_key):
            raise ArtifactError("invalid_input", "invalid idempotency key")
        application = _application(authority)
        with self._ledger.transaction() as connection:
            self._expire(connection)
            existing = connection.execute(
                "SELECT upload_id, application, size, sha256, media_type, state, artifact_id"
                " FROM uploads WHERE owner = ? AND idempotency_key = ?",
                (owner, idempotency_key),
            ).fetchone()
            if existing is not None:
                if tuple(existing[2:5]) != (size, sha256, media_type):
                    raise ArtifactError("conflict", "idempotency key reused with changed input")
                return self._status(connection, tuple(existing))
            tenant = _tenant(authority)
            connection.execute(
                "INSERT OR IGNORE INTO quotas(application, tenant, reserved, stored)"
                " VALUES (?, ?, 0, 0)",
                (application, tenant),
            )
            reserved, stored = connection.execute(
                "SELECT reserved, stored FROM quotas WHERE application = ?", (application,)
            ).fetchone()
            (tenant_used,) = connection.execute(
                "SELECT COALESCE(SUM(reserved + stored), 0) FROM quotas WHERE tenant = ?",
                (tenant,),
            ).fetchone()
            if reserved + stored + size > self._quota or tenant_used + size > self._tenant_quota:
                raise ArtifactError("quota_exceeded")
            upload_id = "upload:" + secrets.token_hex(16)
            now = self._clock()
            connection.execute(
                "INSERT INTO uploads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL)",
                (
                    upload_id,
                    owner,
                    application,
                    idempotency_key,
                    size,
                    sha256,
                    media_type,
                    now,
                    now + self._ttl,
                ),
            )
            connection.execute(
                "UPDATE quotas SET reserved = reserved + ? WHERE application = ?",
                (size, application),
            )
            return UploadStatus(upload_id, "open", size, sha256, media_type, ())

    def write_chunk(
        self, authority: AuthorityKey, upload_id: str, index: int, data: bytes
    ) -> UploadStatus:
        """Store one numbered chunk; the same digest is idempotent, a change conflicts."""
        owner = self._admit(authority)
        if type(index) is not int or index < 0 or type(data) is not bytes:
            raise ArtifactError("invalid_input", "chunk index or bytes are invalid")
        with self._ledger.transaction() as connection:
            self._expire(connection)
            row = self._upload(connection, owner, upload_id)
        if row[5] != "open":
            raise ArtifactError("conflict", "upload is not open")
        size = int(row[2])  # type: ignore[call-overload]
        count = -(-size // CHUNK_BYTES)
        expected = CHUNK_BYTES if index < count - 1 else size - CHUNK_BYTES * (count - 1)
        if index >= count or len(data) != expected:
            raise ArtifactError("invalid_input", "chunk index or length does not fit the upload")
        digest = hashlib.sha256(data).hexdigest()
        directory = private_directory(self._root / "uploads" / upload_id[7:])
        target = directory / f"{index}.{digest}"
        if not target.exists():
            write_private(target, data)
        with self._ledger.transaction() as connection:
            row = self._upload(connection, owner, upload_id)
            if row[5] != "open":
                # A concurrent abort or commit won; leave no unreferenced bytes behind.
                if row[5] != "committing":
                    self._discard(upload_id)
                raise ArtifactError("conflict", "upload is not open")
            prior = connection.execute(
                "SELECT sha256 FROM chunks WHERE upload_id = ? AND idx = ?", (upload_id, index)
            ).fetchone()
            if prior is not None and prior[0] != digest:
                raise ArtifactError("conflict", "chunk was already received with other bytes")
            if prior is None:
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)", (upload_id, index, digest, len(data))
                )
            return self._status(connection, row)

    def status(self, authority: AuthorityKey, upload_id: str) -> UploadStatus:
        """Return progress only to the owning authority."""
        owner = self._admit(authority)
        with self._ledger.transaction() as connection:
            self._expire(connection)
            return self._status(connection, self._upload(connection, owner, upload_id))

    def abort(self, authority: AuthorityKey, upload_id: str) -> None:
        """Release an incomplete upload exactly once; a started commit wins."""
        owner = self._admit(authority)
        with self._ledger.transaction() as connection:
            row = self._upload(connection, owner, upload_id)
            if row[5] in {"aborted", "expired", "rejected"}:
                return
            if row[5] != "open":
                raise ArtifactError("conflict", "upload is committing or committed")
            connection.execute(
                "UPDATE uploads SET state = 'aborted' WHERE upload_id = ?", (upload_id,)
            )
            connection.execute(
                "UPDATE quotas SET reserved = reserved - ? WHERE application = ?",
                (row[2], row[1]),
            )
        self._discard(upload_id)

    def _discard(self, upload_id: str) -> None:
        directory = self._root / "uploads" / upload_id[7:]
        if directory.is_dir() and not directory.is_symlink():
            for path in directory.iterdir():
                path.unlink(missing_ok=True)
            directory.rmdir()
        (self._root / "quarantine" / upload_id[7:]).unlink(missing_ok=True)

    def _reject(self, upload_id: str, application: object, size: object, code: str) -> None:
        with self._ledger.transaction() as connection:
            connection.execute(
                "UPDATE uploads SET state = 'rejected' WHERE upload_id = ? AND state = 'committing'",
                (upload_id,),
            )
            connection.execute(
                "UPDATE quotas SET reserved = reserved - ? WHERE application = ?",
                (size, application),
            )
        self._discard(upload_id)
        raise ArtifactError(code)

    def commit(self, authority: AuthorityKey, upload_id: str) -> ArtifactRef:
        """Verify contiguous chunks, size, hash and media type, then publish once."""
        owner = self._admit(authority)
        with self._ledger.transaction() as connection:
            self._expire(connection)
            row = self._upload(connection, owner, upload_id)
            if row[5] == "committed":
                return self._reference(connection, str(row[6]))
            if row[5] not in {"open", "committing"}:
                raise ArtifactError("conflict", "upload is not open")
            size = int(row[2])  # type: ignore[call-overload]
            chunks = connection.execute(
                "SELECT idx, sha256 FROM chunks WHERE upload_id = ? ORDER BY idx", (upload_id,)
            ).fetchall()
            if [index for index, _ in chunks] != list(range(-(-size // CHUNK_BYTES))):
                raise ArtifactError("incomplete", "chunks are missing")
            connection.execute(
                "UPDATE uploads SET state = 'committing' WHERE upload_id = ?", (upload_id,)
            )
        application, sha256, media_type = row[1], str(row[3]), str(row[4])
        directory = self._root / "uploads" / upload_id[7:]
        quarantine = self._root / "quarantine" / upload_id[7:]
        assembled = hashlib.sha256()
        parts = []
        for index, digest in chunks:
            part = read_private(directory / f"{index}.{digest}")
            if hashlib.sha256(part).hexdigest() != digest:
                self._reject(upload_id, application, size, "integrity_mismatch")
            assembled.update(part)
            parts.append(part)
        if assembled.hexdigest() != sha256 or sum(map(len, parts)) != size:
            self._reject(upload_id, application, size, "integrity_mismatch")
        write_private(quarantine, b"".join(parts))
        if not _media_matches(media_type, read_private(quarantine, 0, 16), quarantine):
            self._reject(upload_id, application, size, "media_mismatch")
        artifact_id = (
            "artifact:" + hashlib.sha256(("artifact|" + upload_id).encode()).hexdigest()[:32]
        )
        quarantine.replace(self._root / "artifacts" / artifact_id[9:])
        self._barrier("after_publish_file")
        with self._ledger.transaction() as connection:
            current = self._upload(connection, owner, upload_id)
            if current[5] == "committed":
                return self._reference(connection, artifact_id)
            connection.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, 1, '[]', ?, 'live')",
                (
                    artifact_id,
                    owner,
                    application,
                    upload_id,
                    sha256,
                    size,
                    media_type,
                    self._clock(),
                ),
            )
            connection.execute(
                "UPDATE uploads SET state = 'committed', artifact_id = ? WHERE upload_id = ?",
                (artifact_id, upload_id),
            )
            connection.execute(
                "UPDATE quotas SET reserved = reserved - ?, stored = stored + ? WHERE application = ?",
                (size, size, application),
            )
            reference = self._reference(connection, artifact_id)
        self._discard(upload_id)
        return reference

    def _reference(self, connection: object, artifact_id: str) -> ArtifactRef:
        row = connection.execute(  # type: ignore[attr-defined]
            "SELECT sha256, size, media_type, version, lineage FROM artifacts"
            " WHERE artifact_id = ? AND state = 'live'",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise ArtifactError("not_found")
        return ArtifactRef(artifact_id, row[0], row[1], row[2], row[3], tuple(json.loads(row[4])))

    def _owned(self, connection: object, owner: str, artifact_id: str) -> ArtifactRef:
        if not isinstance(artifact_id, str) or not _ARTIFACT.fullmatch(artifact_id):
            raise ArtifactError("not_found")
        found = connection.execute(  # type: ignore[attr-defined]
            "SELECT 1 FROM artifacts WHERE artifact_id = ? AND owner = ? AND state = 'live'",
            (artifact_id, owner),
        ).fetchone()
        if found is None:
            raise ArtifactError("not_found")
        return self._reference(connection, artifact_id)

    def stat(self, authority: AuthorityKey, artifact_id: str) -> ArtifactRef:
        """Return the reference only to its owner while it is live and authorized."""
        owner = self._admit(authority)
        with self._ledger.transaction() as connection:
            return self._owned(connection, owner, artifact_id)

    def read_chunk(
        self, authority: AuthorityKey, artifact_id: str, offset: int, length: int
    ) -> bytes:
        """Read at most one chunk; authorization is rechecked on every call."""
        owner = self._admit(authority)
        if type(offset) is not int or type(length) is not int or offset < 0:
            raise ArtifactError("invalid_input", "offset must be a nonnegative integer")
        if not 1 <= length <= CHUNK_BYTES:
            raise ArtifactError("invalid_input", "length must be 1..262144")
        with self._ledger.transaction() as connection:
            reference = self._owned(connection, owner, artifact_id)
        if offset >= reference.size:
            return b""
        return read_private(self._root / "artifacts" / artifact_id[9:], offset, length)

    def list_artifacts(self, authority: AuthorityKey) -> tuple[ArtifactRef, ...]:
        """List the caller's live artifacts in creation order."""
        owner = self._admit(authority)
        with self._ledger.transaction() as connection:
            rows = connection.execute(
                "SELECT artifact_id FROM artifacts WHERE owner = ? AND state = 'live'"
                " ORDER BY created, artifact_id",
                (owner,),
            ).fetchall()
            return tuple(self._reference(connection, row[0]) for row in rows)

    def delete(self, authority: AuthorityKey, artifact_id: str) -> None:
        """Tombstone an artifact, release its stored quota and remove its bytes."""
        owner = self._admit(authority)
        with self._ledger.transaction() as connection:
            reference = self._owned(connection, owner, artifact_id)
            if self._tombstones is not None:
                # Record first: a crash after this point still deletes on next open.
                self._tombstones.record("artifact", artifact_id)
            connection.execute(
                "UPDATE artifacts SET state = 'deleted' WHERE artifact_id = ?", (artifact_id,)
            )
            connection.execute(
                "UPDATE quotas SET stored = stored - ? WHERE application = ?",
                (reference.size, _application(authority)),
            )
        (self._root / "artifacts" / artifact_id[9:]).unlink(missing_ok=True)

    def usage(self, authority: AuthorityKey) -> dict[str, int]:
        """Return the caller application's aggregate reserved and stored bytes."""
        self._admit(authority)
        with self._ledger.transaction() as connection:
            self._expire(connection)
            row = connection.execute(
                "SELECT reserved, stored FROM quotas WHERE application = ?",
                (_application(authority),),
            ).fetchone()
        return (
            {"reserved": 0, "stored": 0} if row is None else {"reserved": row[0], "stored": row[1]}
        )

    def purge(self, *, retention_seconds: float) -> dict[str, int]:
        """Expire stale uploads and delete artifacts older than the retention period.

        Only objects inside this store are touched; canonical source evidence lives
        elsewhere and is never deleted implicitly.
        """
        cutoff = self._clock() - retention_seconds
        with self._ledger.transaction() as connection:
            expired = connection.execute(
                "SELECT upload_id FROM uploads WHERE state = 'open' AND expires <= ?",
                (self._clock(),),
            ).fetchall()
            uploads = self._expire(connection)
            rows = connection.execute(
                "SELECT artifact_id, application, size FROM artifacts"
                " WHERE state = 'live' AND created <= ?",
                (cutoff,),
            ).fetchall()
            for artifact_id, application, size in rows:
                if self._tombstones is not None:
                    self._tombstones.record("artifact", artifact_id)
                connection.execute(
                    "UPDATE artifacts SET state = 'deleted' WHERE artifact_id = ?", (artifact_id,)
                )
                connection.execute(
                    "UPDATE quotas SET stored = stored - ? WHERE application = ?",
                    (size, application),
                )
        for (upload_id,) in expired:
            self._discard(upload_id)
        for artifact_id, _, _ in rows:
            (self._root / "artifacts" / artifact_id[9:]).unlink(missing_ok=True)
        return {"artifacts": len(rows), "uploads": uploads}
