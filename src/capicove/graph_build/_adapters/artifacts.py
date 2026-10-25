"""Owner-only, content-addressed artifacts with persisted scope and source lineage."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.codec import canonical_bytes
from capicove.graph_build.ports import AccessGate
from capicove.graph_build.receipts import require_content_id


class ArtifactStoreError(RuntimeError):
    """Report a bounded storage/integrity failure without paths or payload text."""


def _private_stat(info: os.stat_result, *, directory: bool = False) -> None:
    correct_kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if not correct_kind or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ArtifactStoreError("protected storage permissions or file type are invalid")


@contextmanager
def _private_directory(root: Path, *, create: bool = False) -> Iterator[int]:
    """Walk from / with no-follow directory descriptors, pinning the final root.

    Existing ancestor permissions belong to the host; the explicit storage root
    must be owner-only. Newly created ancestors are also owner-only. Never resolve
    symlinks or repair permissions on pre-existing paths. During creation, persist
    every observed child in its parent before descending, even if it already
    exists: another constructor or a failed attempt may not have synced its link.
    """
    if (
        not isinstance(root, Path)
        or not root.is_absolute()
        or ".." in root.parts
        or root == Path("/")
    ):
        raise ValueError("storage root must be an explicit absolute non-root Path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in root.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            try:
                if create:
                    os.fsync(descriptor)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        _private_stat(os.fstat(descriptor), directory=True)
        yield descriptor
    except PermissionError:
        raise
    except OSError as error:
        raise ArtifactStoreError("protected storage is unavailable or unsafe") from error
    finally:
        os.close(descriptor)


def _read_private(directory: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        _private_stat(os.fstat(stream.fileno()))
        return stream.read()


class FilesystemProtectedArtifacts:
    """Bind each writer to a fixed source set; persist that set in every artifact.

    The artifact ID hashes a canonical envelope containing compound scope, source
    refs, payload SHA-256, and base64 bytes. Receipt result digests hash raw bytes.
    Different source sets therefore cannot silently share provenance. Root paths
    are explicit host configuration and must contain no symlink components.
    """

    def __init__(self, root: Path, gate: AccessGate, *, evidence_refs: tuple[str, ...]) -> None:
        raw_refs: object = evidence_refs
        if isinstance(raw_refs, str) or not evidence_refs:
            raise ValueError("an explicit nonempty evidence set is required")
        if any(not isinstance(ref, str) or not ref.strip() for ref in evidence_refs):
            raise ValueError("evidence refs must be nonblank strings")
        self._evidence_refs = tuple(sorted(set(evidence_refs)))
        self._root = root
        self._gate = gate
        with _private_directory(root, create=True):
            pass

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        """Return the source binding used for all writes by this instance."""
        return self._evidence_refs

    def put(self, scope: GraphScope, payload: bytes) -> str:
        """Publish an immutable complete envelope, without overwriting existing bytes."""
        self._gate.require_scope(scope)
        self._gate.require_access(self.evidence_refs, scope)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        encoded = canonical_bytes(
            {
                "schema_version": 1,
                "workspace_ref": scope.workspace_ref,
                "scope_ref": scope.scope_ref,
                "evidence_refs": self.evidence_refs,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
            }
        )
        ref = "artifact:" + hashlib.sha256(encoded).hexdigest()
        name = ref.removeprefix("artifact:") + ".json"
        with _private_directory(self._root) as directory:
            temporary = ".pending-" + uuid.uuid4().hex
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=directory,
                        dst_dir_fd=directory,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    self._decode(scope, ref, _read_private(directory, name))
                os.fsync(directory)
            finally:
                # Only this call's exclusively created temporary is removed.
                os.unlink(temporary, dir_fd=directory)
                os.fsync(directory)
        return ref

    def read(self, scope: GraphScope, artifact_ref: str) -> bytes:
        """Verify an opaque content ID and reauthorize the envelope's own sources."""
        self._gate.require_scope(scope)
        require_content_id(artifact_ref, "artifact")
        with _private_directory(self._root) as directory:
            encoded = _read_private(directory, artifact_ref.removeprefix("artifact:") + ".json")
        return self._decode(scope, artifact_ref, encoded)

    def _decode(self, scope: GraphScope, ref: str, encoded: bytes) -> bytes:
        if hashlib.sha256(encoded).hexdigest() != ref.removeprefix("artifact:"):
            raise ArtifactStoreError("artifact content digest mismatch")
        try:
            value = json.loads(encoded)
            if not isinstance(value, dict) or set(value) != {
                "schema_version",
                "workspace_ref",
                "scope_ref",
                "evidence_refs",
                "payload_sha256",
                "payload_base64",
            }:
                raise ValueError("invalid envelope")
            if type(value["schema_version"]) is not int or value["schema_version"] != 1:
                raise ValueError("invalid version")
            artifact_scope = GraphScope(value["workspace_ref"], value["scope_ref"])
            refs = value["evidence_refs"]
            if (
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
            ):
                raise ValueError("invalid evidence")
            if refs != sorted(set(refs)) or canonical_bytes(value) != encoded:
                raise ValueError("noncanonical envelope")
            payload = base64.b64decode(value["payload_base64"], validate=True)
            if hashlib.sha256(payload).hexdigest() != value["payload_sha256"]:
                raise ValueError("invalid payload digest")
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            raise ArtifactStoreError("artifact envelope is corrupt") from error
        if artifact_scope != scope:
            raise PermissionError("artifact belongs to another scope")
        self._gate.require_access(tuple(refs), scope)
        return payload
