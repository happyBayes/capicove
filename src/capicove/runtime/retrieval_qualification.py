"""Trusted local preview-qualification evidence for retrieval activation.

Issuing evidence is an explicit administrator-side capability.  Search requests
never provide a verdict and possession of the returned locator conveys no
authority: activation still reopens and verifies the signed local receipt.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from capicove.retrieval._adapters.sqlite_image_generations import (
    GenerationQualificationError,
    PreviewActivationQualification,
    PublishedGenerationRecord,
    VerifiedPreviewQualification,
)


PREVIEW_QUALIFICATION_SCHEMA = "capicove.local-preview-qualification"
PREVIEW_QUALIFICATION_SCHEMA_VERSION = 1
DEFAULT_MAXIMUM_EVIDENCE_BYTES = 16 * 1024

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_READ_CHUNK_BYTES = 16 * 1024
_SIGNATURE_DOMAIN = b"capicove.local-preview-qualification.v1\0"
_KEY_ID_DOMAIN = b"capicove.local-preview-qualification.key.v1\0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_QUALIFICATION_ID = re.compile(r"preview-[0-9a-f]{32}\Z")
_DOCUMENT_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "qualification_tier",
        "qualification_id",
        "activation_set_id",
        "manifest_sha256",
        "profile_ref",
        "report_sha256",
        "policy_sha256",
        "verification_receipt_sha256",
        "verified_at",
        "issuer_key_sha256",
        "hmac_sha256",
    }
)
_SIGNED_FIELDS = _DOCUMENT_FIELDS - {"hmac_sha256"}


class LocalPreviewQualificationStore:
    """Issue and verify bounded, signed preview qualifications below one root.

    The HMAC key is a startup capability and is never persisted.  Construct this
    object only in the trusted local administration composition root; do not
    expose :meth:`issue` through a search request or tool schema.
    """

    def __init__(
        self,
        trusted_root: Path,
        *,
        hmac_key: bytes,
        maximum_evidence_bytes: int = DEFAULT_MAXIMUM_EVIDENCE_BYTES,
    ) -> None:
        if not isinstance(trusted_root, Path) or not trusted_root.is_absolute():
            raise ValueError("trusted_root must be an absolute Path")
        if not isinstance(hmac_key, bytes) or len(hmac_key) < 32:
            raise ValueError("hmac_key must contain at least 32 bytes")
        if (
            isinstance(maximum_evidence_bytes, bool)
            or not isinstance(maximum_evidence_bytes, int)
            or maximum_evidence_bytes < 1_024
            or maximum_evidence_bytes > 1024 * 1024
        ):
            raise ValueError("maximum_evidence_bytes must be between 1024 and 1048576")
        self._root = trusted_root
        self._key = bytes(hmac_key)
        self._key_id = hashlib.sha256(_KEY_ID_DOMAIN + self._key).hexdigest()
        self._maximum_evidence_bytes = maximum_evidence_bytes
        self._initialize_root()

    @property
    def trusted_root(self) -> Path:
        """Return the absolute root containing opaque qualification receipts."""
        return self._root

    def issue(
        self,
        published: PublishedGenerationRecord,
        *,
        report_sha256: str,
        policy_sha256: str,
        verification_receipt_sha256: str,
        verified_at: datetime,
        deadline: datetime,
    ) -> PreviewActivationQualification:
        """Persist one administrator-issued preview receipt without activating it."""
        if not isinstance(published, PublishedGenerationRecord):
            raise TypeError("published must be a PublishedGenerationRecord")
        _check_deadline(deadline)
        for name, value in (
            ("report_sha256", report_sha256),
            ("policy_sha256", policy_sha256),
            ("verification_receipt_sha256", verification_receipt_sha256),
        ):
            _require_sha256(name, value)
        normalized_verified_at = _validated_verified_at(verified_at, published)
        qualification_id = f"preview-{secrets.token_hex(16)}"
        signed_document: dict[str, object] = {
            "schema": PREVIEW_QUALIFICATION_SCHEMA,
            "schema_version": PREVIEW_QUALIFICATION_SCHEMA_VERSION,
            "qualification_tier": "preview",
            "qualification_id": qualification_id,
            "activation_set_id": published.activation_set_id,
            "manifest_sha256": published.manifest_sha256,
            "profile_ref": published.retrieval_profile_ref,
            "report_sha256": report_sha256,
            "policy_sha256": policy_sha256,
            "verification_receipt_sha256": verification_receipt_sha256,
            "verified_at": _encode_datetime(normalized_verified_at),
            "issuer_key_sha256": self._key_id,
        }
        signature = self._sign(signed_document)
        document = {**signed_document, "hmac_sha256": signature}
        payload = _canonical_json(document) + b"\n"
        if len(payload) > self._maximum_evidence_bytes:
            raise GenerationQualificationError("qualification evidence exceeds its size limit")
        _check_deadline(deadline)
        self._write_new_receipt(qualification_id, payload, deadline=deadline)
        return PreviewActivationQualification(qualification_id)

    def verify(
        self,
        qualification: PreviewActivationQualification,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> VerifiedPreviewQualification:
        """Reopen and authenticate one receipt against the exact publication."""
        if not isinstance(qualification, PreviewActivationQualification):
            raise TypeError("qualification must be a PreviewActivationQualification")
        if not isinstance(published, PublishedGenerationRecord):
            raise TypeError("published must be a PublishedGenerationRecord")
        _check_deadline(deadline)
        qualification_id = qualification.qualification_id
        if _QUALIFICATION_ID.fullmatch(qualification_id) is None:
            raise GenerationQualificationError("qualification locator has an invalid format")
        payload = self._read_receipt(qualification_id, deadline=deadline)
        document = _decode_canonical_document(payload)
        _require_exact_schema(document)
        signed_document = {name: document[name] for name in _SIGNED_FIELDS}
        stored_signature = _require_sha256_value("hmac_sha256", document["hmac_sha256"])
        if not hmac.compare_digest(stored_signature, self._sign(signed_document)):
            raise GenerationQualificationError("qualification signature is invalid")
        if document["issuer_key_sha256"] != self._key_id:
            raise GenerationQualificationError(
                "qualification issuer key does not match startup key"
            )

        stored_qualification_id = _require_string("qualification_id", document["qualification_id"])
        activation_set_id = _require_string("activation_set_id", document["activation_set_id"])
        manifest_sha256 = _require_sha256_value("manifest_sha256", document["manifest_sha256"])
        profile_ref = _require_string("profile_ref", document["profile_ref"])
        report_sha256 = _require_sha256_value("report_sha256", document["report_sha256"])
        policy_sha256 = _require_sha256_value("policy_sha256", document["policy_sha256"])
        verification_receipt_sha256 = _require_sha256_value(
            "verification_receipt_sha256",
            document["verification_receipt_sha256"],
        )
        verified_at = _decode_datetime(document["verified_at"])
        if (
            stored_qualification_id != qualification_id
            or activation_set_id != published.activation_set_id
            or manifest_sha256 != published.manifest_sha256
            or profile_ref != published.retrieval_profile_ref
        ):
            raise GenerationQualificationError(
                "qualification does not match the requested publication"
            )
        _validated_verified_at(verified_at, published)
        _check_deadline(deadline)
        return VerifiedPreviewQualification(
            qualification_id=stored_qualification_id,
            activation_set_id=activation_set_id,
            manifest_sha256=manifest_sha256,
            profile_ref=profile_ref,
            report_sha256=report_sha256,
            policy_sha256=policy_sha256,
            verification_receipt_sha256=verification_receipt_sha256,
            verified_at=verified_at,
        )

    def _sign(self, document: dict[str, object]) -> str:
        return hmac.new(
            self._key,
            _SIGNATURE_DOMAIN + _canonical_json(document),
            hashlib.sha256,
        ).hexdigest()

    def _initialize_root(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
            metadata = self._root.lstat()
        except OSError as error:
            raise ValueError("cannot initialize trusted qualification root") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("trusted_root must be a non-symlink directory")
        if metadata.st_uid != os.geteuid():
            raise ValueError("trusted_root must be owned by the current user")
        root_descriptor = self._open_root(require_private_mode=False)
        try:
            os.fchmod(root_descriptor, _PRIVATE_DIRECTORY_MODE)
            os.fsync(root_descriptor)
        except OSError as error:
            raise ValueError("cannot secure trusted qualification root") from error
        finally:
            os.close(root_descriptor)

    def _open_root(self, *, require_private_mode: bool = True) -> int:
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if not no_follow or not directory:
            raise GenerationQualificationError(
                "descriptor-relative O_NOFOLLOW and O_DIRECTORY support is required"
            )
        flags = os.O_RDONLY | no_follow | directory | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self._root, flags)
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise GenerationQualificationError(
                "cannot safely open trusted qualification root"
            ) from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or (require_private_mode and stat.S_IMODE(metadata.st_mode) & 0o077)
        ):
            os.close(descriptor)
            raise GenerationQualificationError(
                "trusted qualification root is not an owner-only directory"
            )
        return descriptor

    def _write_new_receipt(
        self,
        qualification_id: str,
        payload: bytes,
        *,
        deadline: datetime,
    ) -> None:
        root_descriptor = self._open_root()
        temporary_name = f".{qualification_id}.{secrets.token_hex(8)}.tmp"
        final_name = _receipt_name(qualification_id)
        temporary_created = False
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                temporary_name,
                flags,
                _PRIVATE_FILE_MODE,
                dir_fd=root_descriptor,
            )
            temporary_created = True
            try:
                os.fchmod(descriptor, _PRIVATE_FILE_MODE)
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                    raise GenerationQualificationError(
                        "temporary qualification receipt is not a private regular file"
                    )
                view = memoryview(payload)
                offset = 0
                while offset < len(view):
                    written = os.write(descriptor, view[offset:])
                    if written <= 0:
                        raise GenerationQualificationError("short write for qualification receipt")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _check_deadline(deadline)
            os.link(
                temporary_name,
                final_name,
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=root_descriptor)
            temporary_created = False
            os.fsync(root_descriptor)
        except GenerationQualificationError:
            raise
        except OSError as error:
            raise GenerationQualificationError(
                "cannot atomically publish qualification receipt"
            ) from error
        finally:
            if temporary_created:
                try:
                    os.unlink(temporary_name, dir_fd=root_descriptor)
                except FileNotFoundError:
                    pass
            os.close(root_descriptor)

    def _read_receipt(self, qualification_id: str, *, deadline: datetime) -> bytes:
        root_descriptor = self._open_root()
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
            descriptor = os.open(
                _receipt_name(qualification_id),
                flags,
                dir_fd=root_descriptor,
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_nlink != 1
            ):
                raise GenerationQualificationError(
                    "qualification receipt is not a private regular file"
                )
            if before.st_size > self._maximum_evidence_bytes:
                raise GenerationQualificationError("qualification evidence exceeds its size limit")
            content = bytearray()
            while len(content) <= self._maximum_evidence_bytes:
                _check_deadline(deadline)
                chunk = os.read(
                    descriptor,
                    min(
                        _READ_CHUNK_BYTES,
                        self._maximum_evidence_bytes + 1 - len(content),
                    ),
                )
                if not chunk:
                    break
                content.extend(chunk)
            after = os.fstat(descriptor)
            entry = os.stat(
                _receipt_name(qualification_id),
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
            if len(content) > self._maximum_evidence_bytes or len(content) != before.st_size:
                raise GenerationQualificationError(
                    "qualification evidence exceeds its size limit or changed while read"
                )
            if _file_identity(before) != _file_identity(after) or (
                entry.st_dev,
                entry.st_ino,
            ) != (after.st_dev, after.st_ino):
                raise GenerationQualificationError("qualification receipt changed while being read")
            return bytes(content)
        except GenerationQualificationError:
            raise
        except OSError as error:
            raise GenerationQualificationError(
                "cannot safely read qualification receipt"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(root_descriptor)


def _receipt_name(qualification_id: str) -> str:
    if _QUALIFICATION_ID.fullmatch(qualification_id) is None:
        raise GenerationQualificationError("qualification locator has an invalid format")
    return f"{qualification_id}.json"


def _decode_canonical_document(payload: bytes) -> dict[str, object]:
    try:
        decoded = payload.decode("utf-8")
        loaded = cast(object, json.loads(decoded))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationQualificationError(
            "qualification receipt is not valid UTF-8 JSON"
        ) from error
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise GenerationQualificationError("qualification receipt must be a JSON object")
    document = cast(dict[str, object], loaded)
    if payload != _canonical_json(document) + b"\n":
        raise GenerationQualificationError("qualification receipt is not canonical JSON")
    return document


def _require_exact_schema(document: dict[str, object]) -> None:
    if set(document) != _DOCUMENT_FIELDS:
        raise GenerationQualificationError("qualification receipt schema fields are invalid")
    if (
        document["schema"] != PREVIEW_QUALIFICATION_SCHEMA
        or document["schema_version"] != PREVIEW_QUALIFICATION_SCHEMA_VERSION
        or document["qualification_tier"] != "preview"
    ):
        raise GenerationQualificationError("qualification receipt schema is unsupported")


def _validated_verified_at(
    verified_at: datetime,
    published: PublishedGenerationRecord,
) -> datetime:
    if (
        not isinstance(verified_at, datetime)
        or verified_at.tzinfo is None
        or verified_at.utcoffset() is None
    ):
        raise GenerationQualificationError("verified_at must be timezone-aware")
    normalized = verified_at.astimezone(UTC)
    if normalized < published.published_at.astimezone(UTC):
        raise GenerationQualificationError("qualification predates its publication")
    if normalized > datetime.now(UTC):
        raise GenerationQualificationError("qualification timestamp is in the future")
    return normalized


def _encode_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _decode_datetime(value: object) -> datetime:
    text = _require_string("verified_at", value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise GenerationQualificationError("verified_at is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GenerationQualificationError("verified_at must be timezone-aware")
    normalized = parsed.astimezone(UTC)
    if text != _encode_datetime(normalized):
        raise GenerationQualificationError("verified_at is not canonical UTC")
    return normalized


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 512:
        raise GenerationQualificationError(f"{name} is invalid")
    return value


def _require_sha256(name: str, value: object) -> None:
    _require_sha256_value(name, value)


def _require_sha256_value(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GenerationQualificationError(f"{name} must be a lowercase SHA-256")
    return value


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise GenerationQualificationError("deadline must be timezone-aware")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise GenerationQualificationError("deadline exceeded")


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_uid,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "DEFAULT_MAXIMUM_EVIDENCE_BYTES",
    "PREVIEW_QUALIFICATION_SCHEMA",
    "PREVIEW_QUALIFICATION_SCHEMA_VERSION",
    "LocalPreviewQualificationStore",
]
