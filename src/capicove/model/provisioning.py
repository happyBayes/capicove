"""Content-addressed offline provisioning for personal reverse-image models."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from capicove.model.bundle import (
    VerifiedImageEncoderBundle,
    load_verified_image_encoder_bundle,
)
from capicove.model.manifest import (
    ImageModelManifest,
    ImageModelSpec,
    parse_image_model_manifest,
)
from capicove.model.receipt import (
    RECEIPT_NAME,
    parse_receipt,
    verify_model_installation,
)


_DEFAULT_MAXIMUM_CONTROL_BYTES = 1_048_576
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "entries",
        "model_lock_relative_path",
        "model_lock_sha256",
        "provisioning_id",
        "schema_version",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "bundle_manifest_sha256",
        "installation_relative_path",
        "lane",
        "model_key",
        "receipt_relative_path",
        "receipt_sha256",
    }
)
_LANES = ("instance", "concept")


@dataclass(frozen=True, slots=True)
class ReverseImageProvisionedBundleEntry:
    """Bind one semantic lane to one exact local lock, receipt, and bundle."""

    lane: str
    model_key: str
    installation_relative_path: PurePosixPath
    receipt_relative_path: PurePosixPath
    receipt_sha256: str
    bundle_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ReverseImageModelProvisioningManifest:
    """Contain the closed content-addressed two-bundle startup declaration."""

    provisioning_id: str
    model_lock_relative_path: PurePosixPath
    model_lock_sha256: str
    entries: tuple[ReverseImageProvisionedBundleEntry, ...]
    source_sha256: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class ProvisionedReverseImageModels:
    """Expose exact verified instance and concept bundles to runtime composition."""

    manifest: ReverseImageModelProvisioningManifest
    instance_bundle: VerifiedImageEncoderBundle
    concept_bundle: VerifiedImageEncoderBundle


@dataclass(frozen=True, slots=True)
class _BoundDirectory:
    """Keep one no-follow directory handle and its identity alive during verification."""

    path: Path
    fd: int
    identity: tuple[int, int]

    def require_open_identity(self) -> None:
        """Reject an invalid or unexpectedly rebound directory descriptor."""
        try:
            metadata = os.fstat(self.fd)
        except OSError as error:
            raise ValueError("bound provisioning directory became unavailable") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != self.identity
        ):
            raise ValueError("bound provisioning directory identity changed")


def load_reverse_image_model_provisioning(
    root: Path,
    manifest_relative_path: PurePosixPath,
    *,
    expected_manifest_sha256: str,
    maximum_control_bytes: int = _DEFAULT_MAXIMUM_CONTROL_BYTES,
) -> ProvisionedReverseImageModels:
    """Resolve exactly two receipt-verified local ONNX bundles without networking."""
    _require_secure_directory_api()
    manifest_path = _require_relative_path_value(
        manifest_relative_path,
        "provisioning manifest",
    )
    _require_sha256("expected_manifest_sha256", expected_manifest_sha256)
    _require_positive_int("maximum_control_bytes", maximum_control_bytes)
    with _open_bound_root(root) as trusted_root:
        payload = _read_regular_file(
            trusted_root,
            manifest_path,
            maximum_bytes=maximum_control_bytes,
        )
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != expected_manifest_sha256:
            raise ValueError(
                "reverse-image provisioning manifest digest does not match startup policy"
            )
        document = _parse_canonical_json(payload, "reverse-image provisioning manifest")
        manifest = _parse_manifest(document, source_sha256=observed_sha256)

        lock_payload = _read_regular_file(
            trusted_root,
            manifest.model_lock_relative_path,
            maximum_bytes=maximum_control_bytes,
        )
        if hashlib.sha256(lock_payload).hexdigest() != manifest.model_lock_sha256:
            raise ValueError("image-model lock digest does not match provisioning manifest")
        lock = parse_image_model_manifest(lock_payload)
        if lock.source_sha256 != manifest.model_lock_sha256:
            raise ValueError("parsed image-model lock identity changed")

        bundles = tuple(
            _load_entry(
                trusted_root,
                lock,
                entry,
                maximum_control_bytes=maximum_control_bytes,
            )
            for entry in manifest.entries
        )
        instance_bundle, concept_bundle = bundles
        if instance_bundle.manifest_sha256 == concept_bundle.manifest_sha256:
            raise ValueError("instance and concept must use distinct bundle manifest identities")
        if instance_bundle.manifest.embedding_space == concept_bundle.manifest.embedding_space:
            raise ValueError("instance and concept must use distinct complete embedding spaces")

        _assert_file_sha256(
            trusted_root,
            manifest_path,
            expected_manifest_sha256,
            maximum_bytes=maximum_control_bytes,
            context="reverse-image provisioning manifest",
        )
        _assert_file_sha256(
            trusted_root,
            manifest.model_lock_relative_path,
            manifest.model_lock_sha256,
            maximum_bytes=maximum_control_bytes,
            context="image-model lock",
        )
        trusted_root.require_open_identity()
        _require_current_root_identity(trusted_root)
        return ProvisionedReverseImageModels(
            manifest=manifest,
            instance_bundle=instance_bundle,
            concept_bundle=concept_bundle,
        )


def _load_entry(
    root: _BoundDirectory,
    lock: ImageModelManifest,
    entry: ReverseImageProvisionedBundleEntry,
    *,
    maximum_control_bytes: int,
) -> VerifiedImageEncoderBundle:
    model = _model_for_key(lock, entry.model_key)
    if model.adapter != "onnx-image-embedding":
        raise ValueError(f"provisioned model is not an ONNX image encoder: {entry.model_key}")
    expected_receipt_path = entry.installation_relative_path / RECEIPT_NAME
    if entry.receipt_relative_path != expected_receipt_path:
        raise ValueError("receipt path must be the installation-owned Capicove receipt")

    receipt_payload = _read_regular_file(
        root,
        entry.receipt_relative_path,
        maximum_bytes=maximum_control_bytes,
    )
    if hashlib.sha256(receipt_payload).hexdigest() != entry.receipt_sha256:
        raise ValueError(f"provisioned {entry.lane} receipt digest does not match")
    receipt = parse_receipt(receipt_payload)
    model_manifest_sha256 = receipt.get("model_manifest_sha256")
    if not isinstance(model_manifest_sha256, str):
        raise ValueError("installed model receipt lacks a valid file-manifest digest")
    _require_sha256("model_manifest_sha256", model_manifest_sha256)
    with _open_bound_child_directory(root, entry.installation_relative_path) as bound:
        installation = verify_model_installation(
            bound.path,
            model,
            manifest_sha256=lock.source_sha256,
            expected_model_manifest_sha256=model_manifest_sha256,
            expected_receipt_sha256=entry.receipt_sha256,
        )
        if installation.path != bound.path or installation.directory_identity != bound.identity:
            raise ValueError(
                f"provisioned {entry.lane} installation escaped or changed during verification"
            )
        bound.require_open_identity()
        bundle = load_verified_image_encoder_bundle(installation)
        if bundle.manifest_sha256 != entry.bundle_manifest_sha256:
            raise ValueError(f"provisioned {entry.lane} bundle manifest digest does not match")
        bound.require_open_identity()
        _require_same_directory(root, entry.installation_relative_path, bound.identity)
        _assert_file_sha256(
            root,
            entry.receipt_relative_path,
            entry.receipt_sha256,
            maximum_bytes=maximum_control_bytes,
            context=f"provisioned {entry.lane} receipt",
        )
        return bundle


def _parse_manifest(
    document: dict[str, object],
    *,
    source_sha256: str,
) -> ReverseImageModelProvisioningManifest:
    values = _closed_object(
        document,
        _TOP_LEVEL_FIELDS,
        "reverse-image provisioning manifest",
    )
    if _integer(values, "schema_version", "reverse-image provisioning manifest") != 1:
        raise ValueError("reverse-image provisioning schema_version must be 1")
    raw_entries = values.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("reverse-image provisioning entries must be an array")
    entries = tuple(_parse_entry(item, index) for index, item in enumerate(raw_entries))
    if tuple(entry.lane for entry in entries) != _LANES:
        raise ValueError("provisioning must contain exactly one instance and concept entry")
    if len({entry.model_key for entry in entries}) != len(entries):
        raise ValueError("instance and concept provisioning model keys must be distinct")
    if len({entry.installation_relative_path for entry in entries}) != len(entries):
        raise ValueError("instance and concept installation paths must be distinct")
    if len({entry.receipt_relative_path for entry in entries}) != len(entries):
        raise ValueError("instance and concept receipt paths must be distinct")
    return ReverseImageModelProvisioningManifest(
        provisioning_id=_string(values, "provisioning_id", "provisioning manifest"),
        model_lock_relative_path=_safe_relative_path(
            _string(values, "model_lock_relative_path", "provisioning manifest"),
            "model lock",
        ),
        model_lock_sha256=_sha256(values, "model_lock_sha256", "provisioning manifest"),
        entries=entries,
        source_sha256=source_sha256,
        schema_version=1,
    )


def _parse_entry(raw: object, index: int) -> ReverseImageProvisionedBundleEntry:
    context = f"reverse-image provisioning entry {index}"
    values = _closed_object(raw, _ENTRY_FIELDS, context)
    lane = _string(values, "lane", context)
    if lane not in _LANES:
        raise ValueError(f"unsupported reverse-image provisioning lane: {lane}")
    return ReverseImageProvisionedBundleEntry(
        lane=lane,
        model_key=_string(values, "model_key", context),
        installation_relative_path=_safe_relative_path(
            _string(values, "installation_relative_path", context),
            f"{context} installation",
        ),
        receipt_relative_path=_safe_relative_path(
            _string(values, "receipt_relative_path", context),
            f"{context} receipt",
        ),
        receipt_sha256=_sha256(values, "receipt_sha256", context),
        bundle_manifest_sha256=_sha256(values, "bundle_manifest_sha256", context),
    )


def _model_for_key(lock: ImageModelManifest, key: str) -> ImageModelSpec:
    matches = tuple(model for model in lock.models if model.key == key)
    if len(matches) != 1:
        raise ValueError(f"provisioning model key is not unique in the trusted lock: {key}")
    return matches[0]


def _require_secure_directory_api() -> None:
    required_flags = ("O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK")
    missing_flags = tuple(name for name in required_flags if not hasattr(os, name))
    supports_dir_fd: object = getattr(os, "supports_dir_fd", set())
    lacks_open_dir_fd = not isinstance(supports_dir_fd, set) or os.open not in supports_dir_fd
    if missing_flags or lacks_open_dir_fd:
        unavailable = ", ".join(missing_flags) if missing_flags else "os.open(dir_fd=...)"
        raise RuntimeError(
            "secure reverse-image provisioning is unavailable on this platform "
            f"({unavailable}); use a POSIX platform with dir_fd/no-follow support or pass "
            "already verified model bundles directly"
        )


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


@contextmanager
def _open_bound_root(root: Path) -> Iterator[_BoundDirectory]:
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("model provisioning root must be an absolute Path")
    normalized = Path(os.path.abspath(root))
    if normalized.anchor != os.path.sep:
        raise RuntimeError("secure reverse-image provisioning requires a POSIX absolute root")
    fds: list[int] = []
    try:
        try:
            current = os.open(os.path.sep, _directory_open_flags())
            fds.append(current)
            for component in normalized.parts[1:]:
                current = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current,
                )
                fds.append(current)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError("model provisioning root must not contain symlinks") from error
            raise ValueError("model provisioning root is unavailable") from error
        metadata = os.fstat(current)
        bound = _BoundDirectory(
            path=normalized,
            fd=current,
            identity=(metadata.st_dev, metadata.st_ino),
        )
        bound.require_open_identity()
        yield bound
    finally:
        _close_read_fds(fds)


@contextmanager
def _open_bound_child_directory(
    root: _BoundDirectory,
    relative_path: PurePosixPath,
) -> Iterator[_BoundDirectory]:
    root.require_open_identity()
    fds: list[int] = []
    try:
        try:
            current = os.dup(root.fd)
            fds.append(current)
            for component in relative_path.parts:
                current = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=current,
                )
                fds.append(current)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    f"provisioning directory contains a symlink: {relative_path}"
                ) from error
            raise ValueError(
                f"cannot safely open provisioning directory: {relative_path}"
            ) from error
        metadata = os.fstat(current)
        bound = _BoundDirectory(
            path=root.path.joinpath(*relative_path.parts),
            fd=current,
            identity=(metadata.st_dev, metadata.st_ino),
        )
        bound.require_open_identity()
        yield bound
    finally:
        _close_read_fds(fds)


def _read_regular_file(
    root: _BoundDirectory,
    relative_path: PurePosixPath,
    *,
    maximum_bytes: int,
) -> bytes:
    root.require_open_identity()
    fds: list[int] = []
    try:
        current = os.dup(root.fd)
        fds.append(current)
        for component in relative_path.parts[:-1]:
            current = os.open(
                component,
                _directory_open_flags(),
                dir_fd=current,
            )
            fds.append(current)
        file_fd = os.open(
            relative_path.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=current,
        )
        fds.append(file_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"provisioning path is not a regular file: {relative_path}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"provisioning control file exceeds {maximum_bytes} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(file_fd, min(1024 * 1024, maximum_bytes + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError(f"provisioning control file exceeds {maximum_bytes} bytes")
        after = os.fstat(file_fd)
        if _file_identity(before) != _file_identity(after) or size != after.st_size:
            raise ValueError(f"provisioning control file changed while read: {relative_path}")
        return b"".join(chunks)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"provisioning path contains a symlink: {relative_path}") from error
        raise ValueError(f"cannot safely read provisioning path: {relative_path}") from error
    finally:
        _close_read_fds(fds)


def _require_same_directory(
    root: _BoundDirectory,
    relative_path: PurePosixPath,
    expected_identity: tuple[int, int],
) -> None:
    with _open_bound_child_directory(root, relative_path) as observed:
        if observed.identity != expected_identity:
            raise ValueError(f"provisioned installation changed while verified: {relative_path}")


def _require_current_root_identity(root: _BoundDirectory) -> None:
    with _open_bound_root(root.path) as observed:
        if observed.identity != root.identity:
            raise ValueError("model provisioning root changed while being verified")


def _assert_file_sha256(
    root: _BoundDirectory,
    relative_path: PurePosixPath,
    expected_sha256: str,
    *,
    maximum_bytes: int,
    context: str,
) -> None:
    payload = _read_regular_file(root, relative_path, maximum_bytes=maximum_bytes)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{context} changed while provisioning was verified")


def _parse_canonical_json(payload: bytes, context: str) -> dict[str, object]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{context} must be a JSON object")
    values = cast(dict[str, object], document)
    if payload != _canonical_json(values):
        raise ValueError(f"{context} must use canonical JSON encoding")
    return values


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, value in pairs:
        if name in values:
            raise ValueError(f"duplicate JSON field: {name}")
        values[name] = value
    return values


def _reject_non_finite_json_number(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _closed_object(raw: object, fields: frozenset[str], context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a JSON object")
    values = cast(dict[str, object], raw)
    unknown = sorted(set(values) - fields)
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(unknown)}")
    missing = sorted(fields - set(values))
    if missing:
        raise ValueError(f"missing {context} fields: {', '.join(missing)}")
    return values


def _safe_relative_path(value: str, context: str) -> PurePosixPath:
    if (
        "\\" in value
        or "\x00" in value
        or ":" in value
        or value.startswith("/")
        or value.endswith("/")
    ):
        raise ValueError(f"{context} path must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{context} path must be a safe relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{context} path must be canonical")
    return path


def _require_relative_path_value(value: object, context: str) -> PurePosixPath:
    if not isinstance(value, PurePosixPath):
        raise TypeError(f"{context} path must be a PurePosixPath")
    return _safe_relative_path(value.as_posix(), context)


def _string(values: dict[str, object], name: str, context: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{context} {name} must be a canonical nonblank string")
    return value


def _integer(values: dict[str, object], name: str, context: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} {name} must be an integer")
    return value


def _sha256(values: dict[str, object], name: str, context: str) -> str:
    value = _string(values, name, context)
    _require_sha256(name, value)
    return value


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _close_read_fds(fds: list[int]) -> None:
    for fd in reversed(fds):
        try:
            os.close(fd)
        except OSError:
            # Read-only close errors cannot change accepted bytes; continue closing
            # the remaining descriptors instead of masking the primary outcome.
            continue


__all__ = [
    "ProvisionedReverseImageModels",
    "ReverseImageModelProvisioningManifest",
    "ReverseImageProvisionedBundleEntry",
    "load_reverse_image_model_provisioning",
]
