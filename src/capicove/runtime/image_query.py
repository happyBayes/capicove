"""Authorized, same-object image query resolution for reverse-image search."""

from __future__ import annotations

import hashlib
import hmac
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol

from capicove.contracts.retrieval import (
    ArtifactSource,
    ImageQuery,
    InlineBytesSource,
    LocalPathSource,
    MountSource,
    SearchFailureCode,
)
from capicove.processing.image.preparation import (
    DEFAULT_MAX_IMAGE_INPUT_BYTES,
    DEFAULT_MAX_IMAGE_PIXELS,
    ImageInputTooLargeError,
    ImagePixelLimitError,
    ImagePreparationError,
    PreparedImage,
    prepare_image,
)
from capicove.retrieval.search_models import (
    InvocationAuthorizationContext,
    ResolvedImageQuery,
    SearchFailure,
)


MAX_INLINE_IMAGE_BYTES = 4 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_REQUIRED_OPERATION = "capicove.search.image"
_REQUIRED_PERMISSION = "retrieval:search"


@dataclass(frozen=True, slots=True)
class ResolvedImageArtifact:
    """Carry immutable artifact bytes already resolved by trusted runtime policy."""

    data: bytes
    expected_sha256: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("artifact data must contain non-empty bytes")
        if (
            not isinstance(self.expected_sha256, str)
            or len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 hash")
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
        ):
            raise ValueError("media_type must be a non-blank string")


class ImageArtifactResolver(Protocol):
    """Resolve one authorized artifact without exposing a caller-selected path."""

    def resolve(
        self,
        artifact_ref: str,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> ResolvedImageArtifact: ...


@dataclass(frozen=True, slots=True)
class _TrustedRoot:
    path: Path
    device: int
    inode: int


ImagePreparer = Callable[..., PreparedImage]


class AuthorizedImageQueryResolver:
    """Authorize, read, hash, and decode an image from the same opened object."""

    def __init__(
        self,
        *,
        workspace_id: str,
        artifact_resolver: ImageArtifactResolver,
        local_roots_by_policy: Mapping[str, Mapping[str, Path]],
        maximum_inline_bytes: int = MAX_INLINE_IMAGE_BYTES,
        maximum_local_bytes: int = DEFAULT_MAX_IMAGE_INPUT_BYTES,
        maximum_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        preparer: ImagePreparer = prepare_image,
    ) -> None:
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("workspace_id must be a non-blank string")
        for field_name, value in (
            ("maximum_inline_bytes", maximum_inline_bytes),
            ("maximum_local_bytes", maximum_local_bytes),
            ("maximum_image_pixels", maximum_image_pixels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        roots: dict[str, Mapping[str, _TrustedRoot]] = {}
        root_identities: dict[str, _TrustedRoot] = {}
        for policy_ref, configured in local_roots_by_policy.items():
            _require_non_blank("data_policy_ref", policy_ref)
            if not isinstance(configured, Mapping):
                raise TypeError("data policy roots must be a root-ref mapping")
            materialized: dict[str, _TrustedRoot] = {}
            for root_ref, path in configured.items():
                _require_non_blank("source_root_ref", root_ref)
                trusted = _trusted_root(path)
                known_identity = root_identities.get(root_ref)
                if known_identity is not None and known_identity != trusted:
                    raise ValueError(
                        "source root refs must have one stable identity across data policies"
                    )
                root_identities[root_ref] = trusted
                materialized[root_ref] = trusted
            if not materialized:
                raise ValueError("every data policy must authorize at least one local root")
            roots[policy_ref] = MappingProxyType(materialized)
        if not roots:
            raise ValueError("local_roots_by_policy must not be empty")

        self._workspace_id = workspace_id
        self._artifact_resolver = artifact_resolver
        self._roots_by_policy = MappingProxyType(roots)
        self._maximum_inline_bytes = maximum_inline_bytes
        self._maximum_local_bytes = maximum_local_bytes
        self._maximum_image_pixels = maximum_image_pixels
        self._preparer = preparer

    def resolve(
        self,
        query: ImageQuery,
        *,
        authorization: InvocationAuthorizationContext,
        data_policy_ref: str,
        deadline: datetime,
    ) -> ResolvedImageQuery:
        """Return one authorized query image without reopening its source path."""
        self._authorize(authorization, data_policy_ref)
        _check_deadline(deadline)
        try:
            payload, expected_sha256, media_type, maximum_bytes = self._read_source(
                query,
                authorization=authorization,
                data_policy_ref=data_policy_ref,
                deadline=deadline,
            )
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            if expected_sha256 is not None and not hmac.compare_digest(
                expected_sha256,
                observed_sha256,
            ):
                raise SearchFailure(SearchFailureCode.SOURCE_CHANGED)
            prepared = self._preparer(
                payload,
                target_bytes=MAX_INLINE_IMAGE_BYTES,
                max_input_bytes=maximum_bytes,
                max_image_pixels=self._maximum_image_pixels,
            )
            if media_type is not None and media_type != prepared.media_type:
                raise SearchFailure(SearchFailureCode.INVALID_MEDIA)
            _check_deadline(deadline)
        except SearchFailure:
            raise
        except (ImageInputTooLargeError, ImagePixelLimitError) as error:
            raise SearchFailure(SearchFailureCode.INPUT_TOO_LARGE) from error
        except ImagePreparationError as error:
            raise SearchFailure(SearchFailureCode.INVALID_MEDIA) from error
        return ResolvedImageQuery(
            image=prepared.image.convert("RGB"),
            source_sha256=observed_sha256,
            media_type=prepared.media_type,
            encoded_byte_size=len(payload),
            width=prepared.width,
            height=prepared.height,
        )

    def _authorize(
        self,
        authorization: InvocationAuthorizationContext,
        data_policy_ref: str,
    ) -> None:
        if not isinstance(authorization, InvocationAuthorizationContext):
            raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
        if authorization.workspace_id != self._workspace_id:
            raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
        if authorization.operation != _REQUIRED_OPERATION:
            raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)
        if _REQUIRED_PERMISSION not in authorization.permissions:
            raise SearchFailure(SearchFailureCode.PERMISSION_DENIED)
        if data_policy_ref not in authorization.allowed_data_policy_refs:
            raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
        if data_policy_ref not in self._roots_by_policy:
            raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)

    def _read_source(
        self,
        query: ImageQuery,
        *,
        authorization: InvocationAuthorizationContext,
        data_policy_ref: str,
        deadline: datetime,
    ) -> tuple[bytes, str | None, str | None, int]:
        if not isinstance(query, ImageQuery):
            raise SearchFailure(SearchFailureCode.INVALID_QUERY)
        source = query.source
        if isinstance(source, InlineBytesSource):
            if len(source.data) > self._maximum_inline_bytes:
                raise SearchFailure(SearchFailureCode.INPUT_TOO_LARGE)
            return source.data, None, source.media_type, self._maximum_inline_bytes
        if isinstance(source, ArtifactSource):
            artifact = self._artifact_resolver.resolve(
                source.artifact_ref,
                authorization=authorization,
                deadline=deadline,
            )
            if len(artifact.data) > self._maximum_local_bytes:
                raise SearchFailure(SearchFailureCode.INPUT_TOO_LARGE)
            return (
                artifact.data,
                artifact.expected_sha256,
                artifact.media_type,
                self._maximum_local_bytes,
            )
        if isinstance(source, MountSource):
            root_ref = f"mount:{source.mount_ref}"
            root = self._roots_by_policy[data_policy_ref].get(root_ref)
            if root is None or root_ref not in authorization.allowed_source_root_refs:
                raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
            parts = _safe_relative_parts(source.relative_path)
            return (
                _read_beneath(root, parts, self._maximum_local_bytes),
                None,
                None,
                self._maximum_local_bytes,
            )
        if isinstance(source, LocalPathSource):
            candidate = Path(source.path)
            if not candidate.is_absolute() or ".." in candidate.parts:
                raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
            for root_ref, root in self._roots_by_policy[data_policy_ref].items():
                if root_ref not in authorization.allowed_source_root_refs:
                    continue
                try:
                    relative = candidate.relative_to(root.path)
                except ValueError:
                    continue
                return (
                    _read_beneath(root, relative.parts, self._maximum_local_bytes),
                    None,
                    None,
                    self._maximum_local_bytes,
                )
            raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
        raise SearchFailure(SearchFailureCode.INVALID_QUERY)


def _trusted_root(path: Path) -> _TrustedRoot:
    if not isinstance(path, Path):
        raise TypeError("configured roots must be Path values")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("configured root must exist") from error
    if not resolved.is_absolute() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("configured root must be an absolute directory")
    return _TrustedRoot(resolved, metadata.st_dev, metadata.st_ino)


def _safe_relative_parts(value: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
    return relative.parts


def _read_beneath(root: _TrustedRoot, parts: tuple[str, ...], maximum_bytes: int) -> bytes:
    if not parts:
        raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_descriptors: list[int] = []
    file_descriptor: int | None = None
    try:
        root_descriptor = os.open(root.path, directory_flags)
        directory_descriptors.append(root_descriptor)
        metadata = os.fstat(root_descriptor)
        if (metadata.st_dev, metadata.st_ino) != (root.device, root.inode):
            raise SearchFailure(SearchFailureCode.SOURCE_CHANGED)
        current = root_descriptor
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=current)
            directory_descriptors.append(next_descriptor)
            current = next_descriptor
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        return _read_regular_descriptor(file_descriptor, maximum_bytes)
    except SearchFailure:
        raise
    except OSError as error:
        raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED) from error
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)


def _read_regular_descriptor(descriptor: int, maximum_bytes: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
    if before.st_size < 1:
        raise SearchFailure(SearchFailureCode.INVALID_MEDIA)
    if before.st_size > maximum_bytes:
        raise SearchFailure(SearchFailureCode.INPUT_TOO_LARGE)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum_bytes:
            raise SearchFailure(SearchFailureCode.INPUT_TOO_LARGE)
    after = os.fstat(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before or total != before.st_size:
        raise SearchFailure(SearchFailureCode.SOURCE_CHANGED)
    return b"".join(chunks)


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise SearchFailure(SearchFailureCode.INVALID_QUERY)
    if datetime.now(UTC) >= deadline:
        raise SearchFailure(SearchFailureCode.DEADLINE_EXCEEDED)


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


__all__ = [
    "MAX_INLINE_IMAGE_BYTES",
    "AuthorizedImageQueryResolver",
    "ImageArtifactResolver",
    "ResolvedImageArtifact",
]
