"""Bounded personal image-build input and representation persistence."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import Condition, RLock, get_ident
from typing import cast

from capicove.contracts.retrieval import HashSpaceRef, ProcessingFingerprint, TargetType
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity
from capicove.processing.image.representations import (
    EncodedImageVector,
    ImageRepresentationSource,
    ImageTargetRepresentation,
    RepresentationCheckpoint,
    RepresentationPayloadDigest,
    RepresentationSealRequest,
    RepresentationShardReceipt,
    SealedTargetRepresentationManifest,
)
from capicove.retrieval._adapters.sqlite_unit_catalog import IndexedUnitRecord


PERSONAL_IMAGE_MAX_UNITS = 10_000
_MAX_REPRESENTATION_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
_PERSONAL_CLEANUP_LOCK = RLock()
_REPRESENTATION_DIRECTORY = "representations"
_TARGET_MANIFEST_PATH = PurePosixPath(
    _REPRESENTATION_DIRECTORY,
    "target-representation-manifest.json",
)


@dataclass(frozen=True, slots=True)
class ImageBuildSummary:
    """Summarize one frozen, path-free, repeatable image build input."""

    unit_count: int
    processing_fingerprint: ProcessingFingerprint
    frozen_unit_manifest_proof_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.unit_count, bool)
            or not 1 <= self.unit_count <= PERSONAL_IMAGE_MAX_UNITS
        ):
            raise ValueError("unit_count must be from 1 to 10,000")
        if not isinstance(self.processing_fingerprint, ProcessingFingerprint):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
        _require_sha256(
            "frozen_unit_manifest_proof_sha256",
            self.frozen_unit_manifest_proof_sha256,
        )


@dataclass(frozen=True, slots=True)
class StagedRepresentationEvidenceIdentity:
    """Bind one private representation spool to its complete staged identity."""

    activation_set_id: str
    workspace_id: str
    corpus_ref: str
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    unit_count: int
    processing_fingerprint: ProcessingFingerprint
    hash_space: HashSpaceRef
    semantic_targets: tuple[SemanticTargetIdentity, ...]
    representation_payload_set_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name in ("activation_set_id", "workspace_id", "corpus_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_sha256(
            "target_representation_manifest_sha256",
            self.target_representation_manifest_sha256,
        )
        _require_sha256(
            "representation_payload_set_sha256",
            self.representation_payload_set_sha256,
        )
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        if (
            isinstance(self.unit_count, bool)
            or not isinstance(self.unit_count, int)
            or not 1 <= self.unit_count <= PERSONAL_IMAGE_MAX_UNITS
        ):
            raise ValueError("unit_count must be from 1 to 10,000")
        if not isinstance(self.processing_fingerprint, ProcessingFingerprint):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
        if not isinstance(self.hash_space, HashSpaceRef):
            raise TypeError("hash_space must be a HashSpaceRef")
        if not isinstance(self.semantic_targets, tuple) or any(
            not isinstance(item, SemanticTargetIdentity) for item in self.semantic_targets
        ):
            raise TypeError("semantic_targets must contain SemanticTargetIdentity values")
        if tuple(item.lane for item in self.semantic_targets) != (
            ImageSemanticLane.INSTANCE,
            ImageSemanticLane.CONCEPT,
        ):
            raise ValueError("semantic_targets must contain instance and concept in order")


@dataclass(frozen=True, slots=True)
class _SealedEvidenceFile:
    relative_path: PurePosixPath
    sha256: str
    size_bytes: int
    device_id: int
    inode: int


@dataclass(frozen=True, slots=True)
class _RepresentationShardFile:
    relative_path: PurePosixPath
    sha256: str
    size_bytes: int
    first_ordinal: int
    last_ordinal: int
    device_id: int
    inode: int


class _BoundEvidenceDirectory:
    """Pin the staging root and representation directory to open descriptors."""

    __slots__ = (
        "_descriptor",
        "_identity",
        "_root_descriptor",
        "_root_identity",
        "_root_path",
    )

    def __init__(
        self,
        root_path: Path,
        root_descriptor: int,
        descriptor: int,
        root_identity: tuple[int, int],
        identity: tuple[int, int],
    ) -> None:
        self._root_path = root_path
        self._root_descriptor = root_descriptor
        self._descriptor = descriptor
        self._root_identity = root_identity
        self._identity = identity

    @classmethod
    def create(cls, root: Path) -> _BoundEvidenceDirectory:
        """Create and pin a new private representation directory under one staging root."""
        root_descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptor = -1
        try:
            root_status = os.fstat(root_descriptor)
            root_visible = os.stat(root, follow_symlinks=False)
            root_identity = (root_status.st_dev, root_status.st_ino)
            if (
                not stat.S_ISDIR(root_status.st_mode)
                or not stat.S_ISDIR(root_visible.st_mode)
                or (root_visible.st_dev, root_visible.st_ino) != root_identity
            ):
                raise ValueError("representation evidence staging identity changed")
            try:
                os.mkdir(_REPRESENTATION_DIRECTORY, 0o700, dir_fd=root_descriptor)
            except FileExistsError as error:
                raise ValueError("representation evidence directory already exists") from error
            descriptor = os.open(
                _REPRESENTATION_DIRECTORY,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_descriptor,
            )
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ValueError("representation evidence directory is invalid")
            os.fsync(descriptor)
            os.fsync(root_descriptor)
            bound = cls(
                root,
                root_descriptor,
                descriptor,
                root_identity,
                (status.st_dev, status.st_ino),
            )
            bound.verify_visible()
            root_descriptor = -1
            descriptor = -1
            return bound
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> _BoundEvidenceDirectory:
        root_descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptor = -1
        try:
            root_status = os.fstat(root_descriptor)
            descriptor = os.open(
                _REPRESENTATION_DIRECTORY,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_descriptor,
            )
            status = os.fstat(descriptor)
            root_identity = (root_status.st_dev, root_status.st_ino)
            identity = (status.st_dev, status.st_ino)
            if (
                not stat.S_ISDIR(root_status.st_mode)
                or not stat.S_ISDIR(status.st_mode)
                or (expected_identity is not None and identity != expected_identity)
            ):
                raise ValueError("representation evidence directory identity changed")
            bound = cls(root, root_descriptor, descriptor, root_identity, identity)
            bound.verify_visible()
            root_descriptor = -1
            descriptor = -1
            return bound
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)

    @property
    def descriptor(self) -> int:
        if self._descriptor < 0:
            raise RuntimeError("representation evidence directory is closed")
        return self._descriptor

    @property
    def identity(self) -> tuple[int, int]:
        return self._identity

    def is_bound_to_root(self, root: Path) -> bool:
        """Return whether this descriptor pair owns the exact staging path."""
        return root == self._root_path

    def verify_visible(self) -> None:
        if self._root_descriptor < 0 or self._descriptor < 0:
            raise RuntimeError("representation evidence directory is closed")
        root_descriptor_status = os.fstat(self._root_descriptor)
        descriptor_status = os.fstat(self._descriptor)
        root_path_status = os.stat(self._root_path, follow_symlinks=False)
        path_status = os.stat(
            _REPRESENTATION_DIRECTORY,
            dir_fd=self._root_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(root_descriptor_status.st_mode)
            or not stat.S_ISDIR(root_path_status.st_mode)
            or not stat.S_ISDIR(descriptor_status.st_mode)
            or not stat.S_ISDIR(path_status.st_mode)
            or (root_descriptor_status.st_dev, root_descriptor_status.st_ino) != self._root_identity
            or (root_path_status.st_dev, root_path_status.st_ino) != self._root_identity
            or (descriptor_status.st_dev, descriptor_status.st_ino) != self._identity
            or (path_status.st_dev, path_status.st_ino) != self._identity
        ):
            raise ValueError("representation evidence directory identity changed")

    def fsync(self) -> None:
        self.verify_visible()
        os.fsync(self.descriptor)
        self.verify_visible()

    def close(self) -> None:
        descriptor = self._descriptor
        root_descriptor = self._root_descriptor
        self._descriptor = -1
        self._root_descriptor = -1
        for opened in (descriptor, root_descriptor):
            if opened >= 0:
                try:
                    os.close(opened)
                except OSError:  # pragma: no cover - preserve completed cleanup
                    pass


class _EvidenceLeaseState:
    def __init__(self) -> None:
        self._condition = Condition()
        self._closed = False
        self._closing = False
        self._closing_owner: int | None = None
        self._cleanup_complete = False
        self._active_token: object | None = None
        self._active_owner: int | None = None
        self._completed_traversals = 0
        self._incomplete_traversal = False

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    def begin_stream(self) -> object:
        with self._condition:
            if self._closed or self._closing or self._cleanup_complete:
                raise RuntimeError("staged representation evidence lease is closed")
            if self._active_token is not None:
                raise RuntimeError("a staged representation evidence traversal is already active")
            if self._incomplete_traversal:
                raise RuntimeError(
                    "the previous staged representation evidence traversal did not complete"
                )
            if self._completed_traversals >= 2:
                raise RuntimeError(
                    "staged representation evidence permits only two complete traversals"
                )
            token = object()
            self._active_token = token
            self._active_owner = get_ident()
            return token

    def require_stream(self, token: object) -> None:
        with self._condition:
            if (
                self._active_token is not token
                or self._active_owner != get_ident()
                or self._closing
                or self._cleanup_complete
            ):
                raise RuntimeError("staged representation evidence traversal is no longer active")

    def finish_stream(self, token: object, *, completed: bool) -> None:
        with self._condition:
            if self._active_token is not token:
                return
            self._active_token = None
            self._active_owner = None
            if completed:
                self._completed_traversals += 1
            else:
                self._incomplete_traversal = True
            self._condition.notify_all()

    def abandon_active_stream(self) -> None:
        with self._condition:
            if self._active_token is not None:
                if self._active_owner != get_ident():
                    raise RuntimeError(
                        "cannot abandon staged representation evidence read by another thread"
                    )
                self._active_token = None
                self._active_owner = None
                self._incomplete_traversal = True
                self._condition.notify_all()

    def close(self, cleanup: Callable[[], None], on_close: Callable[[], None]) -> None:
        caller = get_ident()
        with self._condition:
            while self._closing:
                if self._closing_owner == caller:
                    raise RuntimeError("recursive staged representation evidence close")
                self._condition.wait()
            if self._closed:
                return
            if self._active_token is not None:
                raise RuntimeError("cannot close staged representation evidence while reading")
            self._closing = True
            self._closing_owner = caller
            cleanup_required = not self._cleanup_complete

        if cleanup_required:
            try:
                cleanup()
            except BaseException:
                with self._condition:
                    self._incomplete_traversal = True
                    self._closing = False
                    self._closing_owner = None
                    self._condition.notify_all()
                raise
            with self._condition:
                self._cleanup_complete = True

        try:
            on_close()
        except BaseException:
            with self._condition:
                self._closing = False
                self._closing_owner = None
                self._condition.notify_all()
            raise

        with self._condition:
            self._closed = True
            self._closing = False
            self._closing_owner = None
            self._condition.notify_all()


class StagedRepresentationBatchSource:
    """Stream verified representation shards without exposing their private paths."""

    __slots__ = ("_directory", "_identity", "_manifest", "_shards", "_state")

    def __init__(
        self,
        directory: _BoundEvidenceDirectory,
        identity: StagedRepresentationEvidenceIdentity,
        manifest: _SealedEvidenceFile,
        shards: tuple[_RepresentationShardFile, ...],
        state: _EvidenceLeaseState,
    ) -> None:
        self._directory = directory
        self._identity = identity
        self._manifest = manifest
        self._shards = shards
        self._state = state

    @property
    def identity(self) -> StagedRepresentationEvidenceIdentity:
        """Return the immutable identity verified by every source pass."""
        return self._identity

    def iter_batches(
        self,
        *,
        start_ordinal: int,
        maximum_batch_size: int,
        deadline: datetime,
    ) -> Generator[tuple[ImageTargetRepresentation, ...], None, None]:
        """Yield one of two full verified passes in caller-sized output batches.

        ``maximum_batch_size`` bounds emitted tuples. The separately enforced
        evidence-file byte cap bounds each verified read and shard decode.
        """
        if (
            isinstance(start_ordinal, bool)
            or not isinstance(start_ordinal, int)
            or start_ordinal != 0
        ):
            raise ValueError("staged representation evidence traversals must start at zero")
        if (
            isinstance(maximum_batch_size, bool)
            or not isinstance(maximum_batch_size, int)
            or not 1 <= maximum_batch_size <= PERSONAL_IMAGE_MAX_UNITS
        ):
            raise ValueError("maximum_batch_size must be from 1 to 10,000")
        _check_deadline(deadline)
        token = self._state.begin_stream()
        completed = False
        try:
            self._state.require_stream(token)
            _read_verified_evidence_file(
                self._directory,
                self._manifest,
                deadline=deadline,
            )
            self._state.require_stream(token)
            expected_ordinal = 0
            pending: list[ImageTargetRepresentation] = []
            for shard in self._shards:
                self._state.require_stream(token)
                _check_deadline(deadline)
                if shard.first_ordinal != expected_ordinal:
                    raise ValueError("representation shard ordinal ranges are not contiguous")
                payload = _read_verified_evidence_file(
                    self._directory,
                    shard,
                    deadline=deadline,
                )
                rows = _decode_representation_shard(payload, shard, self._identity)
                del payload
                self._state.require_stream(token)
                for row in rows:
                    _check_deadline(deadline)
                    if row.ordinal != expected_ordinal:
                        raise ValueError("representation row ordinals are not contiguous")
                    expected_ordinal += 1
                    if row.ordinal < start_ordinal:
                        continue
                    pending.append(row)
                    if len(pending) == maximum_batch_size:
                        self._state.require_stream(token)
                        yield tuple(pending)
                        self._state.require_stream(token)
                        pending.clear()
            if expected_ordinal != self._identity.unit_count:
                raise ValueError("representation evidence does not cover its declared unit count")
            if pending:
                self._state.require_stream(token)
                yield tuple(pending)
                self._state.require_stream(token)
            completed = True
        finally:
            self._state.finish_stream(token, completed=completed)


class StagedRepresentationEvidenceLease:
    """Own one read-only, two-pass view and cleanup of its transient shards."""

    __slots__ = (
        "_directory",
        "_identity",
        "_on_close",
        "_removed_shards",
        "_shards",
        "_source",
        "_state",
    )

    def __init__(
        self,
        directory: _BoundEvidenceDirectory,
        identity: StagedRepresentationEvidenceIdentity,
        manifest: _SealedEvidenceFile,
        shards: tuple[_RepresentationShardFile, ...],
        *,
        on_close: Callable[[], None],
    ) -> None:
        state = _EvidenceLeaseState()
        self._state = state
        self._identity = identity
        self._source = StagedRepresentationBatchSource(
            directory,
            identity,
            manifest,
            shards,
            state,
        )
        self._directory = directory
        self._shards = shards
        self._removed_shards: set[PurePosixPath] = set()
        self._on_close = on_close

    @property
    def identity(self) -> StagedRepresentationEvidenceIdentity:
        """Return the exact staged identity bound to this lease."""
        return self._identity

    @property
    def source(self) -> StagedRepresentationBatchSource:
        """Return the path-free, bounded batch source."""
        return self._source

    @property
    def is_closed(self) -> bool:
        """Report whether transient ownership has been released."""
        return self._state.is_closed

    def is_bound_to_staging_directory(self, staging_directory: Path) -> bool:
        """Report whether this lease owns evidence beneath the exact staging path."""
        return self._directory.is_bound_to_root(staging_directory)

    def close(self) -> None:
        """Idempotently remove transient shards and release the build barrier."""
        self._state.close(self._cleanup, self._on_close)

    def _cleanup(self) -> None:
        _remove_representation_shards(
            self._directory,
            self._shards,
            self._removed_shards,
        )
        self._directory.close()

    def __enter__(self) -> StagedRepresentationEvidenceLease:
        if self.is_closed:
            raise RuntimeError("staged representation evidence lease is closed")
        return self

    def __exit__(self, _exc_type: object, exc: object, _traceback: object) -> None:
        try:
            self._state.abandon_active_stream()
            self.close()
        except BaseException as cleanup_error:
            if not isinstance(exc, BaseException):
                raise
            exc.add_note(
                "staged representation evidence cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


class ImageBuildSource:
    """Materialize one bounded source sequence so every build pass is identical."""

    def __init__(
        self,
        sources: Iterable[ImageRepresentationSource],
        *,
        max_units: int = PERSONAL_IMAGE_MAX_UNITS,
    ) -> None:
        _require_capacity(max_units)
        materialized: list[ImageRepresentationSource] = []
        seen_unit_ids: set[str] = set()
        processing_fingerprint: ProcessingFingerprint | None = None
        proof_rows: list[dict[str, object]] = []
        for ordinal, source in enumerate(sources):
            if ordinal >= max_units:
                raise ValueError(f"image build source exceeds the {max_units:,}-unit capacity")
            if not isinstance(source, ImageRepresentationSource):
                raise TypeError("sources must contain ImageRepresentationSource values")
            if source.ordinal != ordinal:
                raise ValueError("source ordinals must be contiguous from zero")
            if source.unit.unit_id in seen_unit_ids:
                raise ValueError("source unit_id values must be unique")
            seen_unit_ids.add(source.unit.unit_id)
            if source.unit.unit_type is not TargetType.IMAGE:
                raise ValueError("personal image builds accept only image units")
            declared_media_type = source.source.media_type
            if not isinstance(declared_media_type, str) or not declared_media_type.startswith(
                "image/"
            ):
                raise ValueError("every build source must declare an image media_type")
            if processing_fingerprint is None:
                processing_fingerprint = source.unit.processing_fingerprint
            elif source.unit.processing_fingerprint != processing_fingerprint:
                raise ValueError("all build units must share one processing fingerprint")
            materialized.append(source)
            proof_rows.append(_source_proof_document(source, declared_media_type))
        if not materialized or processing_fingerprint is None:
            raise ValueError("image build source must contain at least one unit")
        self._sources = tuple(materialized)
        self._summary = ImageBuildSummary(
            unit_count=len(materialized),
            processing_fingerprint=processing_fingerprint,
            frozen_unit_manifest_proof_sha256=hashlib.sha256(
                _canonical_json(
                    {
                        "kind": "capicove-frozen-image-build-source",
                        "schema_version": 1,
                        "units": proof_rows,
                    }
                )
            ).hexdigest(),
        )

    @property
    def summary(self) -> ImageBuildSummary:
        """Return the preflight summary without reading or decoding image content."""
        return self._summary

    def __iter__(self) -> Iterator[ImageRepresentationSource]:
        """Return a fresh iterator over the exact frozen source tuple."""
        return iter(self._sources)

    def catalog_rows(self) -> tuple[IndexedUnitRecord, ...]:
        """Project stable catalog rows without retaining paths in the catalog."""
        return tuple(
            IndexedUnitRecord(
                ordinal=source.ordinal,
                unit=source.unit,
                source_sha256=source.unit.source_hash,
                media_type=_declared_media_type(source),
            )
            for source in self._sources
        )

    def source_at(self, ordinal: int) -> ImageRepresentationSource:
        """Return one expected source for sink-side ordinal and metadata checks."""
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise TypeError("ordinal must be an integer")
        if ordinal < 0:
            raise ValueError("ordinal is outside the frozen build source")
        try:
            return self._sources[ordinal]
        except IndexError as error:
            raise ValueError("ordinal is outside the frozen build source") from error


class PersonalImageRepresentationSink:
    """Durably checkpoint bounded derived rows while retaining no decoded media."""

    def __init__(
        self,
        staging_directory: Path,
        build_source: ImageBuildSource,
        *,
        max_units: int = PERSONAL_IMAGE_MAX_UNITS,
    ) -> None:
        _require_capacity(max_units)
        if not isinstance(staging_directory, Path) or not staging_directory.is_absolute():
            raise ValueError("staging_directory must be an absolute Path")
        if not staging_directory.is_dir() or staging_directory.is_symlink():
            raise ValueError("staging_directory must be an existing non-symlink directory")
        if not isinstance(build_source, ImageBuildSource):
            raise TypeError("build_source must be an ImageBuildSource")
        if build_source.summary.unit_count > max_units:
            raise ValueError("image build exceeds the configured personal capacity")
        self._staging_directory = staging_directory
        self._build_source = build_source
        self._max_units = max_units
        self._checkpoint: RepresentationCheckpoint | None = None
        self._representations: list[ImageTargetRepresentation] = []
        self._unit_ids: set[str] = set()
        self._created_files: list[_RepresentationShardFile] = []
        self._removed_files: set[PurePosixPath] = set()
        directory = _BoundEvidenceDirectory.create(staging_directory)
        try:
            self._representation_directory_identity = directory.identity
        finally:
            directory.close()
        self._sealed: SealedTargetRepresentationManifest | None = None

    def resume_checkpoint(self) -> RepresentationCheckpoint | None:
        """Return the checkpoint owned by this single build attempt."""
        return self._checkpoint

    def append_batch(
        self,
        batch: tuple[ImageTargetRepresentation, ...],
        *,
        unit_manifest_sha256: str,
        previous_checkpoint: RepresentationCheckpoint | None,
        deadline: datetime,
    ) -> RepresentationShardReceipt:
        """Validate and fsync one canonical metadata/vector shard."""
        _check_deadline(deadline)
        _require_sha256("unit_manifest_sha256", unit_manifest_sha256)
        if self._sealed is not None:
            raise RuntimeError("a sealed representation sink cannot accept another batch")
        if previous_checkpoint != self._checkpoint:
            raise ValueError("previous_checkpoint does not match the sink ledger")
        if not isinstance(batch, tuple) or not batch:
            raise ValueError("batch must be a non-empty tuple")
        if len(self._representations) + len(batch) > self._max_units:
            raise ValueError("representation rows exceed the configured personal capacity")
        expected_ordinal = len(self._representations)
        for row in batch:
            if not isinstance(row, ImageTargetRepresentation):
                raise TypeError("batch must contain ImageTargetRepresentation values")
            if row.ordinal != expected_ordinal:
                raise ValueError("representation ordinals must be contiguous from zero")
            expected_source = self._build_source.source_at(row.ordinal)
            if row.unit_id != expected_source.unit.unit_id:
                raise ValueError("representation unit_id does not match the frozen source")
            if row.content_sha256 != expected_source.unit.source_hash:
                raise ValueError("representation content hash does not match the frozen source")
            if row.media_type != _declared_media_type(expected_source):
                raise ValueError("decoded media_type does not match the frozen source declaration")
            if row.unit_id in self._unit_ids:
                raise ValueError("representation unit_id values must be unique")
            expected_ordinal += 1

        shard_index = len(self._created_files)
        shard_path = PurePosixPath(
            _REPRESENTATION_DIRECTORY,
            f"shard-{shard_index:05d}.json",
        )
        shard_bytes = _bounded_evidence_json(
            {
                "first_ordinal": batch[0].ordinal,
                "kind": "capicove-image-representation-shard",
                "last_ordinal": batch[-1].ordinal,
                "rows": [_representation_document(row) for row in batch],
                "schema_version": 1,
                "unit_manifest_sha256": unit_manifest_sha256,
            }
        )
        directory = _BoundEvidenceDirectory.open(
            self._staging_directory,
            expected_identity=self._representation_directory_identity,
        )
        try:
            _write_private_file(directory, shard_path, shard_bytes)
            device_id, inode = _representation_file_identity(
                directory,
                shard_path,
                expected_size=len(shard_bytes),
            )
            self._representation_directory_identity = directory.identity
        finally:
            directory.close()
        shard_sha256 = hashlib.sha256(shard_bytes).hexdigest()
        self._created_files.append(
            _RepresentationShardFile(
                relative_path=shard_path,
                sha256=shard_sha256,
                size_bytes=len(shard_bytes),
                first_ordinal=batch[0].ordinal,
                last_ordinal=batch[-1].ordinal,
                device_id=device_id,
                inode=inode,
            )
        )
        checkpoint = RepresentationCheckpoint.advance(
            previous_checkpoint=previous_checkpoint,
            batch=batch,
            unit_manifest_sha256=unit_manifest_sha256,
            shard_manifest_sha256=shard_sha256,
        )
        receipt = RepresentationShardReceipt(
            shard_id=f"representation-shard-{shard_index:05d}",
            payloads=(
                RepresentationPayloadDigest(
                    role=f"representation-shard-{shard_index:05d}",
                    sha256=shard_sha256,
                    size_bytes=len(shard_bytes),
                    first_ordinal=batch[0].ordinal,
                    last_ordinal=batch[-1].ordinal,
                ),
            ),
            shard_manifest_sha256=shard_sha256,
            checkpoint=checkpoint,
        )
        self._representations.extend(batch)
        self._unit_ids.update(row.unit_id for row in batch)
        self._checkpoint = checkpoint
        return receipt

    def seal(
        self,
        request: RepresentationSealRequest,
        *,
        deadline: datetime,
    ) -> SealedTargetRepresentationManifest:
        """Seal and fsync the canonical logical target manifest."""
        _check_deadline(deadline)
        if not isinstance(request, RepresentationSealRequest):
            raise TypeError("request must be a RepresentationSealRequest")
        if self._sealed is not None:
            if self._sealed.manifest != request.logical_manifest:
                raise ValueError("sink is already sealed for a different manifest")
            return self._sealed
        if self._checkpoint is None:
            raise ValueError("cannot seal a sink without representation rows")
        if len(self._representations) != self._build_source.summary.unit_count:
            raise ValueError("representation rows do not cover the complete frozen source")
        if request.report.final_checkpoint_sha256 != self._checkpoint.checkpoint_sha256:
            raise ValueError("representation report does not match the durable checkpoint")
        sealed = SealedTargetRepresentationManifest.create(
            request.logical_manifest,
            relative_path=_TARGET_MANIFEST_PATH,
        )
        _require_evidence_file_size("target representation manifest", sealed.size_bytes)
        directory = _BoundEvidenceDirectory.open(
            self._staging_directory,
            expected_identity=self._representation_directory_identity,
        )
        try:
            _write_private_file(
                directory,
                sealed.relative_path,
                sealed.canonical_bytes,
            )
            _representation_file_identity(
                directory,
                sealed.relative_path,
                expected_size=sealed.size_bytes,
            )
            self._representation_directory_identity = directory.identity
        finally:
            directory.close()
        self._sealed = sealed
        return sealed

    def representation_rows(self) -> tuple[ImageTargetRepresentation, ...]:
        """Return complete derived rows only after their logical manifest is sealed."""
        if self._sealed is None:
            raise RuntimeError("representation rows are unavailable until the sink is sealed")
        return tuple(self._representations)

    def cleanup_transient_payloads(self) -> None:
        """Remove transient shards while retaining the sealed manifest payload."""
        if self._sealed is None:
            raise RuntimeError("cannot clean an unsealed representation sink")
        if self._representation_directory_identity is None:
            raise RuntimeError("representation evidence directory identity is unavailable")
        directory = _BoundEvidenceDirectory.open(
            self._staging_directory,
            expected_identity=self._representation_directory_identity,
        )
        try:
            _remove_representation_shards(
                directory,
                tuple(self._created_files),
                self._removed_files,
            )
        finally:
            directory.close()
        self._created_files.clear()
        self._removed_files.clear()

    def retain_evidence(
        self,
        identity: StagedRepresentationEvidenceIdentity,
        *,
        on_close: Callable[[], None],
    ) -> StagedRepresentationEvidenceLease:
        """Transfer transient shard ownership to one path-free evidence lease."""
        if self._sealed is None or self._checkpoint is None:
            raise RuntimeError("cannot retain evidence from an unsealed representation sink")
        if not callable(on_close):
            raise TypeError("on_close must be callable")
        if self._removed_files:
            raise RuntimeError("cannot retain partially cleaned representation evidence")
        if self._representation_directory_identity is None:
            raise RuntimeError("representation evidence directory identity is unavailable")
        manifest = self._sealed.manifest
        if (
            identity.unit_manifest_sha256 != manifest.unit_manifest_sha256
            or identity.target_representation_manifest_sha256 != self._sealed.sha256
            or identity.unit_count != manifest.unit_count
            or identity.processing_fingerprint != manifest.processing_fingerprint
            or identity.hash_space != manifest.hash_space
            or identity.semantic_targets != manifest.semantic_targets
            or identity.representation_payload_set_sha256
            != self._checkpoint.completed_shard_manifest_merkle_root_sha256
            or identity.checkpoint_sha256 != self._checkpoint.checkpoint_sha256
        ):
            raise ValueError("staged representation evidence identity does not match the sink")
        shards = tuple(self._created_files)
        if not shards or len(shards) != self._checkpoint.completed_shard_count:
            raise ValueError("staged representation shard count does not match the checkpoint")
        expected_ordinal = 0
        for shard in shards:
            if shard.first_ordinal != expected_ordinal:
                raise ValueError("staged representation shard ranges are not contiguous")
            expected_ordinal = shard.last_ordinal + 1
        if expected_ordinal != identity.unit_count:
            raise ValueError("staged representation shards do not cover the identity")
        _require_evidence_file_size(
            "target representation manifest",
            self._sealed.size_bytes,
        )
        for shard in shards:
            _require_evidence_file_size("representation shard", shard.size_bytes)
        directory = _BoundEvidenceDirectory.open(
            self._staging_directory,
            expected_identity=self._representation_directory_identity,
        )
        try:
            manifest_device_id, manifest_inode = _representation_file_identity(
                directory,
                self._sealed.relative_path,
                expected_size=self._sealed.size_bytes,
            )
            for shard in shards:
                observed_identity = _representation_file_identity(
                    directory,
                    shard.relative_path,
                    expected_size=shard.size_bytes,
                )
                if observed_identity != (shard.device_id, shard.inode):
                    raise ValueError("representation evidence inode identity changed")
            lease = StagedRepresentationEvidenceLease(
                directory,
                identity,
                _SealedEvidenceFile(
                    relative_path=self._sealed.relative_path,
                    sha256=self._sealed.sha256,
                    size_bytes=self._sealed.size_bytes,
                    device_id=manifest_device_id,
                    inode=manifest_inode,
                ),
                shards,
                on_close=on_close,
            )
        except BaseException:
            directory.close()
            raise
        self._created_files.clear()
        return lease


def _source_proof_document(
    source: ImageRepresentationSource,
    declared_media_type: str,
) -> dict[str, object]:
    unit = source.unit
    return {
        "artifact_ref": unit.artifact_ref,
        "embedding_spaces": [
            {
                "dimension": space.dimension,
                "distance_metric": space.distance_metric.value,
                "model_revision": space.model_revision,
                "normalization": space.normalization.value,
                "preprocessing_version": space.preprocessing_version,
                "provider_id": space.provider_id,
                "query_modalities": sorted(item.value for item in space.query_modalities),
                "schema_version": space.schema_version,
                "space_id": space.space_id,
                "target_modalities": sorted(item.value for item in space.target_modalities),
            }
            for space in unit.embedding_spaces
        ],
        "evidence_ref": unit.evidence_ref,
        "media_type": declared_media_type,
        "ordinal": source.ordinal,
        "processing_fingerprint_sha256": unit.processing_fingerprint.sha256,
        "processing_run_ref": unit.processing_run_ref,
        "processing_version": unit.processing_version,
        "representation_refs": list(unit.representation_refs),
        "scene_range": _temporal_document(unit.scene_range),
        "source_sha256": unit.source_hash,
        "time_range": _temporal_document(unit.time_range),
        "timestamp_ms": unit.timestamp_ms,
        "unit_id": unit.unit_id,
        "unit_type": unit.unit_type.value,
    }


def _temporal_document(value: object) -> dict[str, int] | None:
    if value is None:
        return None
    start_ms = getattr(value, "start_ms", None)
    end_ms = getattr(value, "end_ms", None)
    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        raise TypeError("temporal range must expose integer start_ms and end_ms")
    return {"end_ms": end_ms, "start_ms": start_ms}


def _representation_document(row: ImageTargetRepresentation) -> dict[str, object]:
    return {
        "content_sha256": row.content_sha256,
        "embeddings": [
            {
                "bundle_manifest_sha256": embedding.identity.bundle_manifest_sha256,
                "lane": embedding.identity.lane.value,
                "space_id": embedding.identity.space.space_id,
                "values": embedding.values,
            }
            for embedding in row.embeddings
        ],
        "hash_space_id": row.hash_space.space_id,
        "media_type": row.media_type,
        "ordinal": row.ordinal,
        "perceptual_hash": row.perceptual_hash,
        "processing_fingerprint_sha256": row.processing_fingerprint.sha256,
        "unit_id": row.unit_id,
    }


def _decode_representation_shard(
    payload: bytes | bytearray,
    descriptor: _RepresentationShardFile,
    identity: StagedRepresentationEvidenceIdentity,
) -> tuple[ImageTargetRepresentation, ...]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("representation shard is not valid JSON") from error
    document = _require_exact_keys(
        document,
        {
            "first_ordinal",
            "kind",
            "last_ordinal",
            "rows",
            "schema_version",
            "unit_manifest_sha256",
        },
        "representation shard",
    )
    if document["kind"] != "capicove-image-representation-shard":
        raise ValueError("representation shard kind is invalid")
    if document["schema_version"] != 1:
        raise ValueError("representation shard schema_version is unsupported")
    if document["unit_manifest_sha256"] != identity.unit_manifest_sha256:
        raise ValueError("representation shard unit manifest identity changed")
    if (
        document["first_ordinal"] != descriptor.first_ordinal
        or document["last_ordinal"] != descriptor.last_ordinal
    ):
        raise ValueError("representation shard ordinal bounds changed")
    raw_rows = document["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("representation shard rows must be a non-empty list")
    rows = tuple(_decode_representation_row(item, identity) for item in raw_rows)
    if (
        rows[0].ordinal != descriptor.first_ordinal
        or rows[-1].ordinal != descriptor.last_ordinal
        or len(rows) != descriptor.last_ordinal - descriptor.first_ordinal + 1
    ):
        raise ValueError("representation shard rows do not match their declared range")
    return rows


def _decode_representation_row(
    document: object,
    identity: StagedRepresentationEvidenceIdentity,
) -> ImageTargetRepresentation:
    document = _require_exact_keys(
        document,
        {
            "content_sha256",
            "embeddings",
            "hash_space_id",
            "media_type",
            "ordinal",
            "perceptual_hash",
            "processing_fingerprint_sha256",
            "unit_id",
        },
        "representation row",
    )
    if document["hash_space_id"] != identity.hash_space.space_id:
        raise ValueError("representation row hash-space identity changed")
    if document["processing_fingerprint_sha256"] != identity.processing_fingerprint.sha256:
        raise ValueError("representation row processing identity changed")
    raw_embeddings = document["embeddings"]
    if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(
        identity.semantic_targets
    ):
        raise ValueError("representation row semantic targets are incomplete")
    embeddings = tuple(
        _decode_representation_vector(item, target)
        for item, target in zip(raw_embeddings, identity.semantic_targets, strict=True)
    )
    return ImageTargetRepresentation(
        ordinal=cast(int, document["ordinal"]),
        unit_id=cast(str, document["unit_id"]),
        content_sha256=cast(str, document["content_sha256"]),
        media_type=cast(str, document["media_type"]),
        perceptual_hash=cast(int, document["perceptual_hash"]),
        hash_space=identity.hash_space,
        embeddings=embeddings,
        processing_fingerprint=identity.processing_fingerprint,
    )


def _decode_representation_vector(
    document: object,
    target: SemanticTargetIdentity,
) -> EncodedImageVector:
    document = _require_exact_keys(
        document,
        {"bundle_manifest_sha256", "lane", "space_id", "values"},
        "representation vector",
    )
    if (
        document["lane"] != target.lane.value
        or document["space_id"] != target.space.space_id
        or document["bundle_manifest_sha256"] != target.bundle_manifest_sha256
    ):
        raise ValueError("representation vector semantic identity changed")
    values = document["values"]
    if not isinstance(values, list):
        raise TypeError("representation vector values must be a list")
    return EncodedImageVector(identity=target, values=tuple(values))


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must contain exactly its closed schema")
    return value


def _read_verified_evidence_file(
    directory: _BoundEvidenceDirectory,
    evidence: _SealedEvidenceFile | _RepresentationShardFile,
    *,
    deadline: datetime,
) -> bytearray:
    _require_evidence_file_size("representation evidence", evidence.size_bytes)
    _check_deadline(deadline)
    directory.verify_visible()
    descriptor = os.open(
        _representation_file_name(evidence.relative_path),
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory.descriptor,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != evidence.size_bytes
        ):
            raise ValueError("representation evidence is not a regular file with the sealed size")
        if (before.st_dev, before.st_ino) != (evidence.device_id, evidence.inode):
            raise ValueError("representation evidence inode identity changed")
        digest = hashlib.sha256()
        data = bytearray()
        while len(data) < evidence.size_bytes:
            _check_deadline(deadline)
            chunk = os.read(
                descriptor,
                min(1024 * 1024, evidence.size_bytes - len(data)),
            )
            if not chunk:
                break
            digest.update(chunk)
            data.extend(chunk)
        _check_deadline(deadline)
        trailing = os.read(descriptor, 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    directory.verify_visible()
    if (before.st_dev, before.st_ino, before.st_nlink, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("representation evidence changed while it was read")
    if trailing or len(data) != evidence.size_bytes or digest.hexdigest() != evidence.sha256:
        raise ValueError("representation evidence checksum or size changed")
    return data


def _remove_representation_shards(
    directory: _BoundEvidenceDirectory,
    shards: tuple[_RepresentationShardFile, ...],
    removed: set[PurePosixPath],
) -> None:
    # Threat model: this private 0700 tree is mutated only by cooperating
    # Capicove writers in this process. The shared lock serializes those writers.
    # A malicious same-UID process racing the random quarantine name is out of scope.
    with _PERSONAL_CLEANUP_LOCK:
        directory.verify_visible()
        if stat.S_IMODE(os.fstat(directory.descriptor).st_mode) != 0o700:
            raise ValueError("representation evidence directory is not private")
        for shard in reversed(shards):
            if shard.relative_path in removed:
                continue
            directory.verify_visible()
            name = _representation_file_name(shard.relative_path)
            try:
                observed = os.stat(
                    name,
                    dir_fd=directory.descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError as error:
                raise ValueError(
                    "owned representation evidence disappeared before cleanup"
                ) from error
            if (
                not stat.S_ISREG(observed.st_mode)
                or observed.st_nlink != 1
                or (observed.st_dev, observed.st_ino) != (shard.device_id, shard.inode)
            ):
                raise ValueError("refusing to remove replaced representation evidence")
            quarantine = _quarantine_representation_shard(
                directory.descriptor,
                name,
                expected_identity=(shard.device_id, shard.inode),
            )
            try:
                os.unlink(quarantine, dir_fd=directory.descriptor)
            except OSError:
                _restore_representation_shard(
                    directory.descriptor,
                    quarantine,
                    name,
                )
                raise
            removed.add(shard.relative_path)
        directory.fsync()


def _quarantine_representation_shard(
    directory_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
) -> str:
    """Move one shard to an unpredictable internal name before deleting it."""
    quarantine = f".capicove-cleanup-{uuid.uuid4().hex}"
    try:
        os.stat(quarantine, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - cryptographically improbable UUID collision
        raise RuntimeError("representation cleanup quarantine unexpectedly exists")
    renamed = False
    try:
        os.rename(
            name,
            quarantine,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        renamed = True
        observed = os.stat(
            quarantine,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        if renamed:
            _restore_representation_shard(
                directory_descriptor,
                quarantine,
                name,
            )
        raise ValueError("refusing to remove replaced representation evidence") from error
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (observed.st_dev, observed.st_ino) != expected_identity
    ):
        _restore_representation_shard(directory_descriptor, quarantine, name)
        raise ValueError("refusing to remove replaced representation evidence")
    return quarantine


def _restore_representation_shard(
    directory_descriptor: int,
    quarantine: str,
    original_name: str,
) -> None:
    try:
        os.stat(original_name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.rename(
                quarantine,
                original_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
        except OSError:
            pass


def _representation_file_name(relative_path: PurePosixPath) -> str:
    if relative_path.parent != PurePosixPath(_REPRESENTATION_DIRECTORY):
        raise ValueError("representation evidence path escaped its private directory")
    return relative_path.name


def _representation_file_identity(
    directory: _BoundEvidenceDirectory,
    relative_path: PurePosixPath,
    *,
    expected_size: int,
) -> tuple[int, int]:
    _require_evidence_file_size("representation evidence", expected_size)
    directory.verify_visible()
    descriptor = os.open(
        _representation_file_name(relative_path),
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory.descriptor,
    )
    try:
        status = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    directory.verify_visible()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or status.st_size != expected_size:
        raise ValueError("representation evidence file identity is invalid")
    return status.st_dev, status.st_ino


def _declared_media_type(source: ImageRepresentationSource) -> str:
    media_type = source.source.media_type
    if not isinstance(media_type, str):  # protected by ImageBuildSource validation
        raise ValueError("build source media_type is unavailable")
    return media_type


def _write_private_file(
    directory: _BoundEvidenceDirectory,
    relative_path: PurePosixPath,
    payload: bytes | bytearray,
) -> None:
    _require_evidence_file_size("representation evidence", len(payload))
    directory.verify_visible()
    descriptor = os.open(
        _representation_file_name(relative_path),
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory.descriptor,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("private payload write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory.fsync()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _bounded_evidence_json(value: object) -> bytearray:
    """Encode one canonical evidence file without first materializing an oversized string."""
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload = bytearray()
    for fragment in encoder.iterencode(value):
        encoded = fragment.encode("utf-8")
        if len(payload) + len(encoded) + 1 > _MAX_REPRESENTATION_EVIDENCE_FILE_BYTES:
            raise ValueError(
                "representation shard must be from 1 to "
                f"{_MAX_REPRESENTATION_EVIDENCE_FILE_BYTES:,} bytes"
            )
        payload.extend(encoded)
    payload.extend(b"\n")
    _require_evidence_file_size("representation shard", len(payload))
    return payload


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be a timezone-aware datetime")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise TimeoutError("personal image representation deadline exceeded")


def _require_capacity(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
        raise ValueError("max_units must be from 1 to 10,000")


def _require_evidence_file_size(name: str, size_bytes: object) -> None:
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or not 0 < size_bytes <= _MAX_REPRESENTATION_EVIDENCE_FILE_BYTES
    ):
        raise ValueError(
            f"{name} must be from 1 to {_MAX_REPRESENTATION_EVIDENCE_FILE_BYTES:,} bytes"
        )


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "PERSONAL_IMAGE_MAX_UNITS",
    "ImageBuildSource",
    "ImageBuildSummary",
    "PersonalImageRepresentationSink",
    "StagedRepresentationBatchSource",
    "StagedRepresentationEvidenceIdentity",
    "StagedRepresentationEvidenceLease",
]
