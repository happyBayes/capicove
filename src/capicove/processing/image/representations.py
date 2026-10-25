"""Bounded production of immutable image target representations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from PIL import Image

from capicove.contracts.retrieval import (
    EmbeddingSpaceRef,
    HashSpaceRef,
    ProcessingFingerprint,
    SearchableUnit,
    TargetType,
    VectorNormalization,
)
from capicove.processing.image.io import LoadedImage
from capicove.processing.image.models import ImageSource
from capicove.processing.image.ports import (
    ImageSemanticLane,
    ImageTargetEncoder,
    SemanticTargetIdentity,
)


if TYPE_CHECKING:
    from capicove.processing.image.deduplication.perceptual import PerceptualHashProvider


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_LANES = (ImageSemanticLane.INSTANCE, ImageSemanticLane.CONCEPT)
_ROW_DIGEST_ALGORITHM = "sha256-ordered-row-chain-v1"
_INITIAL_ROW_STATE = hashlib.sha256(_ROW_DIGEST_ALGORITHM.encode("ascii")).digest()
_SHARD_CHAIN_ALGORITHM = b"sha256-shard-manifest-chain-v1"
_INITIAL_SHARD_ROOT = hashlib.sha256(_SHARD_CHAIN_ALGORITHM).digest()
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ImageRepresentationSource:
    """Bind one stable unit ordinal to exactly one local image source."""

    ordinal: int
    unit: SearchableUnit
    source: ImageSource

    def __post_init__(self) -> None:
        _require_non_negative_int("ordinal", self.ordinal)
        if not isinstance(self.unit, SearchableUnit):
            raise TypeError("unit must be a SearchableUnit")
        if self.unit.unit_type is not TargetType.IMAGE:
            raise ValueError("unit must be an image SearchableUnit")
        if not isinstance(self.source, ImageSource):
            raise TypeError("source must be an ImageSource")


@dataclass(frozen=True, slots=True)
class EncodedImageVector:
    """Carry one normalized vector with its complete target identity."""

    identity: SemanticTargetIdentity
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SemanticTargetIdentity):
            raise TypeError("identity must be a SemanticTargetIdentity")
        if not isinstance(self.values, tuple):
            raise TypeError("values must be an immutable tuple")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in self.values):
            raise TypeError("vector values must be real numbers")
        values = tuple(float(value) for value in self.values)
        if len(values) != self.identity.space.dimension:
            raise ValueError("vector dimension must match its complete embedding space")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("vector values must be finite")
        if self.identity.space.normalization is VectorNormalization.L2:
            norm = math.sqrt(math.fsum(value * value for value in values))
            if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("vector must be L2 normalized")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class ImageTargetRepresentation:
    """Contain all ordinal-aligned image target representations for one unit."""

    ordinal: int
    unit_id: str
    content_sha256: str
    media_type: str
    perceptual_hash: int
    hash_space: HashSpaceRef
    embeddings: tuple[EncodedImageVector, ...]
    processing_fingerprint: ProcessingFingerprint

    def __post_init__(self) -> None:
        _require_non_negative_int("ordinal", self.ordinal)
        _require_non_blank("unit_id", self.unit_id)
        _require_sha256("content_sha256", self.content_sha256)
        _require_non_blank("media_type", self.media_type)
        if not self.media_type.startswith("image/"):
            raise ValueError("media_type must identify an image")
        _require_uint64("perceptual_hash", self.perceptual_hash)
        if not isinstance(self.hash_space, HashSpaceRef):
            raise TypeError("hash_space must be a HashSpaceRef")
        if TargetType.IMAGE not in self.hash_space.target_modalities:
            raise ValueError("hash_space must support image targets")
        if not isinstance(self.embeddings, tuple) or any(
            not isinstance(item, EncodedImageVector) for item in self.embeddings
        ):
            raise TypeError("embeddings must be an immutable tuple of EncodedImageVector values")
        lanes = tuple(item.identity.lane for item in self.embeddings)
        if lanes != _SEMANTIC_LANES:
            raise ValueError(
                "embeddings must contain the complete semantic lanes in canonical order"
            )
        space_ids = tuple(item.identity.space.space_id for item in self.embeddings)
        if len(set(space_ids)) != len(space_ids):
            raise ValueError("semantic lanes must not use duplicate embedding spaces")
        if not isinstance(self.processing_fingerprint, ProcessingFingerprint):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")


@dataclass(frozen=True, slots=True)
class RepresentationCheckpoint:
    """Bind the last durably completed row and physical shard closure."""

    unit_manifest_sha256: str
    last_completed_ordinal: int
    processing_fingerprint: ProcessingFingerprint
    hash_space: HashSpaceRef
    semantic_targets: tuple[SemanticTargetIdentity, ...]
    ordered_row_digest_state_sha256: str
    completed_shard_count: int
    completed_shard_manifest_merkle_root_sha256: str
    checkpoint_sha256: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_non_negative_int("last_completed_ordinal", self.last_completed_ordinal)
        _require_processing_identity(
            self.processing_fingerprint,
            self.hash_space,
            self.semantic_targets,
        )
        _require_sha256(
            "ordered_row_digest_state_sha256",
            self.ordered_row_digest_state_sha256,
        )
        _require_positive_int("completed_shard_count", self.completed_shard_count)
        _require_sha256(
            "completed_shard_manifest_merkle_root_sha256",
            self.completed_shard_manifest_merkle_root_sha256,
        )
        _require_schema_version(self.schema_version)
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        self.verify_integrity()

    def verify_integrity(self) -> None:
        """Recompute the canonical digest to detect a mutated checkpoint object."""
        expected = hashlib.sha256(
            _canonical_json(_checkpoint_document(self, include_digest=False))
        ).hexdigest()
        if self.checkpoint_sha256 != expected:
            raise ValueError("checkpoint digest does not match its canonical body")

    @classmethod
    def create(
        cls,
        *,
        unit_manifest_sha256: str,
        last_completed_ordinal: int,
        processing_fingerprint: ProcessingFingerprint,
        hash_space: HashSpaceRef,
        semantic_targets: tuple[SemanticTargetIdentity, ...],
        ordered_row_digest_state_sha256: str,
        completed_shard_count: int,
        completed_shard_manifest_merkle_root_sha256: str,
        schema_version: int = _SCHEMA_VERSION,
    ) -> RepresentationCheckpoint:
        """Create a checkpoint with its canonical self-digest."""
        document = _checkpoint_body_document(
            unit_manifest_sha256=unit_manifest_sha256,
            last_completed_ordinal=last_completed_ordinal,
            processing_fingerprint=processing_fingerprint,
            hash_space=hash_space,
            semantic_targets=semantic_targets,
            ordered_row_digest_state_sha256=ordered_row_digest_state_sha256,
            completed_shard_count=completed_shard_count,
            completed_shard_manifest_merkle_root_sha256=(
                completed_shard_manifest_merkle_root_sha256
            ),
            schema_version=schema_version,
        )
        digest = hashlib.sha256(_canonical_json(document)).hexdigest()
        return cls(
            unit_manifest_sha256=unit_manifest_sha256,
            last_completed_ordinal=last_completed_ordinal,
            processing_fingerprint=processing_fingerprint,
            hash_space=hash_space,
            semantic_targets=semantic_targets,
            ordered_row_digest_state_sha256=ordered_row_digest_state_sha256,
            completed_shard_count=completed_shard_count,
            completed_shard_manifest_merkle_root_sha256=(
                completed_shard_manifest_merkle_root_sha256
            ),
            checkpoint_sha256=digest,
            schema_version=schema_version,
        )

    @classmethod
    def advance(
        cls,
        *,
        previous_checkpoint: RepresentationCheckpoint | None,
        batch: tuple[ImageTargetRepresentation, ...],
        unit_manifest_sha256: str,
        shard_manifest_sha256: str,
    ) -> RepresentationCheckpoint:
        """Advance the logical row chain and physical shard chain by one batch."""
        _require_sha256("unit_manifest_sha256", unit_manifest_sha256)
        _require_sha256("shard_manifest_sha256", shard_manifest_sha256)
        if not isinstance(batch, tuple) or not batch:
            raise ValueError("checkpoint advancement requires one non-empty tuple batch")
        if any(not isinstance(item, ImageTargetRepresentation) for item in batch):
            raise TypeError("batch must contain ImageTargetRepresentation values")
        first = batch[0]
        targets = tuple(item.identity for item in first.embeddings)
        expected_first = 0
        row_state = _INITIAL_ROW_STATE.hex()
        shard_count = 1
        shard_root = _INITIAL_SHARD_ROOT.hex()
        if previous_checkpoint is not None:
            if not isinstance(previous_checkpoint, RepresentationCheckpoint):
                raise TypeError("previous_checkpoint must be a RepresentationCheckpoint")
            previous_checkpoint.verify_integrity()
            _require_checkpoint_identity(
                previous_checkpoint,
                unit_manifest_sha256=unit_manifest_sha256,
                processing_fingerprint=first.processing_fingerprint,
                hash_space=first.hash_space,
                semantic_targets=targets,
            )
            expected_first = previous_checkpoint.last_completed_ordinal + 1
            row_state = previous_checkpoint.ordered_row_digest_state_sha256
            shard_count = previous_checkpoint.completed_shard_count + 1
            shard_root = previous_checkpoint.completed_shard_manifest_merkle_root_sha256
        expected_ordinal = expected_first
        for item in batch:
            if item.ordinal != expected_ordinal:
                raise ValueError("batch ordinals must continue contiguously from the checkpoint")
            if (
                item.processing_fingerprint != first.processing_fingerprint
                or item.hash_space != first.hash_space
                or tuple(vector.identity for vector in item.embeddings) != targets
            ):
                raise ValueError("batch rows must share one complete processing identity")
            row_state = _advance_row_state(row_state, item)
            expected_ordinal += 1
        shard_root = _advance_shard_root(shard_root, shard_manifest_sha256)
        return cls.create(
            unit_manifest_sha256=unit_manifest_sha256,
            last_completed_ordinal=batch[-1].ordinal,
            processing_fingerprint=first.processing_fingerprint,
            hash_space=first.hash_space,
            semantic_targets=targets,
            ordered_row_digest_state_sha256=row_state,
            completed_shard_count=shard_count,
            completed_shard_manifest_merkle_root_sha256=shard_root,
        )


@dataclass(frozen=True, slots=True)
class RepresentationPayloadDigest:
    """Describe one durable payload bound by a shard manifest."""

    role: str
    sha256: str
    size_bytes: int
    first_ordinal: int
    last_ordinal: int

    def __post_init__(self) -> None:
        _require_non_blank("role", self.role)
        _require_sha256("sha256", self.sha256)
        _require_positive_int("size_bytes", self.size_bytes)
        _require_non_negative_int("first_ordinal", self.first_ordinal)
        _require_non_negative_int("last_ordinal", self.last_ordinal)
        if self.last_ordinal < self.first_ordinal:
            raise ValueError("payload ordinal bounds are reversed")


@dataclass(frozen=True, slots=True)
class RepresentationShardReceipt:
    """Acknowledge one durably written representation shard."""

    shard_id: str
    payloads: tuple[RepresentationPayloadDigest, ...]
    shard_manifest_sha256: str
    checkpoint: RepresentationCheckpoint

    def __post_init__(self) -> None:
        _require_non_blank("shard_id", self.shard_id)
        if not isinstance(self.payloads, tuple) or not self.payloads:
            raise ValueError("payloads must be one non-empty immutable tuple")
        if any(not isinstance(item, RepresentationPayloadDigest) for item in self.payloads):
            raise TypeError("payloads must contain RepresentationPayloadDigest values")
        roles = tuple(item.role for item in self.payloads)
        if len(set(roles)) != len(roles):
            raise ValueError("payload roles must be unique within a shard")
        _require_sha256("shard_manifest_sha256", self.shard_manifest_sha256)
        if not isinstance(self.checkpoint, RepresentationCheckpoint):
            raise TypeError("checkpoint must be a RepresentationCheckpoint")
        self.checkpoint.verify_integrity()


@dataclass(frozen=True, slots=True)
class RepresentationProductionReport:
    """Describe physical production without changing logical target identity."""

    unit_manifest_sha256: str
    unit_count: int
    batch_count: int
    first_ordinal: int | None
    last_ordinal: int | None
    hash_space: HashSpaceRef
    semantic_targets: tuple[SemanticTargetIdentity, ...]
    processing_fingerprint: ProcessingFingerprint
    shard_count: int
    representation_payload_set_sha256: str
    final_checkpoint_sha256: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_positive_int("unit_count", self.unit_count)
        _require_positive_int("batch_count", self.batch_count)
        if self.first_ordinal != 0 or self.last_ordinal != self.unit_count - 1:
            raise ValueError("report ordinal bounds must cover every unit exactly once")
        _require_processing_identity(
            self.processing_fingerprint,
            self.hash_space,
            self.semantic_targets,
        )
        _require_positive_int("shard_count", self.shard_count)
        if self.batch_count != self.shard_count:
            raise ValueError("batch_count must equal the durable shard_count")
        _require_sha256(
            "representation_payload_set_sha256",
            self.representation_payload_set_sha256,
        )
        _require_sha256("final_checkpoint_sha256", self.final_checkpoint_sha256)
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class TargetRepresentationManifest:
    """Contain the layout-independent target representation identity."""

    unit_manifest_sha256: str
    unit_count: int
    row_digest_algorithm: str
    ordered_row_digest_sha256: str
    hash_space: HashSpaceRef
    semantic_targets: tuple[SemanticTargetIdentity, ...]
    processing_fingerprint: ProcessingFingerprint
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_positive_int("unit_count", self.unit_count)
        if self.row_digest_algorithm != _ROW_DIGEST_ALGORITHM:
            raise ValueError(f"row_digest_algorithm must be {_ROW_DIGEST_ALGORITHM}")
        _require_sha256("ordered_row_digest_sha256", self.ordered_row_digest_sha256)
        _require_processing_identity(
            self.processing_fingerprint,
            self.hash_space,
            self.semantic_targets,
        )
        _require_schema_version(self.schema_version)

    def canonical_bytes(self) -> bytes:
        """Return strict canonical JSON bytes for this logical manifest."""
        return _canonical_json(_logical_manifest_document(self))


@dataclass(frozen=True, slots=True)
class RepresentationSealRequest:
    """Ask a sink to verify physical receipts and seal one logical manifest."""

    report: RepresentationProductionReport
    logical_manifest: TargetRepresentationManifest

    def __post_init__(self) -> None:
        if not isinstance(self.report, RepresentationProductionReport):
            raise TypeError("report must be a RepresentationProductionReport")
        if not isinstance(self.logical_manifest, TargetRepresentationManifest):
            raise TypeError("logical_manifest must be a TargetRepresentationManifest")
        manifest = self.logical_manifest
        if (
            self.report.unit_manifest_sha256 != manifest.unit_manifest_sha256
            or self.report.unit_count != manifest.unit_count
            or self.report.hash_space != manifest.hash_space
            or self.report.semantic_targets != manifest.semantic_targets
            or self.report.processing_fingerprint != manifest.processing_fingerprint
        ):
            raise ValueError("report and logical manifest identities do not match")


@dataclass(frozen=True, slots=True)
class SealedTargetRepresentationManifest:
    """Carry one canonical manifest payload returned by a durable sink."""

    manifest: TargetRepresentationManifest
    canonical_bytes: bytes
    sha256: str
    relative_path: PurePosixPath
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TargetRepresentationManifest):
            raise TypeError("manifest must be a TargetRepresentationManifest")
        if not isinstance(self.canonical_bytes, bytes):
            raise TypeError("canonical_bytes must be bytes")
        expected_bytes = self.manifest.canonical_bytes()
        if self.canonical_bytes != expected_bytes:
            raise ValueError("sealed manifest bytes are not canonical for the declared manifest")
        _require_sha256("sha256", self.sha256)
        if self.sha256 != hashlib.sha256(self.canonical_bytes).hexdigest():
            raise ValueError("sealed manifest SHA-256 does not match its canonical bytes")
        if self.size_bytes != len(self.canonical_bytes):
            raise ValueError("sealed manifest size does not match its canonical bytes")
        _require_safe_relative_path(self.relative_path)

    @classmethod
    def create(
        cls,
        manifest: TargetRepresentationManifest,
        *,
        relative_path: PurePosixPath,
    ) -> SealedTargetRepresentationManifest:
        """Create a canonical sealed-manifest value for a sink implementation."""
        if not isinstance(manifest, TargetRepresentationManifest):
            raise TypeError("manifest must be a TargetRepresentationManifest")
        canonical_bytes = manifest.canonical_bytes()
        return cls(
            manifest=manifest,
            canonical_bytes=canonical_bytes,
            sha256=hashlib.sha256(canonical_bytes).hexdigest(),
            relative_path=relative_path,
            size_bytes=len(canonical_bytes),
        )


@dataclass(frozen=True, slots=True)
class RepresentationProductionResult:
    """Return the operational report and sealed logical identity."""

    report: RepresentationProductionReport
    sealed_manifest: SealedTargetRepresentationManifest

    def __post_init__(self) -> None:
        if not isinstance(self.report, RepresentationProductionReport):
            raise TypeError("report must be a RepresentationProductionReport")
        if not isinstance(self.sealed_manifest, SealedTargetRepresentationManifest):
            raise TypeError("sealed_manifest must be a SealedTargetRepresentationManifest")
        manifest = self.sealed_manifest.manifest
        if (
            self.report.unit_manifest_sha256 != manifest.unit_manifest_sha256
            or self.report.unit_count != manifest.unit_count
            or self.report.hash_space != manifest.hash_space
            or self.report.semantic_targets != manifest.semantic_targets
            or self.report.processing_fingerprint != manifest.processing_fingerprint
        ):
            raise ValueError("production report and sealed manifest do not describe one build")


@runtime_checkable
class ImageRepresentationSink(Protocol):
    """Persist bounded batches and own the durable checkpoint ledger."""

    def resume_checkpoint(self) -> RepresentationCheckpoint | None:
        """Return the last durably completed batch identity."""
        ...

    def append_batch(
        self,
        batch: tuple[ImageTargetRepresentation, ...],
        *,
        unit_manifest_sha256: str,
        previous_checkpoint: RepresentationCheckpoint | None,
        deadline: datetime,
    ) -> RepresentationShardReceipt:
        """Reject repeated unit IDs, fsync one batch, then return its receipt."""
        ...

    def seal(
        self,
        request: RepresentationSealRequest,
        *,
        deadline: datetime,
    ) -> SealedTargetRepresentationManifest:
        """Verify the receipt ledger and seal one layout-independent manifest."""
        ...


class ImageTargetRepresentationProducer:
    """Decode once and stream complete target representations in bounded batches."""

    def __init__(
        self,
        *,
        decoder: Callable[[ImageSource], LoadedImage],
        perceptual_hasher: PerceptualHashProvider,
        encoders: Mapping[ImageSemanticLane, ImageTargetEncoder],
        hash_space: HashSpaceRef,
        processing_fingerprint: ProcessingFingerprint,
    ) -> None:
        if not callable(decoder):
            raise TypeError("decoder must be callable")
        if not isinstance(encoders, Mapping):
            raise TypeError("encoders must be a mapping")
        closed = dict(encoders)
        if set(closed) != set(_SEMANTIC_LANES):
            raise ValueError("encoders must contain exactly instance and concept lanes")
        ordered: list[tuple[ImageSemanticLane, ImageTargetEncoder]] = []
        for lane in _SEMANTIC_LANES:
            encoder = closed[lane]
            if not isinstance(encoder, ImageTargetEncoder):
                raise TypeError("encoders must satisfy ImageTargetEncoder")
            if encoder.identity.lane is not lane:
                raise ValueError("encoder mapping key must equal encoder identity lane")
            ordered.append((lane, encoder))
        targets = tuple(encoder.identity for _, encoder in ordered)
        _require_processing_identity(processing_fingerprint, hash_space, targets)
        if any(target.space.normalization is not VectorNormalization.L2 for target in targets):
            raise ValueError("semantic target spaces must declare L2-normalized vectors")
        _require_hasher_identity(perceptual_hasher, hash_space)
        self._decoder = decoder
        self._perceptual_hasher = perceptual_hasher
        self._encoders = tuple(ordered)
        self._hash_space = hash_space
        self._processing_fingerprint = processing_fingerprint
        self._semantic_targets = targets

    def verify_ready(self, *, deadline: datetime) -> None:
        """Verify both frozen semantic targets before reading any source."""
        _check_deadline(deadline)
        _require_hasher_identity(self._perceptual_hasher, self._hash_space)
        for lane, encoder in self._encoders:
            target = self._semantic_targets[_SEMANTIC_LANES.index(lane)]
            if encoder.identity != target:
                raise ValueError("encoder identity changed after producer construction")
            encoder.verify_ready(deadline=deadline)
            _check_deadline(deadline)
            if encoder.identity != target:
                raise ValueError("encoder identity changed during readiness verification")

    def produce(
        self,
        sources: Iterable[ImageRepresentationSource],
        *,
        sink: ImageRepresentationSink,
        batch_size: int,
        unit_manifest_sha256: str,
        deadline: datetime,
    ) -> RepresentationProductionResult:
        """Produce, checkpoint, and seal one ordered image representation stream."""
        _require_positive_int("batch_size", batch_size)
        _require_sha256("unit_manifest_sha256", unit_manifest_sha256)
        if not isinstance(sink, ImageRepresentationSink):
            raise TypeError("sink must satisfy ImageRepresentationSink")
        try:
            iterator = iter(sources)
        except TypeError as error:
            raise TypeError("sources must be iterable") from error
        self.verify_ready(deadline=deadline)
        _check_deadline(deadline)
        checkpoint = sink.resume_checkpoint()
        _check_deadline(deadline)
        if checkpoint is not None:
            if not isinstance(checkpoint, RepresentationCheckpoint):
                raise TypeError("sink checkpoint must be a RepresentationCheckpoint")
            checkpoint.verify_integrity()
            _require_checkpoint_identity(
                checkpoint,
                unit_manifest_sha256=unit_manifest_sha256,
                processing_fingerprint=self._processing_fingerprint,
                hash_space=self._hash_space,
                semantic_targets=self._semantic_targets,
            )

        expected_ordinal = 0
        unit_count = 0
        pending: list[ImageRepresentationSource] = []
        for source in iterator:
            _check_deadline(deadline)
            _validate_source(
                source,
                expected_ordinal=expected_ordinal,
                processing_fingerprint=self._processing_fingerprint,
            )
            expected_ordinal += 1
            unit_count += 1
            if checkpoint is not None and source.ordinal <= checkpoint.last_completed_ordinal:
                continue
            pending.append(source)
            if len(pending) == batch_size:
                checkpoint = self._produce_batch(
                    tuple(pending),
                    sink=sink,
                    previous_checkpoint=checkpoint,
                    unit_manifest_sha256=unit_manifest_sha256,
                    deadline=deadline,
                )
                pending.clear()

        if unit_count == 0:
            raise ValueError("sources must contain at least one image representation source")
        if checkpoint is not None and checkpoint.last_completed_ordinal >= unit_count:
            raise ValueError("resume checkpoint extends beyond the ordered source stream")
        if pending:
            checkpoint = self._produce_batch(
                tuple(pending),
                sink=sink,
                previous_checkpoint=checkpoint,
                unit_manifest_sha256=unit_manifest_sha256,
                deadline=deadline,
            )
            pending.clear()
        if checkpoint is None or checkpoint.last_completed_ordinal != unit_count - 1:
            raise ValueError("final checkpoint does not close the complete source stream")

        logical_manifest = TargetRepresentationManifest(
            unit_manifest_sha256=unit_manifest_sha256,
            unit_count=unit_count,
            row_digest_algorithm=_ROW_DIGEST_ALGORITHM,
            ordered_row_digest_sha256=checkpoint.ordered_row_digest_state_sha256,
            hash_space=self._hash_space,
            semantic_targets=self._semantic_targets,
            processing_fingerprint=self._processing_fingerprint,
        )
        report = RepresentationProductionReport(
            unit_manifest_sha256=unit_manifest_sha256,
            unit_count=unit_count,
            batch_count=checkpoint.completed_shard_count,
            first_ordinal=0,
            last_ordinal=unit_count - 1,
            hash_space=self._hash_space,
            semantic_targets=self._semantic_targets,
            processing_fingerprint=self._processing_fingerprint,
            shard_count=checkpoint.completed_shard_count,
            representation_payload_set_sha256=(
                checkpoint.completed_shard_manifest_merkle_root_sha256
            ),
            final_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )
        request = RepresentationSealRequest(report=report, logical_manifest=logical_manifest)
        _check_deadline(deadline)
        sealed = sink.seal(request, deadline=deadline)
        _check_deadline(deadline)
        if not isinstance(sealed, SealedTargetRepresentationManifest):
            raise TypeError("sink must return a SealedTargetRepresentationManifest")
        if sealed.manifest != logical_manifest:
            raise ValueError("sink sealed a different logical representation manifest")
        return RepresentationProductionResult(report=report, sealed_manifest=sealed)

    def _produce_batch(
        self,
        sources: tuple[ImageRepresentationSource, ...],
        *,
        sink: ImageRepresentationSink,
        previous_checkpoint: RepresentationCheckpoint | None,
        unit_manifest_sha256: str,
        deadline: datetime,
    ) -> RepresentationCheckpoint:
        unit_ids = tuple(source.unit.unit_id for source in sources)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("source unit IDs must be unique within each bounded batch")
        loaded_items: list[LoadedImage] = []
        try:
            for source in sources:
                _check_deadline(deadline)
                loaded = self._decoder(source.source)
                if isinstance(loaded, LoadedImage):
                    loaded_items.append(loaded)
                _check_deadline(deadline)
                _validate_loaded_image(loaded, source)
            images = tuple(item.image for item in loaded_items)
            _require_hasher_identity(self._perceptual_hasher, self._hash_space)
            perceptual_hashes = []
            for image in images:
                _check_deadline(deadline)
                value = self._perceptual_hasher.phash(image)
                _require_uint64("perceptual hash provider result", value)
                perceptual_hashes.append(value)
            encoded: dict[ImageSemanticLane, tuple[tuple[float, ...], ...]] = {}
            for lane, encoder in self._encoders:
                _check_deadline(deadline)
                target = self._semantic_targets[_SEMANTIC_LANES.index(lane)]
                if encoder.identity != target:
                    raise ValueError("encoder identity changed during representation production")
                values = encoder.encode_images(images, target.space, deadline)
                _check_deadline(deadline)
                if encoder.identity != target:
                    raise ValueError("encoder identity changed during representation production")
                if not isinstance(values, tuple) or len(values) != len(images):
                    raise ValueError("target encoder row count must equal the decoded batch size")
                encoded[lane] = values

            representations = []
            for index, (source, loaded) in enumerate(zip(sources, loaded_items, strict=True)):
                embeddings = tuple(
                    EncodedImageVector(
                        identity=self._semantic_targets[_SEMANTIC_LANES.index(lane)],
                        values=encoded[lane][index],
                    )
                    for lane, encoder in self._encoders
                )
                representations.append(
                    ImageTargetRepresentation(
                        ordinal=source.ordinal,
                        unit_id=source.unit.unit_id,
                        content_sha256=loaded.sha256,
                        media_type=loaded.media_type,
                        perceptual_hash=perceptual_hashes[index],
                        hash_space=self._hash_space,
                        embeddings=embeddings,
                        processing_fingerprint=self._processing_fingerprint,
                    )
                )
            batch = tuple(representations)
            _check_deadline(deadline)
            receipt = sink.append_batch(
                batch,
                unit_manifest_sha256=unit_manifest_sha256,
                previous_checkpoint=previous_checkpoint,
                deadline=deadline,
            )
            _check_deadline(deadline)
            if not isinstance(receipt, RepresentationShardReceipt):
                raise TypeError("sink must return a RepresentationShardReceipt")
            for payload in receipt.payloads:
                if (
                    payload.first_ordinal != batch[0].ordinal
                    or payload.last_ordinal != batch[-1].ordinal
                ):
                    raise ValueError("receipt payload bounds do not match the appended batch")
            expected = RepresentationCheckpoint.advance(
                previous_checkpoint=previous_checkpoint,
                batch=batch,
                unit_manifest_sha256=unit_manifest_sha256,
                shard_manifest_sha256=receipt.shard_manifest_sha256,
            )
            if receipt.checkpoint != expected:
                raise ValueError("sink receipt checkpoint does not match the appended batch")
            return receipt.checkpoint
        finally:
            for loaded in loaded_items:
                if isinstance(loaded.image, Image.Image):
                    loaded.image.close()
            loaded_items.clear()


def _validate_source(
    source: object,
    *,
    expected_ordinal: int,
    processing_fingerprint: ProcessingFingerprint,
) -> None:
    if not isinstance(source, ImageRepresentationSource):
        raise TypeError("sources must contain ImageRepresentationSource values")
    if source.ordinal != expected_ordinal:
        raise ValueError("source ordinals must be contiguous from zero")
    if source.unit.processing_fingerprint != processing_fingerprint:
        raise ValueError("source unit processing fingerprint does not match the producer")


def _validate_loaded_image(loaded: object, source: ImageRepresentationSource) -> None:
    if not isinstance(loaded, LoadedImage):
        raise TypeError("decoder must return a LoadedImage")
    if not isinstance(loaded.image, Image.Image):
        raise TypeError("loaded image must be a PIL Image")
    if loaded.image.mode != "RGB":
        raise ValueError("decoder must return one EXIF-normalized RGB image")
    if not isinstance(loaded.source_bytes, bytes) or not loaded.source_bytes:
        raise ValueError("loaded source bytes must be non-empty immutable bytes")
    content_sha256 = hashlib.sha256(loaded.source_bytes).hexdigest()
    if loaded.sha256 != content_sha256 or source.unit.source_hash != content_sha256:
        raise ValueError(
            "loaded source hash does not match original encoded bytes and unit identity"
        )
    _require_non_blank("loaded media_type", loaded.media_type)
    if not loaded.media_type.startswith("image/"):
        raise ValueError("loaded media type must identify an image")
    if loaded.byte_size != len(loaded.source_bytes):
        raise ValueError("loaded byte size does not match original encoded bytes")
    if (loaded.width, loaded.height) != loaded.image.size:
        raise ValueError("loaded dimensions do not match the decoded image")


def _require_hasher_identity(hasher: object, hash_space: HashSpaceRef) -> None:
    if not callable(getattr(hasher, "phash", None)):
        raise TypeError("perceptual_hasher must provide phash(image)")
    observed = (
        getattr(hasher, "provider_id", None),
        getattr(hasher, "provider_version", None),
        getattr(hasher, "hash_version", None),
    )
    expected = (
        hash_space.provider_id,
        hash_space.provider_version,
        hash_space.hash_version,
    )
    if observed != expected:
        raise ValueError("perceptual hasher identity does not match the complete hash space")


def _require_processing_identity(
    processing_fingerprint: object,
    hash_space: object,
    semantic_targets: object,
) -> None:
    if not isinstance(processing_fingerprint, ProcessingFingerprint):
        raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
    if not isinstance(hash_space, HashSpaceRef):
        raise TypeError("hash_space must be a HashSpaceRef")
    if TargetType.IMAGE not in hash_space.target_modalities:
        raise ValueError("hash_space must support image targets")
    if not isinstance(semantic_targets, tuple) or any(
        not isinstance(item, SemanticTargetIdentity) for item in semantic_targets
    ):
        raise TypeError("semantic_targets must contain SemanticTargetIdentity values")
    lanes = tuple(item.lane for item in semantic_targets)
    if lanes != _SEMANTIC_LANES:
        raise ValueError("semantic_targets must contain instance and concept in canonical order")
    space_ids = tuple(item.space.space_id for item in semantic_targets)
    if len(set(space_ids)) != len(space_ids):
        raise ValueError("semantic target embedding spaces must be distinct")
    if any(TargetType.IMAGE not in item.space.target_modalities for item in semantic_targets):
        raise ValueError("semantic target spaces must support image targets")


def _require_checkpoint_identity(
    checkpoint: RepresentationCheckpoint,
    *,
    unit_manifest_sha256: str,
    processing_fingerprint: ProcessingFingerprint,
    hash_space: HashSpaceRef,
    semantic_targets: tuple[SemanticTargetIdentity, ...],
) -> None:
    if checkpoint.unit_manifest_sha256 != unit_manifest_sha256:
        raise ValueError("resume checkpoint unit manifest does not match")
    if checkpoint.processing_fingerprint != processing_fingerprint:
        raise ValueError("resume checkpoint processing fingerprint does not match")
    if checkpoint.hash_space != hash_space:
        raise ValueError("resume checkpoint hash space does not match")
    if checkpoint.semantic_targets != semantic_targets:
        raise ValueError("resume checkpoint semantic target identities do not match")


def _advance_row_state(previous_sha256: str, row: ImageTargetRepresentation) -> str:
    _require_sha256("previous row digest state", previous_sha256)
    row_digest = _canonical_row_digest(row)
    return hashlib.sha256(
        b"\x01" + bytes.fromhex(previous_sha256) + struct.pack("<Q", row.ordinal) + row_digest
    ).hexdigest()


def _canonical_row_digest(row: ImageTargetRepresentation) -> bytes:
    header = {
        "content_sha256": row.content_sha256,
        "hash_space": _hash_space_document(row.hash_space),
        "media_type": row.media_type,
        "ordinal": row.ordinal,
        "perceptual_hash": row.perceptual_hash,
        "processing_fingerprint_sha256": row.processing_fingerprint.sha256,
        "semantic_targets": [_semantic_target_document(item.identity) for item in row.embeddings],
        "unit_id": row.unit_id,
    }
    digest = hashlib.sha256(b"capicove-target-representation-row-v1\x00")
    digest.update(_canonical_json(header))
    for item in row.embeddings:
        values = b"".join(struct.pack("<f", value) for value in item.values)
        digest.update(struct.pack("<I", len(values)))
        digest.update(values)
    return digest.digest()


def _advance_shard_root(previous_sha256: str, shard_manifest_sha256: str) -> str:
    _require_sha256("previous shard root", previous_sha256)
    _require_sha256("shard_manifest_sha256", shard_manifest_sha256)
    return hashlib.sha256(
        b"\x02" + bytes.fromhex(previous_sha256) + bytes.fromhex(shard_manifest_sha256)
    ).hexdigest()


def _checkpoint_document(
    checkpoint: RepresentationCheckpoint,
    *,
    include_digest: bool,
) -> dict[str, object]:
    document = _checkpoint_body_document(
        unit_manifest_sha256=checkpoint.unit_manifest_sha256,
        last_completed_ordinal=checkpoint.last_completed_ordinal,
        processing_fingerprint=checkpoint.processing_fingerprint,
        hash_space=checkpoint.hash_space,
        semantic_targets=checkpoint.semantic_targets,
        ordered_row_digest_state_sha256=checkpoint.ordered_row_digest_state_sha256,
        completed_shard_count=checkpoint.completed_shard_count,
        completed_shard_manifest_merkle_root_sha256=(
            checkpoint.completed_shard_manifest_merkle_root_sha256
        ),
        schema_version=checkpoint.schema_version,
    )
    if include_digest:
        document["checkpoint_sha256"] = checkpoint.checkpoint_sha256
    return document


def _checkpoint_body_document(
    *,
    unit_manifest_sha256: str,
    last_completed_ordinal: int,
    processing_fingerprint: ProcessingFingerprint,
    hash_space: HashSpaceRef,
    semantic_targets: tuple[SemanticTargetIdentity, ...],
    ordered_row_digest_state_sha256: str,
    completed_shard_count: int,
    completed_shard_manifest_merkle_root_sha256: str,
    schema_version: int,
) -> dict[str, object]:
    return {
        "completed_shard_count": completed_shard_count,
        "completed_shard_manifest_merkle_root_sha256": (
            completed_shard_manifest_merkle_root_sha256
        ),
        "hash_space": _hash_space_document(hash_space),
        "last_completed_ordinal": last_completed_ordinal,
        "ordered_row_digest_state_sha256": ordered_row_digest_state_sha256,
        "processing_fingerprint_sha256": processing_fingerprint.sha256,
        "schema_version": schema_version,
        "semantic_targets": [_semantic_target_document(item) for item in semantic_targets],
        "unit_manifest_sha256": unit_manifest_sha256,
    }


def _logical_manifest_document(manifest: TargetRepresentationManifest) -> dict[str, object]:
    return {
        "hash_space": _hash_space_document(manifest.hash_space),
        "kind": "capicove-target-representation-manifest",
        "ordered_row_digest_sha256": manifest.ordered_row_digest_sha256,
        "processing_fingerprint_sha256": manifest.processing_fingerprint.sha256,
        "row_digest_algorithm": manifest.row_digest_algorithm,
        "schema_version": manifest.schema_version,
        "semantic_targets": [_semantic_target_document(item) for item in manifest.semantic_targets],
        "unit_count": manifest.unit_count,
        "unit_manifest_sha256": manifest.unit_manifest_sha256,
    }


def _semantic_target_document(identity: SemanticTargetIdentity) -> dict[str, object]:
    return {
        "bundle_manifest_sha256": identity.bundle_manifest_sha256,
        "lane": identity.lane.value,
        "space": _embedding_space_document(identity.space),
    }


def _embedding_space_document(space: EmbeddingSpaceRef) -> dict[str, object]:
    return {
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


def _hash_space_document(space: object) -> dict[str, object]:
    if not isinstance(space, HashSpaceRef):
        raise TypeError("hash_space must be a HashSpaceRef")
    return {
        "algorithm": space.algorithm,
        "bit_width": space.bit_width,
        "hash_version": space.hash_version,
        "preprocessing_version": space.preprocessing_version,
        "provider_id": space.provider_id,
        "provider_version": space.provider_version,
        "schema_version": space.schema_version,
        "space_id": space.space_id,
        "target_modalities": sorted(item.value for item in space.target_modalities),
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be a timezone-aware datetime")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise TimeoutError("image target representation deadline exceeded")


def _require_non_blank(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_uint64(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")


def _require_schema_version(value: object) -> None:
    if value != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")


def _require_safe_relative_path(value: object) -> None:
    if not isinstance(value, PurePosixPath):
        raise TypeError("relative_path must be a PurePosixPath")
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise ValueError("relative_path must be a safe relative POSIX path")


__all__ = [
    "EncodedImageVector",
    "ImageRepresentationSink",
    "ImageRepresentationSource",
    "ImageTargetRepresentation",
    "ImageTargetRepresentationProducer",
    "RepresentationCheckpoint",
    "RepresentationPayloadDigest",
    "RepresentationProductionReport",
    "RepresentationProductionResult",
    "RepresentationSealRequest",
    "RepresentationShardReceipt",
    "SealedTargetRepresentationManifest",
    "TargetRepresentationManifest",
]
