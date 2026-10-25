"""Closed personal-lite generation values and canonical manifest codecs.

This module deliberately contains no activation or mutable lifecycle state.  A
published manifest describes immutable bytes only; a separate lifecycle catalog
may later decide whether that generation is active.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, TypeVar, cast

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    IndexGeneration,
    IndexParametersRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.search_models import RetrievalLane


GENERATION_MANIFEST_SCHEMA_VERSION = 1
MAX_GENERATION_MANIFEST_BYTES = 1_048_576
PERSONAL_LITE_PROFILE = "personal-lite"
PERSONAL_LITE_MAX_UNITS = 10_000
PERSONAL_LITE_MAX_PAYLOADS = 64
PERSONAL_LITE_MAX_SHARDS_PER_LANE = 16
TARGET_REPRESENTATION_MANIFEST_ROLE = "target-representation-manifest"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,254}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
)
_LANE_ORDER = {
    RetrievalLane.PERCEPTUAL: 0,
    RetrievalLane.INSTANCE: 1,
    RetrievalLane.CONCEPT: 2,
}
_REQUIRED_PERSONAL_LANES = frozenset(_LANE_ORDER)

_T = TypeVar("_T")
_StrEnumT = TypeVar("_StrEnumT", bound=StrEnum)


class GenerationManifestError(ValueError):
    """Raised when final manifest bytes do not satisfy the closed schema."""


class GenerationValidationError(ValueError):
    """Raised when staged or published generation bytes fail validation."""


class GenerationPublicationError(RuntimeError):
    """Raised when an immutable generation cannot be safely published."""


class GenerationDirectoryState(StrEnum):
    """Classify one directory discovered below the publication root."""

    PUBLISHED = "published"
    ORPHANED = "orphaned"
    CORRUPT = "corrupt"


@dataclass(frozen=True, slots=True)
class PayloadFileRef:
    """Bind one logical payload role to one safe content-addressed file."""

    relative_path: PurePosixPath
    sha256: str
    size_bytes: int
    role: str

    def __post_init__(self) -> None:
        _require_safe_payload_path(self.relative_path)
        _require_sha256("sha256", self.sha256)
        _require_non_negative_int("size_bytes", self.size_bytes)
        _require_opaque_ref("role", self.role)


@dataclass(frozen=True, slots=True)
class UnitCatalogManifest:
    """Describe the immutable ordinal catalog shared by all personal lanes."""

    catalog_id: str
    unit_count: int
    unit_manifest_sha256: str
    catalog_format: str
    payload_ref: str
    validation_report_ref: str
    schema_version: int = GENERATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_opaque_ref("catalog_id", self.catalog_id)
        _require_positive_int("unit_count", self.unit_count)
        if self.unit_count > PERSONAL_LITE_MAX_UNITS:
            raise ValueError("personal-lite unit_count must be no greater than 10,000")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_opaque_ref("catalog_format", self.catalog_format)
        _require_opaque_ref("payload_ref", self.payload_ref)
        _require_opaque_ref("validation_report_ref", self.validation_report_ref)
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class LaneGeneration:
    """Bind one closed retrieval lane to its complete index generation."""

    lane: RetrievalLane
    generation: IndexGeneration
    target_bundle_manifest_sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        if self.lane not in _REQUIRED_PERSONAL_LANES:
            raise ValueError("personal-lite lanes must be perceptual, instance, or concept")
        if not isinstance(self.generation, IndexGeneration):
            raise TypeError("generation must be an IndexGeneration")
        _validate_lane_identity(
            self.lane,
            self.generation.embedding_spaces,
            self.generation.hash_spaces,
            self.target_bundle_manifest_sha256,
        )


@dataclass(frozen=True, slots=True)
class GenerationSetManifest:
    """Describe one complete, immutable, but not necessarily active generation set."""

    activation_set_id: str
    workspace_id: str
    corpus_ref: str
    retrieval_profile_ref: str
    target_type: TargetType
    unit_manifest_sha256: str
    frozen_unit_manifest_proof_sha256: str
    target_representation_manifest_sha256: str
    processing_fingerprint: ProcessingFingerprint
    unit_catalog: UnitCatalogManifest
    lanes: tuple[LaneGeneration, ...]
    payloads: tuple[PayloadFileRef, ...]
    created_at: datetime
    schema_version: int = GENERATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_opaque_ref("activation_set_id", self.activation_set_id)
        _require_opaque_ref("workspace_id", self.workspace_id)
        _require_opaque_ref("corpus_ref", self.corpus_ref)
        if self.retrieval_profile_ref != PERSONAL_LITE_PROFILE:
            raise ValueError(
                "only the personal-lite generation profile is implemented; enterprise is reserved"
            )
        if self.target_type is not TargetType.IMAGE:
            raise ValueError("personal-lite generations currently support only image targets")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_sha256(
            "frozen_unit_manifest_proof_sha256",
            self.frozen_unit_manifest_proof_sha256,
        )
        _require_sha256(
            "target_representation_manifest_sha256",
            self.target_representation_manifest_sha256,
        )
        if not isinstance(self.processing_fingerprint, ProcessingFingerprint):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
        if not isinstance(self.unit_catalog, UnitCatalogManifest):
            raise TypeError("unit_catalog must be a UnitCatalogManifest")
        lanes = _freeze_tuple("lanes", self.lanes, LaneGeneration)
        payloads = _freeze_tuple("payloads", self.payloads, PayloadFileRef)
        object.__setattr__(
            self, "lanes", tuple(sorted(lanes, key=lambda item: _LANE_ORDER[item.lane]))
        )
        object.__setattr__(
            self,
            "payloads",
            tuple(sorted(payloads, key=lambda item: (item.role, item.relative_path.as_posix()))),
        )
        _require_aware_datetime("created_at", self.created_at)
        _require_schema_version(self.schema_version)
        validate_generation_set_manifest(self)


@dataclass(frozen=True, slots=True)
class StagedUnitCatalog:
    """Describe a catalog before an immutable validation report exists."""

    catalog_id: str
    unit_count: int
    unit_manifest_sha256: str
    catalog_format: str
    payload_ref: str

    def __post_init__(self) -> None:
        _require_opaque_ref("catalog_id", self.catalog_id)
        _require_positive_int("unit_count", self.unit_count)
        if self.unit_count > PERSONAL_LITE_MAX_UNITS:
            raise ValueError("personal-lite unit_count must be no greater than 10,000")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_opaque_ref("catalog_format", self.catalog_format)
        _require_opaque_ref("payload_ref", self.payload_ref)


@dataclass(frozen=True, slots=True)
class StagedIndexGeneration:
    """Describe one lane before its immutable validation report exists."""

    generation_id: str
    activation_set_id: str
    lane: RetrievalLane
    target_bundle_manifest_sha256: str | None
    unit_manifest_sha256: str
    embedding_spaces: tuple[EmbeddingSpaceRef, ...]
    hash_spaces: tuple[HashSpaceRef, ...]
    retriever_id: str
    index_format: str
    index_parameters_ref: IndexParametersRef
    payload_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_opaque_ref("generation_id", self.generation_id)
        _require_opaque_ref("activation_set_id", self.activation_set_id)
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        embedding_spaces = _freeze_tuple(
            "embedding_spaces",
            self.embedding_spaces,
            EmbeddingSpaceRef,
        )
        hash_spaces = _freeze_tuple("hash_spaces", self.hash_spaces, HashSpaceRef)
        object.__setattr__(self, "embedding_spaces", embedding_spaces)
        object.__setattr__(self, "hash_spaces", hash_spaces)
        _require_opaque_ref("retriever_id", self.retriever_id)
        _require_opaque_ref("index_format", self.index_format)
        if not isinstance(self.index_parameters_ref, IndexParametersRef):
            raise TypeError("index_parameters_ref must be an IndexParametersRef")
        payload_refs = _freeze_opaque_refs("payload_refs", self.payload_refs)
        if not payload_refs:
            raise ValueError("payload_refs must not be empty")
        if len(payload_refs) > PERSONAL_LITE_MAX_SHARDS_PER_LANE:
            raise ValueError("personal-lite lanes contain at most 16 shard payloads")
        if self.index_parameters_ref.payload_ref in payload_refs:
            raise ValueError("parameter payload_ref must not also be a shard payload_ref")
        object.__setattr__(self, "payload_refs", payload_refs)
        _validate_lane_identity(
            self.lane,
            embedding_spaces,
            hash_spaces,
            self.target_bundle_manifest_sha256,
        )


@dataclass(frozen=True, slots=True)
class StagedGenerationSetManifest:
    """Closed build input which deliberately cannot be published directly."""

    build_id: str
    activation_set_id: str
    workspace_id: str
    corpus_ref: str
    retrieval_profile_ref: str
    target_type: TargetType
    unit_manifest_sha256: str
    target_representation_manifest_sha256: str
    processing_fingerprint: ProcessingFingerprint
    unit_catalog: StagedUnitCatalog
    lanes: tuple[StagedIndexGeneration, ...]
    payloads: tuple[PayloadFileRef, ...]
    frozen_unit_manifest_proof_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        for name in ("build_id", "activation_set_id", "workspace_id", "corpus_ref"):
            _require_opaque_ref(name, getattr(self, name))
        if self.retrieval_profile_ref != PERSONAL_LITE_PROFILE:
            raise ValueError(
                "only the personal-lite generation profile is implemented; enterprise is reserved"
            )
        if self.target_type is not TargetType.IMAGE:
            raise ValueError("personal-lite generations currently support only image targets")
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        _require_sha256(
            "target_representation_manifest_sha256",
            self.target_representation_manifest_sha256,
        )
        _require_sha256(
            "frozen_unit_manifest_proof_sha256",
            self.frozen_unit_manifest_proof_sha256,
        )
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        if not isinstance(self.processing_fingerprint, ProcessingFingerprint):
            raise TypeError("processing_fingerprint must be a ProcessingFingerprint")
        if not isinstance(self.unit_catalog, StagedUnitCatalog):
            raise TypeError("unit_catalog must be a StagedUnitCatalog")
        lanes = _freeze_tuple("lanes", self.lanes, StagedIndexGeneration)
        payloads = _freeze_tuple("payloads", self.payloads, PayloadFileRef)
        object.__setattr__(
            self, "lanes", tuple(sorted(lanes, key=lambda item: _LANE_ORDER[item.lane]))
        )
        object.__setattr__(
            self,
            "payloads",
            tuple(sorted(payloads, key=lambda item: (item.role, item.relative_path.as_posix()))),
        )
        validate_staged_generation_set_manifest(self)


@dataclass(frozen=True, slots=True)
class SealedFileIdentity:
    """Record the file identity rechecked immediately before publication."""

    relative_path: PurePosixPath
    size_bytes: int
    sha256: str
    device_id: int | None
    inode: int | None
    modified_time_ns: int | None
    mode: int


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Record one closed owner-level validation observation."""

    report_id: str
    owner_kind: str
    owner_id: str
    validator_id: str
    staged_identity_sha256: str
    observed_payload_sha256s: tuple[str, ...]
    capability_codes: tuple[str, ...]
    passed: bool
    created_at: datetime
    schema_version: int = GENERATION_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("report_id", "owner_kind", "owner_id", "validator_id"):
            _require_opaque_ref(name, getattr(self, name))
        _require_sha256("staged_identity_sha256", self.staged_identity_sha256)
        observed = tuple(self.observed_payload_sha256s)
        if not observed:
            raise ValueError("observed_payload_sha256s must not be empty")
        for digest in observed:
            _require_sha256("observed_payload_sha256", digest)
        if len(set(observed)) != len(observed):
            raise ValueError("observed_payload_sha256s must be unique")
        object.__setattr__(self, "observed_payload_sha256s", tuple(sorted(observed)))
        capabilities = _freeze_opaque_refs("capability_codes", self.capability_codes)
        object.__setattr__(self, "capability_codes", tuple(sorted(capabilities)))
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a boolean")
        _require_aware_datetime("created_at", self.created_at)
        _require_schema_version(self.schema_version)


@dataclass(frozen=True, slots=True)
class ValidatedGenerationSet:
    """Carry the sealed staging token required for publication."""

    staging_directory: Path
    manifest: GenerationSetManifest
    manifest_bytes: bytes
    manifest_sha256: str
    sealed_files: tuple[SealedFileIdentity, ...]


@dataclass(frozen=True, slots=True)
class PublishedGenerationSet:
    """Identify one immutable filesystem generation; this is not an activation."""

    root: Path
    manifest: GenerationSetManifest
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationDirectoryInspection:
    """Describe one publication-root directory without mutating it."""

    root: Path
    state: GenerationDirectoryState
    activation_set_id: str | None
    manifest_sha256: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class PayloadValidationKey:
    """Close native validation routing by payload role and index format."""

    role: str
    index_format: str

    def __post_init__(self) -> None:
        if "*" in self.role or "*" in self.index_format:
            raise ValueError("wildcard payload validators are not supported")
        _require_opaque_ref("role", self.role)
        _require_opaque_ref("index_format", self.index_format)


PayloadOwner = StagedIndexGeneration | StagedUnitCatalog


class PayloadValidator(Protocol):
    """Validate one checksum-proven payload without changing it."""

    @property
    def validator_id(self) -> str: ...

    def validate(
        self,
        root: Path,
        payload: PayloadFileRef,
        owner: PayloadOwner,
        *,
        deadline: datetime,
    ) -> ValidationReport:
        """Return one positive or negative immutable validation observation."""


def validate_generation_set_manifest(manifest: GenerationSetManifest) -> None:
    """Validate cross-lane identity and the complete final payload-reference closure."""
    _validate_complete_lane_set(tuple(item.lane for item in manifest.lanes))
    _validate_payload_uniqueness(manifest.payloads)
    if manifest.unit_catalog.unit_manifest_sha256 != manifest.unit_manifest_sha256:
        raise ValueError("unit catalog unit_manifest_sha256 must match the generation set")

    references = [
        TARGET_REPRESENTATION_MANIFEST_ROLE,
        manifest.unit_catalog.payload_ref,
        manifest.unit_catalog.validation_report_ref,
    ]
    semantic_spaces: dict[RetrievalLane, tuple[EmbeddingSpaceRef, ...]] = {}
    payload_by_role = {payload.role: payload for payload in manifest.payloads}
    representation_payload = payload_by_role.get(TARGET_REPRESENTATION_MANIFEST_ROLE)
    if representation_payload is None:
        raise ValueError("missing target representation manifest payload")
    if representation_payload.sha256 != manifest.target_representation_manifest_sha256:
        raise ValueError(
            "target representation manifest payload SHA-256 must match the generation set"
        )
    generation_ids: set[str] = set()
    for item in manifest.lanes:
        generation = item.generation
        if generation.generation_id in generation_ids:
            raise ValueError("lane generation_id values must be unique")
        generation_ids.add(generation.generation_id)
        comparisons = (
            ("activation_set_id", generation.activation_set_id, manifest.activation_set_id),
            ("workspace_id", generation.workspace_id, manifest.workspace_id),
            ("corpus_ref", generation.corpus_ref, manifest.corpus_ref),
            ("target_type", generation.target_type, manifest.target_type),
            (
                "unit_manifest_sha256",
                generation.unit_manifest_sha256,
                manifest.unit_manifest_sha256,
            ),
            (
                "processing_fingerprint",
                generation.build_processing_fingerprint,
                manifest.processing_fingerprint,
            ),
            ("created_at", generation.created_at, manifest.created_at),
        )
        for name, observed, expected in comparisons:
            if observed != expected:
                raise ValueError(f"lane {name} must match the generation set")
        if item.lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
            semantic_spaces[item.lane] = generation.embedding_spaces
        parameter_ref = generation.index_parameters_ref.payload_ref
        if len(generation.shard_refs) > PERSONAL_LITE_MAX_SHARDS_PER_LANE:
            raise ValueError("personal-lite lanes contain at most 16 shard payloads")
        if generation.index_parameters_ref.parameter_schema != generation.index_format:
            raise ValueError("parameter_schema must match the lane index_format")
        references.extend((parameter_ref, *generation.shard_refs, generation.validation_report_ref))
        parameter_payload = payload_by_role.get(parameter_ref)
        if parameter_payload is None:
            raise ValueError(f"missing parameter payload role: {parameter_ref}")
        if parameter_payload.sha256 != generation.index_parameters_ref.sha256:
            raise ValueError("parameter payload SHA-256 must match IndexParametersRef")

    if manifest.unit_catalog.catalog_id in generation_ids:
        raise ValueError("catalog_id and lane generation_id values must be distinct")
    if semantic_spaces[RetrievalLane.INSTANCE] == semantic_spaces[RetrievalLane.CONCEPT]:
        raise ValueError("personal-lite instance and concept embedding spaces must be distinct")
    _validate_reference_closure(references, manifest.payloads)


def validate_staged_generation_set_manifest(manifest: StagedGenerationSetManifest) -> None:
    """Validate the closed build input without inventing final report references."""
    _validate_complete_lane_set(tuple(item.lane for item in manifest.lanes))
    _validate_payload_uniqueness(manifest.payloads)
    if manifest.unit_catalog.unit_manifest_sha256 != manifest.unit_manifest_sha256:
        raise ValueError("unit catalog unit_manifest_sha256 must match the generation set")
    payload_by_role = {payload.role: payload for payload in manifest.payloads}
    references = [TARGET_REPRESENTATION_MANIFEST_ROLE, manifest.unit_catalog.payload_ref]
    representation_payload = payload_by_role.get(TARGET_REPRESENTATION_MANIFEST_ROLE)
    if representation_payload is None:
        raise ValueError("missing target representation manifest payload")
    if representation_payload.sha256 != manifest.target_representation_manifest_sha256:
        raise ValueError(
            "target representation manifest payload SHA-256 must match the generation set"
        )
    generation_ids: set[str] = set()
    semantic_spaces: dict[RetrievalLane, tuple[EmbeddingSpaceRef, ...]] = {}
    for lane in manifest.lanes:
        if lane.generation_id in generation_ids:
            raise ValueError("lane generation_id values must be unique")
        generation_ids.add(lane.generation_id)
        if lane.activation_set_id != manifest.activation_set_id:
            raise ValueError("lane activation_set_id must match the generation set")
        if lane.unit_manifest_sha256 != manifest.unit_manifest_sha256:
            raise ValueError("lane unit_manifest_sha256 must match the generation set")
        if lane.lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
            semantic_spaces[lane.lane] = lane.embedding_spaces
        parameter_ref = lane.index_parameters_ref.payload_ref
        if lane.index_parameters_ref.parameter_schema != lane.index_format:
            raise ValueError("parameter_schema must match the lane index_format")
        references.extend((parameter_ref, *lane.payload_refs))
        parameter_payload = payload_by_role.get(parameter_ref)
        if parameter_payload is None:
            raise ValueError(f"missing parameter payload role: {parameter_ref}")
        if parameter_payload.sha256 != lane.index_parameters_ref.sha256:
            raise ValueError("parameter payload SHA-256 must match IndexParametersRef")
    if manifest.unit_catalog.catalog_id in generation_ids:
        raise ValueError("catalog_id and lane generation_id values must be distinct")
    if semantic_spaces[RetrievalLane.INSTANCE] == semantic_spaces[RetrievalLane.CONCEPT]:
        raise ValueError("personal-lite instance and concept embedding spaces must be distinct")
    _validate_reference_closure(references, manifest.payloads)


def encode_generation_set_manifest(manifest: GenerationSetManifest) -> bytes:
    """Encode one manifest as deterministic, newline-terminated strict JSON."""
    if not isinstance(manifest, GenerationSetManifest):
        raise TypeError("manifest must be a GenerationSetManifest")
    validate_generation_set_manifest(manifest)
    encoded = _encode_json_document(_generation_set_to_document(manifest))
    if len(encoded) > MAX_GENERATION_MANIFEST_BYTES:
        raise GenerationManifestError("canonical manifest exceeds the 1 MiB limit")
    return encoded


def decode_generation_set_manifest(payload: bytes) -> GenerationSetManifest:
    """Decode only the closed version-one final manifest schema."""
    document = _load_json_object(payload, maximum_bytes=MAX_GENERATION_MANIFEST_BYTES)
    try:
        manifest = _generation_set_from_document(document)
        validate_generation_set_manifest(manifest)
    except GenerationManifestError:
        raise
    except (TypeError, ValueError) as error:
        raise GenerationManifestError(str(error)) from error
    if encode_generation_set_manifest(manifest) != payload:
        raise GenerationManifestError("manifest must use canonical JSON encoding")
    return manifest


def encode_validation_report(report: ValidationReport) -> bytes:
    """Encode one validation report using the same canonical JSON discipline."""
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    return _encode_json_document(
        {
            "capability_codes": list(report.capability_codes),
            "created_at": _encode_datetime(report.created_at),
            "observed_payload_sha256s": list(report.observed_payload_sha256s),
            "owner_id": report.owner_id,
            "owner_kind": report.owner_kind,
            "passed": report.passed,
            "report_id": report.report_id,
            "schema_version": report.schema_version,
            "staged_identity_sha256": report.staged_identity_sha256,
            "validator_id": report.validator_id,
        }
    )


def staged_generation_identity_sha256(manifest: StagedGenerationSetManifest) -> str:
    """Return a deterministic digest for report binding without a final-manifest cycle."""
    document = {
        "activation_set_id": manifest.activation_set_id,
        "build_id": manifest.build_id,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "corpus_ref": manifest.corpus_ref,
        "frozen_unit_manifest_proof_sha256": manifest.frozen_unit_manifest_proof_sha256,
        "lanes": [
            {
                "activation_set_id": lane.activation_set_id,
                "embedding_spaces": [
                    _embedding_space_to_document(item) for item in lane.embedding_spaces
                ],
                "generation_id": lane.generation_id,
                "hash_spaces": [_hash_space_to_document(item) for item in lane.hash_spaces],
                "index_format": lane.index_format,
                "index_parameters_ref": _index_parameters_to_document(lane.index_parameters_ref),
                "lane": lane.lane.value,
                "payload_refs": list(lane.payload_refs),
                "retriever_id": lane.retriever_id,
                "target_bundle_manifest_sha256": lane.target_bundle_manifest_sha256,
                "unit_manifest_sha256": lane.unit_manifest_sha256,
            }
            for lane in manifest.lanes
        ],
        "payloads": [_payload_to_document(item) for item in manifest.payloads],
        "processing_fingerprint": {"sha256": manifest.processing_fingerprint.sha256},
        "retrieval_profile_ref": manifest.retrieval_profile_ref,
        "target_representation_manifest_sha256": manifest.target_representation_manifest_sha256,
        "target_type": manifest.target_type.value,
        "unit_catalog": {
            "catalog_format": manifest.unit_catalog.catalog_format,
            "catalog_id": manifest.unit_catalog.catalog_id,
            "payload_ref": manifest.unit_catalog.payload_ref,
            "unit_count": manifest.unit_catalog.unit_count,
            "unit_manifest_sha256": manifest.unit_catalog.unit_manifest_sha256,
        },
        "unit_manifest_sha256": manifest.unit_manifest_sha256,
        "workspace_id": manifest.workspace_id,
    }
    return hashlib.sha256(_encode_json_document(document)).hexdigest()


def _generation_set_to_document(manifest: GenerationSetManifest) -> dict[str, object]:
    return {
        "activation_set_id": manifest.activation_set_id,
        "corpus_ref": manifest.corpus_ref,
        "created_at": _encode_datetime(manifest.created_at),
        "frozen_unit_manifest_proof_sha256": manifest.frozen_unit_manifest_proof_sha256,
        "lanes": [
            {
                "generation": _index_generation_to_document(item.generation),
                "lane": item.lane.value,
                "target_bundle_manifest_sha256": item.target_bundle_manifest_sha256,
            }
            for item in manifest.lanes
        ],
        "payloads": [_payload_to_document(item) for item in manifest.payloads],
        "processing_fingerprint": {"sha256": manifest.processing_fingerprint.sha256},
        "retrieval_profile_ref": manifest.retrieval_profile_ref,
        "schema_version": manifest.schema_version,
        "target_representation_manifest_sha256": manifest.target_representation_manifest_sha256,
        "target_type": manifest.target_type.value,
        "unit_catalog": {
            "catalog_format": manifest.unit_catalog.catalog_format,
            "catalog_id": manifest.unit_catalog.catalog_id,
            "payload_ref": manifest.unit_catalog.payload_ref,
            "schema_version": manifest.unit_catalog.schema_version,
            "unit_count": manifest.unit_catalog.unit_count,
            "unit_manifest_sha256": manifest.unit_catalog.unit_manifest_sha256,
            "validation_report_ref": manifest.unit_catalog.validation_report_ref,
        },
        "unit_manifest_sha256": manifest.unit_manifest_sha256,
        "workspace_id": manifest.workspace_id,
    }


def _index_generation_to_document(generation: IndexGeneration) -> dict[str, object]:
    return {
        "activation_set_id": generation.activation_set_id,
        "build_processing_fingerprint": {"sha256": generation.build_processing_fingerprint.sha256},
        "calibration_profile_ref": generation.calibration_profile_ref,
        "corpus_ref": generation.corpus_ref,
        "created_at": _encode_datetime(generation.created_at),
        "embedding_spaces": [
            _embedding_space_to_document(item) for item in generation.embedding_spaces
        ],
        "generation_id": generation.generation_id,
        "hash_spaces": [_hash_space_to_document(item) for item in generation.hash_spaces],
        "index_format": generation.index_format,
        "index_parameters_ref": _index_parameters_to_document(generation.index_parameters_ref),
        "retriever_id": generation.retriever_id,
        "schema_version": generation.schema_version,
        "shard_refs": list(generation.shard_refs),
        "target_type": generation.target_type.value,
        "unit_manifest_sha256": generation.unit_manifest_sha256,
        "validation_report_ref": generation.validation_report_ref,
        "workspace_id": generation.workspace_id,
    }


def _embedding_space_to_document(space: EmbeddingSpaceRef) -> dict[str, object]:
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


def _hash_space_to_document(space: HashSpaceRef) -> dict[str, object]:
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


def _index_parameters_to_document(parameters: IndexParametersRef) -> dict[str, object]:
    return {
        "parameter_schema": parameters.parameter_schema,
        "payload_ref": parameters.payload_ref,
        "schema_version": parameters.schema_version,
        "sha256": parameters.sha256,
    }


def _payload_to_document(payload: PayloadFileRef) -> dict[str, object]:
    return {
        "relative_path": payload.relative_path.as_posix(),
        "role": payload.role,
        "sha256": payload.sha256,
        "size_bytes": payload.size_bytes,
    }


def _generation_set_from_document(value: object) -> GenerationSetManifest:
    document = _closed_object(
        value,
        "generation manifest",
        {
            "activation_set_id",
            "corpus_ref",
            "created_at",
            "frozen_unit_manifest_proof_sha256",
            "lanes",
            "payloads",
            "processing_fingerprint",
            "retrieval_profile_ref",
            "schema_version",
            "target_representation_manifest_sha256",
            "target_type",
            "unit_catalog",
            "unit_manifest_sha256",
            "workspace_id",
        },
    )
    _require_document_schema(document)
    fingerprint = _processing_fingerprint_from_document(document["processing_fingerprint"])
    lanes = tuple(_lane_from_document(item) for item in _object_list(document["lanes"], "lanes"))
    payloads = tuple(
        _payload_from_document(item) for item in _object_list(document["payloads"], "payloads")
    )
    return GenerationSetManifest(
        activation_set_id=_string(document["activation_set_id"], "activation_set_id"),
        workspace_id=_string(document["workspace_id"], "workspace_id"),
        corpus_ref=_string(document["corpus_ref"], "corpus_ref"),
        retrieval_profile_ref=_string(
            document["retrieval_profile_ref"],
            "retrieval_profile_ref",
        ),
        target_type=_enum_value(TargetType, document["target_type"], "target_type"),
        unit_manifest_sha256=_string(
            document["unit_manifest_sha256"],
            "unit_manifest_sha256",
        ),
        frozen_unit_manifest_proof_sha256=_string(
            document["frozen_unit_manifest_proof_sha256"],
            "frozen_unit_manifest_proof_sha256",
        ),
        target_representation_manifest_sha256=_string(
            document["target_representation_manifest_sha256"],
            "target_representation_manifest_sha256",
        ),
        processing_fingerprint=fingerprint,
        unit_catalog=_unit_catalog_from_document(document["unit_catalog"]),
        lanes=lanes,
        payloads=payloads,
        created_at=_datetime_value(document["created_at"], "created_at"),
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _unit_catalog_from_document(value: object) -> UnitCatalogManifest:
    document = _closed_object(
        value,
        "unit_catalog",
        {
            "catalog_format",
            "catalog_id",
            "payload_ref",
            "schema_version",
            "unit_count",
            "unit_manifest_sha256",
            "validation_report_ref",
        },
    )
    _require_document_schema(document)
    return UnitCatalogManifest(
        catalog_id=_string(document["catalog_id"], "catalog_id"),
        unit_count=_integer(document["unit_count"], "unit_count"),
        unit_manifest_sha256=_string(
            document["unit_manifest_sha256"],
            "unit_manifest_sha256",
        ),
        catalog_format=_string(document["catalog_format"], "catalog_format"),
        payload_ref=_string(document["payload_ref"], "payload_ref"),
        validation_report_ref=_string(
            document["validation_report_ref"],
            "validation_report_ref",
        ),
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _lane_from_document(value: object) -> LaneGeneration:
    document = _closed_object(
        value,
        "lane",
        {"generation", "lane", "target_bundle_manifest_sha256"},
    )
    bundle_value = document["target_bundle_manifest_sha256"]
    bundle = (
        None if bundle_value is None else _string(bundle_value, "target_bundle_manifest_sha256")
    )
    return LaneGeneration(
        lane=_enum_value(RetrievalLane, document["lane"], "lane"),
        generation=_index_generation_from_document(document["generation"]),
        target_bundle_manifest_sha256=bundle,
    )


def _index_generation_from_document(value: object) -> IndexGeneration:
    document = _closed_object(
        value,
        "generation",
        {
            "activation_set_id",
            "build_processing_fingerprint",
            "calibration_profile_ref",
            "corpus_ref",
            "created_at",
            "embedding_spaces",
            "generation_id",
            "hash_spaces",
            "index_format",
            "index_parameters_ref",
            "retriever_id",
            "schema_version",
            "shard_refs",
            "target_type",
            "unit_manifest_sha256",
            "validation_report_ref",
            "workspace_id",
        },
    )
    _require_document_schema(document)
    return IndexGeneration(
        generation_id=_string(document["generation_id"], "generation_id"),
        activation_set_id=_string(document["activation_set_id"], "activation_set_id"),
        workspace_id=_string(document["workspace_id"], "workspace_id"),
        corpus_ref=_string(document["corpus_ref"], "corpus_ref"),
        target_type=_enum_value(TargetType, document["target_type"], "target_type"),
        unit_manifest_sha256=_string(
            document["unit_manifest_sha256"],
            "unit_manifest_sha256",
        ),
        embedding_spaces=tuple(
            _embedding_space_from_document(item)
            for item in _object_list(document["embedding_spaces"], "embedding_spaces")
        ),
        hash_spaces=tuple(
            _hash_space_from_document(item)
            for item in _object_list(document["hash_spaces"], "hash_spaces")
        ),
        retriever_id=_string(document["retriever_id"], "retriever_id"),
        index_format=_string(document["index_format"], "index_format"),
        index_parameters_ref=_index_parameters_from_document(document["index_parameters_ref"]),
        shard_refs=tuple(
            _string(item, "shard_ref")
            for item in _object_list(document["shard_refs"], "shard_refs")
        ),
        calibration_profile_ref=_string(
            document["calibration_profile_ref"],
            "calibration_profile_ref",
        ),
        build_processing_fingerprint=_processing_fingerprint_from_document(
            document["build_processing_fingerprint"]
        ),
        created_at=_datetime_value(document["created_at"], "created_at"),
        validation_report_ref=_string(
            document["validation_report_ref"],
            "validation_report_ref",
        ),
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _embedding_space_from_document(value: object) -> EmbeddingSpaceRef:
    document = _closed_object(
        value,
        "embedding_space",
        {
            "dimension",
            "distance_metric",
            "model_revision",
            "normalization",
            "preprocessing_version",
            "provider_id",
            "query_modalities",
            "schema_version",
            "space_id",
            "target_modalities",
        },
    )
    _require_document_schema(document)
    return EmbeddingSpaceRef(
        space_id=_string(document["space_id"], "space_id"),
        provider_id=_string(document["provider_id"], "provider_id"),
        model_revision=_string(document["model_revision"], "model_revision"),
        preprocessing_version=_string(
            document["preprocessing_version"],
            "preprocessing_version",
        ),
        dimension=_integer(document["dimension"], "dimension"),
        normalization=_enum_value(
            VectorNormalization,
            document["normalization"],
            "normalization",
        ),
        distance_metric=_enum_value(
            DistanceMetric,
            document["distance_metric"],
            "distance_metric",
        ),
        query_modalities=frozenset(
            _enum_value(QueryModality, item, "query_modality")
            for item in _object_list(document["query_modalities"], "query_modalities")
        ),
        target_modalities=frozenset(
            _enum_value(TargetType, item, "target_modality")
            for item in _object_list(document["target_modalities"], "target_modalities")
        ),
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _hash_space_from_document(value: object) -> HashSpaceRef:
    document = _closed_object(
        value,
        "hash_space",
        {
            "algorithm",
            "bit_width",
            "hash_version",
            "preprocessing_version",
            "provider_id",
            "provider_version",
            "schema_version",
            "space_id",
            "target_modalities",
        },
    )
    _require_document_schema(document)
    return HashSpaceRef(
        space_id=_string(document["space_id"], "space_id"),
        provider_id=_string(document["provider_id"], "provider_id"),
        provider_version=_string(document["provider_version"], "provider_version"),
        preprocessing_version=_string(
            document["preprocessing_version"],
            "preprocessing_version",
        ),
        algorithm=_string(document["algorithm"], "algorithm"),
        bit_width=_integer(document["bit_width"], "bit_width"),
        hash_version=_string(document["hash_version"], "hash_version"),
        target_modalities=frozenset(
            _enum_value(TargetType, item, "target_modality")
            for item in _object_list(document["target_modalities"], "target_modalities")
        ),
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _index_parameters_from_document(value: object) -> IndexParametersRef:
    document = _closed_object(
        value,
        "index_parameters_ref",
        {"parameter_schema", "payload_ref", "schema_version", "sha256"},
    )
    _require_document_schema(document)
    return IndexParametersRef(
        parameter_schema=_string(document["parameter_schema"], "parameter_schema"),
        payload_ref=_string(document["payload_ref"], "payload_ref"),
        sha256=_string(document["sha256"], "sha256"),
        schema_version=_integer(document["schema_version"], "schema_version"),
    )


def _processing_fingerprint_from_document(value: object) -> ProcessingFingerprint:
    document = _closed_object(value, "processing_fingerprint", {"sha256"})
    return ProcessingFingerprint(_string(document["sha256"], "sha256"))


def _payload_from_document(value: object) -> PayloadFileRef:
    document = _closed_object(
        value,
        "payload",
        {"relative_path", "role", "sha256", "size_bytes"},
    )
    return PayloadFileRef(
        relative_path=PurePosixPath(_string(document["relative_path"], "relative_path")),
        sha256=_string(document["sha256"], "sha256"),
        size_bytes=_integer(document["size_bytes"], "size_bytes"),
        role=_string(document["role"], "role"),
    )


def _validate_lane_identity(
    lane: RetrievalLane,
    embedding_spaces: tuple[EmbeddingSpaceRef, ...],
    hash_spaces: tuple[HashSpaceRef, ...],
    bundle_sha256: str | None,
) -> None:
    if lane is RetrievalLane.PERCEPTUAL:
        if embedding_spaces or len(hash_spaces) != 1:
            raise ValueError("perceptual lane requires exactly one complete pHash space")
        if bundle_sha256 is not None:
            raise ValueError("perceptual lane target_bundle_manifest_sha256 must be null")
        return
    if lane not in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
        raise ValueError("personal-lite generation lanes exclude content and enterprise extensions")
    if len(embedding_spaces) != 1 or hash_spaces:
        raise ValueError(f"{lane.value} lane requires exactly one complete embedding space")
    if bundle_sha256 is None:
        raise ValueError(f"{lane.value} lane requires target_bundle_manifest_sha256")
    _require_sha256("target_bundle_manifest_sha256", bundle_sha256)


def _validate_complete_lane_set(lanes: tuple[RetrievalLane, ...]) -> None:
    if len(lanes) != len(set(lanes)):
        raise ValueError("generation lanes must be unique")
    if frozenset(lanes) != _REQUIRED_PERSONAL_LANES:
        raise ValueError("personal-lite requires exactly perceptual, instance, and concept lanes")


def _validate_payload_uniqueness(payloads: tuple[PayloadFileRef, ...]) -> None:
    if not payloads:
        raise ValueError("payloads must not be empty")
    if len(payloads) > PERSONAL_LITE_MAX_PAYLOADS:
        raise ValueError("personal-lite manifests contain at most 64 payloads")
    roles = [payload.role for payload in payloads]
    paths = [payload.relative_path for payload in payloads]
    if len(roles) != len(set(roles)):
        raise ValueError("payload roles must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("payload relative paths must be unique")


def _validate_reference_closure(
    references: list[str],
    payloads: tuple[PayloadFileRef, ...],
) -> None:
    counts = Counter(references)
    duplicate_refs = sorted(role for role, count in counts.items() if count != 1)
    if duplicate_refs:
        raise ValueError(f"payload roles must be referenced exactly once: {duplicate_refs}")
    declared = {payload.role for payload in payloads}
    missing = sorted(set(references) - declared)
    if missing:
        raise ValueError(f"missing payload roles: {missing}")
    unreferenced = sorted(declared - set(references))
    if unreferenced:
        raise ValueError(f"unreferenced payload roles: {unreferenced}")


def _freeze_tuple(
    name: str,
    values: tuple[_T, ...],
    item_type: type[_T],
) -> tuple[_T, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    if any(not isinstance(value, item_type) for value in values):
        raise TypeError(f"{name} contains an invalid value")
    return values


def _freeze_opaque_refs(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    for value in values:
        _require_opaque_ref(name, value)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    return values


def _require_safe_payload_path(path: PurePosixPath) -> None:
    if not isinstance(path, PurePosixPath):
        raise TypeError("relative_path must be a PurePosixPath")
    text = path.as_posix()
    if (
        not text
        or text == "."
        or path.is_absolute()
        or text.startswith("//")
        or _WINDOWS_DRIVE_PATTERN.match(text)
        or "\\" in text
        or "\x00" in text
        or ":" in text
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError("relative_path must be a safe relative POSIX path")
    for part in path.parts:
        stem = part.rstrip(" .").split(".", maxsplit=1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise ValueError("relative_path contains a reserved device name")


def _require_opaque_ref(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not _OPAQUE_REF_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a safe non-blank opaque reference")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_schema_version(value: int) -> None:
    if value != GENERATION_MANIFEST_SCHEMA_VERSION or isinstance(value, bool):
        raise ValueError(f"schema_version must be {GENERATION_MANIFEST_SCHEMA_VERSION}")


def _require_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _encode_datetime(value: datetime) -> str:
    _require_aware_datetime("datetime", value)
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _datetime_value(value: object, name: str) -> datetime:
    text = _string(value, name)
    if not text.endswith("Z"):
        raise GenerationManifestError(f"{name} must use canonical UTC Z form")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise GenerationManifestError(f"{name} must be an ISO-8601 datetime") from error
    if _encode_datetime(parsed) != text:
        raise GenerationManifestError(f"{name} must include canonical microseconds")
    return parsed


def _encode_json_document(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_json_object(payload: bytes, *, maximum_bytes: int) -> dict[str, object]:
    if not isinstance(payload, bytes):
        raise TypeError("manifest payload must be bytes")
    if not payload or len(payload) > maximum_bytes:
        raise GenerationManifestError("manifest size is outside the allowed range")
    try:
        text = payload.decode("utf-8", errors="strict")
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GenerationManifestError("manifest must be valid UTF-8 JSON") from error
    return _closed_object(decoded, "generation manifest", None)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise GenerationManifestError(f"non-finite JSON constant is forbidden: {value}")


def _closed_object(
    value: object,
    name: str,
    expected_fields: set[str] | None,
) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise GenerationManifestError(f"{name} must be an object")
    document = cast(dict[str, object], value)
    if expected_fields is not None:
        actual = set(document)
        unknown = sorted(actual - expected_fields)
        if unknown:
            raise GenerationManifestError(f"{name} has unknown field: {unknown[0]}")
        missing = sorted(expected_fields - actual)
        if missing:
            raise GenerationManifestError(f"{name} is missing field: {missing[0]}")
    return document


def _object_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise GenerationManifestError(f"{name} must be an array")
    return cast(list[object], value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise GenerationManifestError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenerationManifestError(f"{name} must be an integer")
    return value


def _enum_value(
    enum_type: type[_StrEnumT],
    value: object,
    name: str,
) -> _StrEnumT:
    text = _string(value, name)
    try:
        return enum_type(text)
    except ValueError as error:
        raise GenerationManifestError(f"{name} has an unsupported value") from error


def _require_document_schema(document: dict[str, object]) -> None:
    value = _integer(document["schema_version"], "schema_version")
    if value != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise GenerationManifestError(
            f"schema_version must be {GENERATION_MANIFEST_SCHEMA_VERSION}"
        )
