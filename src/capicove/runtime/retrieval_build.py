"""Production administration for inert personal image-search generations."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from capicove.contracts.retrieval import (
    IndexGeneration,
    IndexParametersRef,
    TargetType,
)
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity
from capicove.processing.image.representations import (
    ImageTargetRepresentation,
    ImageTargetRepresentationProducer,
    RepresentationProductionResult,
)
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval._adapters.personal_numpy import (
    NumpyExactIndexReader,
    NumpyExactIndexWriter,
    OrdinalPhash,
    OrdinalVector,
    PersonalIndexPayload,
    PhashFlatIndexReader,
    PhashFlatIndexWriter,
)
from capicove.retrieval._adapters.sqlite_unit_catalog import (
    SQLITE_UNIT_CATALOG_FORMAT,
    SqliteUnitCatalog,
    SqliteUnitCatalogDescriptor,
    SqliteUnitCatalogWriter,
)
from capicove.retrieval.image_generations import (
    PERSONAL_LITE_PROFILE,
    TARGET_REPRESENTATION_MANIFEST_ROLE,
    PayloadFileRef,
    PayloadOwner,
    PayloadValidationKey,
    PayloadValidator,
    PublishedGenerationSet,
    StagedGenerationSetManifest,
    StagedIndexGeneration,
    StagedUnitCatalog,
    ValidatedGenerationSet,
    ValidationReport,
    staged_generation_identity_sha256,
)
from capicove.retrieval.search_models import RetrievalLane
from capicove.runtime.retrieval_personal import (
    PersonalIndexParameters,
    decode_personal_index_parameters,
    encode_personal_index_parameters,
)
from capicove.runtime.retrieval_representations import (
    _PERSONAL_CLEANUP_LOCK,
    PERSONAL_IMAGE_MAX_UNITS,
    ImageBuildSource,
    PersonalImageRepresentationSink,
    StagedRepresentationEvidenceIdentity,
    StagedRepresentationEvidenceLease,
)


_SEMANTIC_FORMAT = "numpy-flat-f32-v1"
_PHASH_FORMAT = "phash-flat-u64-v1"
_LANES = (
    RetrievalLane.PERCEPTUAL,
    RetrievalLane.INSTANCE,
    RetrievalLane.CONCEPT,
)


@dataclass(frozen=True, slots=True)
class PersonalImageBuildPlan:
    """Name one personal build without granting validation, publication, or activation."""

    build_id: str
    activation_set_id: str
    workspace_id: str
    corpus_ref: str
    batch_size: int = 64

    def __post_init__(self) -> None:
        for field_name in (
            "build_id",
            "activation_set_id",
            "workspace_id",
            "corpus_ref",
        ):
            _require_opaque(field_name, getattr(self, field_name))
        if isinstance(self.batch_size, bool) or not 1 <= self.batch_size <= 1_000:
            raise ValueError("batch_size must be from 1 to 1,000")


@dataclass(frozen=True, slots=True)
class BuiltPersonalImageGeneration:
    """Carry inert staged bytes and their exact native validators."""

    staging_directory: Path
    staged_manifest: StagedGenerationSetManifest
    representation_result: RepresentationProductionResult
    native_validators: Mapping[PayloadValidationKey, PayloadValidator] = field(repr=False)
    staging_directory_identity: tuple[int, int] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    representation_evidence_lease: StagedRepresentationEvidenceLease | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.staging_directory, Path) or not self.staging_directory.is_absolute():
            raise ValueError("staging_directory must be an absolute Path")
        if not isinstance(self.staged_manifest, StagedGenerationSetManifest):
            raise TypeError("staged_manifest must be a StagedGenerationSetManifest")
        if not isinstance(self.representation_result, RepresentationProductionResult):
            raise TypeError("representation_result must be a RepresentationProductionResult")
        staging_identity = self.staging_directory_identity
        if staging_identity is None:
            try:
                metadata = os.lstat(self.staging_directory)
            except OSError as error:
                raise ValueError(
                    "staging_directory must exist when its identity is omitted"
                ) from error
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("staging_directory identity fallback requires a directory")
            staging_identity = (metadata.st_dev, metadata.st_ino)
            object.__setattr__(self, "staging_directory_identity", staging_identity)
        if (
            not isinstance(staging_identity, tuple)
            or len(staging_identity) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in staging_identity
            )
        ):
            raise ValueError("staging_directory_identity must contain device and inode integers")
        report = self.representation_result.report
        sealed = self.representation_result.sealed_manifest
        if (
            sealed.sha256 != self.staged_manifest.target_representation_manifest_sha256
            or report.unit_count != self.staged_manifest.unit_catalog.unit_count
            or report.unit_manifest_sha256 != self.staged_manifest.unit_manifest_sha256
            or report.processing_fingerprint != self.staged_manifest.processing_fingerprint
            or report.final_checkpoint_sha256 != self.staged_manifest.checkpoint_sha256
        ):
            raise ValueError("representation evidence does not match the staged generation")
        validators = dict(self.native_validators)
        if not validators or any(not isinstance(key, PayloadValidationKey) for key in validators):
            raise TypeError("native_validators must use PayloadValidationKey values")
        object.__setattr__(self, "native_validators", MappingProxyType(validators))
        lease = self.representation_evidence_lease
        if lease is not None:
            if not isinstance(lease, StagedRepresentationEvidenceLease):
                raise TypeError(
                    "representation_evidence_lease must be a StagedRepresentationEvidenceLease"
                )
            if lease.identity != _representation_evidence_identity(
                self.staged_manifest,
                self.representation_result,
            ):
                raise ValueError("representation evidence identity does not match the build")
            if not lease.is_bound_to_staging_directory(self.staging_directory):
                raise ValueError(
                    "representation evidence lease does not belong to the build staging"
                )


class PersonalImageGenerationAdministrator:
    """Build, validate, and publish personal generations as separate inert steps."""

    def __init__(
        self,
        store: FilesystemGenerationStore,
        producer: ImageTargetRepresentationProducer,
        *,
        max_units: int = PERSONAL_IMAGE_MAX_UNITS,
    ) -> None:
        if not isinstance(store, FilesystemGenerationStore):
            raise TypeError("store must be a FilesystemGenerationStore")
        if not isinstance(producer, ImageTargetRepresentationProducer):
            raise TypeError("producer must be an ImageTargetRepresentationProducer")
        if isinstance(max_units, bool) or not 1 <= max_units <= PERSONAL_IMAGE_MAX_UNITS:
            raise ValueError("max_units must be from 1 to 10,000")
        self._store = store
        self._producer = producer
        self._max_units = max_units
        self._open_representation_evidence: dict[Path, StagedRepresentationEvidenceLease] = {}

    def build(
        self,
        source: ImageBuildSource,
        plan: PersonalImageBuildPlan,
        *,
        deadline: datetime,
        retain_representation_evidence: bool = False,
    ) -> BuiltPersonalImageGeneration:
        """Build closed staged bytes without validating, publishing, or activating."""
        _check_deadline(deadline)
        if not isinstance(source, ImageBuildSource):
            raise TypeError("source must be an ImageBuildSource")
        if not isinstance(plan, PersonalImageBuildPlan):
            raise TypeError("plan must be a PersonalImageBuildPlan")
        if not isinstance(retain_representation_evidence, bool):
            raise TypeError("retain_representation_evidence must be a boolean")

        # Capacity and readiness are deliberately checked before the first decode
        # and before a staging directory exists.
        summary = source.summary
        if summary.unit_count > self._max_units:
            raise ValueError("image build summary exceeds the configured personal capacity")
        self._producer.verify_ready(deadline=deadline)
        _check_deadline(deadline)

        staging = self._store.create_staging(plan.build_id)
        staging_identity = _owned_directory_identity(staging)
        try:
            return self._build_in_staging(
                source,
                summary.unit_count,
                plan,
                staging,
                staging_identity,
                deadline,
                retain_representation_evidence=retain_representation_evidence,
            )
        except BaseException as build_error:
            try:
                _remove_owned_staging(
                    staging,
                    self._store.staging_root,
                    expected_identity=staging_identity,
                )
            except BaseException as cleanup_error:
                build_error.add_note(
                    "owned staging cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def validate(
        self,
        built: BuiltPersonalImageGeneration,
        *,
        deadline: datetime,
    ) -> ValidatedGenerationSet:
        """Validate canonical representation evidence and every native payload."""
        _check_deadline(deadline)
        if not isinstance(built, BuiltPersonalImageGeneration):
            raise TypeError("built must be a BuiltPersonalImageGeneration")
        if built.staging_directory.parent != self._store.staging_root:
            raise ValueError("built generation does not belong to this administrator store")
        self._require_representation_evidence_closed(built.staging_directory)
        lease = built.representation_evidence_lease
        if lease is not None and not lease.is_closed:
            raise RuntimeError("staged representation evidence lease is open")
        sealed = built.representation_result.sealed_manifest
        if (
            hashlib.sha256(sealed.canonical_bytes).hexdigest() != sealed.sha256
            or sealed.manifest.canonical_bytes() != sealed.canonical_bytes
            or sealed.sha256 != built.staged_manifest.target_representation_manifest_sha256
        ):
            raise ValueError("target representation manifest failed canonical validation")
        return self._store.validate(
            built.staging_directory,
            built.staged_manifest,
            native_validators=built.native_validators,
            deadline=deadline,
        )

    def publish(
        self,
        validated: ValidatedGenerationSet,
        *,
        deadline: datetime,
    ) -> PublishedGenerationSet:
        """Publish one validated immutable set without activating any alias."""
        _check_deadline(deadline)
        if not isinstance(validated, ValidatedGenerationSet):
            raise TypeError("validated must be a ValidatedGenerationSet")
        self._require_representation_evidence_closed(validated.staging_directory)
        return self._store.publish(validated, deadline=deadline)

    def abort(self, built: BuiltPersonalImageGeneration) -> None:
        """Close retained evidence and remove only this administrator's inert staging."""
        if not isinstance(built, BuiltPersonalImageGeneration):
            raise TypeError("built must be a BuiltPersonalImageGeneration")
        staging = built.staging_directory
        if staging.parent != self._store.staging_root:
            raise ValueError("built generation does not belong to this administrator store")
        registered_lease = self._open_representation_evidence.get(staging)
        carried_lease = built.representation_evidence_lease
        if carried_lease is not None and not carried_lease.is_bound_to_staging_directory(staging):
            raise ValueError("built representation evidence lease does not belong to its staging")
        if (
            registered_lease is not None
            and carried_lease is not None
            and registered_lease is not carried_lease
        ):
            raise ValueError("built representation evidence lease does not match the administrator")
        lease = carried_lease if carried_lease is not None else registered_lease
        if lease is not None:
            lease.close()
        staging_identity = built.staging_directory_identity
        if staging_identity is None:  # Defensive: __post_init__ always resolves it.
            raise RuntimeError("built staging directory identity is unavailable")
        _remove_owned_staging(
            staging,
            self._store.staging_root,
            expected_identity=staging_identity,
        )

    def _build_in_staging(
        self,
        source: ImageBuildSource,
        unit_count: int,
        plan: PersonalImageBuildPlan,
        staging: Path,
        staging_identity: tuple[int, int],
        deadline: datetime,
        *,
        retain_representation_evidence: bool,
    ) -> BuiltPersonalImageGeneration:
        payload_root = staging / "payloads"
        payload_root.mkdir(mode=0o700)
        catalog_descriptor = SqliteUnitCatalogWriter(max_units=self._max_units).build(
            source.catalog_rows(),
            payload_root / "catalog.sqlite3",
            activation_set_id=plan.activation_set_id,
            corpus_ref=plan.corpus_ref,
            deadline=deadline,
        )
        catalog_payload = _payload_ref(
            staging,
            catalog_descriptor.path,
            role="catalog-payload",
            expected_sha256=catalog_descriptor.sha256,
            expected_size=catalog_descriptor.size_bytes,
        )

        sink = PersonalImageRepresentationSink(
            staging,
            source,
            max_units=self._max_units,
        )
        representation_result = self._producer.produce(
            source,
            sink=sink,
            batch_size=plan.batch_size,
            unit_manifest_sha256=catalog_descriptor.unit_manifest_sha256,
            deadline=deadline,
        )
        rows = sink.representation_rows()
        if len(rows) != unit_count:
            raise ValueError("representation production did not cover the build summary")

        lane_builds = tuple(
            self._build_lane(
                lane,
                rows=rows,
                plan=plan,
                staging=staging,
                payload_root=payload_root,
                unit_manifest_sha256=catalog_descriptor.unit_manifest_sha256,
                representation_result=representation_result,
                deadline=deadline,
            )
            for lane in _LANES
        )
        sealed_representation = representation_result.sealed_manifest
        representation_payload = _payload_ref(
            staging,
            staging.joinpath(*sealed_representation.relative_path.parts),
            role=TARGET_REPRESENTATION_MANIFEST_ROLE,
            expected_sha256=sealed_representation.sha256,
            expected_size=sealed_representation.size_bytes,
        )
        staged_manifest = StagedGenerationSetManifest(
            build_id=plan.build_id,
            activation_set_id=plan.activation_set_id,
            workspace_id=plan.workspace_id,
            corpus_ref=plan.corpus_ref,
            retrieval_profile_ref=PERSONAL_LITE_PROFILE,
            target_type=TargetType.IMAGE,
            unit_manifest_sha256=catalog_descriptor.unit_manifest_sha256,
            target_representation_manifest_sha256=(representation_result.sealed_manifest.sha256),
            processing_fingerprint=representation_result.report.processing_fingerprint,
            unit_catalog=StagedUnitCatalog(
                catalog_id=catalog_descriptor.catalog_id,
                unit_count=unit_count,
                unit_manifest_sha256=catalog_descriptor.unit_manifest_sha256,
                catalog_format=SQLITE_UNIT_CATALOG_FORMAT,
                payload_ref=catalog_payload.role,
            ),
            lanes=tuple(item.staged_lane for item in lane_builds),
            payloads=(
                representation_payload,
                catalog_payload,
                *(item.parameter_payload for item in lane_builds),
                *(item.shard_payload for item in lane_builds),
            ),
            frozen_unit_manifest_proof_sha256=(source.summary.frozen_unit_manifest_proof_sha256),
            checkpoint_sha256=representation_result.report.final_checkpoint_sha256,
        )
        staged_digest = staged_generation_identity_sha256(staged_manifest)
        validators: dict[PayloadValidationKey, PayloadValidator] = {
            PayloadValidationKey(
                catalog_payload.role,
                SQLITE_UNIT_CATALOG_FORMAT,
            ): _CatalogPayloadValidator(catalog_descriptor, staged_digest)
        }
        for lane_build in lane_builds:
            validators[
                PayloadValidationKey(
                    lane_build.parameter_payload.role,
                    lane_build.staged_lane.index_format,
                )
            ] = _ParameterPayloadValidator(
                lane_build.parameters,
                lane_build.parameter_payload.role,
                staged_digest,
            )
            validators[
                PayloadValidationKey(
                    lane_build.shard_payload.role,
                    lane_build.staged_lane.index_format,
                )
            ] = _IndexPayloadValidator(
                generation=lane_build.generation,
                payload_descriptor=lane_build.index_payload,
                expected_role=lane_build.shard_payload.role,
                catalog_descriptor=catalog_descriptor,
                staged_digest=staged_digest,
            )
        _check_deadline(deadline)
        evidence_lease: StagedRepresentationEvidenceLease | None = None
        if retain_representation_evidence:
            evidence_lease = sink.retain_evidence(
                _representation_evidence_identity(staged_manifest, representation_result),
                on_close=lambda: self._release_representation_evidence(staging),
            )
        else:
            sink.cleanup_transient_payloads()
        try:
            built = BuiltPersonalImageGeneration(
                staging_directory=staging,
                staged_manifest=staged_manifest,
                representation_result=representation_result,
                native_validators=validators,
                staging_directory_identity=staging_identity,
                representation_evidence_lease=evidence_lease,
            )
            if evidence_lease is not None:
                self._open_representation_evidence[staging] = evidence_lease
            return built
        except BaseException as build_error:
            if evidence_lease is not None:
                try:
                    evidence_lease.close()
                except BaseException as cleanup_error:
                    build_error.add_note(
                        "representation evidence ownership cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    def _require_representation_evidence_closed(self, staging: Path) -> None:
        lease = self._open_representation_evidence.get(staging)
        if lease is not None and not lease.is_closed:
            raise RuntimeError("staged representation evidence lease is open")

    def _release_representation_evidence(self, staging: Path) -> None:
        self._open_representation_evidence.pop(staging, None)

    def _build_lane(
        self,
        lane: RetrievalLane,
        *,
        rows: tuple[ImageTargetRepresentation, ...],
        plan: PersonalImageBuildPlan,
        staging: Path,
        payload_root: Path,
        unit_manifest_sha256: str,
        representation_result: RepresentationProductionResult,
        deadline: datetime,
    ) -> _LaneBuild:
        target_manifest = representation_result.sealed_manifest.manifest
        identity_by_lane = {
            ImageSemanticLane.INSTANCE: target_manifest.semantic_targets[0],
            ImageSemanticLane.CONCEPT: target_manifest.semantic_targets[1],
        }
        parameter_role = f"{lane.value}-parameters"
        shard_role = f"{lane.value}-shard"
        index_format = _PHASH_FORMAT if lane is RetrievalLane.PERCEPTUAL else _SEMANTIC_FORMAT
        shape = (
            (len(rows),)
            if lane is RetrievalLane.PERCEPTUAL
            else (
                len(rows),
                identity_by_lane[ImageSemanticLane(lane.value)].space.dimension,
            )
        )
        parameters = PersonalIndexParameters(
            lane=lane,
            index_format=index_format,
            unit_count=len(rows),
            shape=shape,
            dtype="<u8" if lane is RetrievalLane.PERCEPTUAL else "<f4",
            c_order=True,
            shard_role=shard_role,
        )
        parameter_path = payload_root / f"{parameter_role}.json"
        parameter_bytes = encode_personal_index_parameters(parameters)
        _write_private_payload(parameter_path, parameter_bytes)
        parameter_payload = _payload_ref(
            staging,
            parameter_path,
            role=parameter_role,
            expected_sha256=hashlib.sha256(parameter_bytes).hexdigest(),
            expected_size=len(parameter_bytes),
        )
        index_parameters = IndexParametersRef(
            parameter_schema=index_format,
            payload_ref=parameter_role,
            sha256=parameter_payload.sha256,
        )
        semantic_identity = (
            None
            if lane is RetrievalLane.PERCEPTUAL
            else identity_by_lane[ImageSemanticLane(lane.value)]
        )
        generation = IndexGeneration(
            generation_id=f"{plan.activation_set_id}-{lane.value}",
            activation_set_id=plan.activation_set_id,
            workspace_id=plan.workspace_id,
            corpus_ref=plan.corpus_ref,
            target_type=TargetType.IMAGE,
            unit_manifest_sha256=unit_manifest_sha256,
            embedding_spaces=() if semantic_identity is None else (semantic_identity.space,),
            hash_spaces=(target_manifest.hash_space,) if semantic_identity is None else (),
            retriever_id=(
                target_manifest.hash_space.provider_id
                if semantic_identity is None
                else semantic_identity.space.provider_id
            ),
            index_format=index_format,
            index_parameters_ref=index_parameters,
            shard_refs=(shard_role,),
            calibration_profile_ref=f"{lane.value}-personal-lite-v1",
            build_processing_fingerprint=target_manifest.processing_fingerprint,
            created_at=datetime.now(UTC),
            validation_report_ref=f"pending-{lane.value}-validation",
        )
        index_directory = payload_root / "indexes"
        index_directory.mkdir(mode=0o700, exist_ok=True)
        index_path = index_directory / f"{lane.value}.npy"
        if lane is RetrievalLane.PERCEPTUAL:
            index_payload = PhashFlatIndexWriter(max_units=self._max_units).build(
                generation,
                (
                    OrdinalPhash(row.ordinal, target_manifest.hash_space, row.perceptual_hash)
                    for row in rows
                ),
                index_path,
                expected_units=len(rows),
                deadline=deadline,
            )
        else:
            semantic_identity = _require_semantic_identity(semantic_identity)
            embedding_index = 0 if lane is RetrievalLane.INSTANCE else 1
            index_payload = NumpyExactIndexWriter(max_units=self._max_units).build(
                generation,
                (
                    OrdinalVector(
                        row.ordinal,
                        semantic_identity.space,
                        row.embeddings[embedding_index].values,
                    )
                    for row in rows
                ),
                index_path,
                expected_units=len(rows),
                lane=lane,
                deadline=deadline,
            )
        index_payload.path.chmod(0o600)
        _fsync_file(index_payload.path)
        shard_payload = _payload_ref(
            staging,
            index_payload.path,
            role=shard_role,
            expected_sha256=index_payload.sha256,
            expected_size=index_payload.size_bytes,
        )
        staged_lane = StagedIndexGeneration(
            generation_id=generation.generation_id,
            activation_set_id=plan.activation_set_id,
            lane=lane,
            target_bundle_manifest_sha256=(
                None if semantic_identity is None else semantic_identity.bundle_manifest_sha256
            ),
            unit_manifest_sha256=unit_manifest_sha256,
            embedding_spaces=generation.embedding_spaces,
            hash_spaces=generation.hash_spaces,
            retriever_id=generation.retriever_id,
            index_format=index_format,
            index_parameters_ref=index_parameters,
            payload_refs=(shard_role,),
        )
        return _LaneBuild(
            staged_lane=staged_lane,
            generation=generation,
            parameters=parameters,
            parameter_payload=parameter_payload,
            shard_payload=shard_payload,
            index_payload=index_payload,
        )


@dataclass(frozen=True, slots=True)
class _LaneBuild:
    staged_lane: StagedIndexGeneration
    generation: IndexGeneration
    parameters: PersonalIndexParameters
    parameter_payload: PayloadFileRef
    shard_payload: PayloadFileRef
    index_payload: PersonalIndexPayload


def _representation_evidence_identity(
    staged: StagedGenerationSetManifest,
    result: RepresentationProductionResult,
) -> StagedRepresentationEvidenceIdentity:
    report = result.report
    return StagedRepresentationEvidenceIdentity(
        activation_set_id=staged.activation_set_id,
        workspace_id=staged.workspace_id,
        corpus_ref=staged.corpus_ref,
        unit_manifest_sha256=staged.unit_manifest_sha256,
        target_representation_manifest_sha256=(staged.target_representation_manifest_sha256),
        unit_count=staged.unit_catalog.unit_count,
        processing_fingerprint=staged.processing_fingerprint,
        hash_space=report.hash_space,
        semantic_targets=report.semantic_targets,
        representation_payload_set_sha256=report.representation_payload_set_sha256,
        checkpoint_sha256=staged.checkpoint_sha256,
    )


@dataclass(frozen=True, slots=True)
class _CatalogPayloadValidator:
    descriptor: SqliteUnitCatalogDescriptor
    staged_digest: str
    validator_id: str = "sqlite-unit-catalog-native-v1"

    def validate(
        self,
        root: Path,
        payload: PayloadFileRef,
        owner: PayloadOwner,
        *,
        deadline: datetime,
    ) -> ValidationReport:
        if not isinstance(owner, StagedUnitCatalog):
            raise TypeError("catalog validator requires a StagedUnitCatalog owner")
        if (
            owner.catalog_id != self.descriptor.catalog_id
            or owner.catalog_format != self.descriptor.catalog_format
            or owner.unit_count != self.descriptor.unit_count
            or owner.unit_manifest_sha256 != self.descriptor.unit_manifest_sha256
            or payload.sha256 != self.descriptor.sha256
            or payload.size_bytes != self.descriptor.size_bytes
        ):
            raise ValueError("catalog payload does not match its staged owner and descriptor")
        path = _verified_payload_path(root, payload)
        descriptor = replace(self.descriptor, path=path)
        catalog = SqliteUnitCatalog.open_readonly(descriptor, deadline=deadline)
        try:
            if (
                catalog.unit_count != owner.unit_count
                or catalog.unit_manifest_sha256 != owner.unit_manifest_sha256
            ):
                raise ValueError("native catalog identity does not match its staged owner")
        finally:
            catalog.close()
        return _validation_report(
            owner_kind="catalog",
            owner_id=owner.catalog_id,
            payload=payload,
            validator_id=self.validator_id,
            staged_digest=self.staged_digest,
            capability="sqlite-integrity-and-schema-checked",
        )


@dataclass(frozen=True, slots=True)
class _ParameterPayloadValidator:
    expected: PersonalIndexParameters
    expected_role: str
    staged_digest: str
    validator_id: str = "personal-index-parameters-native-v1"

    def validate(
        self,
        root: Path,
        payload: PayloadFileRef,
        owner: PayloadOwner,
        *,
        deadline: datetime,
    ) -> ValidationReport:
        _check_deadline(deadline)
        if not isinstance(owner, StagedIndexGeneration):
            raise TypeError("parameter validator requires a StagedIndexGeneration owner")
        if (
            payload.role != self.expected_role
            or owner.lane is not self.expected.lane
            or owner.index_format != self.expected.index_format
            or owner.index_parameters_ref.payload_ref != self.expected_role
            or owner.index_parameters_ref.sha256 != payload.sha256
        ):
            raise ValueError("parameter payload does not match its staged lane")
        data = _read_verified_payload(root, payload, maximum_bytes=65_536)
        if decode_personal_index_parameters(data) != self.expected:
            raise ValueError("canonical index parameters do not match the staged lane")
        return _validation_report(
            owner_kind="lane",
            owner_id=owner.generation_id,
            payload=payload,
            validator_id=self.validator_id,
            staged_digest=self.staged_digest,
            capability="canonical-index-parameters-checked",
        )


@dataclass(frozen=True, slots=True)
class _IndexPayloadValidator:
    generation: IndexGeneration
    payload_descriptor: PersonalIndexPayload
    expected_role: str
    catalog_descriptor: SqliteUnitCatalogDescriptor
    staged_digest: str
    validator_id: str = "personal-index-native-reader-v1"

    def validate(
        self,
        root: Path,
        payload: PayloadFileRef,
        owner: PayloadOwner,
        *,
        deadline: datetime,
    ) -> ValidationReport:
        if not isinstance(owner, StagedIndexGeneration):
            raise TypeError("index validator requires a StagedIndexGeneration owner")
        if (
            owner.generation_id != self.generation.generation_id
            or owner.lane is not self.payload_descriptor.lane
            or owner.index_format != self.generation.index_format
            or owner.index_parameters_ref != self.generation.index_parameters_ref
            or owner.embedding_spaces != self.generation.embedding_spaces
            or owner.hash_spaces != self.generation.hash_spaces
            or owner.payload_refs != (self.expected_role,)
            or payload.role != self.expected_role
            or payload.sha256 != self.payload_descriptor.sha256
            or payload.size_bytes != self.payload_descriptor.size_bytes
        ):
            raise ValueError("index payload does not match its staged lane and descriptor")
        catalog_path = self.catalog_descriptor.path
        if catalog_path.parent.parent != root:
            raise ValueError("catalog descriptor does not belong to the staged generation")
        catalog = SqliteUnitCatalog.open_readonly(
            replace(self.catalog_descriptor, path=catalog_path),
            deadline=deadline,
        )
        reader: NumpyExactIndexReader | PhashFlatIndexReader | None = None
        try:
            descriptor = replace(
                self.payload_descriptor,
                path=_verified_payload_path(root, payload),
            )
            if owner.lane is RetrievalLane.PERCEPTUAL:
                reader = PhashFlatIndexReader.open_readonly(
                    self.generation,
                    descriptor,
                    ordinal_resolver=catalog,
                    deadline=deadline,
                )
            else:
                reader = NumpyExactIndexReader.open_readonly(
                    self.generation,
                    descriptor,
                    ordinal_resolver=catalog,
                    deadline=deadline,
                )
        finally:
            if reader is not None:
                reader.close()
            catalog.close()
        return _validation_report(
            owner_kind="lane",
            owner_id=owner.generation_id,
            payload=payload,
            validator_id=self.validator_id,
            staged_digest=self.staged_digest,
            capability="native-readonly-index-open-checked",
        )


def _validation_report(
    *,
    owner_kind: str,
    owner_id: str,
    payload: PayloadFileRef,
    validator_id: str,
    staged_digest: str,
    capability: str,
) -> ValidationReport:
    return ValidationReport(
        report_id=f"native-{hashlib.sha256((owner_id + payload.role).encode()).hexdigest()[:32]}",
        owner_kind=owner_kind,
        owner_id=owner_id,
        validator_id=validator_id,
        staged_identity_sha256=staged_digest,
        observed_payload_sha256s=(payload.sha256,),
        capability_codes=(capability,),
        passed=True,
        created_at=datetime.now(UTC),
    )


def _payload_ref(
    staging: Path,
    path: Path,
    *,
    role: str,
    expected_sha256: str,
    expected_size: int,
) -> PayloadFileRef:
    if not path.is_absolute() or staging not in path.parents:
        raise ValueError("payload path must remain inside the staging directory")
    observed_sha256, observed_size = _digest_regular_file(path)
    if observed_sha256 != expected_sha256 or observed_size != expected_size:
        raise ValueError("sealed writer payload identity changed")
    return PayloadFileRef(
        relative_path=PurePosixPath(path.relative_to(staging).as_posix()),
        sha256=observed_sha256,
        size_bytes=observed_size,
        role=role,
    )


def _verified_payload_path(root: Path, payload: PayloadFileRef) -> Path:
    path = root.joinpath(*payload.relative_path.parts)
    observed_sha256, observed_size = _digest_regular_file(path)
    if observed_sha256 != payload.sha256 or observed_size != payload.size_bytes:
        raise ValueError("native payload checksum or size changed")
    return path


def _read_verified_payload(
    root: Path,
    payload: PayloadFileRef,
    *,
    maximum_bytes: int,
) -> bytes:
    if payload.size_bytes > maximum_bytes:
        raise ValueError("payload exceeds its native validation size limit")
    path = root.joinpath(*payload.relative_path.parts)
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != payload.size_bytes:
            raise ValueError("payload is not a regular file with the declared size")
        data = bytearray()
        while len(data) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("payload changed during native validation")
    if len(data) != payload.size_bytes or hashlib.sha256(data).hexdigest() != payload.sha256:
        raise ValueError("payload failed native size or checksum validation")
    return bytes(data)


def _digest_regular_file(path: Path) -> tuple[str, int]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("payload must be a regular non-symlink file")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("payload changed while its identity was sealed")
    if size != before.st_size:
        raise ValueError("payload size changed while its identity was sealed")
    return digest.hexdigest(), size


def _write_private_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        written = 0
        view = memoryview(payload)
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("private payload write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _owned_directory_identity(path: Path) -> tuple[int, int]:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        visible = os.stat(path, follow_symlinks=False)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(visible.st_mode)
        or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
    ):
        raise RuntimeError("owned staging directory identity changed")
    return opened.st_dev, opened.st_ino


def _remove_owned_staging(
    staging: Path,
    staging_root: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Remove one verified staging tree owned by this administrator.

    The cleanup threat model is deliberately narrow: the store and staging roots
    are private 0700 directories, and only cooperative Capicove writers in this
    process mutate them.  The shared lock serializes those writers.  POSIX does
    not provide unlink-if-inode; a malicious same-UID process racing a random
    quarantine name is outside this contract.
    """
    if staging.parent != staging_root or not staging.name:
        raise RuntimeError("refusing to clean a staging directory outside the owned root")
    with _PERSONAL_CLEANUP_LOCK:
        root_descriptor = os.open(
            staging_root,
            os.O_RDONLY
            | os.O_NONBLOCK
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            root_status = os.fstat(root_descriptor)
            if not stat.S_ISDIR(root_status.st_mode) or stat.S_IMODE(root_status.st_mode) != 0o700:
                raise RuntimeError("refusing to clean staging outside a private store root")
            try:
                descriptor = os.open(
                    staging.name,
                    os.O_RDONLY
                    | os.O_NONBLOCK
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_descriptor,
                )
            except FileNotFoundError:
                return
            except OSError as error:
                raise RuntimeError("refusing to clean a replaced staging directory") from error
            try:
                opened = os.fstat(descriptor)
                visible = os.stat(
                    staging.name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or not stat.S_ISDIR(visible.st_mode)
                    or stat.S_IMODE(opened.st_mode) != 0o700
                    or (opened.st_dev, opened.st_ino) != expected_identity
                    or (visible.st_dev, visible.st_ino) != expected_identity
                ):
                    raise RuntimeError("refusing to clean a replaced staging directory")
                _remove_directory_contents(descriptor)
            finally:
                os.close(descriptor)

            quarantine = _quarantine_owned_entry(
                root_descriptor,
                staging.name,
                expected_identity=expected_identity,
                expected_directory=True,
                error_message=(
                    "owned staging was replaced during cleanup; the replacement was preserved"
                ),
            )
            try:
                os.rmdir(quarantine, dir_fd=root_descriptor)
            except OSError:
                _restore_quarantined_entry(root_descriptor, quarantine, staging.name)
                raise
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)


def _remove_directory_contents(directory_descriptor: int) -> None:
    """Remove a private directory's contents after quarantining each identity."""
    for name in os.listdir(directory_descriptor):
        observed = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        identity = (observed.st_dev, observed.st_ino)
        if stat.S_ISDIR(observed.st_mode):
            quarantine = _quarantine_owned_entry(
                directory_descriptor,
                name,
                expected_identity=identity,
                expected_directory=True,
                error_message="staging directory identity changed during cleanup",
            )
            child_descriptor = -1
            try:
                child_descriptor = os.open(
                    quarantine,
                    os.O_RDONLY
                    | os.O_NONBLOCK
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                opened = os.fstat(child_descriptor)
                visible = os.stat(
                    quarantine,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != identity
                    or (visible.st_dev, visible.st_ino) != identity
                ):
                    raise RuntimeError("staging directory identity changed during cleanup")
                _remove_directory_contents(child_descriptor)
                os.close(child_descriptor)
                child_descriptor = -1
                os.rmdir(quarantine, dir_fd=directory_descriptor)
            except BaseException:
                if child_descriptor >= 0:
                    os.close(child_descriptor)
                _restore_quarantined_entry(directory_descriptor, quarantine, name)
                raise
        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
            quarantine = _quarantine_owned_entry(
                directory_descriptor,
                name,
                expected_identity=identity,
                expected_directory=False,
                error_message="staging file identity changed during cleanup",
            )
            file_descriptor = -1
            try:
                file_descriptor = os.open(
                    quarantine,
                    os.O_RDONLY
                    | os.O_NONBLOCK
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_descriptor,
                )
                opened = os.fstat(file_descriptor)
                visible = os.stat(
                    quarantine,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != identity
                    or (visible.st_dev, visible.st_ino) != identity
                ):
                    raise RuntimeError("staging file identity changed during cleanup")
                os.close(file_descriptor)
                file_descriptor = -1
                os.unlink(quarantine, dir_fd=directory_descriptor)
            except BaseException:
                if file_descriptor >= 0:
                    os.close(file_descriptor)
                _restore_quarantined_entry(directory_descriptor, quarantine, name)
                raise
        else:
            raise RuntimeError("refusing to remove an unsupported staging entry")
    os.fsync(directory_descriptor)


def _quarantine_owned_entry(
    directory_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
    expected_directory: bool,
    error_message: str,
) -> str:
    """Move one entry to an unpredictable sibling name and verify its identity."""
    quarantine = f".capicove-cleanup-{uuid.uuid4().hex}"
    try:
        os.stat(quarantine, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:  # pragma: no cover - cryptographically improbable UUID collision
        raise RuntimeError("cleanup quarantine unexpectedly exists")
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
            _restore_quarantined_entry(directory_descriptor, quarantine, name)
        raise RuntimeError(error_message) from error
    expected_kind = stat.S_ISDIR if expected_directory else stat.S_ISREG
    if (
        not expected_kind(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != expected_identity
        or (not expected_directory and observed.st_nlink != 1)
    ):
        _restore_quarantined_entry(directory_descriptor, quarantine, name)
        raise RuntimeError(error_message)
    return quarantine


def _restore_quarantined_entry(
    directory_descriptor: int,
    quarantine: str,
    original_name: str,
) -> None:
    """Best-effort restore a quarantined entry when its original name is free."""
    try:
        os.stat(
            original_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
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


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be a timezone-aware datetime")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise TimeoutError("personal image generation deadline exceeded")


def _require_semantic_identity(
    identity: SemanticTargetIdentity | None,
) -> SemanticTargetIdentity:
    if identity is None:
        raise RuntimeError("semantic index build requires a semantic identity")
    return identity


def _require_opaque(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")
    if value.startswith(("/", "\\")) or ".." in value or "\x00" in value:
        raise ValueError(f"{name} must be a safe opaque reference")


__all__ = [
    "BuiltPersonalImageGeneration",
    "PersonalImageBuildPlan",
    "PersonalImageGenerationAdministrator",
]
