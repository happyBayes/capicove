"""Preview composition for offline personal reverse-image retrieval."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Protocol, runtime_checkable

from capicove.contracts.retrieval import ExecutionProfile, ProcessingFingerprint
from capicove.model.bundle import VerifiedImageEncoderBundle
from capicove.model.provisioning import load_reverse_image_model_provisioning
from capicove.processing.image._adapters.onnx_target import OnnxImageTargetEncoder
from capicove.processing.image.deduplication.perceptual import OpenCvPhashProvider
from capicove.processing.image.io import load_image
from capicove.processing.image.ports import ImageSemanticLane
from capicove.processing.image.representations import ImageTargetRepresentationProducer
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval._adapters.onnx_image import OnnxImageQueryEncoder
from capicove.retrieval._adapters.opencv_phash import (
    OPENCV_PHASH_HASH_SPACE,
    OpenCvPerceptualQueryEncoder,
)
from capicove.retrieval._adapters.sqlite_image_generations import (
    GenerationAliasSnapshot,
    PreviewActivationQualification,
    PreviewActivationVerifier,
    PublishedGenerationRecord,
    SqliteGenerationCatalog,
)
from capicove.retrieval.cursor import HmacSearchCursorCodec
from capicove.retrieval.fusion import WeightedReciprocalRankFusion
from capicove.retrieval.image_generations import PublishedGenerationSet
from capicove.retrieval.search import RetrievalService
from capicove.retrieval.search_models import DeferredCleanupSupervisor, RetrievalLane
from capicove.runtime.image_query import AuthorizedImageQueryResolver, ImageArtifactResolver
from capicove.runtime.retrieval import (
    BoundedLaneScheduler,
    CpuExecutionBudget,
    GenerationSetManager,
    TrustedQueryEncoderRegistration,
    TrustedQueryEncoderRegistry,
    default_retrieval_profiles,
)
from capicove.runtime.retrieval_build import PersonalImageGenerationAdministrator
from capicove.runtime.retrieval_personal import PersonalGenerationHandleFactory
from capicove.tools.image_search import SearchToolHandler, StaticSearchAuthorizationFactory


_REQUIRED_PERMISSION = "retrieval:search"


class PublishedGenerationRegistration:
    """Opaque process-local token for one runtime-owned publication record."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class ReverseImageModelProvisioning:
    """Pin one local two-bundle provisioning manifest in startup policy."""

    root: Path
    manifest_relative_path: PurePosixPath
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("model provisioning root must be an absolute Path")
        if not isinstance(self.manifest_relative_path, PurePosixPath):
            raise TypeError("model provisioning manifest path must be a PurePosixPath")
        value = self.manifest_relative_path.as_posix()
        if (
            self.manifest_relative_path.is_absolute()
            or self.manifest_relative_path == PurePosixPath(".")
            or ".." in self.manifest_relative_path.parts
            or "\\" in value
            or ":" in value
        ):
            raise ValueError("model provisioning manifest path must be safely relative")
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValueError("model provisioning manifest_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class _ResolvedReverseImageModels:
    """Retain trusted startup provenance without expanding the preview API."""

    instance_bundle: VerifiedImageEncoderBundle
    concept_bundle: VerifiedImageEncoderBundle
    bundle_provisioning_manifest_sha256: str | None


@dataclass(frozen=True, slots=True)
class ActiveGenerationSnapshot:
    """Public immutable CAS snapshot of one active retrieval alias."""

    workspace_id: str
    profile_ref: str
    activation_set_id: str
    manifest_sha256: str
    qualification_id: str
    qualification_tier: str
    version: int


@runtime_checkable
class _PreviewQualificationIssuer(Protocol):
    def issue(
        self,
        published: PublishedGenerationRecord,
        *,
        report_sha256: str,
        policy_sha256: str,
        verification_receipt_sha256: str,
        verified_at: datetime,
        deadline: datetime,
    ) -> PreviewActivationQualification: ...


@dataclass(slots=True)
class ReverseImageRuntime:
    """Expose one composed personal runtime and its explicit build lifecycle."""

    store: FilesystemGenerationStore
    administrator: PersonalImageGenerationAdministrator
    service: RetrievalService
    tool_handler: SearchToolHandler
    processing_fingerprint: ProcessingFingerprint
    _workspace_id: str
    _lifecycle: SqliteGenerationCatalog
    _preview_activation_verifier: PreviewActivationVerifier
    _bundle_provisioning_manifest_sha256: str | None = field(default=None, repr=False)
    _registrations: dict[PublishedGenerationRegistration, PublishedGenerationRecord] = field(
        default_factory=dict, init=False, repr=False
    )
    _registration_lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def register_published(
        self,
        published: PublishedGenerationSet,
        *,
        published_at: datetime,
        deadline: datetime,
    ) -> PublishedGenerationRegistration:
        """Register an immutable publication without qualifying or activating it."""
        if not isinstance(published, PublishedGenerationSet):
            raise TypeError("published must be a PublishedGenerationSet")
        record = PublishedGenerationRecord(
            activation_set_id=published.manifest.activation_set_id,
            manifest_sha256=published.manifest_sha256,
            published_root=published.root,
            workspace_id=published.manifest.workspace_id,
            corpus_ref=published.manifest.corpus_ref,
            retrieval_profile_ref=published.manifest.retrieval_profile_ref,
            published_at=published_at,
        )
        if record.workspace_id != self._workspace_id:
            raise ValueError("published generation belongs to another workspace")
        registration = PublishedGenerationRegistration()
        with self._registration_lock:
            self._require_open()
            self._lifecycle.register_published(record, deadline=deadline)
            self._registrations[registration] = record
        return registration

    def issue_preview_qualification(
        self,
        registration: PublishedGenerationRegistration,
        *,
        report_sha256: str,
        policy_sha256: str,
        verification_receipt_sha256: str,
        verified_at: datetime,
        deadline: datetime,
    ) -> str:
        """Issue preview evidence explicitly; this never activates the generation."""
        with self._registration_lock:
            self._require_open()
            published = self._resolve_registration(registration)
            issuer = self._preview_activation_verifier
            if not isinstance(issuer, _PreviewQualificationIssuer):
                raise RuntimeError(
                    "configured preview activation verifier cannot issue qualifications"
                )
            qualification = issuer.issue(
                published,
                report_sha256=report_sha256,
                policy_sha256=policy_sha256,
                verification_receipt_sha256=verification_receipt_sha256,
                verified_at=verified_at,
                deadline=deadline,
            )
        if not isinstance(qualification, PreviewActivationQualification):
            raise TypeError("preview qualification issuer returned an invalid locator")
        return qualification.qualification_id

    def resolve_active_snapshot(
        self,
        profile_ref: str,
        *,
        deadline: datetime,
    ) -> ActiveGenerationSnapshot | None:
        """Resolve the current public alias snapshot for explicit activation CAS."""
        with self._registration_lock:
            self._require_open()
            snapshot = self._lifecycle.resolve_active(
                self._workspace_id,
                profile_ref,
                deadline=deadline,
            )
        return None if snapshot is None else _public_snapshot(snapshot)

    def activate_preview(
        self,
        registration: PublishedGenerationRegistration,
        *,
        qualification_id: str,
        expected_previous_snapshot: ActiveGenerationSnapshot | None,
        activated_at: datetime,
        deadline: datetime,
    ) -> ActiveGenerationSnapshot:
        """CAS-activate one separately registered and qualified preview generation."""
        if expected_previous_snapshot is not None and not isinstance(
            expected_previous_snapshot,
            ActiveGenerationSnapshot,
        ):
            raise TypeError("expected_previous_snapshot must be an ActiveGenerationSnapshot")
        expected = (
            None
            if expected_previous_snapshot is None
            else _private_snapshot(expected_previous_snapshot)
        )
        qualification = PreviewActivationQualification(qualification_id)
        with self._registration_lock:
            self._require_open()
            published = self._resolve_registration(registration)
            activated = self._lifecycle.activate(
                workspace_id=self._workspace_id,
                profile_ref=published.retrieval_profile_ref,
                activation_set_id=published.activation_set_id,
                expected_previous_snapshot=expected,
                qualification=qualification,
                activated_at=activated_at,
                deadline=deadline,
            )
        return _public_snapshot(activated)

    def close(self) -> None:
        """Close the persistent lifecycle catalog idempotently."""
        with self._registration_lock:
            if self._closed:
                return
            self._lifecycle.close()
            self._registrations.clear()
            self._closed = True

    def __enter__(self) -> ReverseImageRuntime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _resolve_registration(
        self,
        registration: PublishedGenerationRegistration,
    ) -> PublishedGenerationRecord:
        if not isinstance(registration, PublishedGenerationRegistration):
            raise TypeError("registration must be a PublishedGenerationRegistration")
        try:
            return self._registrations[registration]
        except KeyError as error:
            raise ValueError("registration was not issued by this runtime") from error

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("reverse-image runtime is closed")


def build_reverse_image_runtime(
    *,
    workspace_id: str,
    generation_root: Path,
    lifecycle_database_path: Path,
    instance_bundle: VerifiedImageEncoderBundle | None = None,
    concept_bundle: VerifiedImageEncoderBundle | None = None,
    model_provisioning: ReverseImageModelProvisioning | None = None,
    artifact_resolver: ImageArtifactResolver,
    local_roots_by_policy: Mapping[str, Mapping[str, Path]],
    preview_activation_verifier: PreviewActivationVerifier,
    cursor_signing_key: bytes,
    cpu_budget: CpuExecutionBudget,
    authorization_policy: StaticSearchAuthorizationFactory,
    owner_id: str,
    owner_session_epoch: str,
    clock: Callable[[], datetime] | None = None,
) -> ReverseImageRuntime:
    """Compose the offline preview runtime without publishing or activating data."""
    effective_clock = clock or _utc_now
    resolved_models = _resolve_model_bundles(
        instance_bundle=instance_bundle,
        concept_bundle=concept_bundle,
        model_provisioning=model_provisioning,
    )
    instance_bundle = resolved_models.instance_bundle
    concept_bundle = resolved_models.concept_bundle
    _validate_startup_policy(
        workspace_id=workspace_id,
        generation_root=generation_root,
        lifecycle_database_path=lifecycle_database_path,
        instance_bundle=instance_bundle,
        concept_bundle=concept_bundle,
        artifact_resolver=artifact_resolver,
        local_roots_by_policy=local_roots_by_policy,
        preview_activation_verifier=preview_activation_verifier,
        cpu_budget=cpu_budget,
        authorization_policy=authorization_policy,
        owner_id=owner_id,
        owner_session_epoch=owner_session_epoch,
        clock=effective_clock,
    )
    instance_bundle.verify()
    concept_bundle.verify()
    if (
        instance_bundle.manifest.embedding_space.space_id
        == concept_bundle.manifest.embedding_space.space_id
    ):
        raise ValueError("instance and concept bundles must use distinct embedding spaces")

    query_resolver = AuthorizedImageQueryResolver(
        workspace_id=workspace_id,
        artifact_resolver=artifact_resolver,
        local_roots_by_policy=local_roots_by_policy,
    )
    processing_fingerprint = _processing_fingerprint(instance_bundle, concept_bundle)
    target_encoders = {
        ImageSemanticLane.INSTANCE: OnnxImageTargetEncoder(
            instance_bundle,
            ImageSemanticLane.INSTANCE,
            intra_op_threads=cpu_budget.per_lane_compute_threads,
            inter_op_threads=cpu_budget.onnx_inter_op_threads,
        ),
        ImageSemanticLane.CONCEPT: OnnxImageTargetEncoder(
            concept_bundle,
            ImageSemanticLane.CONCEPT,
            intra_op_threads=cpu_budget.per_lane_compute_threads,
            inter_op_threads=cpu_budget.onnx_inter_op_threads,
        ),
    }
    producer = ImageTargetRepresentationProducer(
        decoder=load_image,
        perceptual_hasher=OpenCvPhashProvider(),
        encoders=target_encoders,
        hash_space=OPENCV_PHASH_HASH_SPACE,
        processing_fingerprint=processing_fingerprint,
    )
    query_encoders = TrustedQueryEncoderRegistry(
        (
            _query_registration(RetrievalLane.INSTANCE, instance_bundle),
            _query_registration(RetrievalLane.CONCEPT, concept_bundle),
        ),
        cpu_budget=cpu_budget,
    )
    cursor_codec = HmacSearchCursorCodec(
        signing_key=cursor_signing_key,
        workspace_id=workspace_id,
    )
    scheduler = BoundedLaneScheduler(
        cpu_budget.schedule_snapshot(schedule_id="personal-reverse-image-v1")
    )

    store = FilesystemGenerationStore(generation_root)
    lifecycle = SqliteGenerationCatalog.open(
        lifecycle_database_path,
        verifier=preview_activation_verifier,
    )
    try:
        administrator = PersonalImageGenerationAdministrator(store=store, producer=producer)
        cleanup_supervisor = DeferredCleanupSupervisor()
        generation_sets = GenerationSetManager(
            workspace_id=workspace_id,
            lifecycle=lifecycle,
            handle_factory=PersonalGenerationHandleFactory(store),
            owner_id=owner_id,
            owner_session_epoch=owner_session_epoch,
            clock=effective_clock,
            cleanup_supervisor=cleanup_supervisor,
        )
        service = RetrievalService(
            workspace_id=workspace_id,
            generation_sets=generation_sets,
            query_resolver=query_resolver,
            perceptual_encoder=OpenCvPerceptualQueryEncoder(),
            query_encoders=query_encoders,
            fusion=WeightedReciprocalRankFusion(),
            cursor_codec=cursor_codec,
            lane_scheduler=scheduler,
            profiles=default_retrieval_profiles(),
            clock=effective_clock,
            cleanup_supervisor=cleanup_supervisor,
        )
        handler = SearchToolHandler(
            service,
            authorization_policy,
            clock=effective_clock,
        )
    except BaseException:
        lifecycle.close()
        raise
    return ReverseImageRuntime(
        store=store,
        administrator=administrator,
        service=service,
        tool_handler=handler,
        processing_fingerprint=processing_fingerprint,
        _workspace_id=workspace_id,
        _lifecycle=lifecycle,
        _preview_activation_verifier=preview_activation_verifier,
        _bundle_provisioning_manifest_sha256=(resolved_models.bundle_provisioning_manifest_sha256),
    )


def _resolve_model_bundles(
    *,
    instance_bundle: VerifiedImageEncoderBundle | None,
    concept_bundle: VerifiedImageEncoderBundle | None,
    model_provisioning: ReverseImageModelProvisioning | None,
) -> _ResolvedReverseImageModels:
    if model_provisioning is not None:
        if not isinstance(model_provisioning, ReverseImageModelProvisioning):
            raise TypeError("model_provisioning must be a ReverseImageModelProvisioning")
        if instance_bundle is not None or concept_bundle is not None:
            raise ValueError("model_provisioning cannot be combined with direct model bundles")
        provisioned = load_reverse_image_model_provisioning(
            model_provisioning.root,
            model_provisioning.manifest_relative_path,
            expected_manifest_sha256=model_provisioning.manifest_sha256,
        )
        return _ResolvedReverseImageModels(
            instance_bundle=provisioned.instance_bundle,
            concept_bundle=provisioned.concept_bundle,
            bundle_provisioning_manifest_sha256=provisioned.manifest.source_sha256,
        )
    if not isinstance(instance_bundle, VerifiedImageEncoderBundle) or not isinstance(
        concept_bundle,
        VerifiedImageEncoderBundle,
    ):
        raise TypeError("provide model_provisioning or both verified instance and concept bundles")
    return _ResolvedReverseImageModels(
        instance_bundle=instance_bundle,
        concept_bundle=concept_bundle,
        bundle_provisioning_manifest_sha256=None,
    )


def _public_snapshot(snapshot: GenerationAliasSnapshot) -> ActiveGenerationSnapshot:
    return ActiveGenerationSnapshot(
        workspace_id=snapshot.workspace_id,
        profile_ref=snapshot.profile_ref,
        activation_set_id=snapshot.activation_set_id,
        manifest_sha256=snapshot.manifest_sha256,
        qualification_id=snapshot.qualification_id,
        qualification_tier=snapshot.qualification_tier,
        version=snapshot.version,
    )


def _private_snapshot(snapshot: ActiveGenerationSnapshot) -> GenerationAliasSnapshot:
    return GenerationAliasSnapshot(
        workspace_id=snapshot.workspace_id,
        profile_ref=snapshot.profile_ref,
        activation_set_id=snapshot.activation_set_id,
        manifest_sha256=snapshot.manifest_sha256,
        qualification_id=snapshot.qualification_id,
        qualification_tier=snapshot.qualification_tier,
        version=snapshot.version,
    )


def _query_registration(
    lane: RetrievalLane,
    bundle: VerifiedImageEncoderBundle,
) -> TrustedQueryEncoderRegistration:
    return TrustedQueryEncoderRegistration(
        lane=lane,
        space=bundle.manifest.embedding_space,
        target_bundle_manifest_sha256=bundle.manifest_sha256,
        execution_profile=ExecutionProfile.CPU,
        factory=partial(OnnxImageQueryEncoder, bundle),
    )


def _processing_fingerprint(
    instance_bundle: VerifiedImageEncoderBundle,
    concept_bundle: VerifiedImageEncoderBundle,
) -> ProcessingFingerprint:
    return ProcessingFingerprint.from_manifest(
        {
            "pipeline": "personal-reverse-image-v1",
            "decoder": "capicove.processing.image.io.load_image-v1",
            "perceptual_hash": {
                "provider_id": OPENCV_PHASH_HASH_SPACE.provider_id,
                "provider_version": OPENCV_PHASH_HASH_SPACE.provider_version,
                "hash_version": OPENCV_PHASH_HASH_SPACE.hash_version,
                "preprocessing_version": OPENCV_PHASH_HASH_SPACE.preprocessing_version,
            },
            "semantic_bundles": {
                "instance": instance_bundle.manifest_sha256,
                "concept": concept_bundle.manifest_sha256,
            },
        }
    )


def _validate_startup_policy(
    *,
    workspace_id: str,
    generation_root: Path,
    lifecycle_database_path: Path,
    instance_bundle: object,
    concept_bundle: object,
    artifact_resolver: object,
    local_roots_by_policy: Mapping[str, Mapping[str, Path]],
    preview_activation_verifier: object,
    cpu_budget: object,
    authorization_policy: object,
    owner_id: str,
    owner_session_epoch: str,
    clock: object,
) -> None:
    for name, value in (
        ("workspace_id", workspace_id),
        ("owner_id", owner_id),
        ("owner_session_epoch", owner_session_epoch),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-blank string")
    if not isinstance(generation_root, Path) or not generation_root.is_absolute():
        raise ValueError("generation_root must be an absolute Path")
    if not isinstance(lifecycle_database_path, Path) or not lifecycle_database_path.is_absolute():
        raise ValueError("lifecycle_database_path must be an absolute Path")
    if not isinstance(instance_bundle, VerifiedImageEncoderBundle) or not isinstance(
        concept_bundle,
        VerifiedImageEncoderBundle,
    ):
        raise TypeError("instance_bundle and concept_bundle must be verified bundles")
    if not callable(getattr(artifact_resolver, "resolve", None)):
        raise TypeError("artifact_resolver must provide resolve")
    if not callable(getattr(preview_activation_verifier, "verify", None)):
        raise TypeError("preview_activation_verifier must provide verify")
    if not isinstance(cpu_budget, CpuExecutionBudget):
        raise TypeError("cpu_budget must be a CpuExecutionBudget")
    if not isinstance(authorization_policy, StaticSearchAuthorizationFactory):
        raise TypeError("authorization_policy must be a StaticSearchAuthorizationFactory")
    if _REQUIRED_PERMISSION not in authorization_policy.permissions:
        raise ValueError("authorization policy must grant retrieval:search")
    configured_policies = set(local_roots_by_policy)
    if not authorization_policy.allowed_data_policy_refs.issubset(configured_policies):
        raise ValueError("authorization policy references an unconfigured local data policy")
    configured_source_roots = {
        root_ref
        for policy_ref in authorization_policy.allowed_data_policy_refs
        for root_ref in local_roots_by_policy[policy_ref]
    }
    if not authorization_policy.allowed_source_root_refs.issubset(configured_source_roots):
        raise ValueError("authorization policy references an unconfigured source root")
    if not callable(clock):
        raise TypeError("clock must be callable")


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "ActiveGenerationSnapshot",
    "PublishedGenerationRegistration",
    "ReverseImageModelProvisioning",
    "ReverseImageRuntime",
    "build_reverse_image_runtime",
]
