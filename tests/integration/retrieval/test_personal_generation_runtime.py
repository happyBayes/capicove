"""Synthetic end-to-end personal generation publication and search vertical."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    ImageQuery,
    IndexGeneration,
    IndexParametersRef,
    InlineBytesSource,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFailureCode,
    SearchIntent,
    SearchRequest,
    SearchStatus,
    TargetType,
    VectorNormalization,
)
from capicove.contracts.tools import PrivacyClass
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval._adapters.opencv_phash import OPENCV_PHASH_HASH_SPACE
from capicove.retrieval._adapters.personal_numpy import (
    NumpyExactIndexWriter,
    PhashFlatIndexWriter,
)
from capicove.retrieval._adapters.sqlite_image_generations import (
    PreviewActivationQualification,
    PublishedGenerationRecord,
    SqliteGenerationCatalog,
    VerifiedPreviewQualification,
)
from capicove.retrieval._adapters.sqlite_unit_catalog import (
    SQLITE_UNIT_CATALOG_FORMAT,
    IndexedUnitRecord,
    SqliteUnitCatalogWriter,
)
from capicove.retrieval.cursor import HmacSearchCursorCodec
from capicove.retrieval.fusion import WeightedReciprocalRankFusion
from capicove.retrieval.image_generations import (
    TARGET_REPRESENTATION_MANIFEST_ROLE,
    PayloadFileRef,
    PayloadOwner,
    PayloadValidationKey,
    PayloadValidator,
    PublishedGenerationSet,
    StagedGenerationSetManifest,
    StagedIndexGeneration,
    StagedUnitCatalog,
    ValidationReport,
)
from capicove.retrieval.search import CandidateEvaluationRetrievalService, RetrievalService
from capicove.retrieval.search_models import (
    DeferredCleanupSupervisor,
    InvocationAuthorizationContext,
    PerceptualHashQuery,
    QueryVector,
    ResolvedImageQuery,
    RetrievalLane,
    SearchFailure,
)
from capicove.runtime.retrieval import (
    BoundedLaneScheduler,
    CandidateBoundGenerationSetProvider,
    CpuExecutionBudget,
    GenerationSetManager,
    OpenedGenerationHandles,
    default_retrieval_profiles,
)
from capicove.runtime.retrieval_personal import (
    PersonalGenerationHandleFactory,
    PersonalIndexParameters,
    encode_personal_index_parameters,
)


NOW = datetime.now(UTC)
FUTURE = NOW + timedelta(hours=1)
SHA_A = "a" * 64
SHA_B = "b" * 64


def _space(lane: RetrievalLane) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-space-v1",
        provider_id=f"{lane.value}-provider",
        model_revision="revision-1",
        preprocessing_version="rgb-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _unit(index: int, source_sha256: str) -> IndexedUnitRecord:
    unit = SearchableUnit(
        unit_id=f"unit-{index}",
        unit_type=TargetType.IMAGE,
        source_hash=source_sha256,
        artifact_ref=f"artifact-{index}",
        evidence_ref=f"evidence-{index}",
        representation_refs=(f"representation-{index}",),
        processing_run_ref="run-1",
        processing_version="image-v1",
        processing_fingerprint=ProcessingFingerprint(SHA_A),
    )
    return IndexedUnitRecord(index, unit, source_sha256, "image/png")


def _payload(path: Path, role: str, staging: Path) -> PayloadFileRef:
    content = path.read_bytes()
    return PayloadFileRef(
        relative_path=PurePosixPath(path.relative_to(staging).as_posix()),
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        role=role,
    )


@dataclass(frozen=True)
class _Validator:
    validator_id: str = "synthetic-structural-v1"

    def validate(
        self,
        root: Path,
        payload: PayloadFileRef,
        owner: PayloadOwner,
        *,
        deadline: datetime,
    ) -> ValidationReport:
        del root, deadline
        owner_kind = "catalog" if isinstance(owner, StagedUnitCatalog) else "lane"
        owner_id = owner.catalog_id if isinstance(owner, StagedUnitCatalog) else owner.generation_id
        return ValidationReport(
            report_id=f"report-{payload.role}",
            owner_kind=owner_kind,
            owner_id=owner_id,
            validator_id=self.validator_id,
            staged_identity_sha256=SHA_A,
            observed_payload_sha256s=(payload.sha256,),
            capability_codes=("synthetic-structure-checked",),
            passed=True,
            created_at=NOW,
        )


@dataclass(frozen=True)
class _Verifier:
    verified: VerifiedPreviewQualification

    def verify(
        self,
        qualification: PreviewActivationQualification,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> VerifiedPreviewQualification:
        del published, deadline
        assert qualification.qualification_id == self.verified.qualification_id
        return self.verified


def _publish_generation(
    tmp_path: Path,
) -> tuple[FilesystemGenerationStore, PublishedGenerationSet]:
    store = FilesystemGenerationStore(tmp_path / "published")
    staging = store.create_staging("synthetic-build")
    payload_root = staging / "payloads"
    payload_root.mkdir(mode=0o700)

    source_hashes = (SHA_B, "c" * 64, "d" * 64)
    rows = tuple(_unit(index, digest) for index, digest in enumerate(source_hashes))
    catalog_descriptor = SqliteUnitCatalogWriter().build(
        rows,
        payload_root / "catalog.sqlite3",
        activation_set_id="set-1",
        corpus_ref="photos",
        deadline=FUTURE,
    )
    unit_manifest_sha256 = catalog_descriptor.unit_manifest_sha256
    catalog_payload = _payload(catalog_descriptor.path, "catalog", staging)
    fingerprint = ProcessingFingerprint(SHA_A)

    lane_specs = {
        RetrievalLane.PERCEPTUAL: ((3,), "<u8", "phash-flat-u64-v1"),
        RetrievalLane.INSTANCE: ((3, 2), "<f4", "numpy-flat-f32-v1"),
        RetrievalLane.CONCEPT: ((3, 2), "<f4", "numpy-flat-f32-v1"),
    }
    parameter_payloads: list[PayloadFileRef] = []
    shard_payloads: list[PayloadFileRef] = []
    generations: dict[RetrievalLane, IndexGeneration] = {}
    staged_lanes: list[StagedIndexGeneration] = []
    for lane, (shape, dtype, index_format) in lane_specs.items():
        parameter_role = f"{lane.value}-parameters"
        shard_role = f"{lane.value}-shard"
        parameter_bytes = encode_personal_index_parameters(
            PersonalIndexParameters(
                lane=lane,
                index_format=index_format,
                unit_count=3,
                shape=shape,
                dtype=dtype,
                c_order=True,
                shard_role=shard_role,
            )
        )
        parameter_path = payload_root / f"{parameter_role}.json"
        parameter_path.write_bytes(parameter_bytes)
        parameter_path.chmod(0o600)
        parameter_payload = _payload(parameter_path, parameter_role, staging)
        parameter_payloads.append(parameter_payload)
        generation = IndexGeneration(
            generation_id=f"generation-{lane.value}",
            activation_set_id="set-1",
            workspace_id="workspace-1",
            corpus_ref="photos",
            target_type=TargetType.IMAGE,
            unit_manifest_sha256=unit_manifest_sha256,
            embedding_spaces=(() if lane is RetrievalLane.PERCEPTUAL else (_space(lane),)),
            hash_spaces=((OPENCV_PHASH_HASH_SPACE,) if lane is RetrievalLane.PERCEPTUAL else ()),
            retriever_id=f"provider-{lane.value}",
            index_format=index_format,
            index_parameters_ref=IndexParametersRef(
                parameter_schema=index_format,
                payload_ref=parameter_role,
                sha256=parameter_payload.sha256,
            ),
            shard_refs=(shard_role,),
            calibration_profile_ref=f"{lane.value}-preview-v1",
            build_processing_fingerprint=fingerprint,
            created_at=NOW,
            validation_report_ref=f"validation:{lane.value}",
        )
        generations[lane] = generation
        destination = payload_root / f"{shard_role}.npy"
        if lane is RetrievalLane.PERCEPTUAL:
            descriptor = PhashFlatIndexWriter().build_values(
                (0, 1, 3),
                generation,
                destination,
                deadline=FUTURE,
            )
        else:
            values = (
                np.asarray(((1.0, 0.0), (0.8, 0.6), (0.0, 1.0)), dtype="<f4")
                if lane is RetrievalLane.INSTANCE
                else np.asarray(((0.0, 1.0), (1.0, 0.0), (0.6, 0.8)), dtype="<f4")
            )
            descriptor = NumpyExactIndexWriter().build_array(
                values,
                generation,
                destination,
                lane=lane,
                deadline=FUTURE,
            )
        descriptor.path.chmod(0o600)
        shard_payloads.append(_payload(descriptor.path, shard_role, staging))
        staged_lanes.append(
            StagedIndexGeneration(
                generation_id=generation.generation_id,
                activation_set_id="set-1",
                lane=lane,
                target_bundle_manifest_sha256=(None if lane is RetrievalLane.PERCEPTUAL else SHA_B),
                unit_manifest_sha256=unit_manifest_sha256,
                embedding_spaces=generation.embedding_spaces,
                hash_spaces=generation.hash_spaces,
                retriever_id=generation.retriever_id,
                index_format=generation.index_format,
                index_parameters_ref=generation.index_parameters_ref,
                payload_refs=generation.shard_refs,
            )
        )

    target_representation_path = payload_root / "target-representation-manifest.json"
    target_representation_path.write_bytes(b'{"schema_version":1}\n')
    target_representation_path.chmod(0o600)
    target_representation_payload = _payload(
        target_representation_path,
        TARGET_REPRESENTATION_MANIFEST_ROLE,
        staging,
    )

    staged = StagedGenerationSetManifest(
        build_id="synthetic-build",
        activation_set_id="set-1",
        workspace_id="workspace-1",
        corpus_ref="photos",
        retrieval_profile_ref="personal-lite",
        target_type=TargetType.IMAGE,
        unit_manifest_sha256=unit_manifest_sha256,
        target_representation_manifest_sha256=target_representation_payload.sha256,
        processing_fingerprint=fingerprint,
        unit_catalog=StagedUnitCatalog(
            catalog_id=catalog_descriptor.catalog_id,
            unit_count=3,
            unit_manifest_sha256=unit_manifest_sha256,
            catalog_format=SQLITE_UNIT_CATALOG_FORMAT,
            payload_ref="catalog",
        ),
        lanes=tuple(staged_lanes),
        payloads=(
            target_representation_payload,
            catalog_payload,
            *parameter_payloads,
            *shard_payloads,
        ),
        frozen_unit_manifest_proof_sha256="f" * 64,
        checkpoint_sha256="1" * 64,
    )
    validator = _Validator()
    validators: dict[PayloadValidationKey, PayloadValidator] = {
        PayloadValidationKey("catalog", SQLITE_UNIT_CATALOG_FORMAT): validator
    }
    for staged_lane in staged.lanes:
        validators[
            PayloadValidationKey(
                staged_lane.index_parameters_ref.payload_ref,
                staged_lane.index_format,
            )
        ] = validator
        for role in staged_lane.payload_refs:
            validators[PayloadValidationKey(role, staged_lane.index_format)] = validator
    validated = store.validate(
        staging,
        staged,
        native_validators=validators,
        deadline=FUTURE,
    )
    return store, store.publish(validated, deadline=FUTURE)


class _Resolver:
    def resolve(
        self,
        query: ImageQuery,
        *,
        authorization: InvocationAuthorizationContext,
        data_policy_ref: str,
        deadline: datetime,
    ) -> ResolvedImageQuery:
        del query, authorization, data_policy_ref, deadline
        return ResolvedImageQuery(Image.new("RGB", (2, 2)), SHA_B, "image/png", 16, 2, 2)


class _PhashEncoder:
    def encode(
        self,
        image: Image.Image,
        source_sha256: str,
        max_hamming_distance: int,
        deadline: datetime,
    ) -> PerceptualHashQuery:
        del image, max_hamming_distance, deadline
        return PerceptualHashQuery(source_sha256, OPENCV_PHASH_HASH_SPACE, 0, 2)


@dataclass
class _QueryEncoder:
    embedding_space: EmbeddingSpaceRef
    bundle_manifest_sha256: str = SHA_B

    def encode(
        self,
        image: Image.Image,
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> QueryVector:
        del image, deadline
        values = (1.0, 0.0) if space.space_id.startswith("instance") else (0.0, 1.0)
        return QueryVector(space, values)


class _QueryEncoderResolver:
    def resolve(
        self,
        lane: RetrievalLane,
        space: EmbeddingSpaceRef,
        target_bundle_manifest_sha256: str,
        execution_profile: object,
    ) -> _QueryEncoder:
        del lane, execution_profile
        assert target_bundle_manifest_sha256 == SHA_B
        return _QueryEncoder(space)


class _CountingHandleFactory:
    def __init__(self, delegate: PersonalGenerationHandleFactory) -> None:
        self._delegate = delegate
        self.open_calls = 0
        self.close_calls = 0
        self.close_callback_count = 0

    def open(
        self,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> OpenedGenerationHandles:
        self.open_calls += 1
        opened = self._delegate.open(published, deadline=deadline)
        self.close_callback_count += len(opened.close_callbacks)

        def tracked(callback: Callable[[], None]) -> Callable[[], None]:
            def close() -> None:
                callback()
                self.close_calls += 1

            return close

        return OpenedGenerationHandles(
            manifest=opened.manifest,
            unit_catalog=opened.unit_catalog,
            readers=opened.readers,
            close_callbacks=tuple(tracked(callback) for callback in opened.close_callbacks),
        )


def _candidate_authorization() -> InvocationAuthorizationContext:
    return InvocationAuthorizationContext(
        request_id="candidate-request-1",
        workspace_id="workspace-1",
        actor_id="offline-evaluator-1",
        environment_id="test",
        operation="capicove.offline.evaluate_candidate",
        authorization_scope_ref="offline-candidate-scope-1",
        permissions=frozenset({"retrieval:evaluate_candidate"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset(),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def _candidate_request() -> SearchRequest:
    return SearchRequest(
        query=ImageQuery(InlineBytesSource(b"synthetic", "image/png")),
        targets=frozenset({TargetType.IMAGE}),
        detail_level=DetailLevel.FULL,
        retrieval_profile_ref="personal-lite",
        search_intent=SearchIntent.BALANCED,
    )


def _candidate_service(
    provider: CandidateBoundGenerationSetProvider,
) -> CandidateEvaluationRetrievalService:
    return CandidateEvaluationRetrievalService(
        workspace_id="workspace-1",
        candidate_generation_sets=provider,
        query_resolver=_Resolver(),
        perceptual_encoder=_PhashEncoder(),
        query_encoders=_QueryEncoderResolver(),
        fusion=WeightedReciprocalRankFusion(),
        cursor_codec=HmacSearchCursorCodec(
            signing_key=b"k" * 32,
            workspace_id="workspace-1",
        ),
        lane_scheduler=BoundedLaneScheduler(
            CpuExecutionBudget(1, 1, 1).schedule_snapshot(schedule_id="candidate-vertical")
        ),
        profiles=default_retrieval_profiles(),
        clock=lambda: NOW,
    )


def test_published_personal_generation_reopens_and_searches_after_preview_activation(
    tmp_path: Path,
) -> None:
    store, published = _publish_generation(tmp_path)
    verified = VerifiedPreviewQualification(
        qualification_id="qualification-1",
        activation_set_id="set-1",
        manifest_sha256=published.manifest_sha256,
        profile_ref="personal-lite",
        report_sha256="2" * 64,
        policy_sha256="3" * 64,
        verification_receipt_sha256="4" * 64,
        verified_at=NOW,
    )
    lifecycle = SqliteGenerationCatalog.open(
        tmp_path / "lifecycle.sqlite3",
        verifier=_Verifier(verified),
    )
    lifecycle.register_published(
        PublishedGenerationRecord(
            activation_set_id="set-1",
            manifest_sha256=published.manifest_sha256,
            published_root=published.root,
            workspace_id="workspace-1",
            corpus_ref="photos",
            retrieval_profile_ref="personal-lite",
            published_at=NOW,
        ),
        deadline=FUTURE,
    )
    lifecycle.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-1",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-1"),
        activated_at=NOW,
        deadline=FUTURE,
    )
    lease_ids = iter(("lease-search", "lease-default"))
    cleanup_supervisor = DeferredCleanupSupervisor(max_pending=1)
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=lifecycle,
        handle_factory=PersonalGenerationHandleFactory(store),
        owner_id="worker-1",
        owner_session_epoch="session-1",
        lease_id_factory=lambda: next(lease_ids),
        clock=lambda: datetime.now(UTC),
        cleanup_supervisor=cleanup_supervisor,
    )
    service = RetrievalService(
        workspace_id="workspace-1",
        generation_sets=manager,
        query_resolver=_Resolver(),
        perceptual_encoder=_PhashEncoder(),
        query_encoders=_QueryEncoderResolver(),
        fusion=WeightedReciprocalRankFusion(),
        cursor_codec=HmacSearchCursorCodec(
            signing_key=b"k" * 32,
            workspace_id="workspace-1",
        ),
        lane_scheduler=BoundedLaneScheduler(
            CpuExecutionBudget(1, 1, 1).schedule_snapshot(schedule_id="vertical")
        ),
        profiles=default_retrieval_profiles(),
        clock=lambda: NOW,
        cleanup_supervisor=cleanup_supervisor,
    )
    authorization = InvocationAuthorizationContext(
        request_id="request-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="test",
        operation="capicove.search.image",
        authorization_scope_ref="scope-1",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset(),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )

    result = service.search(
        SearchRequest(
            query=ImageQuery(InlineBytesSource(b"synthetic", "image/png")),
            targets=frozenset({TargetType.IMAGE}),
            detail_level=DetailLevel.FULL,
            retrieval_profile_ref="personal-lite",
            search_intent=SearchIntent.BALANCED,
        ),
        authorization=authorization,
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert [hit.id for hit in result.hits] == ["unit-0", "unit-1", "unit-2"]
    assert result.degraded
    assert result.execution_summary is not None
    assert result.execution_summary.activation_tier == "preview"

    default_alias_result = service.search(
        SearchRequest(
            query=ImageQuery(InlineBytesSource(b"synthetic", "image/png")),
            targets=frozenset({TargetType.IMAGE}),
            detail_level=DetailLevel.FULL,
            search_intent=SearchIntent.BALANCED,
        ),
        authorization=authorization,
        deadline=FUTURE,
    )

    assert default_alias_result.status is SearchStatus.SUCCEEDED
    assert default_alias_result.execution_summary is not None
    assert default_alias_result.execution_summary.resolved_profile_ref == "personal-lite"
    cleanup_report = cleanup_supervisor.drain(timeout_seconds=0)
    assert cleanup_report.completed
    assert cleanup_report.admission_rejections == 0


def test_real_candidate_evaluator_reuses_one_persistent_session_without_active_alias(
    tmp_path: Path,
) -> None:
    store, published = _publish_generation(tmp_path)
    verified = VerifiedPreviewQualification(
        qualification_id="qualification-candidate-shared",
        activation_set_id="set-1",
        manifest_sha256=published.manifest_sha256,
        profile_ref="personal-lite",
        report_sha256="2" * 64,
        policy_sha256="3" * 64,
        verification_receipt_sha256="4" * 64,
        verified_at=NOW,
    )
    database = tmp_path / "candidate-lifecycle.sqlite3"
    lifecycle = SqliteGenerationCatalog.open(database, verifier=_Verifier(verified))
    lifecycle.register_published(
        PublishedGenerationRecord(
            activation_set_id="set-1",
            manifest_sha256=published.manifest_sha256,
            published_root=published.root,
            workspace_id="workspace-1",
            corpus_ref="photos",
            retrieval_profile_ref="personal-lite",
            published_at=NOW,
        ),
        deadline=FUTURE,
    )
    factory = _CountingHandleFactory(PersonalGenerationHandleFactory(store))
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=lifecycle,
        candidate_lifecycle=lifecycle,
        handle_factory=factory,
        owner_id="offline-worker",
        owner_session_epoch="offline-session",
        lease_id_factory=lambda: "candidate-session",
        clock=lambda: datetime.now(UTC),
    )
    provider = CandidateBoundGenerationSetProvider(
        manager.acquire_published(
            "set-1",
            published.manifest_sha256,
            "personal-lite",
            deadline=FUTURE,
        )
    )
    evaluator = _candidate_service(provider)
    statements: list[str] = []
    lifecycle._connection.set_trace_callback(statements.append)

    results = tuple(
        evaluator.evaluate_candidate(
            _candidate_request(),
            authorization=_candidate_authorization(),
            deadline=FUTURE,
        )
        for _ in range(3)
    )

    assert all(result.status is SearchStatus.SUCCEEDED for result in results)
    assert all(result.execution_summary is not None for result in results)
    assert all(
        result.execution_summary is not None
        and result.execution_summary.activation_tier == "candidate_evaluation"
        for result in results
    )
    assert factory.open_calls == 1
    assert factory.close_calls == 0
    assert all(
        "update published_candidate_leases set released_at" not in statement.lower()
        for statement in statements
    )
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM active_aliases").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM query_leases").fetchone() == (0,)
        assert connection.execute(
            "SELECT lease_id, released_at FROM published_candidate_leases",
        ).fetchall() == [("candidate-session", None)]

    changes_before_close = lifecycle._connection.total_changes
    provider.close()
    provider.close()

    assert factory.close_callback_count > 0
    assert factory.close_calls == factory.close_callback_count
    assert lifecycle._connection.total_changes - changes_before_close == 1
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
                ("candidate-session",),
            ).fetchone()[0]
            is not None
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM query_leases",
        ).fetchone() == (0,)
    lifecycle._connection.set_trace_callback(None)
    lifecycle.close()


def test_real_active_and_candidate_sessions_share_handles_until_both_close(
    tmp_path: Path,
) -> None:
    store, published = _publish_generation(tmp_path)
    verified = VerifiedPreviewQualification(
        qualification_id="qualification-shared-handles",
        activation_set_id="set-1",
        manifest_sha256=published.manifest_sha256,
        profile_ref="personal-lite",
        report_sha256="2" * 64,
        policy_sha256="3" * 64,
        verification_receipt_sha256="4" * 64,
        verified_at=NOW,
    )
    database = tmp_path / "shared-handle-lifecycle.sqlite3"
    lifecycle = SqliteGenerationCatalog.open(database, verifier=_Verifier(verified))
    lifecycle.register_published(
        PublishedGenerationRecord(
            activation_set_id="set-1",
            manifest_sha256=published.manifest_sha256,
            published_root=published.root,
            workspace_id="workspace-1",
            corpus_ref="photos",
            retrieval_profile_ref="personal-lite",
            published_at=NOW,
        ),
        deadline=FUTURE,
    )
    lifecycle.activate(
        workspace_id="workspace-1",
        profile_ref="personal-lite",
        activation_set_id="set-1",
        expected_previous_snapshot=None,
        qualification=PreviewActivationQualification("qualification-shared-handles"),
        activated_at=NOW,
        deadline=FUTURE,
    )
    factory = _CountingHandleFactory(PersonalGenerationHandleFactory(store))
    lease_ids = iter(("active-shared", "candidate-shared"))
    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=lifecycle,
        candidate_lifecycle=lifecycle,
        handle_factory=factory,
        owner_id="offline-worker",
        owner_session_epoch="offline-session",
        lease_id_factory=lambda: next(lease_ids),
        clock=lambda: datetime.now(UTC),
    )
    active = manager.acquire("personal-lite", deadline=FUTURE)
    provider = CandidateBoundGenerationSetProvider(
        manager.acquire_published(
            "set-1",
            published.manifest_sha256,
            "personal-lite",
            deadline=FUTURE,
        )
    )
    result = _candidate_service(provider).evaluate_candidate(
        _candidate_request(),
        authorization=_candidate_authorization(),
        deadline=FUTURE,
    )

    assert result.status is SearchStatus.SUCCEEDED
    assert factory.open_calls == 1
    active.close()
    assert factory.close_calls == 0
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT released_at FROM query_leases WHERE lease_id = 'active-shared'",
            ).fetchone()[0]
            is not None
        )
        assert connection.execute(
            "SELECT released_at FROM published_candidate_leases "
            "WHERE lease_id = 'candidate-shared'",
        ).fetchone() == (None,)

    provider.close()

    assert factory.close_calls == factory.close_callback_count
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT released_at FROM published_candidate_leases "
                "WHERE lease_id = 'candidate-shared'",
            ).fetchone()[0]
            is not None
        )
    lifecycle.close()


def test_real_candidate_evaluator_open_failure_releases_persistent_lease(
    tmp_path: Path,
) -> None:
    _store, published = _publish_generation(tmp_path)
    database = tmp_path / "candidate-failure-lifecycle.sqlite3"
    lifecycle = SqliteGenerationCatalog.open(
        database,
        verifier=_Verifier(
            VerifiedPreviewQualification(
                qualification_id="unused-candidate-qualification",
                activation_set_id="set-1",
                manifest_sha256=published.manifest_sha256,
                profile_ref="personal-lite",
                report_sha256="2" * 64,
                policy_sha256="3" * 64,
                verification_receipt_sha256="4" * 64,
                verified_at=NOW,
            )
        ),
    )
    lifecycle.register_published(
        PublishedGenerationRecord(
            activation_set_id="set-1",
            manifest_sha256=published.manifest_sha256,
            published_root=published.root,
            workspace_id="workspace-1",
            corpus_ref="photos",
            retrieval_profile_ref="personal-lite",
            published_at=NOW,
        ),
        deadline=FUTURE,
    )

    class CorruptFactory:
        def open(
            self,
            published: PublishedGenerationRecord,
            *,
            deadline: datetime,
        ) -> OpenedGenerationHandles:
            del published, deadline
            raise ValueError("candidate manifest checksum mismatch")

    manager = GenerationSetManager(
        workspace_id="workspace-1",
        lifecycle=lifecycle,
        candidate_lifecycle=lifecycle,
        handle_factory=CorruptFactory(),
        owner_id="offline-worker",
        owner_session_epoch="offline-session",
        lease_id_factory=lambda: "candidate-open-failure",
        clock=lambda: datetime.now(UTC),
    )
    with pytest.raises(SearchFailure) as raised:
        manager.acquire_published(
            "set-1",
            published.manifest_sha256,
            "personal-lite",
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.INDEX_CORRUPT
    with sqlite3.connect(database) as connection:
        released_at = connection.execute(
            "SELECT released_at FROM published_candidate_leases WHERE lease_id = ?",
            ("candidate-open-failure",),
        ).fetchone()
    assert released_at is not None
    assert released_at[0] is not None
    lifecycle.close()
