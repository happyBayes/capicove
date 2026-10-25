"""Integration tests for immutable personal-lite generation publication."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    IndexParametersRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval.image_generations import (
    TARGET_REPRESENTATION_MANIFEST_ROLE,
    GenerationDirectoryState,
    GenerationPublicationError,
    GenerationValidationError,
    PayloadFileRef,
    PayloadOwner,
    PayloadValidationKey,
    StagedGenerationSetManifest,
    StagedIndexGeneration,
    StagedUnitCatalog,
    ValidatedGenerationSet,
    ValidationReport,
)
from capicove.retrieval.search_models import RetrievalLane


NOW = datetime.now(UTC)
FUTURE = NOW + timedelta(hours=1)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _embedding(space_id: str) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=space_id,
        provider_id="local-test-provider",
        model_revision=f"{space_id}-revision",
        preprocessing_version="rgb-224-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _phash() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="opencv-phash-64-v1",
        provider_id="opencv-phash",
        provider_version="4.12.0",
        preprocessing_version="pillow-luma-opencv-area32-dct8-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


@dataclass
class RecordingValidator:
    validator_id: str = "test-structural-validator-v1"
    calls: list[str] | None = None

    def validate(
        self,
        root: Path,
        payload: PayloadFileRef,
        owner: PayloadOwner,
        *,
        deadline: datetime,
    ) -> ValidationReport:
        del root, deadline
        if self.calls is not None:
            self.calls.append(payload.role)
        owner_id = owner.catalog_id if isinstance(owner, StagedUnitCatalog) else owner.generation_id
        return ValidationReport(
            report_id=f"probe:{payload.role}",
            owner_kind="catalog" if isinstance(owner, StagedUnitCatalog) else "lane",
            owner_id=owner_id,
            validator_id=self.validator_id,
            staged_identity_sha256=SHA_A,
            observed_payload_sha256s=(payload.sha256,),
            capability_codes=("structure-checked",),
            passed=True,
            created_at=NOW,
        )


def _stage_small_generation_set(
    tmp_path: Path,
    *,
    activation_set_id: str = "set-1",
    validator: RecordingValidator | None = None,
) -> tuple[
    FilesystemGenerationStore,
    Path,
    StagedGenerationSetManifest,
    dict[PayloadValidationKey, RecordingValidator],
]:
    store = FilesystemGenerationStore(tmp_path / "published")
    staging = store.create_staging(f"build-{activation_set_id}")
    contents = {
        TARGET_REPRESENTATION_MANIFEST_ROLE: b'{"schema_version":1}\n',
        "catalog": b'{"unit_id":"u1"}\n{"unit_id":"u2"}\n',
        "perceptual-parameters": b'{"radius":2}\n',
        "perceptual-shard": bytes.fromhex("01" * 16),
        "instance-parameters": b'{"metric":"cosine"}\n',
        "instance-shard": b"instance-vectors",
        "concept-parameters": b'{"metric":"cosine"}\n',
        "concept-shard": b"concept-vectors",
    }
    payloads: list[PayloadFileRef] = []
    for role, content in contents.items():
        relative_path = PurePosixPath("payloads") / f"{role}.bin"
        target = staging.joinpath(*relative_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.parent.chmod(0o700)
        target.write_bytes(content)
        target.chmod(0o600)
        payloads.append(
            PayloadFileRef(
                relative_path=relative_path,
                sha256=_sha(content),
                size_bytes=len(content),
                role=role,
            )
        )
    payload_by_role = {payload.role: payload for payload in payloads}
    fingerprint = ProcessingFingerprint(SHA_A)

    def lane(
        lane_name: RetrievalLane,
        *,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...] = (),
        hash_spaces: tuple[HashSpaceRef, ...] = (),
    ) -> StagedIndexGeneration:
        parameter_role = f"{lane_name.value}-parameters"
        return StagedIndexGeneration(
            generation_id=f"generation-{lane_name.value}-{activation_set_id}",
            activation_set_id=activation_set_id,
            lane=lane_name,
            target_bundle_manifest_sha256=(
                None if lane_name is RetrievalLane.PERCEPTUAL else SHA_C
            ),
            unit_manifest_sha256=SHA_B,
            embedding_spaces=embedding_spaces,
            hash_spaces=hash_spaces,
            retriever_id=f"capicove.{lane_name.value}-exact",
            index_format=f"{lane_name.value}-flat-v1",
            index_parameters_ref=IndexParametersRef(
                parameter_schema=f"{lane_name.value}-flat-v1",
                payload_ref=parameter_role,
                sha256=payload_by_role[parameter_role].sha256,
            ),
            payload_refs=(f"{lane_name.value}-shard",),
        )

    staged = StagedGenerationSetManifest(
        build_id=f"build-{activation_set_id}",
        activation_set_id=activation_set_id,
        workspace_id="workspace-1",
        corpus_ref="personal-photos",
        retrieval_profile_ref="personal-lite",
        target_type=TargetType.IMAGE,
        unit_manifest_sha256=SHA_B,
        target_representation_manifest_sha256=(
            payload_by_role[TARGET_REPRESENTATION_MANIFEST_ROLE].sha256
        ),
        processing_fingerprint=fingerprint,
        unit_catalog=StagedUnitCatalog(
            catalog_id=f"catalog-{activation_set_id}",
            unit_count=2,
            unit_manifest_sha256=SHA_B,
            catalog_format="jsonl-v1",
            payload_ref="catalog",
        ),
        lanes=(
            lane(RetrievalLane.PERCEPTUAL, hash_spaces=(_phash(),)),
            lane(RetrievalLane.INSTANCE, embedding_spaces=(_embedding("instance"),)),
            lane(RetrievalLane.CONCEPT, embedding_spaces=(_embedding("concept"),)),
        ),
        payloads=tuple(payloads),
        frozen_unit_manifest_proof_sha256=SHA_C,
        checkpoint_sha256="e" * 64,
    )
    recording = validator or RecordingValidator(calls=[])
    validators = {
        PayloadValidationKey(payload.role, _owner_format(payload.role, staged)): recording
        for payload in staged.payloads
        if payload.role != TARGET_REPRESENTATION_MANIFEST_ROLE
    }
    return store, staging, staged, validators


def _owner_format(role: str, staged: StagedGenerationSetManifest) -> str:
    if role == staged.unit_catalog.payload_ref:
        return staged.unit_catalog.catalog_format
    for lane in staged.lanes:
        if role == lane.index_parameters_ref.payload_ref or role in lane.payload_refs:
            return lane.index_format
    raise AssertionError(f"unowned role: {role}")


def test_publish_is_atomic_immutable_and_inert_and_reopens(tmp_path: Path) -> None:
    store, staging, staged, validators = _stage_small_generation_set(tmp_path)

    validated = store.validate(
        staging,
        staged,
        native_validators=validators,
        deadline=FUTURE,
    )
    manifest_before = (staging / "manifest.json").read_bytes()
    published = store.publish(validated, deadline=FUTURE)
    reopened_store = FilesystemGenerationStore(store.root)
    reopened = reopened_store.open_published(
        staged.activation_set_id,
        expected_manifest_sha256=validated.manifest_sha256,
        deadline=FUTURE,
    )

    assert published == reopened
    assert (published.root / "manifest.json").read_bytes() == manifest_before
    assert published.manifest_sha256 == validated.manifest_sha256
    assert not staging.exists()
    assert not (store.root / "active").exists()
    assert not (store.root / "current").exists()
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0 for path in published.root.rglob("*"))


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "symlink", "non-regular", "group-readable"),
)
def test_validate_rejects_unsafe_or_nonclosed_staging(tmp_path: Path, mutation: str) -> None:
    calls: list[str] = []
    validator = RecordingValidator(calls=calls)
    store, staging, staged, validators = _stage_small_generation_set(
        tmp_path,
        validator=validator,
    )
    first = staging.joinpath(*staged.payloads[0].relative_path.parts)
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        extra = staging / "payloads" / "extra.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)
    elif mutation == "symlink":
        first.unlink()
        first.symlink_to(staging / "outside.bin")
    elif mutation == "non-regular":
        first.unlink()
        first.mkdir(mode=0o700)
    else:
        first.chmod(0o640)

    with pytest.raises(GenerationValidationError, match=r"missing|extra|symlink|permission"):
        store.validate(
            staging,
            staged,
            native_validators=validators,
            deadline=FUTURE,
        )

    assert calls == []
    assert not (store.root / staged.activation_set_id).exists()


def test_validate_rejects_tampering_before_native_open(tmp_path: Path) -> None:
    calls: list[str] = []
    validator = RecordingValidator(calls=calls)
    store, staging, staged, validators = _stage_small_generation_set(
        tmp_path,
        validator=validator,
    )
    payload_path = staging.joinpath(*staged.payloads[1].relative_path.parts)
    payload_path.write_bytes(payload_path.read_bytes() + b"tampered")
    payload_path.chmod(0o600)

    with pytest.raises(GenerationValidationError, match=r"size|checksum"):
        store.validate(
            staging,
            staged,
            native_validators=validators,
            deadline=FUTURE,
        )

    assert calls == []


@pytest.mark.parametrize("mutation", ("byte", "replace", "extra"))
def test_publish_rechecks_sealed_files_before_rename(tmp_path: Path, mutation: str) -> None:
    store, staging, staged, validators = _stage_small_generation_set(tmp_path)
    validated = store.validate(
        staging,
        staged,
        native_validators=validators,
        deadline=FUTURE,
    )
    first = staging.joinpath(*staged.payloads[0].relative_path.parts)
    if mutation == "byte":
        content = bytearray(first.read_bytes())
        content[0] ^= 1
        first.write_bytes(content)
        first.chmod(0o600)
    elif mutation == "replace":
        replacement = first.with_suffix(".replacement")
        replacement.write_bytes(first.read_bytes())
        replacement.chmod(0o600)
        os.replace(replacement, first)
    else:
        extra = staging / "extra.bin"
        extra.write_bytes(b"extra")
        extra.chmod(0o600)

    with pytest.raises(GenerationPublicationError, match=r"changed|extra|sealed"):
        store.publish(validated, deadline=FUTURE)

    assert not (store.root / staged.activation_set_id).exists()


def test_publish_never_overwrites_a_winner(tmp_path: Path) -> None:
    store, staging, staged, validators = _stage_small_generation_set(tmp_path)
    winner = store.publish(
        store.validate(staging, staged, native_validators=validators, deadline=FUTURE),
        deadline=FUTURE,
    )
    winner_manifest = (winner.root / "manifest.json").read_bytes()
    _, loser_staging, loser_staged, loser_validators = _stage_small_generation_set(tmp_path)
    loser = store.validate(
        loser_staging,
        loser_staged,
        native_validators=loser_validators,
        deadline=FUTURE,
    )

    with pytest.raises(GenerationPublicationError, match="already published"):
        store.publish(loser, deadline=FUTURE)

    assert (winner.root / "manifest.json").read_bytes() == winner_manifest
    assert loser_staging.exists()


def test_racing_publishers_have_exactly_one_winner(tmp_path: Path) -> None:
    store_a, staging_a, staged_a, validators_a = _stage_small_generation_set(tmp_path)
    store_b, staging_b, staged_b, validators_b = _stage_small_generation_set(tmp_path)
    validated_a = store_a.validate(
        staging_a,
        staged_a,
        native_validators=validators_a,
        deadline=FUTURE,
    )
    validated_b = store_b.validate(
        staging_b,
        staged_b,
        native_validators=validators_b,
        deadline=FUTURE,
    )
    barrier = threading.Barrier(2)

    def publish(store: FilesystemGenerationStore, validated: ValidatedGenerationSet) -> str:
        barrier.wait()
        try:
            store.publish(validated, deadline=FUTURE)
        except GenerationPublicationError:
            return "collision"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(
            executor.map(
                lambda pair: publish(*pair),
                ((store_a, validated_a), (store_b, validated_b)),
            )
        )

    assert sorted(outcomes) == ["collision", "published"]
    assert (store_a.root / staged_a.activation_set_id / "manifest.json").is_file()


def test_validation_and_publication_fsync_files_and_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, staging, staged, validators = _stage_small_generation_set(tmp_path)
    real_fsync = os.fsync
    fsynced_modes: list[int] = []

    def recording_fsync(file_descriptor: int) -> None:
        fsynced_modes.append(os.fstat(file_descriptor).st_mode)
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    validated = store.validate(
        staging,
        staged,
        native_validators=validators,
        deadline=FUTURE,
    )
    store.publish(validated, deadline=FUTURE)

    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_failed_rename_leaves_no_partial_directory_and_preserves_published_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, staging, staged, validators = _stage_small_generation_set(
        tmp_path, activation_set_id="a"
    )
    first = store.publish(
        store.validate(staging, staged, native_validators=validators, deadline=FUTURE),
        deadline=FUTURE,
    )
    first_digest = (first.root / "manifest.json").read_bytes()
    _, staging_b, staged_b, validators_b = _stage_small_generation_set(
        tmp_path,
        activation_set_id="b",
    )
    validated_b = store.validate(
        staging_b,
        staged_b,
        native_validators=validators_b,
        deadline=FUTURE,
    )

    def fail_rename(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "rename", fail_rename)
    with pytest.raises(GenerationPublicationError, match="rename"):
        store.publish(validated_b, deadline=FUTURE)

    assert not (store.root / "b").exists()
    assert (first.root / "manifest.json").read_bytes() == first_digest


def test_reopen_detects_tampering_and_scan_identifies_renamed_orphan(tmp_path: Path) -> None:
    store, staging, staged, validators = _stage_small_generation_set(tmp_path)
    published = store.publish(
        store.validate(staging, staged, native_validators=validators, deadline=FUTURE),
        deadline=FUTURE,
    )
    payload = published.root.joinpath(*published.manifest.payloads[0].relative_path.parts)
    original = payload.read_bytes()
    payload.write_bytes(original + b"tampered")
    payload.chmod(0o600)

    with pytest.raises(GenerationValidationError, match=r"size|checksum"):
        store.open_published(staged.activation_set_id, deadline=FUTURE)

    payload.write_bytes(original)
    payload.chmod(0o600)
    orphan = store.root / "renamed-orphan"
    published.root.rename(orphan)

    inspections = store.scan_published(deadline=FUTURE)
    assert len(inspections) == 1
    assert inspections[0].root == orphan
    assert inspections[0].state is GenerationDirectoryState.ORPHANED
    assert inspections[0].activation_set_id == staged.activation_set_id


def test_deadline_and_validator_mapping_are_closed(tmp_path: Path) -> None:
    store, staging, staged, validators = _stage_small_generation_set(tmp_path)
    missing_key = next(iter(validators))
    validators.pop(missing_key)
    with pytest.raises(GenerationValidationError, match="validator"):
        store.validate(staging, staged, native_validators=validators, deadline=FUTURE)

    store2, staging2, staged2, validators2 = _stage_small_generation_set(tmp_path / "expired")
    with pytest.raises(GenerationValidationError, match="deadline"):
        store2.validate(
            staging2,
            staged2,
            native_validators=validators2,
            deadline=NOW - timedelta(seconds=1),
        )
