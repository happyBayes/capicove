"""Unit tests for the bounded personal image-generation build boundary."""

from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import Event, Thread
from typing import Any

import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    VectorNormalization,
)
from capicove.processing.image.io import LoadedImage, load_image
from capicove.processing.image.models import ImageSource
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity
from capicove.processing.image.representations import (
    EncodedImageVector,
    ImageRepresentationSource,
    ImageTargetRepresentation,
    ImageTargetRepresentationProducer,
    RepresentationSealRequest,
)
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval.image_generations import TARGET_REPRESENTATION_MANIFEST_ROLE, PayloadFileRef
from capicove.runtime import retrieval_build, retrieval_representations
from capicove.runtime.retrieval_build import (
    BuiltPersonalImageGeneration,
    PersonalImageBuildPlan,
    PersonalImageGenerationAdministrator,
)
from capicove.runtime.retrieval_representations import (
    ImageBuildSource,
    ImageBuildSummary,
    PersonalImageRepresentationSink,
    StagedRepresentationEvidenceIdentity,
    StagedRepresentationEvidenceLease,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)
FINGERPRINT = ProcessingFingerprint("a" * 64)


def test_built_generation_identity_parameter_remains_backward_compatible() -> None:
    parameter = inspect.signature(BuiltPersonalImageGeneration).parameters[
        "staging_directory_identity"
    ]

    assert parameter.default is None


def _png_bytes(index: int) -> bytes:
    payload = BytesIO()
    Image.new("RGB", (2, 2), (index, 0, 0)).save(payload, "PNG")
    return payload.getvalue()


def _space(lane: ImageSemanticLane) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-space-v1",
        provider_id=f"{lane.value}-fake",
        model_revision="revision-1",
        preprocessing_version="rgb-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _identity(lane: ImageSemanticLane) -> SemanticTargetIdentity:
    return SemanticTargetIdentity(
        lane=lane,
        space=_space(lane),
        bundle_manifest_sha256=("b" if lane is ImageSemanticLane.INSTANCE else "c") * 64,
    )


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="fake-phash-v1",
        provider_id="fake-phash",
        provider_version="1",
        preprocessing_version="rgb-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _sources(count: int) -> tuple[ImageRepresentationSource, ...]:
    result = []
    for ordinal in range(count):
        payload = _png_bytes(ordinal + 1)
        result.append(
            ImageRepresentationSource(
                ordinal=ordinal,
                unit=SearchableUnit(
                    unit_id=f"unit-{ordinal}",
                    unit_type=TargetType.IMAGE,
                    source_hash=hashlib.sha256(payload).hexdigest(),
                    artifact_ref=f"artifact-{ordinal}",
                    evidence_ref=f"evidence-{ordinal}",
                    representation_refs=(f"source-{ordinal}",),
                    processing_run_ref="run-1",
                    processing_version="image-v1",
                    processing_fingerprint=FINGERPRINT,
                ),
                source=ImageSource(data=payload, media_type="image/png"),
            )
        )
    return tuple(result)


class _Decoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, source: ImageSource) -> LoadedImage:
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic decode failure")
        return load_image(source)


class _Hasher:
    provider_id = "fake-phash"
    provider_version = "1"
    hash_version = "phash-64-v1"

    def phash(self, image: Image.Image) -> int:
        pixel = image.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        return int(pixel[0])


class _Encoder:
    def __init__(self, lane: ImageSemanticLane, *, readiness_error: bool = False) -> None:
        self.identity = _identity(lane)
        self.readiness_error = readiness_error
        self.ready_calls = 0

    def verify_ready(self, *, deadline: datetime) -> None:
        del deadline
        self.ready_calls += 1
        if self.readiness_error:
            raise RuntimeError("synthetic readiness failure")

    def encode_images(
        self,
        images: tuple[Image.Image, ...],
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[tuple[float, ...], ...]:
        del space, deadline
        value = (1.0, 0.0) if self.identity.lane is ImageSemanticLane.INSTANCE else (0.0, 1.0)
        return tuple(value for _ in images)


def _producer(
    *, readiness_error: bool = False, decode_error: bool = False
) -> tuple[ImageTargetRepresentationProducer, _Decoder, tuple[_Encoder, _Encoder]]:
    decoder = _Decoder(fail=decode_error)
    encoders = (
        _Encoder(ImageSemanticLane.INSTANCE, readiness_error=readiness_error),
        _Encoder(ImageSemanticLane.CONCEPT),
    )
    producer = ImageTargetRepresentationProducer(
        decoder=decoder,
        perceptual_hasher=_Hasher(),
        encoders={
            ImageSemanticLane.INSTANCE: encoders[0],
            ImageSemanticLane.CONCEPT: encoders[1],
        },
        hash_space=_hash_space(),
        processing_fingerprint=FINGERPRINT,
    )
    return producer, decoder, encoders


def _administrator(
    tmp_path: object,
    *,
    max_units: int = 10_000,
    readiness_error: bool = False,
    decode_error: bool = False,
) -> tuple[
    PersonalImageGenerationAdministrator,
    FilesystemGenerationStore,
    _Decoder,
    tuple[_Encoder, _Encoder],
]:
    root = Path(str(tmp_path))
    producer, decoder, encoders = _producer(
        readiness_error=readiness_error,
        decode_error=decode_error,
    )
    store = FilesystemGenerationStore(root / "generations")
    return (
        PersonalImageGenerationAdministrator(store, producer, max_units=max_units),
        store,
        decoder,
        encoders,
    )


def _plan() -> PersonalImageBuildPlan:
    return PersonalImageBuildPlan(
        build_id="build-1",
        activation_set_id="generation-set-1",
        workspace_id="workspace-1",
        corpus_ref="photos",
        batch_size=2,
    )


def _row(
    source: ImageRepresentationSource,
    *,
    ordinal: int | None = None,
    unit_id: str | None = None,
    content_sha256: str | None = None,
    media_type: str | None = None,
) -> ImageTargetRepresentation:
    declared_media_type = source.source.media_type
    assert isinstance(declared_media_type, str)
    return ImageTargetRepresentation(
        ordinal=source.ordinal if ordinal is None else ordinal,
        unit_id=source.unit.unit_id if unit_id is None else unit_id,
        content_sha256=(source.unit.source_hash if content_sha256 is None else content_sha256),
        media_type=declared_media_type if media_type is None else media_type,
        perceptual_hash=source.ordinal + 1,
        hash_space=_hash_space(),
        embeddings=(
            EncodedImageVector(_identity(ImageSemanticLane.INSTANCE), (1.0, 0.0)),
            EncodedImageVector(_identity(ImageSemanticLane.CONCEPT), (0.0, 1.0)),
        ),
        processing_fingerprint=source.unit.processing_fingerprint,
    )


def _build(
    tmp_path: Path,
    *,
    count: int = 2,
) -> tuple[
    PersonalImageGenerationAdministrator,
    FilesystemGenerationStore,
    BuiltPersonalImageGeneration,
]:
    administrator, store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(ImageBuildSource(_sources(count)), _plan(), deadline=FUTURE)
    return administrator, store, built


def test_image_build_source_is_repeatable_and_freezes_a_bounded_summary() -> None:
    build_source = ImageBuildSource(iter(_sources(3)))

    assert tuple(build_source) == tuple(build_source)
    assert build_source.summary.unit_count == 3
    assert build_source.summary.processing_fingerprint == FINGERPRINT
    assert len(build_source.summary.frozen_unit_manifest_proof_sha256) == 64
    assert tuple(row.ordinal for row in build_source.catalog_rows()) == (0, 1, 2)


def test_capacity_is_rejected_before_readiness_staging_or_decode(tmp_path: object) -> None:
    administrator, store, decoder, encoders = _administrator(tmp_path, max_units=1)

    with pytest.raises(ValueError, match="capacity"):
        administrator.build(ImageBuildSource(_sources(2)), _plan(), deadline=FUTURE)

    assert decoder.calls == 0
    assert tuple(encoder.ready_calls for encoder in encoders) == (0, 0)
    assert tuple(store.staging_root.iterdir()) == ()


def test_readiness_failure_occurs_before_staging_or_decode(tmp_path: object) -> None:
    administrator, store, decoder, encoders = _administrator(
        tmp_path,
        readiness_error=True,
    )

    with pytest.raises(RuntimeError, match="readiness"):
        administrator.build(ImageBuildSource(_sources(1)), _plan(), deadline=FUTURE)

    assert decoder.calls == 0
    assert tuple(encoder.ready_calls for encoder in encoders) == (1, 0)
    assert tuple(store.staging_root.iterdir()) == ()


def test_decode_failure_removes_only_the_owned_inert_staging_attempt(tmp_path: object) -> None:
    administrator, store, decoder, encoders = _administrator(
        tmp_path,
        decode_error=True,
    )

    with pytest.raises(RuntimeError, match="decode"):
        administrator.build(
            ImageBuildSource(_sources(1)),
            _plan(),
            deadline=FUTURE,
            retain_representation_evidence=True,
        )

    assert decoder.calls == 1
    assert tuple(encoder.ready_calls for encoder in encoders) == (2, 2)
    assert tuple(store.staging_root.iterdir()) == ()


def test_build_preserves_primary_failure_when_owned_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(
        tmp_path,
        decode_error=True,
    )

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(retrieval_build, "_remove_owned_staging", fail_cleanup)

    with pytest.raises(RuntimeError, match="synthetic decode failure") as captured:
        administrator.build(
            ImageBuildSource(_sources(1)),
            _plan(),
            deadline=FUTURE,
            retain_representation_evidence=True,
        )

    assert any("cleanup failure" in note for note in getattr(captured.value, "__notes__", ()))


def test_build_preserves_construction_failure_when_evidence_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, store, _decoder, _encoders = _administrator(tmp_path)
    original_close = StagedRepresentationEvidenceLease.close

    def fail_construction(*_args: object, **_kwargs: object) -> None:
        raise LookupError("synthetic built-value failure")

    def close_then_fail(lease: StagedRepresentationEvidenceLease) -> None:
        original_close(lease)
        raise RuntimeError("synthetic evidence close failure")

    monkeypatch.setattr(retrieval_build, "BuiltPersonalImageGeneration", fail_construction)
    monkeypatch.setattr(StagedRepresentationEvidenceLease, "close", close_then_fail)

    with pytest.raises(LookupError, match="built-value failure") as captured:
        administrator.build(
            ImageBuildSource(_sources(1)),
            _plan(),
            deadline=FUTURE,
            retain_representation_evidence=True,
        )

    assert any("evidence close failure" in note for note in captured.value.__notes__)
    assert tuple(store.staging_root.iterdir()) == ()


def test_registry_insertion_failure_closes_transferred_evidence_and_preserves_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, store, _decoder, _encoders = _administrator(tmp_path)
    original_close = StagedRepresentationEvidenceLease.close
    close_calls = 0

    class FailingRegistry(dict[Path, StagedRepresentationEvidenceLease]):
        def __setitem__(self, key: Path, value: StagedRepresentationEvidenceLease) -> None:
            del key, value
            raise MemoryError("synthetic registry insertion failure")

    def tracked_close(lease: StagedRepresentationEvidenceLease) -> None:
        nonlocal close_calls
        close_calls += 1
        original_close(lease)

    administrator._open_representation_evidence = FailingRegistry()
    monkeypatch.setattr(StagedRepresentationEvidenceLease, "close", tracked_close)

    with pytest.raises(MemoryError, match="registry insertion failure"):
        administrator.build(
            ImageBuildSource(_sources(1)),
            _plan(),
            deadline=FUTURE,
            retain_representation_evidence=True,
        )

    assert close_calls == 1
    assert tuple(store.staging_root.iterdir()) == ()


def test_build_validate_publish_are_separate_and_preserve_closed_manifest(
    tmp_path: Path,
) -> None:
    administrator, store, built = _build(tmp_path, count=3)
    representation = built.representation_result.sealed_manifest
    representation_path = built.staging_directory.joinpath(*representation.relative_path.parts)

    assert representation_path.read_bytes() == representation.canonical_bytes
    assert built.representation_evidence_lease is None
    assert tuple((built.staging_directory / "representations").iterdir()) == (representation_path,)
    assert not (store.root / _plan().activation_set_id).exists()
    assert TARGET_REPRESENTATION_MANIFEST_ROLE in {
        payload.role for payload in built.staged_manifest.payloads
    }

    validated = administrator.validate(built, deadline=FUTURE)
    assert not (store.root / _plan().activation_set_id).exists()

    published = administrator.publish(validated, deadline=FUTURE)
    published_manifest = next(
        payload
        for payload in published.manifest.payloads
        if payload.role == TARGET_REPRESENTATION_MANIFEST_ROLE
    )
    assert published.root.joinpath(*published_manifest.relative_path.parts).read_bytes() == (
        representation.canonical_bytes
    )
    assert tuple(store.staging_root.iterdir()) == ()


def test_retained_representation_evidence_streams_two_complete_passes_then_closes(
    tmp_path: Path,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(3)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    assert lease.identity == StagedRepresentationEvidenceIdentity(
        activation_set_id=built.staged_manifest.activation_set_id,
        workspace_id=built.staged_manifest.workspace_id,
        corpus_ref=built.staged_manifest.corpus_ref,
        unit_manifest_sha256=built.staged_manifest.unit_manifest_sha256,
        target_representation_manifest_sha256=(
            built.staged_manifest.target_representation_manifest_sha256
        ),
        unit_count=3,
        processing_fingerprint=built.staged_manifest.processing_fingerprint,
        hash_space=built.representation_result.report.hash_space,
        semantic_targets=built.representation_result.report.semantic_targets,
        representation_payload_set_sha256=(
            built.representation_result.report.representation_payload_set_sha256
        ),
        checkpoint_sha256=built.staged_manifest.checkpoint_sha256,
    )
    assert lease.source.identity == lease.identity

    with pytest.raises(ValueError, match="must start at zero"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=1,
                maximum_batch_size=1,
                deadline=FUTURE,
            )
        )

    batches = tuple(
        lease.source.iter_batches(
            start_ordinal=0,
            maximum_batch_size=1,
            deadline=FUTURE,
        )
    )

    assert tuple(len(batch) for batch in batches) == (1, 1, 1)
    assert tuple(row.ordinal for batch in batches for row in batch) == (0, 1, 2)
    assert tuple(row.unit_id for batch in batches for row in batch) == (
        "unit-0",
        "unit-1",
        "unit-2",
    )
    replay = tuple(
        lease.source.iter_batches(
            start_ordinal=0,
            maximum_batch_size=2,
            deadline=FUTURE,
        )
    )
    assert tuple(row.ordinal for batch in replay for row in batch) == (0, 1, 2)
    with pytest.raises(RuntimeError, match="two complete traversals"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=2,
                deadline=FUTURE,
            )
        )

    lease.close()
    lease.close()
    assert lease.is_closed
    representation_directory = built.staging_directory / "representations"
    assert tuple(representation_directory.iterdir()) == (
        representation_directory / "target-representation-manifest.json",
    )
    with pytest.raises(RuntimeError, match="closed"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=2,
                deadline=FUTURE,
            )
        )


def test_representation_evidence_rejects_concurrent_traversal(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    first = lease.source.iter_batches(
        start_ordinal=0,
        maximum_batch_size=1,
        deadline=FUTURE,
    )
    assert next(first)[0].ordinal == 0

    with pytest.raises(RuntimeError, match="already active"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=1,
                deadline=FUTURE,
            )
        )

    assert tuple(row.ordinal for batch in first for row in batch) == (1,)
    assert tuple(
        row.ordinal
        for batch in lease.source.iter_batches(
            start_ordinal=0,
            maximum_batch_size=2,
            deadline=FUTURE,
        )
        for row in batch
    ) == (0, 1)
    lease.close()


def test_partial_representation_traversal_permanently_blocks_replay(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    first = lease.source.iter_batches(
        start_ordinal=0,
        maximum_batch_size=1,
        deadline=FUTURE,
    )
    assert next(first)[0].ordinal == 0
    first.close()

    with pytest.raises(RuntimeError, match="did not complete"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=1,
                deadline=FUTURE,
            )
        )
    lease.close()


@pytest.mark.parametrize("tamper_kind", ["manifest-content", "shard-content", "shard-inode"])
def test_representation_evidence_revalidates_files_between_passes(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    assert (
        len(
            tuple(
                lease.source.iter_batches(
                    start_ordinal=0,
                    maximum_batch_size=2,
                    deadline=FUTURE,
                )
            )
        )
        == 1
    )

    evidence_directory = built.staging_directory / "representations"
    target = evidence_directory / (
        "target-representation-manifest.json"
        if tamper_kind == "manifest-content"
        else "shard-00000.json"
    )
    payload = target.read_bytes()
    owned: Path | None = None
    if tamper_kind == "shard-inode":
        owned = evidence_directory / "owned-shard"
        target.rename(owned)
        target.write_bytes(payload)
    else:
        changed = bytearray(payload)
        changed[0] = ord("[") if changed[0] != ord("[") else ord("{")
        target.write_bytes(changed)
    with pytest.raises(ValueError, match=r"checksum|inode"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=2,
                deadline=FUTURE,
            )
        )
    if owned is not None:
        with pytest.raises(ValueError, match="refusing to remove replaced"):
            lease.close()
        assert target.read_bytes() == payload
        target.unlink()
        owned.rename(target)
    lease.close()


def test_open_representation_evidence_blocks_validate_and_publish(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    retained = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = retained.representation_evidence_lease
    assert lease is not None

    with pytest.raises(RuntimeError, match="evidence lease is open"):
        administrator.validate(retained, deadline=FUTURE)

    ordinary = administrator.build(
        ImageBuildSource(_sources(2)),
        replace(_plan(), build_id="build-2"),
        deadline=FUTURE,
    )
    validated = administrator.validate(ordinary, deadline=FUTURE)
    with pytest.raises(RuntimeError, match="evidence lease is open"):
        administrator.publish(
            replace(validated, staging_directory=retained.staging_directory),
            deadline=FUTURE,
        )

    lease.close()
    administrator.validate(retained, deadline=FUTURE)


def test_representation_evidence_rejects_identity_drift_and_source_tamper(
    tmp_path: Path,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None

    with pytest.raises(ValueError, match="evidence identity"):
        replace(
            built,
            staged_manifest=replace(built.staged_manifest, corpus_ref="other-corpus"),
        )

    shard = built.staging_directory / "representations" / "shard-00000.json"
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"checksum|size"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=2,
                deadline=FUTURE,
            )
        )

    lease.close()
    assert not shard.exists()


def test_representation_evidence_never_follows_a_replaced_source_directory(
    tmp_path: Path,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    representation_directory = built.staging_directory / "representations"
    saved_directory = built.staging_directory / "saved-representations"
    external_directory = tmp_path / "external"
    external_directory.mkdir()
    external_shard = external_directory / "shard-00000.json"
    external_shard.write_bytes(b"must-not-be-read-or-deleted")
    representation_directory.rename(saved_directory)
    representation_directory.symlink_to(external_directory, target_is_directory=True)

    with pytest.raises((OSError, ValueError)):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=1,
                deadline=FUTURE,
            )
        )
    with pytest.raises((OSError, ValueError)):
        lease.close()
    assert external_shard.read_bytes() == b"must-not-be-read-or-deleted"

    representation_directory.unlink()
    saved_directory.rename(representation_directory)
    lease.close()


def test_representation_evidence_rejects_real_directory_replacement_without_deleting_it(
    tmp_path: Path,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    representation_directory = built.staging_directory / "representations"
    saved_directory = built.staging_directory / "saved-representations"
    representation_directory.rename(saved_directory)
    representation_directory.mkdir()
    foreign_shard = representation_directory / "shard-00000.json"
    foreign_shard.write_bytes(b"must-not-be-deleted")

    with pytest.raises(ValueError, match="directory identity changed"):
        lease.close()

    assert not lease.is_closed
    assert foreign_shard.read_bytes() == b"must-not-be-deleted"
    assert (saved_directory / "shard-00000.json").is_file()
    foreign_shard.unlink()
    representation_directory.rmdir()
    saved_directory.rename(representation_directory)
    lease.close()


def test_representation_evidence_cleanup_rejects_a_hardlink_alias(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    shard = built.staging_directory / "representations" / "shard-00000.json"
    alias = shard.with_name("shard-alias.json")
    os.link(shard, alias)

    with pytest.raises(ValueError, match="refusing to remove replaced"):
        lease.close()

    assert shard.is_file()
    assert alias.is_file()
    alias.unlink()
    lease.close()


def test_representation_evidence_cleanup_preserves_a_late_shard_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    shard = built.staging_directory / "representations" / "shard-00000.json"
    saved_shard = built.staging_directory / "saved-owned-shard.json"
    original_rename = os.rename
    swapped = False

    def swap_before_quarantine(
        source: Any,
        destination: Any,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if not swapped and source == shard.name and src_dir_fd is not None:
            swapped = True
            original_rename(shard, saved_shard)
            shard.write_bytes(b"foreign replacement")
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", swap_before_quarantine)

    with pytest.raises(ValueError, match="refusing to remove replaced"):
        lease.close()

    assert shard.read_bytes() == b"foreign replacement"
    assert saved_shard.is_file()
    shard.unlink()
    original_rename(saved_shard, shard)
    lease.close()


def test_representation_evidence_fifo_fails_without_blocking(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    shard = built.staging_directory / "representations" / "shard-00000.json"
    owned_shard = shard.with_name("owned-shard")
    shard.rename(owned_shard)
    os.mkfifo(shard, 0o600)

    with pytest.raises(ValueError, match="regular file"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=1,
                deadline=FUTURE,
            )
        )

    shard.unlink()
    owned_shard.rename(shard)
    lease.close()


def test_lease_context_preserves_consumer_error_and_invalidates_retained_stream(
    tmp_path: Path,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    stream = lease.source.iter_batches(
        start_ordinal=0,
        maximum_batch_size=1,
        deadline=FUTURE,
    )

    with pytest.raises(LookupError, match="consumer failed"):
        with lease:
            for batch in stream:
                assert batch[0].ordinal == 0
                raise LookupError("consumer failed")

    assert lease.is_closed
    with pytest.raises(RuntimeError, match="no longer active"):
        next(stream)


def test_lease_context_does_not_abandon_a_stream_owned_by_another_thread(
    tmp_path: Path,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    stream = lease.source.iter_batches(
        start_ordinal=0,
        maximum_batch_size=1,
        deadline=FUTURE,
    )
    yielded = Event()
    release = Event()
    errors: list[BaseException] = []

    def consume() -> None:
        try:
            assert next(stream)[0].ordinal == 0
            yielded.set()
            assert release.wait(1)
            stream.close()
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = Thread(target=consume)
    thread.start()
    assert yielded.wait(1)
    with pytest.raises(LookupError, match="consumer failed") as captured:
        with lease:
            raise LookupError("consumer failed")

    assert not lease.is_closed
    assert any("another thread" in note for note in getattr(captured.value, "__notes__", ()))
    release.set()
    thread.join(1)
    assert not thread.is_alive()
    assert errors == []
    lease.close()


def test_lease_context_preserves_body_error_when_cleanup_fails(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    representation_directory = built.staging_directory / "representations"
    saved_directory = built.staging_directory / "saved-representations"
    representation_directory.rename(saved_directory)
    representation_directory.mkdir()

    with pytest.raises(LookupError, match="consumer failed") as captured:
        with lease:
            raise LookupError("consumer failed")

    assert not lease.is_closed
    assert any(
        "directory identity changed" in note for note in getattr(captured.value, "__notes__", ())
    )
    representation_directory.rmdir()
    saved_directory.rename(representation_directory)
    lease.close()


def test_lease_close_callback_may_observe_state_without_deadlock(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    observed: list[bool] = []
    errors: list[BaseException] = []
    original_on_close = lease._on_close

    def observing_close() -> None:
        observed.append(lease.is_closed)
        original_on_close()

    lease._on_close = observing_close

    def close_lease() -> None:
        try:
            lease.close()
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = Thread(target=close_lease, daemon=True)
    thread.start()
    thread.join(1)

    assert not thread.is_alive()
    assert errors == []
    assert observed == [False]
    assert lease.is_closed


def test_lease_close_callback_failure_is_retryable_without_repeating_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    original_remove = retrieval_representations._remove_representation_shards
    remove_calls = 0

    def counting_remove(*args: object, **kwargs: object) -> None:
        nonlocal remove_calls
        remove_calls += 1
        original_remove(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        retrieval_representations,
        "_remove_representation_shards",
        counting_remove,
    )
    original_on_close = lease._on_close
    callback_calls = 0

    def flaky_close() -> None:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            raise RuntimeError("synthetic callback failure")
        original_on_close()

    lease._on_close = flaky_close

    with pytest.raises(RuntimeError, match="callback failure"):
        lease.close()
    assert not bool(lease.is_closed)
    with pytest.raises(RuntimeError, match="closed"):
        tuple(
            lease.source.iter_batches(
                start_ordinal=0,
                maximum_batch_size=1,
                deadline=FUTURE,
            )
        )

    lease.close()
    assert bool(lease.is_closed)
    assert remove_calls == 1
    assert callback_calls == 2


def test_concurrent_lease_close_runs_cleanup_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    original_remove = retrieval_representations._remove_representation_shards
    entered = Event()
    release = Event()
    calls = 0

    def delayed_remove(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(1)
        original_remove(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        retrieval_representations,
        "_remove_representation_shards",
        delayed_remove,
    )
    errors: list[BaseException] = []

    def close_lease() -> None:
        try:
            lease.close()
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = Thread(target=close_lease)
    second = Thread(target=close_lease)
    first.start()
    assert entered.wait(1)
    second.start()
    release.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == 1
    assert lease.is_closed


def test_representation_evidence_file_cap_is_enforced_at_write_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path / "reference")
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    evidence_directory = built.staging_directory / "representations"
    maximum_size = max(path.stat().st_size for path in evidence_directory.iterdir())

    with monkeypatch.context() as scoped:
        scoped.setattr(
            retrieval_representations,
            "_MAX_REPRESENTATION_EVIDENCE_FILE_BYTES",
            maximum_size - 1,
        )
        with pytest.raises(ValueError, match="representation evidence"):
            tuple(
                lease.source.iter_batches(
                    start_ordinal=0,
                    maximum_batch_size=1,
                    deadline=FUTURE,
                )
            )
    lease.close()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            retrieval_representations,
            "_MAX_REPRESENTATION_EVIDENCE_FILE_BYTES",
            maximum_size,
        )
        boundary, _boundary_store, _decoder, _encoders = _administrator(tmp_path / "boundary")
        boundary_built = boundary.build(
            ImageBuildSource(_sources(1)),
            _plan(),
            deadline=FUTURE,
            retain_representation_evidence=True,
        )
        assert boundary_built.representation_evidence_lease is not None
        boundary_built.representation_evidence_lease.close()

    with monkeypatch.context() as scoped:
        scoped.setattr(
            retrieval_representations,
            "_MAX_REPRESENTATION_EVIDENCE_FILE_BYTES",
            maximum_size - 1,
        )
        over_limit, over_limit_store, _decoder, _encoders = _administrator(tmp_path / "over-limit")
        with pytest.raises(ValueError, match=r"representation shard|target representation"):
            over_limit.build(
                ImageBuildSource(_sources(1)),
                _plan(),
                deadline=FUTURE,
                retain_representation_evidence=True,
            )
        assert tuple(over_limit_store.staging_root.iterdir()) == ()


def test_abort_closes_representation_evidence_and_removes_owned_staging(
    tmp_path: Path,
) -> None:
    administrator, store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None

    administrator.abort(built)
    administrator.abort(built)

    assert lease.is_closed
    assert not built.staging_directory.exists()
    assert tuple(store.staging_root.iterdir()) == ()


def test_other_administrator_abort_closes_the_built_evidence_lease(tmp_path: Path) -> None:
    first, store, _decoder, _encoders = _administrator(tmp_path)
    built = first.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    producer, _decoder, _encoders = _producer()
    second = PersonalImageGenerationAdministrator(store, producer)

    second.abort(built)

    assert lease.is_closed
    assert not built.staging_directory.exists()
    assert tuple(store.staging_root.iterdir()) == ()


def test_abort_rejects_a_replaced_staging_directory_without_deleting_it(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    built = administrator.build(
        ImageBuildSource(_sources(1)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    lease = built.representation_evidence_lease
    assert lease is not None
    lease.close()
    saved_staging = built.staging_directory.with_name("saved-staging")
    built.staging_directory.rename(saved_staging)
    built.staging_directory.mkdir()
    foreign = built.staging_directory / "foreign"
    foreign.write_bytes(b"must-not-be-deleted")

    with pytest.raises(RuntimeError, match="replaced staging"):
        administrator.abort(built)

    assert foreign.read_bytes() == b"must-not-be-deleted"
    assert saved_staging.is_dir()
    foreign.unlink()
    built.staging_directory.rmdir()
    saved_staging.rename(built.staging_directory)
    administrator.abort(built)


def test_owned_cleanup_preserves_a_file_replacement_swapped_at_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / ".staging"
    staging_root.mkdir(mode=0o700)
    staging = staging_root / "owned-staging"
    staging.mkdir(mode=0o700)
    payload = staging / "payload.bin"
    payload.write_bytes(b"owned payload")
    saved_payload = staging / "saved-owned-payload.bin"
    expected_identity = retrieval_build._owned_directory_identity(staging)
    original_rename = os.rename
    swapped = False

    def swap_before_quarantine(
        source: Any,
        destination: Any,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if not swapped and source == payload.name and src_dir_fd is not None:
            swapped = True
            original_rename(payload, saved_payload)
            payload.write_bytes(b"foreign replacement")
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", swap_before_quarantine)

    with pytest.raises(RuntimeError, match="staging file identity changed"):
        retrieval_build._remove_owned_staging(
            staging,
            staging_root,
            expected_identity=expected_identity,
        )

    assert payload.read_bytes() == b"foreign replacement"
    assert saved_payload.read_bytes() == b"owned payload"
    payload.unlink()
    original_rename(saved_payload, payload)
    retrieval_build._remove_owned_staging(
        staging,
        staging_root,
        expected_identity=expected_identity,
    )
    assert not staging.exists()
    assert tuple(staging_root.iterdir()) == ()


def test_owned_cleanup_is_serialized_within_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / ".staging"
    staging_root.mkdir(mode=0o700)
    staging = staging_root / "owned-staging"
    staging.mkdir(mode=0o700)
    (staging / "payload.bin").write_bytes(b"owned payload")
    expected_identity = retrieval_build._owned_directory_identity(staging)
    original_remove = retrieval_build._remove_directory_contents
    entered = Event()
    release = Event()
    second_done = Event()
    calls = 0
    errors: list[BaseException] = []

    def delayed_remove(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(1)
        original_remove(descriptor)

    monkeypatch.setattr(retrieval_build, "_remove_directory_contents", delayed_remove)

    def clean(*, signal_done: bool = False) -> None:
        try:
            retrieval_build._remove_owned_staging(
                staging,
                staging_root,
                expected_identity=expected_identity,
            )
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            if signal_done:
                second_done.set()

    first = Thread(target=clean)
    second = Thread(target=lambda: clean(signal_done=True))
    first.start()
    assert entered.wait(1)
    second.start()
    assert not second_done.wait(0.05)
    release.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == 1
    assert not staging.exists()
    assert tuple(staging_root.iterdir()) == ()


def test_abort_preserves_a_staging_replacement_swapped_at_final_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    administrator, _store, built = _build(tmp_path, count=1)
    staging = built.staging_directory
    saved_staging = staging.with_name("saved-owned-staging")
    foreign = staging / "foreign"
    original_rename = os.rename
    swapped = False

    def swap_before_quarantine(
        source: Any,
        destination: Any,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        if not swapped and source == staging.name and src_dir_fd is not None:
            swapped = True
            original_rename(staging, saved_staging)
            staging.mkdir()
            foreign.write_bytes(b"must-not-be-deleted")
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(os, "rename", swap_before_quarantine)

    with pytest.raises(RuntimeError, match="replacement was preserved"):
        administrator.abort(built)

    assert foreign.read_bytes() == b"must-not-be-deleted"
    assert saved_staging.is_dir()
    foreign.unlink()
    staging.rmdir()
    original_rename(saved_staging, staging)
    administrator.abort(built)


def test_build_rejects_an_identity_matching_lease_from_other_staging(tmp_path: Path) -> None:
    administrator, _store, _decoder, _encoders = _administrator(tmp_path)
    first = administrator.build(
        ImageBuildSource(_sources(2)),
        _plan(),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    second = administrator.build(
        ImageBuildSource(_sources(2)),
        replace(_plan(), build_id="build-2"),
        deadline=FUTURE,
        retain_representation_evidence=True,
    )
    first_lease = first.representation_evidence_lease
    second_lease = second.representation_evidence_lease
    assert first_lease is not None
    assert second_lease is not None
    assert first_lease.identity == second_lease.identity

    with pytest.raises(ValueError, match="does not belong to the build staging"):
        replace(first, representation_evidence_lease=second_lease)

    first_lease.close()
    second_lease.close()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"build_id": ""}, "build_id"),
        ({"activation_set_id": "../set"}, "activation_set_id"),
        ({"workspace_id": "/workspace"}, "workspace_id"),
        ({"corpus_ref": "bad\x00ref"}, "corpus_ref"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 1_001}, "batch_size"),
    ],
)
def test_build_plan_rejects_unsafe_or_unbounded_values(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "build_id": "build-1",
        "activation_set_id": "set-1",
        "workspace_id": "workspace-1",
        "corpus_ref": "photos",
        "batch_size": 2,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PersonalImageBuildPlan(**values)  # type: ignore[arg-type]


def test_administrator_and_entrypoints_reject_wrong_types_and_deadlines(
    tmp_path: Path,
) -> None:
    producer, _decoder, _encoders = _producer()
    store = FilesystemGenerationStore(tmp_path / "generations")
    administrator = PersonalImageGenerationAdministrator(store, producer)
    source = ImageBuildSource(_sources(1))
    past = datetime.now(UTC) - timedelta(seconds=1)

    with pytest.raises(TypeError, match="FilesystemGenerationStore"):
        PersonalImageGenerationAdministrator(object(), producer)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ImageTargetRepresentationProducer"):
        PersonalImageGenerationAdministrator(store, object())  # type: ignore[arg-type]
    for invalid_capacity in (True, 0, 10_001):
        with pytest.raises(ValueError, match="max_units"):
            PersonalImageGenerationAdministrator(
                store,
                producer,
                max_units=invalid_capacity,
            )
    with pytest.raises(ValueError, match="timezone-aware"):
        administrator.build(source, _plan(), deadline=datetime.now())
    with pytest.raises(TimeoutError, match="deadline"):
        administrator.build(source, _plan(), deadline=past)
    with pytest.raises(TypeError, match="ImageBuildSource"):
        administrator.build(object(), _plan(), deadline=FUTURE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PersonalImageBuildPlan"):
        administrator.build(source, object(), deadline=FUTURE)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BuiltPersonalImageGeneration"):
        administrator.validate(object(), deadline=FUTURE)  # type: ignore[arg-type]


def test_semantic_index_build_fails_closed_without_identity() -> None:
    with pytest.raises(RuntimeError, match="requires a semantic identity"):
        retrieval_build._require_semantic_identity(None)


def test_validate_rejects_foreign_or_tampered_staging_without_publication(
    tmp_path: Path,
) -> None:
    administrator, store, built = _build(tmp_path / "first")
    other, _other_store, _decoder, _encoders = _administrator(tmp_path / "other")

    with pytest.raises(ValueError, match="does not belong"):
        other.validate(built, deadline=FUTURE)

    manifest_path = built.staging_directory.joinpath(
        *built.representation_result.sealed_manifest.relative_path.parts
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"checksum|size"):
        administrator.validate(built, deadline=FUTURE)
    assert not (store.root / built.staged_manifest.activation_set_id).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: replace(source, ordinal=1),
        lambda source: replace(
            source,
            source=ImageSource(data=source.source.data, media_type="application/octet-stream"),
        ),
    ],
)
def test_image_build_source_rejects_noncanonical_source_metadata(
    mutation: object,
) -> None:
    source = _sources(1)[0]

    with pytest.raises(ValueError, match=r"ordinal|image media_type"):
        ImageBuildSource((mutation(source),))  # type: ignore[operator]


def test_image_build_source_rejects_empty_duplicate_mixed_or_invalid_inputs() -> None:
    source_a, source_b = _sources(2)
    duplicate = replace(source_b, unit=replace(source_b.unit, unit_id=source_a.unit.unit_id))
    mixed = replace(
        source_b,
        unit=replace(
            source_b.unit,
            processing_fingerprint=ProcessingFingerprint("f" * 64),
        ),
    )

    with pytest.raises(ValueError, match="at least one"):
        ImageBuildSource(())
    with pytest.raises(TypeError, match="ImageRepresentationSource"):
        ImageBuildSource((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unique"):
        ImageBuildSource((source_a, duplicate))
    with pytest.raises(ValueError, match="fingerprint"):
        ImageBuildSource((source_a, mixed))
    with pytest.raises(ValueError, match="capacity"):
        ImageBuildSource((source_a, source_b), max_units=1)
    for invalid_capacity in (True, 0, 10_001):
        with pytest.raises(ValueError, match="max_units"):
            ImageBuildSource((source_a,), max_units=invalid_capacity)


def test_build_summary_and_source_lookup_validate_boundary_values() -> None:
    source = ImageBuildSource(_sources(1))

    with pytest.raises(ValueError, match="unit_count"):
        ImageBuildSummary(0, FINGERPRINT, "a" * 64)
    with pytest.raises(TypeError, match="ProcessingFingerprint"):
        ImageBuildSummary(1, object(), "a" * 64)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SHA-256"):
        ImageBuildSummary(1, FINGERPRINT, "A" * 64)
    with pytest.raises(TypeError, match="integer"):
        source.source_at(True)
    with pytest.raises(ValueError, match="outside"):
        source.source_at(-1)
    with pytest.raises(ValueError, match="outside"):
        source.source_at(1)


def test_built_generation_value_rejects_mismatched_evidence_and_validators(
    tmp_path: Path,
) -> None:
    _administrator_value, _store, built = _build(tmp_path)

    with pytest.raises(ValueError, match="absolute"):
        replace(built, staging_directory=Path("relative"))
    with pytest.raises(TypeError, match="StagedGenerationSetManifest"):
        replace(built, staged_manifest=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RepresentationProductionResult"):
        replace(built, representation_result=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evidence"):
        replace(
            built,
            staged_manifest=replace(built.staged_manifest, checkpoint_sha256="f" * 64),
        )
    with pytest.raises(TypeError, match="native_validators"):
        replace(built, native_validators={})
    with pytest.raises(TypeError, match="native_validators"):
        replace(
            built,
            native_validators={
                object(): next(iter(built.native_validators.values()))  # type: ignore[dict-item]
            },
        )


def test_representation_sink_constructor_rejects_unowned_or_unbounded_storage(
    tmp_path: Path,
) -> None:
    source = ImageBuildSource(_sources(2))
    staging = tmp_path / "staging"
    staging.mkdir()
    link = tmp_path / "staging-link"
    link.symlink_to(staging, target_is_directory=True)

    with pytest.raises(ValueError, match="absolute"):
        PersonalImageRepresentationSink(Path("relative"), source)
    with pytest.raises(ValueError, match="existing non-symlink"):
        PersonalImageRepresentationSink(tmp_path / "missing", source)
    with pytest.raises(ValueError, match="existing non-symlink"):
        PersonalImageRepresentationSink(link, source)
    with pytest.raises(TypeError, match="ImageBuildSource"):
        PersonalImageRepresentationSink(staging, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capacity"):
        PersonalImageRepresentationSink(staging, source, max_units=1)
    with pytest.raises(ValueError, match="max_units"):
        PersonalImageRepresentationSink(staging, source, max_units=True)


def test_representation_sink_rejects_stale_ledger_and_mismatched_rows(
    tmp_path: Path,
) -> None:
    sources = _sources(2)
    source = ImageBuildSource(sources)

    def new_sink(name: str) -> PersonalImageRepresentationSink:
        staging = tmp_path / name
        staging.mkdir()
        return PersonalImageRepresentationSink(staging, source)

    empty = new_sink("empty")
    with pytest.raises(ValueError, match="non-empty tuple"):
        empty.append_batch(
            (),
            unit_manifest_sha256="d" * 64,
            previous_checkpoint=None,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        empty.append_batch(
            (_row(sources[0]),),
            unit_manifest_sha256="D" * 64,
            previous_checkpoint=None,
            deadline=FUTURE,
        )
    with pytest.raises(TypeError, match="ImageTargetRepresentation"):
        empty.append_batch(
            (object(),),  # type: ignore[arg-type]
            unit_manifest_sha256="d" * 64,
            previous_checkpoint=None,
            deadline=FUTURE,
        )

    mismatches = (
        (_row(sources[0], ordinal=1), "ordinals"),
        (_row(sources[0], unit_id="other-unit"), "unit_id"),
        (_row(sources[0], content_sha256="e" * 64), "content hash"),
        (_row(sources[0], media_type="image/jpeg"), "media_type"),
    )
    for index, (row, message) in enumerate(mismatches):
        sink = new_sink(f"mismatch-{index}")
        with pytest.raises(ValueError, match=message):
            sink.append_batch(
                (row,),
                unit_manifest_sha256="d" * 64,
                previous_checkpoint=None,
                deadline=FUTURE,
            )

    stale = new_sink("stale")
    receipt = stale.append_batch(
        (_row(sources[0]),),
        unit_manifest_sha256="d" * 64,
        previous_checkpoint=None,
        deadline=FUTURE,
    )
    assert receipt.checkpoint == stale.resume_checkpoint()
    with pytest.raises(ValueError, match="previous_checkpoint"):
        stale.append_batch(
            (_row(sources[1]),),
            unit_manifest_sha256="d" * 64,
            previous_checkpoint=None,
            deadline=FUTURE,
        )


def test_representation_sink_never_writes_into_a_replaced_directory(tmp_path: Path) -> None:
    sources = _sources(2)
    staging = tmp_path / "staging"
    staging.mkdir()
    sink = PersonalImageRepresentationSink(staging, ImageBuildSource(sources))
    first = sink.append_batch(
        (_row(sources[0]),),
        unit_manifest_sha256="d" * 64,
        previous_checkpoint=None,
        deadline=FUTURE,
    )
    representation_directory = staging / "representations"
    saved_directory = staging / "saved-representations"
    representation_directory.rename(saved_directory)
    representation_directory.mkdir()
    foreign = representation_directory / "foreign"
    foreign.write_bytes(b"must-not-be-modified")

    with pytest.raises(ValueError, match="directory identity changed"):
        sink.append_batch(
            (_row(sources[1]),),
            unit_manifest_sha256="d" * 64,
            previous_checkpoint=first.checkpoint,
            deadline=FUTURE,
        )

    assert tuple(representation_directory.iterdir()) == (foreign,)
    assert foreign.read_bytes() == b"must-not-be-modified"


def test_representation_sink_seal_is_complete_idempotent_and_cleanup_safe(
    tmp_path: Path,
) -> None:
    source = ImageBuildSource(_sources(2))
    producer, _decoder, _encoders = _producer()
    staging = tmp_path / "complete"
    staging.mkdir()
    sink = PersonalImageRepresentationSink(staging, source)

    with pytest.raises(RuntimeError, match="unavailable"):
        sink.representation_rows()
    with pytest.raises(RuntimeError, match="unsealed"):
        sink.cleanup_transient_payloads()
    with pytest.raises(TypeError, match="RepresentationSealRequest"):
        sink.seal(object(), deadline=FUTURE)  # type: ignore[arg-type]

    result = producer.produce(
        source,
        sink=sink,
        batch_size=2,
        unit_manifest_sha256="d" * 64,
        deadline=FUTURE,
    )
    request = RepresentationSealRequest(result.report, result.sealed_manifest.manifest)
    assert sink.seal(request, deadline=FUTURE) == result.sealed_manifest
    assert sink.representation_rows() == tuple(_row(item) for item in _sources(2))
    with pytest.raises(RuntimeError, match="sealed"):
        sink.append_batch(
            (_row(_sources(2)[0]),),
            unit_manifest_sha256="d" * 64,
            previous_checkpoint=sink.resume_checkpoint(),
            deadline=FUTURE,
        )

    different = replace(
        request,
        logical_manifest=replace(
            request.logical_manifest,
            ordered_row_digest_sha256="e" * 64,
        ),
    )
    with pytest.raises(ValueError, match="different manifest"):
        sink.seal(different, deadline=FUTURE)

    sink.cleanup_transient_payloads()
    assert tuple((staging / "representations").iterdir()) == (
        staging / "representations" / "target-representation-manifest.json",
    )


def test_representation_sink_refuses_empty_incomplete_or_unbound_seals(
    tmp_path: Path,
) -> None:
    sources = _sources(2)
    source = ImageBuildSource(sources)
    producer, _decoder, _encoders = _producer()
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_sink = PersonalImageRepresentationSink(reference_root, source)
    result = producer.produce(
        source,
        sink=reference_sink,
        batch_size=2,
        unit_manifest_sha256="d" * 64,
        deadline=FUTURE,
    )
    request = RepresentationSealRequest(result.report, result.sealed_manifest.manifest)

    empty_root = tmp_path / "empty-seal"
    empty_root.mkdir()
    empty = PersonalImageRepresentationSink(empty_root, source)
    with pytest.raises(ValueError, match="without representation rows"):
        empty.seal(request, deadline=FUTURE)

    partial_root = tmp_path / "partial-seal"
    partial_root.mkdir()
    partial = PersonalImageRepresentationSink(partial_root, source)
    partial.append_batch(
        (_row(sources[0]),),
        unit_manifest_sha256="d" * 64,
        previous_checkpoint=None,
        deadline=FUTURE,
    )
    with pytest.raises(ValueError, match="complete frozen source"):
        partial.seal(request, deadline=FUTURE)

    mismatch_root = tmp_path / "checkpoint-seal"
    mismatch_root.mkdir()
    mismatch = PersonalImageRepresentationSink(mismatch_root, source)
    mismatch.append_batch(
        tuple(_row(item) for item in sources),
        unit_manifest_sha256="d" * 64,
        previous_checkpoint=None,
        deadline=FUTURE,
    )
    mismatched_request = replace(
        request,
        report=replace(request.report, final_checkpoint_sha256="f" * 64),
    )
    with pytest.raises(ValueError, match="durable checkpoint"):
        mismatch.seal(mismatched_request, deadline=FUTURE)


def test_native_validators_reject_wrong_owner_or_lane_binding(tmp_path: Path) -> None:
    _administrator_value, _store, built = _build(tmp_path)
    manifest = built.staged_manifest
    payloads = {payload.role: payload for payload in manifest.payloads}
    for key, validator in built.native_validators.items():
        with pytest.raises(TypeError, match="validator requires"):
            validator.validate(
                built.staging_directory,
                payloads[key.role],
                object(),  # type: ignore[arg-type]
                deadline=FUTURE,
            )

    catalog_validator = next(
        validator
        for key, validator in built.native_validators.items()
        if key.role == manifest.unit_catalog.payload_ref
    )
    with pytest.raises(ValueError, match="catalog payload"):
        catalog_validator.validate(
            built.staging_directory,
            payloads[manifest.unit_catalog.payload_ref],
            replace(manifest.unit_catalog, unit_count=manifest.unit_catalog.unit_count + 1),
            deadline=FUTURE,
        )

    lane = manifest.lanes[0]
    parameter_role = lane.index_parameters_ref.payload_ref
    parameter_validator = built.native_validators[
        next(key for key in built.native_validators if key.role == parameter_role)
    ]
    with pytest.raises(ValueError, match="parameter payload"):
        parameter_validator.validate(
            built.staging_directory,
            payloads[parameter_role],
            replace(
                lane,
                index_parameters_ref=replace(lane.index_parameters_ref, payload_ref="other"),
            ),
            deadline=FUTURE,
        )

    shard_role = lane.payload_refs[0]
    index_validator = built.native_validators[
        next(key for key in built.native_validators if key.role == shard_role)
    ]
    with pytest.raises(ValueError, match="index payload"):
        index_validator.validate(
            built.staging_directory,
            payloads[shard_role],
            replace(lane, payload_refs=("other",)),
            deadline=FUTURE,
        )


def test_payload_helpers_fail_closed_for_escape_tamper_and_unsafe_cleanup(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    payload = staging / "payload.bin"
    payload.write_bytes(b"payload")

    with pytest.raises(ValueError, match="inside"):
        retrieval_build._payload_ref(
            staging,
            tmp_path / "outside.bin",
            role="payload",
            expected_sha256=hashlib.sha256(b"payload").hexdigest(),
            expected_size=7,
        )
    with pytest.raises(ValueError, match="identity"):
        retrieval_build._payload_ref(
            staging,
            payload,
            role="payload",
            expected_sha256="f" * 64,
            expected_size=7,
        )

    reference = PayloadFileRef(
        relative_path=PurePosixPath("payload.bin"),
        sha256="f" * 64,
        size_bytes=7,
        role="payload",
    )
    with pytest.raises(ValueError, match="checksum"):
        retrieval_build._verified_payload_path(staging, reference)
    with pytest.raises(ValueError, match="size limit"):
        retrieval_build._read_verified_payload(
            staging,
            replace(reference, sha256=hashlib.sha256(b"payload").hexdigest()),
            maximum_bytes=6,
        )
    with pytest.raises(ValueError, match="checksum"):
        retrieval_build._read_verified_payload(
            staging,
            reference,
            maximum_bytes=64,
        )
    with pytest.raises(RuntimeError, match="refusing"):
        retrieval_build._remove_owned_staging(
            staging,
            tmp_path / "other",
            expected_identity=retrieval_build._owned_directory_identity(staging),
        )


def test_private_file_writers_reject_short_writes_and_leave_no_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = os.write
    calls = 0

    def short_write(descriptor: int, payload: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0
        return original_write(descriptor, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(OSError, match="no progress"):
        retrieval_build._write_private_payload(tmp_path / "build.bin", b"payload")

    calls = 0
    representation_root = tmp_path / "representation-build"
    representation_root.mkdir()
    directory = retrieval_representations._BoundEvidenceDirectory.create(representation_root)
    try:
        with pytest.raises(OSError, match="no progress"):
            retrieval_representations._write_private_file(
                directory,
                PurePosixPath("representations", "rows.json"),
                b"payload",
            )
    finally:
        directory.close()
