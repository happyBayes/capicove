"""Integration test for production personal image-generation administration."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from io import BytesIO

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
    ImageRepresentationSource,
    ImageTargetRepresentationProducer,
)
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval._adapters.sqlite_image_generations import PublishedGenerationRecord
from capicove.retrieval.image_generations import TARGET_REPRESENTATION_MANIFEST_ROLE
from capicove.retrieval.search_models import RetrievalLane
from capicove.runtime.retrieval_build import (
    PersonalImageBuildPlan,
    PersonalImageGenerationAdministrator,
)
from capicove.runtime.retrieval_personal import PersonalGenerationHandleFactory
from capicove.runtime.retrieval_representations import ImageBuildSource


FUTURE = datetime.now(UTC) + timedelta(hours=1)
FINGERPRINT = ProcessingFingerprint("d" * 64)


def _png_bytes(index: int) -> bytes:
    payload = BytesIO()
    Image.new("RGB", (3, 2), (index * 30, 20, 10)).save(payload, "PNG")
    return payload.getvalue()


def _space(lane: ImageSemanticLane) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-production-space-v1",
        provider_id=f"{lane.value}-synthetic-provider",
        model_revision="revision-1",
        preprocessing_version="rgb-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


class _Hasher:
    provider_id = "synthetic-phash"
    provider_version = "1"
    hash_version = "phash-64-v1"

    def phash(self, image: Image.Image) -> int:
        pixel = image.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        return int(pixel[0])


class _Encoder:
    def __init__(self, lane: ImageSemanticLane) -> None:
        self.identity = SemanticTargetIdentity(
            lane=lane,
            space=_space(lane),
            bundle_manifest_sha256=("e" if lane is ImageSemanticLane.INSTANCE else "f") * 64,
        )
        self.ready_calls = 0

    def verify_ready(self, *, deadline: datetime) -> None:
        assert deadline == FUTURE
        self.ready_calls += 1

    def encode_images(
        self,
        images: tuple[Image.Image, ...],
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[tuple[float, ...], ...]:
        assert space == self.identity.space
        assert deadline == FUTURE
        rows = []
        for index, _image in enumerate(images):
            rows.append((1.0, 0.0) if index % 2 == 0 else (0.0, 1.0))
        return tuple(rows)


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="synthetic-phash-space-v1",
        provider_id="synthetic-phash",
        provider_version="1",
        preprocessing_version="rgb-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _build_source() -> ImageBuildSource:
    sources = []
    for ordinal in range(3):
        payload = _png_bytes(ordinal + 1)
        sources.append(
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
    return ImageBuildSource(sources)


def _decoder(source: ImageSource) -> LoadedImage:
    return load_image(source)


def test_build_validate_publish_is_inert_and_factory_reopens_complete_set(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    store = FilesystemGenerationStore(root / "generations")
    instance = _Encoder(ImageSemanticLane.INSTANCE)
    concept = _Encoder(ImageSemanticLane.CONCEPT)
    producer = ImageTargetRepresentationProducer(
        decoder=_decoder,
        perceptual_hasher=_Hasher(),
        encoders={
            ImageSemanticLane.INSTANCE: instance,
            ImageSemanticLane.CONCEPT: concept,
        },
        hash_space=_hash_space(),
        processing_fingerprint=FINGERPRINT,
    )
    administrator = PersonalImageGenerationAdministrator(store, producer)
    plan = PersonalImageBuildPlan(
        build_id="synthetic-build",
        activation_set_id="generation-set-1",
        workspace_id="workspace-1",
        corpus_ref="photos",
        batch_size=2,
    )

    built = administrator.build(_build_source(), plan, deadline=FUTURE)

    assert built.staged_manifest.unit_catalog.unit_count == 3
    assert built.staged_manifest.target_representation_manifest_sha256 == (
        built.representation_result.sealed_manifest.sha256
    )
    staged_representation_path = (
        built.staging_directory / "representations" / "target-representation-manifest.json"
    )
    assert staged_representation_path.read_bytes() == (
        built.representation_result.sealed_manifest.canonical_bytes
    )
    assert tuple((built.staging_directory / "representations").iterdir()) == (
        staged_representation_path,
    )
    assert not (store.root / plan.activation_set_id).exists()
    assert instance.ready_calls == 2
    assert concept.ready_calls == 2

    validated = administrator.validate(built, deadline=FUTURE)
    assert not (store.root / plan.activation_set_id).exists()

    published = administrator.publish(validated, deadline=FUTURE)
    assert published.root == store.root / plan.activation_set_id
    assert published.root.is_dir()
    representation_payload = next(
        payload
        for payload in published.manifest.payloads
        if payload.role == TARGET_REPRESENTATION_MANIFEST_ROLE
    )
    published_representation_path = published.root.joinpath(
        *representation_payload.relative_path.parts
    )
    assert published_representation_path.read_bytes() == (
        built.representation_result.sealed_manifest.canonical_bytes
    )
    assert hashlib.sha256(published_representation_path.read_bytes()).hexdigest() == (
        published.manifest.target_representation_manifest_sha256
    )

    handles = PersonalGenerationHandleFactory(store).open(
        PublishedGenerationRecord(
            activation_set_id=published.manifest.activation_set_id,
            manifest_sha256=published.manifest_sha256,
            published_root=published.root,
            workspace_id=published.manifest.workspace_id,
            corpus_ref=published.manifest.corpus_ref,
            retrieval_profile_ref=published.manifest.retrieval_profile_ref,
            published_at=datetime.now(UTC),
        ),
        deadline=FUTURE,
    )
    try:
        assert handles.unit_catalog.unit_count == 3
        assert set(handles.readers) == {
            RetrievalLane.PERCEPTUAL,
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }
        assert (
            handles.manifest.target_representation_manifest_sha256
            == built.representation_result.sealed_manifest.sha256
        )
    finally:
        handles.close()
