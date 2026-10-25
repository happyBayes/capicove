"""Tests for bounded image target-representation production."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import PurePosixPath

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
from capicove.processing.image import (
    EncodedImageVector,
    ImageRepresentationSink,
    ImageRepresentationSource,
    ImageSemanticLane,
    ImageTargetRepresentation,
    ImageTargetRepresentationProducer,
    RepresentationCheckpoint,
    RepresentationPayloadDigest,
    RepresentationSealRequest,
    RepresentationShardReceipt,
    SealedTargetRepresentationManifest,
)
from capicove.processing.image.io import LoadedImage, load_image
from capicove.processing.image.models import ImageSource
from capicove.processing.image.ports import ImageTargetEncoder, SemanticTargetIdentity


FUTURE = datetime.now(UTC) + timedelta(hours=1)
PAST = datetime.now(UTC) - timedelta(seconds=1)
UNIT_MANIFEST_SHA = "d" * 64


def _fingerprint(value: str = "c") -> ProcessingFingerprint:
    return ProcessingFingerprint(value * 64)


def _space(lane: ImageSemanticLane) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=f"{lane.value}-space-v1",
        provider_id=f"{lane.value}-encoder",
        model_revision=("1" if lane is ImageSemanticLane.INSTANCE else "2") * 40,
        preprocessing_version="rgb-exif-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _identity(lane: ImageSemanticLane) -> SemanticTargetIdentity:
    return SemanticTargetIdentity(
        lane=lane,
        space=_space(lane),
        bundle_manifest_sha256=("a" if lane is ImageSemanticLane.INSTANCE else "b") * 64,
    )


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="opencv-phash-rgb-v1",
        provider_id="fake-phash",
        provider_version="1.0",
        preprocessing_version="rgb-exif-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _png_bytes(index: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), (index, 255 - index, index // 2)).save(buffer, "PNG")
    return buffer.getvalue()


def _sources(
    count: int, *, fingerprint: ProcessingFingerprint | None = None
) -> tuple[ImageRepresentationSource, ...]:
    effective_fingerprint = fingerprint or _fingerprint()
    result = []
    for ordinal in range(count):
        payload = _png_bytes(ordinal + 1)
        unit = SearchableUnit(
            unit_id=f"unit-{ordinal}",
            unit_type=TargetType.IMAGE,
            source_hash=hashlib.sha256(payload).hexdigest(),
            artifact_ref=f"artifact-{ordinal}",
            evidence_ref=f"evidence-{ordinal}",
            representation_refs=(f"source-{ordinal}",),
            processing_run_ref="run-v1",
            processing_version="1",
            processing_fingerprint=effective_fingerprint,
        )
        result.append(
            ImageRepresentationSource(
                ordinal=ordinal,
                unit=unit,
                source=ImageSource(data=payload),
            )
        )
    return tuple(result)


class CountingDecoder:
    def __init__(self) -> None:
        self.calls: list[ImageSource] = []

    def __call__(self, source: ImageSource) -> LoadedImage:
        self.calls.append(source)
        return load_image(source)


class FakeHasher:
    provider_id = "fake-phash"
    provider_version = "1.0"
    hash_version = "phash-64-v1"

    def __init__(self, *, offset: int = 0) -> None:
        self.images: list[Image.Image] = []
        self.offset = offset

    def phash(self, image: Image.Image) -> int:
        self.images.append(image)
        pixel = image.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        return int(pixel[0]) + self.offset


class FakeEncoder:
    def __init__(self, identity: SemanticTargetIdentity) -> None:
        self.identity = identity
        self.verified = 0
        self.batches: list[tuple[Image.Image, ...]] = []

    def verify_ready(self, *, deadline: datetime) -> None:
        assert deadline == FUTURE
        self.verified += 1

    def encode_images(
        self,
        images: tuple[Image.Image, ...],
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[tuple[float, ...], ...]:
        assert space == self.identity.space
        assert deadline == FUTURE
        self.batches.append(images)
        value = (1.0, 0.0) if self.identity.lane is ImageSemanticLane.INSTANCE else (0.0, 1.0)
        return tuple(value for _ in images)


class WrongRowCountEncoder(FakeEncoder):
    def encode_images(
        self,
        images: tuple[Image.Image, ...],
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[tuple[float, ...], ...]:
        super().encode_images(images, space, deadline)
        return ()


class RecordingSink:
    def __init__(self, *, fail_on_append: int | None = None) -> None:
        self.checkpoint: RepresentationCheckpoint | None = None
        self.batches: list[tuple[ImageTargetRepresentation, ...]] = []
        self.receipts: list[RepresentationShardReceipt] = []
        self.seal_requests: list[RepresentationSealRequest] = []
        self.append_attempts = 0
        self.fail_on_append = fail_on_append
        self.unit_ids: set[str] = set()

    def resume_checkpoint(self) -> RepresentationCheckpoint | None:
        return self.checkpoint

    def append_batch(
        self,
        batch: tuple[ImageTargetRepresentation, ...],
        *,
        unit_manifest_sha256: str,
        previous_checkpoint: RepresentationCheckpoint | None,
        deadline: datetime,
    ) -> RepresentationShardReceipt:
        self.append_attempts += 1
        if self.fail_on_append == self.append_attempts:
            raise OSError("synthetic append failure")
        assert previous_checkpoint == self.checkpoint
        unit_ids = {item.unit_id for item in batch}
        if len(unit_ids) != len(batch) or unit_ids & self.unit_ids:
            raise ValueError("synthetic sink rejects duplicate unit IDs")
        body = ",".join(str(item.ordinal) for item in batch).encode("ascii")
        payload = RepresentationPayloadDigest(
            role="rows",
            sha256=hashlib.sha256(body).hexdigest(),
            size_bytes=max(1, len(body)),
            first_ordinal=batch[0].ordinal,
            last_ordinal=batch[-1].ordinal,
        )
        shard_manifest_sha256 = hashlib.sha256(b"shard\0" + body).hexdigest()
        checkpoint = RepresentationCheckpoint.advance(
            previous_checkpoint=previous_checkpoint,
            batch=batch,
            unit_manifest_sha256=unit_manifest_sha256,
            shard_manifest_sha256=shard_manifest_sha256,
        )
        receipt = RepresentationShardReceipt(
            shard_id=f"shard-{len(self.receipts)}",
            payloads=(payload,),
            shard_manifest_sha256=shard_manifest_sha256,
            checkpoint=checkpoint,
        )
        self.batches.append(batch)
        self.receipts.append(receipt)
        self.checkpoint = checkpoint
        self.unit_ids.update(unit_ids)
        return receipt

    def seal(
        self,
        request: RepresentationSealRequest,
        *,
        deadline: datetime,
    ) -> SealedTargetRepresentationManifest:
        assert deadline == FUTURE
        assert self.checkpoint is not None
        assert request.report.final_checkpoint_sha256 == self.checkpoint.checkpoint_sha256
        self.seal_requests.append(request)
        return SealedTargetRepresentationManifest.create(
            request.logical_manifest,
            relative_path=PurePosixPath("target-representation-manifest.json"),
        )


def _producer(
    *,
    decoder: CountingDecoder | None = None,
    encoders: dict[ImageSemanticLane, FakeEncoder] | None = None,
    hasher: FakeHasher | None = None,
) -> tuple[
    ImageTargetRepresentationProducer,
    CountingDecoder,
    FakeHasher,
    dict[ImageSemanticLane, FakeEncoder],
]:
    effective_decoder = decoder or CountingDecoder()
    effective_encoders = encoders or {
        ImageSemanticLane.INSTANCE: FakeEncoder(_identity(ImageSemanticLane.INSTANCE)),
        ImageSemanticLane.CONCEPT: FakeEncoder(_identity(ImageSemanticLane.CONCEPT)),
    }
    effective_hasher = hasher or FakeHasher()
    return (
        ImageTargetRepresentationProducer(
            decoder=effective_decoder,
            perceptual_hasher=effective_hasher,
            encoders=effective_encoders,
            hash_space=_hash_space(),
            processing_fingerprint=_fingerprint(),
        ),
        effective_decoder,
        effective_hasher,
        effective_encoders,
    )


def test_immutable_values_require_complete_spaces_and_normalized_vectors() -> None:
    embeddings = (
        EncodedImageVector(_identity(ImageSemanticLane.INSTANCE), (1.0, 0.0)),
        EncodedImageVector(_identity(ImageSemanticLane.CONCEPT), (0.0, 1.0)),
    )
    representation = ImageTargetRepresentation(
        ordinal=0,
        unit_id="unit-0",
        content_sha256="f" * 64,
        media_type="image/png",
        perceptual_hash=7,
        hash_space=_hash_space(),
        embeddings=embeddings,
        processing_fingerprint=_fingerprint(),
    )

    assert tuple(item.identity.space for item in representation.embeddings) == (
        _space(ImageSemanticLane.INSTANCE),
        _space(ImageSemanticLane.CONCEPT),
    )
    with pytest.raises(FrozenInstanceError):
        representation.unit_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="normalized"):
        EncodedImageVector(_identity(ImageSemanticLane.INSTANCE), (1.0, 1.0))
    with pytest.raises(ValueError, match="dimension"):
        EncodedImageVector(_identity(ImageSemanticLane.INSTANCE), (1.0,))
    with pytest.raises(ValueError, match="finite"):
        EncodedImageVector(_identity(ImageSemanticLane.INSTANCE), (float("nan"), 0.0))
    with pytest.raises(ValueError, match="semantic lanes"):
        replace(representation, embeddings=(embeddings[0], embeddings[0]))
    with pytest.raises(ValueError, match="64-bit"):
        replace(representation, perceptual_hash=1 << 64)


def test_encoder_and_sink_ports_are_runtime_checkable() -> None:
    encoder = FakeEncoder(_identity(ImageSemanticLane.INSTANCE))
    sink = RecordingSink()

    assert isinstance(encoder, ImageTargetEncoder)
    assert isinstance(sink, ImageRepresentationSink)


def test_producer_decodes_once_uses_bounded_batches_and_retains_no_media() -> None:
    producer, decoder, hasher, encoders = _producer()
    sink = RecordingSink()

    result = producer.produce(
        _sources(5),
        sink=sink,
        batch_size=2,
        unit_manifest_sha256=UNIT_MANIFEST_SHA,
        deadline=FUTURE,
    )

    assert len(decoder.calls) == 5
    assert [len(batch) for batch in sink.batches] == [2, 2, 1]
    assert [item.ordinal for batch in sink.batches for item in batch] == list(range(5))
    assert result.report.unit_count == 5
    assert result.report.shard_count == 3
    assert (
        result.sealed_manifest.sha256
        == hashlib.sha256(result.sealed_manifest.canonical_bytes).hexdigest()
    )
    flattened = tuple(item for batch in sink.batches for item in batch)
    assert tuple(item.content_sha256 for item in flattened) == tuple(
        source.unit.source_hash for source in _sources(5)
    )
    assert {item.media_type for item in flattened} == {"image/png"}
    assert tuple(item.perceptual_hash for item in flattened) == (1, 2, 3, 4, 5)
    assert all(
        not hasattr(item, "image") and not hasattr(item, "source_bytes") for item in flattened
    )
    for lane in (ImageSemanticLane.INSTANCE, ImageSemanticLane.CONCEPT):
        assert [len(batch) for batch in encoders[lane].batches] == [2, 2, 1]
        encoder_images = tuple(image for batch in encoders[lane].batches for image in batch)
        assert all(
            encoded is hashed for encoded, hashed in zip(encoder_images, hasher.images, strict=True)
        )


def test_mapping_order_never_swaps_instance_and_concept_outputs() -> None:
    forward_encoders = {
        ImageSemanticLane.INSTANCE: FakeEncoder(_identity(ImageSemanticLane.INSTANCE)),
        ImageSemanticLane.CONCEPT: FakeEncoder(_identity(ImageSemanticLane.CONCEPT)),
    }
    reverse_encoders = dict(reversed(tuple(forward_encoders.items())))
    forward, _, _, _ = _producer(encoders=forward_encoders)
    reverse, _, _, _ = _producer(
        encoders={lane: FakeEncoder(encoder.identity) for lane, encoder in reverse_encoders.items()}
    )
    forward_sink = RecordingSink()
    reverse_sink = RecordingSink()

    forward_result = forward.produce(
        _sources(2),
        sink=forward_sink,
        batch_size=2,
        unit_manifest_sha256=UNIT_MANIFEST_SHA,
        deadline=FUTURE,
    )
    reverse_result = reverse.produce(
        _sources(2),
        sink=reverse_sink,
        batch_size=2,
        unit_manifest_sha256=UNIT_MANIFEST_SHA,
        deadline=FUTURE,
    )

    for sink in (forward_sink, reverse_sink):
        assert tuple(item.identity.lane for item in sink.batches[0][0].embeddings) == (
            ImageSemanticLane.INSTANCE,
            ImageSemanticLane.CONCEPT,
        )
        assert tuple(item.values for item in sink.batches[0][0].embeddings) == (
            (1.0, 0.0),
            (0.0, 1.0),
        )
    assert (
        forward_result.sealed_manifest.canonical_bytes
        == reverse_result.sealed_manifest.canonical_bytes
    )


def test_constructor_rejects_incomplete_or_misbound_semantic_targets() -> None:
    instance = FakeEncoder(_identity(ImageSemanticLane.INSTANCE))
    concept = FakeEncoder(_identity(ImageSemanticLane.CONCEPT))
    with pytest.raises(ValueError, match="exactly instance and concept"):
        _producer(encoders={ImageSemanticLane.INSTANCE: instance})
    with pytest.raises(ValueError, match="mapping key"):
        _producer(
            encoders={
                ImageSemanticLane.INSTANCE: instance,
                ImageSemanticLane.CONCEPT: FakeEncoder(_identity(ImageSemanticLane.INSTANCE)),
            }
        )
    duplicate_space = SemanticTargetIdentity(
        lane=ImageSemanticLane.CONCEPT,
        space=instance.identity.space,
        bundle_manifest_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="distinct"):
        _producer(
            encoders={
                ImageSemanticLane.INSTANCE: instance,
                ImageSemanticLane.CONCEPT: FakeEncoder(duplicate_space),
            }
        )
    assert concept.identity.lane is ImageSemanticLane.CONCEPT


def test_source_order_duplicates_and_fingerprint_fail_before_batch_decode() -> None:
    sources = _sources(2)
    producer, decoder, _, _ = _producer()
    with pytest.raises(ValueError, match="contiguous"):
        producer.produce(
            (replace(sources[0], ordinal=1),),
            sink=RecordingSink(),
            batch_size=2,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert decoder.calls == []

    duplicate = replace(sources[1], unit=replace(sources[1].unit, unit_id="unit-0"))
    with pytest.raises(ValueError, match="unique"):
        producer.produce(
            (sources[0], duplicate),
            sink=RecordingSink(),
            batch_size=2,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert decoder.calls == []

    cross_batch_producer, cross_batch_decoder, _, _ = _producer()
    cross_batch_sink = RecordingSink()
    with pytest.raises(ValueError, match="duplicate unit IDs"):
        cross_batch_producer.produce(
            (sources[0], duplicate),
            sink=cross_batch_sink,
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert len(cross_batch_decoder.calls) == 2
    assert cross_batch_sink.checkpoint is not None
    assert cross_batch_sink.checkpoint.last_completed_ordinal == 0

    changed = _sources(1, fingerprint=_fingerprint("e"))[0]
    with pytest.raises(ValueError, match="processing fingerprint"):
        producer.produce(
            (changed,),
            sink=RecordingSink(),
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert decoder.calls == []


def test_invalid_encoder_row_count_never_reaches_sink() -> None:
    encoders: dict[ImageSemanticLane, FakeEncoder] = {
        ImageSemanticLane.INSTANCE: WrongRowCountEncoder(_identity(ImageSemanticLane.INSTANCE)),
        ImageSemanticLane.CONCEPT: FakeEncoder(_identity(ImageSemanticLane.CONCEPT)),
    }
    producer, _, _, _ = _producer(encoders=encoders)
    sink = RecordingSink()

    with pytest.raises(ValueError, match="row count"):
        producer.produce(
            _sources(1),
            sink=sink,
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )

    assert sink.append_attempts == 0
    assert sink.checkpoint is None


def test_logical_manifest_is_stable_across_batch_sizes_and_resume() -> None:
    logical_results = []
    physical_roots = []
    for batch_size in (1, 2, 5):
        producer, _, _, _ = _producer()
        sink = RecordingSink()
        result = producer.produce(
            _sources(5),
            sink=sink,
            batch_size=batch_size,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
        logical_results.append(result.sealed_manifest.canonical_bytes)
        physical_roots.append(result.report.representation_payload_set_sha256)

    producer, initial_decoder, _, _ = _producer()
    resumed_sink = RecordingSink(fail_on_append=2)
    with pytest.raises(OSError, match="append failure"):
        producer.produce(
            _sources(5),
            sink=resumed_sink,
            batch_size=2,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert resumed_sink.checkpoint is not None
    assert resumed_sink.checkpoint.last_completed_ordinal == 1
    assert len(initial_decoder.calls) == 4
    resumed_sink.fail_on_append = None
    resumed_producer, resumed_decoder, _, _ = _producer()
    resumed = resumed_producer.produce(
        _sources(5),
        sink=resumed_sink,
        batch_size=2,
        unit_manifest_sha256=UNIT_MANIFEST_SHA,
        deadline=FUTURE,
    )

    assert logical_results[0] == logical_results[1] == logical_results[2]
    assert resumed.sealed_manifest.canonical_bytes == logical_results[0]
    assert len(resumed_decoder.calls) == 3
    assert len(set(physical_roots)) == 3


def test_logical_manifest_changes_when_one_phash_changes() -> None:
    baseline_producer, _, _, _ = _producer()
    changed_producer, _, _, _ = _producer(hasher=FakeHasher(offset=1))

    baseline = baseline_producer.produce(
        _sources(2),
        sink=RecordingSink(),
        batch_size=2,
        unit_manifest_sha256=UNIT_MANIFEST_SHA,
        deadline=FUTURE,
    )
    changed = changed_producer.produce(
        _sources(2),
        sink=RecordingSink(),
        batch_size=2,
        unit_manifest_sha256=UNIT_MANIFEST_SHA,
        deadline=FUTURE,
    )

    assert baseline.sealed_manifest.sha256 != changed.sealed_manifest.sha256


def test_resume_rejects_changed_manifest_or_target_identity_before_decode() -> None:
    producer, _, _, encoders = _producer()
    sink = RecordingSink(fail_on_append=2)
    with pytest.raises(OSError):
        producer.produce(
            _sources(4),
            sink=sink,
            batch_size=2,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )

    changed_decoder = CountingDecoder()
    changed_producer, _, _, _ = _producer(decoder=changed_decoder)
    with pytest.raises(ValueError, match="unit manifest"):
        changed_producer.produce(
            _sources(4),
            sink=sink,
            batch_size=2,
            unit_manifest_sha256="e" * 64,
            deadline=FUTURE,
        )
    assert changed_decoder.calls == []

    encoders[ImageSemanticLane.CONCEPT].identity = SemanticTargetIdentity(
        lane=ImageSemanticLane.CONCEPT,
        space=replace(_space(ImageSemanticLane.CONCEPT), space_id="changed-space"),
        bundle_manifest_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="identity changed"):
        producer.produce(
            _sources(4),
            sink=sink,
            batch_size=2,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )


def test_deadline_and_failures_never_advance_past_a_durable_receipt() -> None:
    producer, decoder, _, _ = _producer()
    sink = RecordingSink()
    with pytest.raises(TimeoutError, match="deadline"):
        producer.produce(
            _sources(1),
            sink=sink,
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=PAST,
        )
    assert decoder.calls == []
    assert sink.checkpoint is None

    failing_producer, failing_decoder, _, _ = _producer()
    failing_sink = RecordingSink(fail_on_append=2)
    with pytest.raises(OSError):
        failing_producer.produce(
            _sources(5),
            sink=failing_sink,
            batch_size=2,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert len(failing_decoder.calls) == 4
    assert failing_sink.checkpoint is not None
    assert failing_sink.checkpoint.last_completed_ordinal == 1


def test_source_and_sink_identity_mismatches_fail_closed() -> None:
    producer, _, _, _ = _producer()
    sink = RecordingSink()
    source = _sources(1)[0]
    mismatched_unit = replace(source.unit, source_hash="f" * 64)
    with pytest.raises(ValueError, match="source hash"):
        producer.produce(
            (replace(source, unit=mismatched_unit),),
            sink=sink,
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert sink.checkpoint is None

    valid_producer, _, _, _ = _producer()
    valid_sink = RecordingSink(fail_on_append=2)
    with pytest.raises(OSError):
        valid_producer.produce(
            _sources(3),
            sink=valid_sink,
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
    assert valid_sink.checkpoint is not None
    tampered = valid_sink.checkpoint
    object.__setattr__(tampered, "ordered_row_digest_state_sha256", "0" * 64)
    valid_sink.checkpoint = tampered
    with pytest.raises(ValueError, match="checkpoint digest"):
        valid_producer.produce(
            _sources(3),
            sink=valid_sink,
            batch_size=1,
            unit_manifest_sha256=UNIT_MANIFEST_SHA,
            deadline=FUTURE,
        )
