"""Evidence, resume and failure tests for offline preparation."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.processing.video.models import AcceptedVideoSource, VideoSource
from capicove.processing.video.retrieval._adapters.local import (
    LocalContentStore,
    LocalDerivedCacheCatalog,
    MetricSceneExtractor,
)
from capicove.processing.video.retrieval.models import (
    FrameObservation,
    RetrievalKeyframe,
    VideoDecodeObservation,
    VideoRetrievalPreparationRequest,
)
from capicove.processing.video.retrieval.pipeline import VideoRetrievalPreparationService
from tests.unit.processing.video.retrieval.test_text_units import transcript


def request() -> VideoRetrievalPreparationRequest:
    fp = ProcessingFingerprint("b" * 64)
    return VideoRetrievalPreparationRequest(
        VideoSource(data=b"video"),
        "artifact:v",
        "evidence:v",
        "representation:asr",
        "run:1",
        "profile:1",
        fp,
        fp,
        fp,
        datetime.now(UTC) + timedelta(minutes=1),
    )


class Acceptor:
    def accept(self, source: VideoSource, *, deadline: datetime) -> AcceptedVideoSource:
        assert source.data is not None
        return AcceptedVideoSource(
            "source", sha256(source.data).hexdigest(), "video/mp4", data=source.data
        )


class Decoder:
    fingerprint = ProcessingFingerprint("c" * 64)
    calls = 0

    def __init__(self, store: LocalContentStore) -> None:
        self.store = store

    def decode(self, source: AcceptedVideoSource, *, deadline: datetime) -> VideoDecodeObservation:
        self.calls += 1
        ref = self.store.put(b"preview")
        return VideoDecodeObservation(
            4000, (FrameObservation(1000, 0.8, ref), FrameObservation(3000, 0.9, ref)), ()
        )


class Encoder:
    calls = 0
    embedding_space = EmbeddingSpaceRef(
        "joint",
        "local",
        "revision",
        "preprocess",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )

    def encode_keyframes(
        self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
    ) -> tuple[tuple[float, ...], ...]:
        self.calls += 1
        return tuple((1.0, 0.0) for _ in keyframes)


def setup_service(
    tmp_path: Path,
) -> tuple[VideoRetrievalPreparationService, Decoder, Encoder, LocalContentStore]:
    store = LocalContentStore(tmp_path / "objects")
    decoder, encoder = Decoder(store), Encoder()
    service = VideoRetrievalPreparationService(
        source_acceptor=Acceptor(),
        decode_pass=decoder,
        scene_extractor=MetricSceneExtractor(),
        target_encoder=encoder,
        preview_store=store,
        manifest_store=store,
        cache_catalog=LocalDerivedCacheCatalog(tmp_path / "cache"),
    )
    return service, decoder, encoder, store


def test_prepare_and_persistent_decode_reuse(tmp_path: Path) -> None:
    service, decoder, encoder, store = setup_service(tmp_path)
    first = service.prepare(request())
    encoder.embedding_space = replace(encoder.embedding_space, model_revision="revision2")
    second = service.prepare(request())
    assert decoder.calls == 1
    assert encoder.calls == 2
    assert first.keyframes == second.keyframes
    assert first.unit_manifest_sha256 != second.unit_manifest_sha256
    assert first.segments[0].keyframe_unit_ids == (first.keyframes[0].unit_id,)
    assert (
        sha256(store.read(first.keyframes[0].preview_resource_ref)).hexdigest()
        == first.keyframes[0].preview_sha256
    )


def test_deadline_and_bad_preview_fail(tmp_path: Path) -> None:
    service, _decoder, _, store = setup_service(tmp_path)
    with pytest.raises(TimeoutError):
        service.prepare(replace(request(), deadline=datetime.now(UTC) - timedelta(seconds=1)))
    first = service.prepare(request())
    (store.root / first.keyframes[0].preview_sha256).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        service.prepare(request())


def test_changed_source_and_nonfinite_vectors_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _decoder, encoder, _ = setup_service(tmp_path)
    monkeypatch.setattr(
        encoder,
        "encode_keyframes",
        lambda frames, deadline: tuple((float("nan"), 0.0) for _ in frames),
    )
    with pytest.raises(ValueError):
        service.prepare(request())


def test_identical_encoder_is_reused_from_checkpoint(tmp_path: Path) -> None:
    service, decoder, encoder, _ = setup_service(tmp_path)
    first = service.prepare(request())
    second = service.prepare(request())
    assert first == second
    assert decoder.calls == encoder.calls == 1


def test_source_changed_after_decode_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, decoder, _, _ = setup_service(tmp_path)
    path = tmp_path / "source.bin"
    path.write_bytes(b"video")
    accepted = AcceptedVideoSource("source", sha256(b"video").hexdigest(), "video/mp4", path=path)
    monkeypatch.setattr(service._acceptor, "accept", lambda source, deadline: accepted)
    original = decoder.decode

    def changed(source: AcceptedVideoSource, *, deadline: datetime) -> VideoDecodeObservation:
        result = original(source, deadline=deadline)
        path.write_bytes(b"changed")
        return result

    monkeypatch.setattr(decoder, "decode", changed)
    with pytest.raises(ValueError, match="source_changed"):
        service.prepare(request())


def test_optional_ocr_unavailable(tmp_path: Path) -> None:
    from capicove.processing.video.ports import BackendUnavailableError
    from capicove.processing.video.retrieval.models import TemporalOcrText

    class Unavailable:
        fingerprint = ProcessingFingerprint("f" * 64)

        def recognize(
            self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
        ) -> tuple[TemporalOcrText, ...]:
            raise BackendUnavailableError("unavailable")

    service, _, _, _ = setup_service(tmp_path)
    service._ocr = Unavailable()
    assert service.prepare(request()).warning_codes == ("ocr_unavailable",)


def test_ocr_provider_fingerprint_invalidates_only_ocr(tmp_path: Path) -> None:
    from capicove.contracts.retrieval import TemporalRange
    from capicove.processing.video.retrieval.models import TemporalOcrText

    class Ocr:
        fingerprint = ProcessingFingerprint("d" * 64)
        calls = 0

        def __init__(self, store: LocalContentStore) -> None:
            self.store = store

        def recognize(
            self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
        ) -> tuple[TemporalOcrText, ...]:
            self.calls += 1
            ref = self.store.put(b"text")
            evidence = self.store.put(self.fingerprint.sha256.encode())
            return (
                TemporalOcrText(
                    "text",
                    TemporalRange(keyframes[0].timestamp_ms, keyframes[0].timestamp_ms + 1),
                    evidence,
                    ref,
                ),
            )

    service, decoder, encoder, store = setup_service(tmp_path)
    ocr = Ocr(store)
    service._ocr = ocr
    first = service.prepare(request())
    assert service.prepare(request()) == first
    assert ocr.calls == 1
    ocr.fingerprint = ProcessingFingerprint("e" * 64)
    second = service.prepare(request())
    assert ocr.calls == 2
    assert decoder.calls == encoder.calls == 1
    assert first.unit_manifest_sha256 != second.unit_manifest_sha256


def test_incompatible_checkpoint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, decoder, _, store = setup_service(tmp_path)
    reference = store.put(b'{"fingerprint":"wrong","payload":{}}')
    monkeypatch.setattr(service._cache, "lookup", lambda fingerprint: reference)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        service.prepare(request())
    assert decoder.calls == 0


def test_asr_checkpoint_survives_restart_and_invalidates_identity(tmp_path: Path) -> None:
    from capicove.processing.audio.models import AudioTranscript

    class Asr:
        fingerprint = ProcessingFingerprint("d" * 64)
        calls = 0

        def transcribe(self, source: AcceptedVideoSource, *, deadline: datetime) -> AudioTranscript:
            self.calls += 1
            return transcript()

    asr = Asr()
    service, _, _, _ = setup_service(tmp_path)
    service._asr = asr
    first = service.prepare(request())
    restarted, _, _, _ = setup_service(tmp_path)
    restarted._asr = asr
    assert restarted.prepare(request()) == first
    assert asr.calls == 1
    asr.fingerprint = ProcessingFingerprint("e" * 64)
    changed = restarted.prepare(request())
    assert asr.calls == 2
    assert changed.unit_manifest_sha256 != first.unit_manifest_sha256


def test_unavailable_asr_is_not_cached(tmp_path: Path) -> None:
    from capicove.processing.audio.models import AudioTranscript
    from capicove.processing.video.ports import BackendUnavailableError

    class Asr:
        fingerprint = ProcessingFingerprint("d" * 64)
        calls = 0

        def transcribe(
            self, source: AcceptedVideoSource, *, deadline: datetime
        ) -> AudioTranscript | None:
            self.calls += 1
            raise BackendUnavailableError("not installed")

    service, _, _, _ = setup_service(tmp_path)
    asr = Asr()
    service._asr = asr
    assert service.prepare(request()).warning_codes == ("asr_unavailable",)
    assert service.prepare(request()).warning_codes == ("asr_unavailable",)
    assert asr.calls == 2
