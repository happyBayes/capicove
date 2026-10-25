"""Authored local media proves a real single-pass decode and preview pipeline."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
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
from capicove.processing.video.models import VideoSource
from capicove.processing.video.retrieval._adapters.local import (
    LocalContentStore,
    LocalDerivedCacheCatalog,
    LocalVideoSourceAcceptor,
    MetricSceneExtractor,
)
from capicove.processing.video.retrieval._adapters.providers import PreviewBatchEncoder
from capicove.processing.video.retrieval._adapters.pyav import PyAVVideoDecodePass
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationRequest
from capicove.processing.video.retrieval.pipeline import VideoRetrievalPreparationService


@pytest.fixture
def authored_video() -> bytes:
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    output = BytesIO()
    with av.open(output, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=8000)
        audio.layout = "mono"
        for index in range(40):
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[:, :, 0 if index < 20 else 1] = 220
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        silence = av.AudioFrame.from_ndarray(
            np.zeros((1, 32000), dtype=np.float32), format="fltp", layout="mono"
        )
        silence.sample_rate = 8000
        for packet in audio.encode(silence):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    payload = output.getvalue()
    with av.open(BytesIO(payload)) as container:
        assert len(container.streams.audio) == 1
        assert len(container.streams.video) == 1
    return payload


def test_authored_video_real_decode(tmp_path: Path, authored_video: bytes) -> None:
    store = LocalContentStore(tmp_path / "objects")
    space = EmbeddingSpaceRef(
        "authored",
        "test",
        "v1",
        "v1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )
    encoder = PreviewBatchEncoder(
        store, space, lambda previews: tuple((1.0, 0.0) for _ in previews)
    )
    decoder = PyAVVideoDecodePass(store, sample_interval_ms=500, max_decoded_frames=100)
    service = VideoRetrievalPreparationService(
        source_acceptor=LocalVideoSourceAcceptor(),
        decode_pass=decoder,
        scene_extractor=MetricSceneExtractor(),
        target_encoder=encoder,
        preview_store=store,
        manifest_store=store,
        cache_catalog=LocalDerivedCacheCatalog(tmp_path / "cache"),
    )
    fp = ProcessingFingerprint("b" * 64)
    request = VideoRetrievalPreparationRequest(
        VideoSource(data=authored_video),
        "artifact:test",
        "evidence:test",
        "representation:asr",
        "run:test",
        "profile:test",
        fp,
        fp,
        fp,
        datetime.now(UTC) + timedelta(seconds=30),
    )
    result = service.prepare(request)
    assert len(result.keyframes) == 2
    visual_units = [
        u
        for u in result.searchable_units
        if u.unit_type in {TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}
    ]
    assert len({u.evidence_ref for u in visual_units}) == len(visual_units)
    for unit in visual_units:
        receipt = json.loads(store.read(unit.evidence_ref))
        assert receipt["unit_id"] == unit.unit_id
        assert receipt["source_evidence_ref"] == "evidence:test"
        assert receipt["preview_resource_ref"] in unit.representation_refs
    assert [(s.time_range.start_ms, s.time_range.end_ms) for s in result.segments] == [
        (0, 2000),
        (2000, 4000),
    ]
    assert all(store.read(k.preview_resource_ref).startswith(b"\x89PNG") for k in result.keyframes)
    assert not tuple(tmp_path.rglob("*.mp4"))
    # Exercise the actual unit catalog and Tool evidence expansion. Every visual
    # hit must address exactly its own persisted receipt and selected preview.
    from capicove.contracts import InvocationContext, ToolErrorCode, ToolStatus
    from capicove.retrieval._adapters.sqlite_video_units import (
        SqliteVideoUnitCatalogReader,
        SqliteVideoUnitCatalogWriter,
    )
    from capicove.retrieval.video_evidence import VideoEvidenceReader, video_evidence_bindings
    from capicove.retrieval.video_models import searchable_units_manifest_sha256
    from capicove.tools.evidence import EvidenceToolHandler

    assert result.unit_manifest_sha256 == searchable_units_manifest_sha256(result.searchable_units)
    catalog_path = tmp_path / "catalog.sqlite3"
    SqliteVideoUnitCatalogWriter(catalog_path).publish(
        units=result.searchable_units,
        unit_manifest_sha256=searchable_units_manifest_sha256(result.searchable_units),
        keyframe_to_segments={},
        text_to_segments={},
    )
    handler = EvidenceToolHandler(
        VideoEvidenceReader(
            catalog=SqliteVideoUnitCatalogReader(catalog_path),
            evidence_units=video_evidence_bindings(result.searchable_units),
            read_resource=store.read,
            authorize_resource=lambda context, evidence_id, reference: (
                context.actor_id == "allowed"
            ),
        )
    )
    context = InvocationContext(
        "r",
        "t",
        "workspace",
        "allowed",
        "environment",
        datetime.now(UTC) + timedelta(seconds=10),
        permissions=frozenset({"evidence:read"}),
    )
    for unit in visual_units:
        expanded = handler(
            context,
            {"evidence_id": unit.evidence_ref, "sections": ["preview", "metadata", "lineage"]},
        )
        assert expanded.status is ToolStatus.SUCCEEDED
        assert expanded.result
        assert expanded.result["preview"] == {
            "resource_ref": unit.representation_refs[0],
            "media_type": "image/png",
        }
    assert (
        handler(
            replace(context, actor_id="denied"), {"evidence_id": visual_units[0].evidence_ref}
        ).status
        is ToolStatus.FAILED
    )
    removed = result.keyframes[0]
    (store.root / removed.preview_sha256).unlink()
    selected = next(u for u in visual_units if u.unit_id == removed.unit_id)
    erased = handler(context, {"evidence_id": selected.evidence_ref, "sections": ["preview"]})
    assert erased.status is ToolStatus.FAILED
    assert erased.error and erased.error.code is ToolErrorCode.UNAVAILABLE


def test_decode_capacity_fails_before_full_video(tmp_path: Path, authored_video: bytes) -> None:
    from capicove.processing.video.models import AcceptedVideoSource

    decoder = PyAVVideoDecodePass(LocalContentStore(tmp_path), max_decoded_frames=1)
    with pytest.raises(ValueError, match="frame capacity"):
        decoder.decode(
            AcceptedVideoSource(
                "fixture", sha256(authored_video).hexdigest(), "video/mp4", data=authored_video
            ),
            deadline=datetime.now(UTC) + timedelta(seconds=1),
        )


def test_public_preparation_with_shared_ocr_and_asr(tmp_path: Path, authored_video: bytes) -> None:
    from PIL import Image

    from capicove.contracts.retrieval import TextOrigin
    from capicove.processing.audio.models import (
        AudioBackendLineage,
        AudioTranscript,
        AudioTranscriptionSegment,
    )
    from capicove.processing.image import ImageOcr, OcrWord
    from capicove.processing.image.ocr import OcrProviderOutput
    from capicove.processing.video.models import AcceptedVideoSource
    from capicove.processing.video.retrieval.local import (
        create_local_video_preparation_service,
        read_local_preparation_resource,
    )

    class Ocr:
        provider_id = "authored-ocr"
        provider_version = "1"
        calls = 0

        def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
            self.calls += 1
            return OcrProviderOutput((OcrWord("visible", 0, 0, 8, 8, 0.9, 1, 1, 1, 1, 1),))

    class Asr:
        fingerprint = ProcessingFingerprint("d" * 64)

        def transcribe(self, source: AcceptedVideoSource, *, deadline: datetime) -> AudioTranscript:
            return AudioTranscript(
                "speech",
                "en",
                (AudioTranscriptionSegment(0, 0.0, 1.5, "speech"),),
                (),
                AudioBackendLineage("fixture", "1", "model", "a" * 64, "cpu", "float32"),
            )

    space = EmbeddingSpaceRef(
        "authored",
        "fixture",
        "1",
        "1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )
    ocr = Ocr()
    root = tmp_path / "workspace"
    fp = ProcessingFingerprint("b" * 64)

    def build(languages: tuple[str, ...]) -> VideoRetrievalPreparationService:
        return create_local_video_preparation_service(
            root,
            embedding_space=space,
            encode_batch=lambda previews: tuple((1.0, 0.0) for _ in previews),
            asr_processor=Asr(),
            image_ocr=ImageOcr(ocr),
            ocr_provider_fingerprint=fp,
            ocr_languages=languages,
        )

    request = VideoRetrievalPreparationRequest(
        VideoSource(data=authored_video),
        "artifact:test",
        "evidence:test",
        "representation:asr",
        "run:test",
        "profile:test",
        fp,
        fp,
        fp,
        datetime.now(UTC) + timedelta(seconds=30),
    )
    first = build(("eng",)).prepare(request)
    second = build(("eng",)).prepare(request)
    assert first == second
    assert ocr.calls == 2
    third = build(("chi_sim", "eng")).prepare(request)
    assert ocr.calls == 4
    assert first.unit_manifest_sha256 != third.unit_manifest_sha256
    assert first.keyframes == third.keyframes
    assert first.source_hash == sha256(authored_video).hexdigest()
    text_units = [u for u in first.searchable_units if u.text_locator]
    assert len(text_units) == 3
    for unit in text_units:
        assert unit.text_locator
        assert read_local_preparation_resource(root, unit.representation_refs[0]) in (
            b"visible",
            b"speech",
        )
        if unit.text_locator.origin == TextOrigin.OCR:
            assert unit.text_locator.time_range
            assert unit.text_locator.time_range.end_ms - unit.text_locator.time_range.start_ms == 1
            receipt = json.loads(read_local_preparation_resource(root, unit.evidence_ref))
            assert receipt["recognition"]["languages"] == ["eng"]
        else:
            receipt = json.loads(read_local_preparation_resource(root, unit.evidence_ref))
            assert receipt["transcript"]["lineage"]["model_sha256"] == "a" * 64
    assert all(k.preview_resource_ref.startswith("resource:preview:") for k in first.keyframes)
    with pytest.raises(ValueError, match="authorizer"):
        path = tmp_path / "video.mp4"
        path.write_bytes(authored_video)
        build(("eng",)).prepare(replace(request, source=VideoSource(path=path)))
