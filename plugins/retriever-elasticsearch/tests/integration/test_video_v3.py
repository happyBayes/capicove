"""Opt-in physical temporal-text lifecycle against an isolated TLS cluster."""

import gc
import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4

import pytest
from capicove_retriever_elasticsearch.adapter import ElasticsearchTextRetriever
from capicove_retriever_elasticsearch.codec import encode_text_index_entry
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    SearchRequest,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
    VectorNormalization,
)
from capicove.retrieval import TextIndexEntry, TextIndexGenerationManifest
from capicove.retrieval._adapters.sqlite_video_units import SqliteVideoUnitCatalogWriter
from capicove.retrieval.video_generations import (
    VideoTextMemberValidator,
    open_exact_video_runtime,
    open_video_runtime,
)
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    VideoGenerationSetManifest,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    searchable_units_manifest_sha256,
    visual_entries_content_sha256,
)
from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry
from capicove.retrieval.video_search import VideoQueryEncoder, VideoSearchService


@pytest.mark.skipif(
    os.environ.get("CAPICOVE_TEST_REAL_ELASTICSEARCH") != "1",
    reason="explicit local TLS integration opt-in required",
)
@pytest.mark.parametrize("real_model", (False, True), ids=("fixture-encoder", "siglip2"))
@pytest.mark.parametrize("visual_provider", ("numpy", "faiss"))
def test_native_temporal_ocr_and_asr_roundtrip_and_time_filter(
    tmp_path: Path,
    real_model: bool,
    visual_provider: str,
) -> None:
    if real_model and not os.environ.get("CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"):
        pytest.skip("verified isolated local SigLIP2 model API opt-in required")
    capacity_path = os.environ.get("CAPICOVE_TEST_VIDEO_CAPACITY_VECTORS")
    capacity_vectors = None
    if capacity_path:
        if not real_model or visual_provider != "faiss":
            pytest.skip("capacity probe requires the real-model FAISS combination")
        import numpy as np

        capacity_vectors = np.load(capacity_path, mmap_mode="r", allow_pickle=False)
        if capacity_vectors.shape != (1_000_000, 768):
            raise ValueError(
                "capacity fixture requires exactly one million 768-dimensional vectors"
            )
        tmp_path = Path(os.environ["CAPICOVE_TEST_VIDEO_CAPACITY_WORKDIR"]).resolve()
        tmp_path.mkdir(parents=True, exist_ok=False)
    backend = ElasticsearchTextRetriever(
        ElasticsearchConfiguration(
            hosts=(os.environ["CAPICOVE_VIDEO_ES_ENDPOINT"],),
            ca_certs=os.environ["CAPICOVE_VIDEO_ES_CA"],
            api_key=os.environ.get("CAPICOVE_TEST_ES_API_KEY"),
            index_prefix="capicove-video-text",
            active_alias="capicove-video-text-active",
        )
    )
    entries = tuple(
        TextIndexEntry(
            unit=SearchableUnit(
                unit_id=f"text-{origin.value}",
                unit_type=TargetType.TEXT_CHUNK,
                source_hash="a" * 64,
                artifact_ref="authored-video",
                evidence_ref=f"evidence:{origin.value}",
                representation_refs=(f"representation:{origin.value}",),
                processing_run_ref="run-1",
                processing_version="temporal-text-v1",
                processing_fingerprint=ProcessingFingerprint("b" * 64),
                text_locator=TextChunkLocator(
                    origin, CharacterRange(0, 13), time_range=TemporalRange(start, start + 1000)
                ),
            ),
            text="authored text",
            embeddings=(),
            corpus_refs=("authored-corpus",),
            media_type="video/mp4",
            language="en",
        )
        for origin, start in ((TextOrigin.ASR, 0), (TextOrigin.OCR, 2000))
    )
    lifecycle = backend.generation_lifecycle
    space = EmbeddingSpaceRef(
        "fixture-joint",
        "authored",
        "1",
        "1",
        2,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )
    vectors: tuple[tuple[float, ...], ...] = ((1.0, 0.0), (0.0, 1.0))

    class Encoder:
        def encode(
            self, query: object, *, embedding_space: EmbeddingSpaceRef, deadline: datetime
        ) -> tuple[float, ...]:
            return vectors[0]

    encoder: VideoQueryEncoder = Encoder()
    if real_model:
        from capicove.model import load_packaged_manifest
        from capicove.retrieval.video_http_encoding import LocalSiglip2QueryEncoder

        spec = next(
            item
            for item in load_packaged_manifest().models
            if item.key == "siglip2-base-patch16-224"
        )
        space = replace(
            space,
            space_id="siglip2-local-test",
            provider_id=spec.repository,
            model_revision=spec.revision,
            preprocessing_version=spec.preprocess_revision,
            dimension=768,
        )
        encoder = LocalSiglip2QueryEncoder(
            endpoint=os.environ["CAPICOVE_TEST_VIDEO_MODEL_ENDPOINT"],
            embedding_space=space,
            model_id=spec.repository,
        )
        vectors = tuple(
            encoder.encode(
                TextQuery(text),
                embedding_space=space,
                deadline=datetime.now(UTC) + timedelta(seconds=30),
            )
            for text in ("authored text", "different scene")
        )
    benchmark_output = os.environ.get("CAPICOVE_TEST_VIDEO_HTTP_REPORT")
    starts = tuple(range(0, 40_000, 2000)) if benchmark_output else (0, 2000)
    visual_units = tuple(
        replace(
            entries[0].unit,
            unit_id=f"{target.value}-{start}",
            unit_type=target,
            evidence_ref=f"evidence:{target.value}:{start}",
            text_locator=None,
            embedding_spaces=(space,),
            representation_refs=(f"resource:preview:{start}",),
            timestamp_ms=start + 500 if target is TargetType.KEYFRAME else None,
            time_range=TemporalRange(start, start + 1000)
            if target is TargetType.VIDEO_SEGMENT
            else None,
        )
        for target in (TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT)
        for start in starts
    )
    if capacity_vectors is not None:
        template = visual_units[0]
        visual_units = tuple(
            replace(
                template,
                unit_id=f"keyframe-{ordinal}",
                timestamp_ms=starts[ordinal % len(starts)] + 500,
                evidence_ref=f"evidence:keyframe:{ordinal}",
                representation_refs=(f"resource:preview:{ordinal}",),
            )
            for ordinal in range(len(capacity_vectors))
        ) + tuple(u for u in visual_units if u.unit_type is TargetType.VIDEO_SEGMENT)
        print("capacity: hashing full unit catalog", flush=True)
    units = (*visual_units, *(e.unit for e in entries))
    unit_hash = searchable_units_manifest_sha256(units)
    content = json.dumps(
        [encode_text_index_entry(e) for e in entries], sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = TextIndexGenerationManifest(
        generation_id=f"video-v3-{uuid4().hex}",
        workspace_ref="qualification",
        corpus_refs=("authored-corpus",),
        searchable_unit_manifest_sha256=unit_hash,
        content_sha256=hashlib.sha256(content).hexdigest(),
        mapping_sha256=lifecycle.mapping_sha256(embedding_spaces=()),
        embedding_spaces=(),
        retriever_id="capicove.retriever.elasticsearch",
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("c" * 64),
        expected_unit_count=2,
    )
    lifecycle.create(manifest)
    backend.index_writer.add(index_generation_id=manifest.generation_id, entries=entries)
    report = lifecycle.validate(manifest)
    assert report.valid
    pin = lifecycle.activate(manifest, report)
    for origin, start in ((TextOrigin.ASR, 0), (TextOrigin.OCR, 2000)):
        found = backend.lexical_retriever.retrieve(
            query=TextQuery("authored"),
            filters=SearchFilters(
                text_origins=frozenset({origin}), time_range=TemporalRange(start, start + 1000)
            ),
            generation=pin,
            top_k=160,
        )
        assert [c.unit_id for c in found] == [f"text-{origin.value}"]
        assert found[0].text_locator.time_range == TemporalRange(start, start + 1000)
    assert not backend.lexical_retriever.retrieve(
        query=TextQuery("authored"),
        filters=SearchFilters(time_range=TemporalRange(4000, 5000)),
        generation=pin,
        top_k=10,
    )
    catalog_path = tmp_path / "units.sqlite3"
    print(f"publishing unit catalog with {len(units)} units", flush=True)
    SqliteVideoUnitCatalogWriter(catalog_path).publish(
        units=units,
        unit_manifest_sha256=unit_hash,
        keyframe_to_segments={
            u.unit_id: (f"video_segment-{u.timestamp_ms - 500}",)
            for u in visual_units
            if u.unit_type is TargetType.KEYFRAME and u.timestamp_ms is not None
        },
        text_to_segments={
            f"text-{origin.value}": ((f"video_segment-{start}", 1000),)
            for origin, start in ((TextOrigin.ASR, 0), (TextOrigin.OCR, 2000))
        },
    )
    if visual_provider == "faiss":
        native = pytest.importorskip("capicove_retriever_faiss.adapter")
        config = pytest.importorskip("capicove_retriever_faiss.config")
        runtime = open_video_runtime(
            generation_database_path=tmp_path / "sets.sqlite3",
            visual_backend=native.FaissVisualRetriever(
                config.FaissConfiguration(
                    tmp_path / "visual",
                    threads=4 if capacity_vectors is not None else 2,
                    ef_search=1024 if capacity_vectors is not None else 128,
                )
            ),
            validate_text_members=VideoTextMemberValidator(asr=lifecycle, ocr=lifecycle),
        )
    else:
        runtime = open_exact_video_runtime(
            generation_database_path=tmp_path / "sets.sqlite3",
            index_root=tmp_path / "visual",
            validate_text_members=VideoTextMemberValidator(asr=lifecycle, ocr=lifecycle),
        )
    visual_manifests = []
    visual_pins = []
    for target in (TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT):

        def members(
            selected_target: TargetType = target,
            source_units: tuple[SearchableUnit, ...] = visual_units,
        ) -> Iterator[VisualIndexEntry]:
            for u in source_units:
                if u.unit_type is not selected_target:
                    continue
                if capacity_vectors is not None and selected_target is TargetType.KEYFRAME:
                    vector = tuple(map(float, capacity_vectors[int(u.unit_id.split("-")[1])]))
                else:
                    vector = vectors[0] if u.unit_id.endswith("-0") else vectors[1]
                yield VisualIndexEntry(u, vector)

        member = VisualIndexGenerationManifest(
            f"visual-{target.value}",
            "qualification",
            ("authored-corpus",),
            target,
            unit_hash,
            visual_entries_content_sha256(members()),
            space,
            runtime.visual_backend.provider_id,
            "faiss-hnsw-flat-ip-v1" if visual_provider == "faiss" else "numpy-exact-v1",
            ProcessingFingerprint("d" * 64),
            sum(u.unit_type is target for u in visual_units),
        )
        print(f"building {member.expected_unit_count} {target.value} vectors", flush=True)
        runtime.visual_backend.create(member)
        if capacity_vectors is not None:
            runtime.visual_backend.add_stream(
                index_generation_id=member.generation_id, entries=members()
            )
        else:
            runtime.visual_backend.add(
                index_generation_id=member.generation_id, entries=tuple(members())
            )
        visual_manifests.append(member)
        visual_pins.append(
            runtime.visual_backend.activate(member, runtime.visual_backend.validate(member))
        )
    full_manifest = VideoGenerationSetManifest(
        "full-set",
        "qualification",
        "video-local-exact",
        ("authored-corpus",),
        str(catalog_path),
        unit_hash,
        visual_manifests[0],
        visual_manifests[1],
        manifest,
        manifest,
        "video-weighted-rrf-v1",
    )
    full_pin = PinnedVideoGenerationSet(
        "full-set",
        full_manifest.manifest_sha256,
        "qualification",
        "video-local-exact",
        str(catalog_path),
        unit_hash,
        visual_pins[0],
        visual_pins[1],
        pin,
        pin,
        "video-weighted-rrf-v1",
    )
    runtime.generation_sets.register(full_manifest, full_pin)
    runtime.generation_sets.activate("full-set")
    del members, units, visual_units
    gc.collect()

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            service = VideoSearchService(
                workspace_ref="qualification",
                generation_sets=runtime.generation_sets,
                profiles=StaticVideoRetrievalProfileRegistry(),
                catalog_resolver=runtime.catalog_resolver,
                visual_retriever=runtime.visual_backend,
                query_encoder=encoder,
                executor=executor,
                concurrency=BoundedSemaphore(4),
                text_backends={"asr_lexical": backend, "ocr_lexical": backend},
            )
            search_result = service.search(
                SearchRequest(
                    TextQuery("authored"),
                    frozenset({TargetType.VIDEO_SEGMENT}),
                    retrieval_profile_ref="video-local-exact",
                )
            )
            assert {hit.id for hit in search_result.hits} == {
                f"video_segment-{start}" for start in starts
            }
            assert not search_result.degraded, search_result.warning_codes
            if benchmark_output and real_model and visual_provider == "faiss":
                from http_benchmark import measure_local_http

                measure_local_http(
                    service,
                    Path(benchmark_output),
                    keyframe_count=1_000_000 if capacity_vectors is not None else len(starts),
                    generation_manifest_sha256=full_manifest.manifest_sha256,
                )
    finally:
        runtime.close()
