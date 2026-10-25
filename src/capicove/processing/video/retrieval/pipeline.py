"""Offline evidence preparation with validated content-addressed checkpoints."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any, cast

from capicove.contracts.retrieval import (
    ProcessingFingerprint,
    SearchableUnit,
    TargetType,
    TemporalRange,
)
from capicove.processing.audio.models import (
    AudioBackendLineage,
    AudioTranscript,
    AudioTranscriptionSegment,
    AudioWarningCode,
)
from capicove.processing.video.models import AcceptedVideoSource
from capicove.processing.video.ports import BackendUnavailableError
from capicove.processing.video.retrieval.keyframes import (
    KeyframeSelectionConfig,
    keyframe_policy_manifest,
    select_retrieval_keyframes,
)
from capicove.processing.video.retrieval.models import (
    FrameObservation,
    RetrievalKeyframe,
    SceneMetric,
    TemporalOcrText,
    VideoDecodeObservation,
    VideoRetrievalPreparationRequest,
    VideoRetrievalPreparationResult,
)
from capicove.processing.video.retrieval.ports import (
    ASRProcessor,
    DerivedCacheCatalog,
    MediaRepresentationEncoder,
    OCRProcessor,
    OfflineVideoDecodePass,
    PreviewStore,
    ProcessingManifestStore,
    SceneBoundaryExtractor,
    SourceAcceptor,
)
from capicove.processing.video.retrieval.segments import (
    SegmentBuildConfig,
    build_candidate_segments,
    pool_segment_vector,
)
from capicove.processing.video.retrieval.text_units import (
    asr_searchable_units,
    ocr_searchable_units,
)


def check_deadline(deadline: datetime) -> None:
    """Check the shared absolute deadline at each bounded work boundary."""
    if deadline.tzinfo is None or datetime.now(UTC) >= deadline:
        raise TimeoutError("deadline_exceeded")


def canonical_json(value: object) -> bytes:
    """Serialize deterministic evidence identities, including enum sets."""

    def default(item: object) -> object:
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError("unsupported manifest value")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=default,
    ).encode()


def verify_source(source: AcceptedVideoSource, deadline: datetime) -> None:
    """Rehash accepted materialized evidence before and after processing."""
    digest = hashlib.sha256()
    if source.path is not None:
        with source.path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                check_deadline(deadline)
                digest.update(chunk)
    elif source.data is not None:
        digest.update(source.data)
    check_deadline(deadline)
    if digest.hexdigest() != source.sha256:
        raise ValueError("source_changed")


class VideoRetrievalPreparationService:
    """Produce one frozen unit manifest without opening any retrieval index."""

    def __init__(
        self,
        *,
        source_acceptor: SourceAcceptor,
        decode_pass: OfflineVideoDecodePass,
        scene_extractor: SceneBoundaryExtractor,
        target_encoder: MediaRepresentationEncoder,
        preview_store: PreviewStore,
        manifest_store: ProcessingManifestStore,
        cache_catalog: DerivedCacheCatalog,
        asr_processor: ASRProcessor | None = None,
        ocr_processor: OCRProcessor | None = None,
        keyframe_config: KeyframeSelectionConfig | None = None,
        segment_config: SegmentBuildConfig | None = None,
    ) -> None:
        self._acceptor = source_acceptor
        self._decoder = decode_pass
        self._scenes = scene_extractor
        self._encoder = target_encoder
        self._previews = preview_store
        self._manifests = manifest_store
        self._cache = cache_catalog
        self._asr = asr_processor
        self._ocr = ocr_processor
        self._keyframe_config = keyframe_config or KeyframeSelectionConfig()
        self._segment_config = segment_config or SegmentBuildConfig()

    def _checkpoint(self, fingerprint: ProcessingFingerprint, payload: dict[str, Any]) -> str:
        reference = self._manifests.put(
            canonical_json({"fingerprint": fingerprint.sha256, "payload": payload})
        )
        self._cache.publish(fingerprint, reference)
        return reference

    def _read_checkpoint(self, fingerprint: ProcessingFingerprint) -> dict[str, Any] | None:
        reference = self._cache.lookup(fingerprint)
        if reference is None:
            return None
        value = json.loads(self._manifests.read(reference))
        if (
            not isinstance(value, dict)
            or value.get("fingerprint") != fingerprint.sha256
            or not isinstance(value.get("payload"), dict)
        ):
            raise ValueError("checkpoint fingerprint mismatch")
        return cast(dict[str, Any], value["payload"])

    def prepare(self, request: VideoRetrievalPreparationRequest) -> VideoRetrievalPreparationResult:
        """Run one bounded pass or reuse verified upstream layer manifests."""
        check_deadline(request.deadline)
        source = self._acceptor.accept(request.source, deadline=request.deadline)
        verify_source(source, request.deadline)
        decode_fp = ProcessingFingerprint.from_manifest(
            {
                "layer": "decode-v1",
                "source": source.sha256,
                "decoder": self._decoder.fingerprint.sha256,
            }
        )
        cached = self._read_checkpoint(decode_fp)
        if cached is None:
            decoded = self._decoder.decode(source, deadline=request.deadline)
            check_deadline(request.deadline)
            verify_source(source, request.deadline)
            self._checkpoint(decode_fp, asdict(decoded))
        else:
            decoded = VideoDecodeObservation(
                cached["duration_ms"],
                tuple(FrameObservation(**f) for f in cached["frames"]),
                tuple(SceneMetric(**m) for m in cached["scene_metrics"]),
            )
        for frame in decoded.frames:
            check_deadline(request.deadline)
            self._previews.read(frame.preview_candidate_ref)
        keyframe_fp = ProcessingFingerprint.from_manifest(
            {
                "layer": "keyframes-v1",
                "decode": decode_fp.sha256,
                "scene": self._scenes.fingerprint.sha256,
                "policy": keyframe_policy_manifest(self._keyframe_config),
            }
        )
        cached = self._read_checkpoint(keyframe_fp)
        if cached is None:
            scenes = self._scenes.extract(
                decoded.scene_metrics, duration_ms=decoded.duration_ms, deadline=request.deadline
            ) or (TemporalRange(0, decoded.duration_ms),)
            selected = select_retrieval_keyframes(scenes, decoded.frames, self._keyframe_config)
            keyframes = tuple(
                RetrievalKeyframe(
                    "keyframe:"
                    + ProcessingFingerprint.from_manifest(
                        {
                            "source": source.sha256,
                            "time": f.timestamp_ms,
                            "fingerprint": keyframe_fp.sha256,
                        }
                    ).sha256,
                    f.timestamp_ms,
                    next(s for s in scenes if s.start_ms <= f.timestamp_ms < s.end_ms),
                    f.preview_candidate_ref.replace("resource:sha256:", "resource:preview:", 1),
                    hashlib.sha256(self._previews.read(f.preview_candidate_ref)).hexdigest(),
                    f.quality_score,
                    keyframe_fp,
                )
                for f in selected
            )
            self._checkpoint(
                keyframe_fp,
                {
                    "scenes": [asdict(s) for s in scenes],
                    "keyframes": [asdict(f) for f in keyframes],
                },
            )
        else:
            scenes = tuple(TemporalRange(**s) for s in cached["scenes"])
            keyframes = tuple(
                RetrievalKeyframe(
                    **{
                        **f,
                        "scene_range": TemporalRange(**f["scene_range"]),
                        "processing_fingerprint": ProcessingFingerprint(
                            **f["processing_fingerprint"]
                        ),
                    }
                )
                for f in cached["keyframes"]
            )
        for keyframe in keyframes:
            if (
                hashlib.sha256(self._previews.read(keyframe.preview_resource_ref)).hexdigest()
                != keyframe.preview_sha256
            ):
                raise ValueError("preview hash mismatch")
        check_deadline(request.deadline)
        warnings: list[str] = []
        transcript = None
        asr_identity = request.asr_processing_fingerprint
        if self._asr is not None:
            if not isinstance(self._asr.fingerprint, ProcessingFingerprint):
                raise ValueError("ASR provider fingerprint is required")
            asr_identity = ProcessingFingerprint.from_manifest(
                {
                    "layer": "asr-v1",
                    "source": source.sha256,
                    "provider": self._asr.fingerprint.sha256,
                    "requested": request.asr_processing_fingerprint.sha256,
                    "time_projection": "round-seconds-half-open-ms-v1",
                }
            )
            asr_cache = self._read_checkpoint(asr_identity)
            try:
                if asr_cache is None:
                    transcript = self._asr.transcribe(source, deadline=request.deadline)
                    check_deadline(request.deadline)
                    verify_source(source, request.deadline)
                    self._checkpoint(
                        asr_identity,
                        {"transcript": asdict(transcript) if transcript is not None else None},
                    )
                elif asr_cache["transcript"] is not None:
                    item = asr_cache["transcript"]
                    transcript = AudioTranscript(
                        **{
                            **item,
                            "segments": tuple(
                                AudioTranscriptionSegment(**s) for s in item["segments"]
                            ),
                            "warnings": tuple(AudioWarningCode(w) for w in item["warnings"]),
                            "lineage": AudioBackendLineage(**item["lineage"]),
                        }
                    )
            except BackendUnavailableError:
                warnings.append("asr_unavailable")
        check_deadline(request.deadline)
        ocr: tuple[TemporalOcrText, ...] = ()
        ocr_fp = request.ocr_processing_fingerprint
        if self._ocr is not None:
            if not isinstance(self._ocr.fingerprint, ProcessingFingerprint):
                raise ValueError("OCR provider fingerprint is required")
            ocr_fp = ProcessingFingerprint.from_manifest(
                {
                    "layer": "ocr-v1",
                    "keyframes": keyframe_fp.sha256,
                    "provider": self._ocr.fingerprint.sha256,
                    "requested": request.ocr_processing_fingerprint.sha256,
                    "previews": [asdict(k) for k in keyframes],
                }
            )
            ocr_cache = self._read_checkpoint(ocr_fp)
            try:
                if ocr_cache is None:
                    ocr = self._ocr.recognize(keyframes, deadline=request.deadline)
                    for item in ocr:
                        if self._manifests.read(item.representation_ref) != item.text.encode():
                            raise ValueError("OCR representation mismatch")
                        self._manifests.read(item.evidence_ref)
                    verify_source(source, request.deadline)
                    self._checkpoint(ocr_fp, {"ocr": [asdict(t) for t in ocr]})
                else:
                    ocr = tuple(
                        TemporalOcrText(**{**t, "time_range": TemporalRange(**t["time_range"])})
                        for t in ocr_cache["ocr"]
                    )
            except BackendUnavailableError:
                warnings.append("ocr_unavailable")
        check_deadline(request.deadline)
        # Persist text before projecting offsets into its addressed representation.
        asr_ref = (
            self._manifests.put(transcript.text.encode())
            if transcript
            else request.asr_representation_ref
        )
        asr_fp = request.asr_processing_fingerprint
        asr_evidence = request.evidence_ref
        if transcript is not None:
            asr_fp = ProcessingFingerprint.from_manifest(
                {
                    "requested": asr_identity.sha256,
                    "lineage": asdict(transcript.lineage),
                    "language": transcript.language,
                    "time_projection": "round-seconds-half-open-ms-v1",
                }
            )
            asr_evidence = self._manifests.put(
                canonical_json(
                    {
                        "schema_version": 1,
                        "source_hash": source.sha256,
                        "source_evidence_ref": request.evidence_ref,
                        "transcript": asdict(transcript),
                        "text_representation_ref": asr_ref,
                        "processing_fingerprint": asr_fp.sha256,
                    }
                )
            )
        asr_units = (
            asr_searchable_units(
                source_hash=source.sha256,
                artifact_ref=request.artifact_ref,
                evidence_ref=asr_evidence,
                transcript=transcript,
                representation_ref=asr_ref,
                processing_run_ref=request.processing_run_ref,
                processing_fingerprint=asr_fp,
            )
            if transcript
            else ()
        )
        for item in ocr:
            if self._manifests.read(item.representation_ref) != item.text.encode():
                raise ValueError("OCR representation mismatch")
            self._manifests.read(item.evidence_ref)
        ocr_units = ocr_searchable_units(
            source_hash=source.sha256,
            artifact_ref=request.artifact_ref,
            ocr=ocr,
            processing_run_ref=request.processing_run_ref,
            processing_fingerprint=ocr_fp,
        )
        if any(
            u.text_locator
            and u.text_locator.time_range
            and u.text_locator.time_range.end_ms > decoded.duration_ms
            for u in (*asr_units, *ocr_units)
        ):
            raise ValueError("temporal text exceeds source duration")
        space = self._encoder.embedding_space
        vector_fp = ProcessingFingerprint.from_manifest(
            {
                "layer": "visual-v1",
                "keyframes": keyframe_fp.sha256,
                "previews": [k.preview_sha256 for k in keyframes],
                "space": json.loads(canonical_json(asdict(space))),
            }
        )
        vector_cache = self._read_checkpoint(vector_fp)
        vectors = (
            self._encoder.encode_keyframes(keyframes, deadline=request.deadline)
            if vector_cache is None
            else tuple(tuple(float(v) for v in row) for row in vector_cache["vectors"])
        )
        check_deadline(request.deadline)
        if len(vectors) != len(keyframes) or any(len(v) != space.dimension for v in vectors):
            raise ValueError("vector count or dimension mismatch")
        # Normalization also rejects non-finite and zero vectors.
        vectors = tuple(pool_segment_vector((v,), (1.0,)) for v in vectors)
        if vector_cache is None:
            verify_source(source, request.deadline)
            self._checkpoint(vector_fp, {"vectors": vectors})
        segment_fp = ProcessingFingerprint.from_manifest(
            {
                "layer": "segments-v1",
                "keyframes": keyframe_fp.sha256,
                "policy": asdict(self._segment_config),
                "pooling": "quality-weighted-mean-l2-v1",
                "space": json.loads(canonical_json(asdict(space))),
                "asr": [asdict(u) for u in asr_units],
                "ocr": [asdict(u) for u in ocr_units],
                "request": request.processing_fingerprint.sha256,
            }
        )
        segments = build_candidate_segments(
            source_hash=source.sha256,
            duration_ms=decoded.duration_ms,
            scenes=scenes,
            keyframes=keyframes,
            asr_units=asr_units,
            ocr_units=ocr_units,
            config=self._segment_config,
            processing_fingerprint=segment_fp,
        )
        by_id = {f.unit_id: (f, v) for f, v in zip(keyframes, vectors, strict=True)}
        segment_vectors = tuple(
            pool_segment_vector(
                tuple(by_id[i][1] for i in s.keyframe_unit_ids),
                tuple(by_id[i][0].quality_score for i in s.keyframe_unit_ids),
            )
            for s in segments
        )
        self._checkpoint(
            segment_fp, {"segments": [asdict(s) for s in segments], "vectors": segment_vectors}
        )
        common: dict[str, Any] = dict(
            source_hash=source.sha256,
            artifact_ref=request.artifact_ref,
            processing_run_ref=request.processing_run_ref,
            processing_version="video-retrieval-v1",
            embedding_spaces=(space,),
        )

        def visual_evidence(
            unit_id: str, preview_ref: str, fingerprint: ProcessingFingerprint
        ) -> str:
            # Each hit resolves to its own immutable receipt rather than an
            # ambiguous source-wide evidence ID shared by every video interval.
            return self._manifests.put(
                canonical_json(
                    {
                        "schema_version": 1,
                        "kind": "video-visual-unit",
                        "unit_id": unit_id,
                        "source_hash": source.sha256,
                        "source_evidence_ref": request.evidence_ref,
                        "preview_resource_ref": preview_ref,
                        "processing_fingerprint": fingerprint.sha256,
                    }
                )
            )

        visual_units = tuple(
            SearchableUnit(
                unit_id=k.unit_id,
                unit_type=TargetType.KEYFRAME,
                evidence_ref=visual_evidence(
                    k.unit_id, k.preview_resource_ref, k.processing_fingerprint
                ),
                representation_refs=(k.preview_resource_ref,),
                processing_fingerprint=k.processing_fingerprint,
                timestamp_ms=k.timestamp_ms,
                scene_range=k.scene_range,
                **common,
            )
            for k in keyframes
        ) + tuple(
            SearchableUnit(
                unit_id=s.unit_id,
                unit_type=TargetType.VIDEO_SEGMENT,
                evidence_ref=visual_evidence(
                    s.unit_id, s.preview_resource_ref, s.processing_fingerprint
                ),
                representation_refs=(s.preview_resource_ref,),
                processing_fingerprint=s.processing_fingerprint,
                time_range=s.time_range,
                **common,
            )
            for s in segments
        )
        units = tuple(
            sorted(
                (*visual_units, *asr_units, *ocr_units),
                key=lambda u: (u.unit_type.value, u.unit_id),
            )
        )
        # Canonical unit manifests encode the fingerprint as its digest string,
        # matching the public index-manifest identity without adapter imports.
        payload = canonical_json(
            [
                {**asdict(u), "processing_fingerprint": u.processing_fingerprint.sha256}
                for u in units
            ]
        )
        manifest_hash = hashlib.sha256(payload).hexdigest()
        verify_source(source, request.deadline)
        reference = self._manifests.put(payload)
        check_deadline(request.deadline)
        return VideoRetrievalPreparationResult(
            source.sha256,
            keyframes,
            segments,
            units,
            vectors,
            segment_vectors,
            space,
            manifest_hash,
            (reference,),
            tuple(warnings),
        )
