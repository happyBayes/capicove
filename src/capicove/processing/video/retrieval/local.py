"""Explicit public composition for one local offline preparation workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.processing.video.ports import ArtifactResolver, PathAuthorizer
from capicove.processing.video.retrieval.keyframes import KeyframeSelectionConfig
from capicove.processing.video.retrieval.pipeline import VideoRetrievalPreparationService
from capicove.processing.video.retrieval.ports import ASRProcessor, OCRProcessor
from capicove.processing.video.retrieval.segments import SegmentBuildConfig


if TYPE_CHECKING:
    from capicove.processing.image import ImageOcr


def create_local_video_preparation_service(
    workspace_root: Path,
    *,
    embedding_space: EmbeddingSpaceRef,
    encode_batch: Callable[[tuple[bytes, ...]], tuple[tuple[float, ...], ...]],
    path_authorizer: PathAuthorizer | None = None,
    policy_ref: str | None = None,
    artifact_resolver: ArtifactResolver | None = None,
    asr_processor: ASRProcessor | None = None,
    image_ocr: ImageOcr | None = None,
    ocr_provider_fingerprint: ProcessingFingerprint | None = None,
    ocr_languages: tuple[str, ...] | None = None,
    keyframe_config: KeyframeSelectionConfig | None = None,
    segment_config: SegmentBuildConfig | None = None,
    sample_interval_ms: int = 500,
    maximum_preview_edge: int = 384,
    max_decoded_frames: int = 1_000_000,
    max_observations: int = 100_000,
    max_pixels: int = 16_777_216,
    video_stream_index: int = 0,
    encoder_batch_size: int = 16,
) -> VideoRetrievalPreparationService:
    """Wire local decode, immutable stores, a provisioned encoder and optional OCR.

    The caller verifies that the supplied local encoder matches embedding_space;
    no model is selected or downloaded here. Path sources still require the
    existing PathAuthorizer and policy reference. The workspace root is trusted
    application configuration, never a search or model-supplied argument.
    Construction performs no media/model calls or filesystem writes.
    """
    from capicove.processing.video.retrieval._adapters.local import (
        LocalContentStore,
        LocalDerivedCacheCatalog,
        LocalVideoSourceAcceptor,
        MetricSceneExtractor,
    )
    from capicove.processing.video.retrieval._adapters.providers import PreviewBatchEncoder
    from capicove.processing.video.retrieval._adapters.pyav import PyAVVideoDecodePass
    from capicove.processing.video.retrieval.shared_ocr import SharedImageOCRProcessor

    required_queries = {QueryModality.TEXT, QueryModality.IMAGE}
    required_targets = {TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}
    if (
        not required_queries <= embedding_space.query_modalities
        or not required_targets <= embedding_space.target_modalities
    ):
        raise ValueError(
            "local video space must support text and image queries for both video targets"
        )
    if (
        embedding_space.normalization is not VectorNormalization.L2
        or embedding_space.distance_metric is not DistanceMetric.INNER_PRODUCT
    ):
        raise ValueError("local video space requires L2-normalized inner product")
    if image_ocr is None and (ocr_provider_fingerprint is not None or ocr_languages is not None):
        raise ValueError("OCR settings require an injected ImageOcr service")
    store = LocalContentStore(workspace_root / "objects")
    ocr: OCRProcessor | None = None
    if image_ocr is not None:
        ocr = SharedImageOCRProcessor(
            image_ocr,
            store,
            store,
            languages=ocr_languages,
            provider_fingerprint=ocr_provider_fingerprint,
        )
    return VideoRetrievalPreparationService(
        source_acceptor=LocalVideoSourceAcceptor(
            path_authorizer=path_authorizer,
            policy_ref=policy_ref,
            artifact_resolver=artifact_resolver,
        ),
        decode_pass=PyAVVideoDecodePass(
            store,
            sample_interval_ms=sample_interval_ms,
            maximum_preview_edge=maximum_preview_edge,
            max_decoded_frames=max_decoded_frames,
            max_observations=max_observations,
            max_pixels=max_pixels,
            video_stream_index=video_stream_index,
        ),
        scene_extractor=MetricSceneExtractor(),
        target_encoder=PreviewBatchEncoder(
            store, embedding_space, encode_batch, batch_size=encoder_batch_size
        ),
        preview_store=store,
        manifest_store=store,
        cache_catalog=LocalDerivedCacheCatalog(workspace_root / "checkpoints"),
        asr_processor=asr_processor,
        ocr_processor=ocr,
        keyframe_config=keyframe_config,
        segment_config=segment_config,
    )


def read_local_preparation_resource(workspace_root: Path, reference: str) -> bytes:
    """Read one hash-verified resource from an already authorized workspace.

    Callers must enforce evidence authorization before exposing the result through
    a transport. This operation is for trusted local composition, not ingress.
    """
    from capicove.processing.video.retrieval._adapters.local import LocalContentStore

    return LocalContentStore(workspace_root / "objects").read(reference)
