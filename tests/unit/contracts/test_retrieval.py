"""Tests for canonical multimodal retrieval contracts."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

import capicove.contracts as public_contracts
from capicove.contracts.retrieval import (
    RETRIEVAL_CONTRACT_VERSION,
    ArtifactSource,
    CharacterRange,
    CompactHit,
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    ExecutionProfile,
    FullHit,
    ImageQuery,
    InlineBytesSource,
    LaneScore,
    LocalPathSource,
    MatchKind,
    MountSource,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    SearchRequest,
    SearchResult,
    SearchStatus,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
    VectorNormalization,
    VideoClipQuery,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def embedding_space() -> EmbeddingSpaceRef:
    """Return one valid joint image-text embedding space."""
    return EmbeddingSpaceRef(
        space_id="siglip2-joint-v1",
        provider_id="huggingface-transformers-siglip2",
        model_revision="revision-1",
        preprocessing_version="siglip2-224-v1",
        dimension=768,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE, TargetType.KEYFRAME}),
    )


def processing_fingerprint() -> ProcessingFingerprint:
    """Return one valid processing identity."""
    return ProcessingFingerprint.from_manifest(
        {
            "source_hash": SHA_A,
            "processor_id": "capicove.keyframes",
            "configuration": {"stride": 2},
        }
    )


@pytest.mark.parametrize(
    ("query", "modality"),
    [
        (TextQuery("rainy bicycle", language="en"), QueryModality.TEXT),
        (ImageQuery(ArtifactSource("artifact-image")), QueryModality.IMAGE),
        (
            VideoClipQuery(
                InlineBytesSource(b"video", "video/mp4"),
                time_range=TemporalRange(1_000, 2_000),
            ),
            QueryModality.VIDEO_CLIP,
        ),
    ],
)
def test_query_variants_expose_stable_discriminators(
    query: TextQuery | ImageQuery | VideoClipQuery,
    modality: QueryModality,
) -> None:
    assert query.type is modality


def test_query_media_sources_are_a_closed_tagged_union() -> None:
    sources = (
        ArtifactSource("artifact-1"),
        LocalPathSource("inputs/query.png"),
        MountSource("media", "queries/query.png"),
        InlineBytesSource(b"image", "image/png"),
    )

    assert [source.type.value for source in sources] == [
        "artifact",
        "local_path",
        "mount",
        "inline_bytes",
    ]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ArtifactSource("  "),
        lambda: LocalPathSource(""),
        lambda: MountSource("mount", " "),
        lambda: MountSource(" ", "image.png"),
        lambda: InlineBytesSource(b"", "image/png"),
        lambda: InlineBytesSource(b"image", " "),
    ],
)
def test_query_media_sources_reject_empty_values(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


@pytest.mark.parametrize("text", ["", "  "])
def test_text_query_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        TextQuery(text)


def test_text_contract_exports_stable_version_and_discriminators() -> None:
    # Version 4 (RFC-0032) is additive: every version-3 discriminator is unchanged.
    assert RETRIEVAL_CONTRACT_VERSION == 4
    assert TargetType.TEXT_CHUNK.value == "text_chunk"
    assert MatchKind.LEXICAL.value == "lexical"
    assert [origin.value for origin in TextOrigin] == ["document", "ocr", "asr"]
    assert public_contracts.CharacterRange is CharacterRange
    assert public_contracts.TextChunkLocator is TextChunkLocator
    assert public_contracts.TextOrigin is TextOrigin


def test_character_range_uses_non_empty_half_open_coordinates() -> None:
    locator_range = CharacterRange(start_char=0, end_char=12)

    assert locator_range.start_char == 0
    assert locator_range.end_char == 12


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [(-1, 1), (1, 1), (2, 1), (True, 2), (0, True)],
)
def test_character_range_rejects_invalid_coordinates(
    start_char: int,
    end_char: int,
) -> None:
    with pytest.raises(ValueError):
        CharacterRange(start_char=start_char, end_char=end_char)


def test_text_chunk_locator_accepts_each_closed_origin_shape() -> None:
    character_range = CharacterRange(0, 12)
    document = TextChunkLocator(TextOrigin.DOCUMENT, character_range)
    ocr = TextChunkLocator(TextOrigin.OCR, character_range, page_number=3)
    asr = TextChunkLocator(
        TextOrigin.ASR,
        character_range,
        time_range=TemporalRange(1_000, 2_000),
    )

    assert document.page_number is None
    assert ocr.page_number == 3
    assert asr.time_range == TemporalRange(1_000, 2_000)


@pytest.mark.parametrize(
    "locator",
    [
        lambda: TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 1), 1),
        lambda: TextChunkLocator(
            TextOrigin.DOCUMENT,
            CharacterRange(0, 1),
            time_range=TemporalRange(0, 1),
        ),
        lambda: TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 1)),
        lambda: TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 1), 0),
        lambda: TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 1), True),
        lambda: TextChunkLocator(
            TextOrigin.OCR,
            CharacterRange(0, 1),
            page_number=1,
            time_range=TemporalRange(0, 1),
        ),
        lambda: TextChunkLocator(TextOrigin.ASR, CharacterRange(0, 1)),
        lambda: TextChunkLocator(
            TextOrigin.ASR,
            CharacterRange(0, 1),
            page_number=1,
            time_range=TemporalRange(0, 1),
        ),
    ],
)
def test_text_chunk_locator_rejects_invalid_origin_shapes(locator: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        locator()  # type: ignore[operator]


def test_search_request_defaults_to_compact_and_twenty_hits() -> None:
    request = SearchRequest(
        query=TextQuery("rainy bicycle"),
        targets=frozenset({TargetType.KEYFRAME}),
    )

    assert request.detail_level is DetailLevel.COMPACT
    assert request.top_k == 20
    assert request.filters == SearchFilters()
    assert request.retrieval_profile_ref == "local-default"
    assert request.execution_profile is ExecutionProfile.AUTO
    assert request.data_policy_ref == "local-only"


@pytest.mark.parametrize("top_k", [0, 101, -1, True])
def test_search_request_rejects_top_k_outside_public_limit(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        SearchRequest(
            query=TextQuery("rain"),
            targets=frozenset({TargetType.IMAGE}),
            top_k=top_k,
        )


def test_search_request_requires_at_least_one_target() -> None:
    with pytest.raises(ValueError, match="targets"):
        SearchRequest(query=TextQuery("rain"), targets=frozenset())


def test_search_filters_are_immutable_and_validate_identifiers() -> None:
    filters = SearchFilters(
        corpus_refs=frozenset({"corpus-1"}),
        artifact_refs=frozenset({"artifact-1"}),
        media_types=frozenset({"image/png"}),
        time_range=TemporalRange(0, 1_000),
    )

    with pytest.raises(FrozenInstanceError):
        filters.time_range = None  # type: ignore[misc]

    with pytest.raises(ValueError, match="corpus_ref"):
        SearchFilters(corpus_refs=frozenset({" "}))


def test_search_filters_freeze_text_origins_and_default_to_schema_version_two() -> None:
    filters = SearchFilters(
        text_origins=[TextOrigin.OCR, TextOrigin.ASR],  # type: ignore[arg-type]
    )

    assert filters.text_origins == frozenset({TextOrigin.OCR, TextOrigin.ASR})
    assert filters.schema_version == 2

    with pytest.raises(TypeError, match="text_origins"):
        SearchFilters(text_origins=frozenset({"ocr"}))  # type: ignore[arg-type]


def test_processing_fingerprint_is_order_independent_for_mapping_keys() -> None:
    left = ProcessingFingerprint.from_manifest(
        {"threshold": 0.4, "nested": {"stride": 2, "mode": "scene"}}
    )
    right = ProcessingFingerprint.from_manifest(
        {"nested": {"mode": "scene", "stride": 2}, "threshold": 0.4}
    )

    assert left == right
    assert len(left.sha256) == 64


def test_processing_fingerprint_preserves_list_order() -> None:
    left = ProcessingFingerprint.from_manifest({"inputs": [SHA_A, SHA_B]})
    right = ProcessingFingerprint.from_manifest({"inputs": [SHA_B, SHA_A]})

    assert left != right


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_processing_fingerprint_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        ProcessingFingerprint.from_manifest({"threshold": value})


def test_temporal_range_is_half_open_and_non_empty() -> None:
    with pytest.raises(ValueError, match="end_ms"):
        TemporalRange(start_ms=100, end_ms=100)


@pytest.mark.parametrize(("start_ms", "end_ms"), [(-1, 2), (2, 1), (True, 2)])
def test_temporal_range_rejects_invalid_coordinates(start_ms: int, end_ms: int) -> None:
    with pytest.raises(ValueError):
        TemporalRange(start_ms=start_ms, end_ms=end_ms)


def test_embedding_space_checks_modality_compatibility() -> None:
    space = embedding_space()

    assert space.accepts(QueryModality.TEXT, TargetType.KEYFRAME)
    assert not space.accepts(QueryModality.VIDEO_CLIP, TargetType.KEYFRAME)
    assert not space.accepts(QueryModality.TEXT, TargetType.VIDEO_SEGMENT)


@pytest.mark.parametrize(("field", "value"), [("dimension", True), ("schema_version", True)])
def test_embedding_space_rejects_boolean_integer_fields(field: str, value: bool) -> None:
    values = {
        "space_id": "space",
        "provider_id": "provider",
        "model_revision": "revision",
        "preprocessing_version": "preprocess",
        "dimension": 3,
        "normalization": VectorNormalization.L2,
        "distance_metric": DistanceMetric.COSINE,
        "query_modalities": frozenset({QueryModality.IMAGE}),
        "target_modalities": frozenset({TargetType.IMAGE}),
        "schema_version": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        EmbeddingSpaceRef(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["space_id", "provider_id", "model_revision"])
def test_embedding_space_rejects_blank_identity_fields(field: str) -> None:
    values = {
        "space_id": "space",
        "provider_id": "provider",
        "model_revision": "revision",
        "preprocessing_version": "preprocess",
        "dimension": 3,
        "normalization": VectorNormalization.L2,
        "distance_metric": DistanceMetric.COSINE,
        "query_modalities": frozenset({QueryModality.IMAGE}),
        "target_modalities": frozenset({TargetType.IMAGE}),
    }
    values[field] = " "

    with pytest.raises(ValueError, match=field):
        EmbeddingSpaceRef(**values)  # type: ignore[arg-type]


def test_searchable_unit_requires_immutable_lineage_identity() -> None:
    unit = SearchableUnit(
        unit_id="keyframe-1",
        unit_type=TargetType.KEYFRAME,
        source_hash=SHA_A,
        artifact_ref="artifact-video",
        evidence_ref="evidence-video",
        representation_refs=("representation-frame",),
        processing_run_ref="processing-run-1",
        processing_version="keyframes-v1",
        processing_fingerprint=processing_fingerprint(),
        embedding_spaces=(embedding_space(),),
        timestamp_ms=2_500,
        scene_range=TemporalRange(2_000, 4_000),
    )

    assert unit.timestamp_ms == 2_500
    assert unit.embedding_spaces[0].dimension == 768
    with pytest.raises(FrozenInstanceError):
        unit.source_hash = SHA_B  # type: ignore[misc]


def test_searchable_text_unit_requires_a_text_locator() -> None:
    locator = TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 20))
    values = {
        "unit_id": "text-1",
        "unit_type": TargetType.TEXT_CHUNK,
        "source_hash": SHA_A,
        "artifact_ref": "artifact-document",
        "evidence_ref": "evidence-document",
        "representation_refs": ("representation-text",),
        "processing_run_ref": "processing-run-1",
        "processing_version": "text-chunk-v1",
        "processing_fingerprint": processing_fingerprint(),
    }

    unit = SearchableUnit(**values, text_locator=locator)  # type: ignore[arg-type]

    assert unit.text_locator is locator
    with pytest.raises(ValueError, match="text_locator"):
        SearchableUnit(**values)  # type: ignore[arg-type]


def test_non_text_searchable_unit_forbids_a_text_locator() -> None:
    with pytest.raises(ValueError, match="text_locator"):
        SearchableUnit(
            unit_id="image-1",
            unit_type=TargetType.IMAGE,
            source_hash=SHA_A,
            artifact_ref="artifact-image",
            evidence_ref="evidence-image",
            representation_refs=("representation-image",),
            processing_run_ref="processing-run-1",
            processing_version="image-v1",
            processing_fingerprint=processing_fingerprint(),
            text_locator=TextChunkLocator(
                TextOrigin.DOCUMENT,
                CharacterRange(0, 5),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_hash", "not-a-hash"),
        ("unit_id", " "),
        ("processing_version", ""),
        ("processing_run_ref", " "),
    ],
)
def test_searchable_unit_rejects_malformed_identity(field: str, value: str) -> None:
    values = {
        "unit_id": "image-1",
        "unit_type": TargetType.IMAGE,
        "source_hash": SHA_A,
        "artifact_ref": "artifact-image",
        "evidence_ref": "evidence-image",
        "representation_refs": ("representation-image",),
        "processing_run_ref": "processing-run-1",
        "processing_version": "image-v1",
        "processing_fingerprint": processing_fingerprint(),
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        SearchableUnit(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "unit",
    [
        SearchableUnit,
    ],
)
def test_searchable_unit_enforces_target_specific_temporal_locator(unit: object) -> None:
    common = {
        "source_hash": SHA_A,
        "artifact_ref": "artifact-1",
        "evidence_ref": "evidence-1",
        "representation_refs": ("representation-1",),
        "processing_run_ref": "run-1",
        "processing_version": "v1",
        "processing_fingerprint": processing_fingerprint(),
    }

    with pytest.raises(ValueError, match="timestamp_ms"):
        unit(unit_id="keyframe", unit_type=TargetType.KEYFRAME, **common)  # type: ignore[operator]
    with pytest.raises(ValueError, match="time_range"):
        unit(unit_id="segment", unit_type=TargetType.VIDEO_SEGMENT, **common)  # type: ignore[operator]
    with pytest.raises(ValueError, match="temporal"):
        unit(  # type: ignore[operator]
            unit_id="image",
            unit_type=TargetType.IMAGE,
            timestamp_ms=1,
            **common,
        )


def test_compact_hits_enforce_target_specific_temporal_locator() -> None:
    image = CompactHit("image-1", TargetType.IMAGE, 0.9, "evidence-image")
    keyframe = CompactHit(
        "keyframe-1",
        TargetType.KEYFRAME,
        0.8,
        "evidence-video",
        timestamp_ms=1_500,
    )
    segment = CompactHit(
        "segment-1",
        TargetType.VIDEO_SEGMENT,
        0.7,
        "evidence-video",
        time_range=TemporalRange(1_000, 3_000),
    )

    assert image.timestamp_ms is None
    assert keyframe.timestamp_ms == 1_500
    assert segment.time_range == TemporalRange(1_000, 3_000)


def test_text_hit_requires_a_bounded_plain_text_snippet_and_locator() -> None:
    locator = TextChunkLocator(TextOrigin.OCR, CharacterRange(20, 40), page_number=2)
    hit = CompactHit(
        "text-1",
        TargetType.TEXT_CHUNK,
        0.9,
        "evidence-text",
        text_locator=locator,
        snippet="A bounded OCR result.",
    )

    assert hit.text_locator is locator
    assert hit.snippet == "A bounded OCR result."
    CompactHit(
        "text-2",
        TargetType.TEXT_CHUNK,
        0.8,
        "evidence-text",
        text_locator=locator,
        snippet="x" * 2_048,
    )


@pytest.mark.parametrize("snippet", [None, " ", "x" * 2_049, "<em>match</em>"])
def test_text_hit_rejects_missing_unbounded_or_highlighted_snippets(
    snippet: str | None,
) -> None:
    with pytest.raises(ValueError, match="snippet"):
        CompactHit(
            "text-1",
            TargetType.TEXT_CHUNK,
            0.9,
            "evidence-text",
            text_locator=TextChunkLocator(
                TextOrigin.DOCUMENT,
                CharacterRange(0, 5),
            ),
            snippet=snippet,
        )


def test_text_hit_requires_locator_and_non_text_hit_forbids_text_fields() -> None:
    with pytest.raises(ValueError, match="text_locator"):
        CompactHit(
            "text-1",
            TargetType.TEXT_CHUNK,
            0.9,
            "evidence-text",
            snippet="result",
        )

    with pytest.raises(ValueError, match="text_locator"):
        CompactHit(
            "image-1",
            TargetType.IMAGE,
            0.9,
            "evidence-image",
            text_locator=TextChunkLocator(
                TextOrigin.DOCUMENT,
                CharacterRange(0, 5),
            ),
            snippet="result",
        )


@pytest.mark.parametrize("brief_score", [math.nan, math.inf, -0.1, 1.1])
def test_compact_hit_rejects_invalid_brief_score(brief_score: float) -> None:
    with pytest.raises(ValueError, match="brief_score"):
        CompactHit("image-1", TargetType.IMAGE, brief_score, "evidence-1")


def test_full_hit_carries_scores_and_lineage_without_embedding_bytes() -> None:
    hit = FullHit(
        id="image-1",
        target_type=TargetType.IMAGE,
        brief_score=0.87,
        evidence_id="evidence-1",
        artifact_ref="artifact-1",
        searchable_unit_ref="image-1",
        representation_refs=("representation-1",),
        match_kinds=(MatchKind.SEMANTIC,),
        lane_scores=(
            LaneScore(
                lane_id="semantic-image",
                raw_score=0.42,
                normalized_score=0.87,
                score_semantics="cosine_similarity",
                calibration_version="rrf-v1",
                rank=1,
            ),
        ),
        embedding_spaces=(embedding_space(),),
        processing_fingerprint=processing_fingerprint(),
        index_generation_ids=("generation-1",),
        provider_ids=("provider-1",),
        execution_profile=ExecutionProfile.CPU,
        lineage_refs=("lineage-1",),
    )

    assert hit.lane_scores[0].raw_score == 0.42
    assert hit.lane_scores[0].normalized_score == 0.87
    assert not hasattr(hit, "embedding")
    assert not hasattr(hit, "embeddings")


def test_full_hit_enforces_keyframe_scene_range() -> None:
    hit = FullHit(
        id="keyframe-1",
        target_type=TargetType.KEYFRAME,
        brief_score=0.87,
        evidence_id="evidence-1",
        timestamp_ms=1_000,
        scene_range=TemporalRange(1_000, 2_000),
    )

    assert hit.timestamp_ms == 1_000
    assert hit.scene_range == TemporalRange(1_000, 2_000)


@pytest.mark.parametrize("timestamp_ms", [999, 2_000])
def test_full_hit_rejects_timestamp_outside_half_open_scene_range(
    timestamp_ms: int,
) -> None:
    with pytest.raises(ValueError, match="scene_range"):
        FullHit(
            id="keyframe-2",
            target_type=TargetType.KEYFRAME,
            brief_score=0.75,
            evidence_id="evidence-2",
            timestamp_ms=timestamp_ms,
            scene_range=TemporalRange(1_000, 2_000),
        )


def test_full_hit_rejects_invalid_scene_range_locator() -> None:
    with pytest.raises(ValueError, match="timestamp_ms"):
        FullHit(
            id="keyframe-1",
            target_type=TargetType.KEYFRAME,
            brief_score=0.75,
            evidence_id="evidence-1",
            scene_range=TemporalRange(1_000, 2_000),
        )

    with pytest.raises(ValueError, match="temporal locator"):
        FullHit(
            id="image-1",
            target_type=TargetType.IMAGE,
            brief_score=0.75,
            evidence_id="evidence-2",
            scene_range=TemporalRange(1_000, 2_000),
        )

    with pytest.raises(TypeError, match="scene_range"):
        FullHit(
            id="keyframe-2",
            target_type=TargetType.KEYFRAME,
            brief_score=0.75,
            evidence_id="evidence-3",
            timestamp_ms=1_500,
            scene_range="not-a-temporal-range",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("raw_score", "normalized_score"),
    [(math.nan, 0.5), (math.inf, 0.5), (0.5, -0.1), (0.5, 1.1)],
)
def test_lane_score_rejects_non_finite_or_unbounded_values(
    raw_score: float,
    normalized_score: float,
) -> None:
    with pytest.raises(ValueError):
        LaneScore(
            lane_id="lane",
            raw_score=raw_score,
            normalized_score=normalized_score,
            score_semantics="distance",
            calibration_version="v1",
            rank=1,
        )


def test_lane_score_rejects_boolean_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        LaneScore(
            lane_id="lane",
            raw_score=0.5,
            normalized_score=0.5,
            score_semantics="distance",
            calibration_version="v1",
            rank=True,
        )


def test_search_result_allows_a_successful_empty_result() -> None:
    result = SearchResult(
        status=SearchStatus.SUCCEEDED,
        query_id="query-1",
        hits=(),
    )

    assert result.hits == ()
    assert not result.degraded
    assert result.next_cursor is None


def test_search_result_requires_warning_when_degraded() -> None:
    with pytest.raises(ValueError, match="warning"):
        SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id="query-1",
            degraded=True,
        )
