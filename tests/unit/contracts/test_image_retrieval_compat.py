"""Tests for canonical multimodal retrieval contracts."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict, fields
from datetime import UTC, datetime

import pytest

import capicove.contracts as canonical_contracts
from capicove.contracts.retrieval import (
    MAX_CURSOR_BYTES,
    MAX_FILTER_VALUES,
    MAX_INLINE_IMAGE_BYTES,
    MAX_LANGUAGE_BYTES,
    MAX_MEDIA_TYPE_BYTES,
    MAX_PATH_BYTES,
    MAX_REF_BYTES,
    MAX_TEMPORAL_MILLISECONDS,
    MAX_TEXT_BYTES,
    ArtifactSource,
    CompactHit,
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    ExecutionProfile,
    FullHit,
    HashSpaceRef,
    ImageQuery,
    IndexGeneration,
    IndexParametersRef,
    InlineBytesSource,
    LaneScore,
    LocalPathSource,
    MatchKind,
    MountSource,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchExecutionSummary,
    SearchFailureCode,
    SearchFilters,
    SearchIntent,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TemporalRange,
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


def hash_space() -> HashSpaceRef:
    """Return the version-one perceptual-hash space."""
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


def index_generation() -> IndexGeneration:
    """Return one valid immutable image-index generation."""
    return IndexGeneration(
        generation_id="generation-instance-1",
        activation_set_id="set-1",
        workspace_id="workspace-1",
        corpus_ref="education-film",
        target_type=TargetType.IMAGE,
        unit_manifest_sha256=SHA_A,
        embedding_spaces=(embedding_space(),),
        hash_spaces=(),
        retriever_id="capicove.numpy-exact",
        index_format="numpy-flat-f32-v1",
        index_parameters_ref=IndexParametersRef(
            parameter_schema="numpy-flat-f32-v1",
            payload_ref="payload:instance-parameters",
            sha256=SHA_B,
        ),
        shard_refs=("payload:instance-vectors",),
        calibration_profile_ref="exact-cosine-v1",
        build_processing_fingerprint=processing_fingerprint(),
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
        validation_report_ref="validation:instance-1",
    )


def execution_summary() -> SearchExecutionSummary:
    """Return one bounded full-detail execution summary."""
    return SearchExecutionSummary(
        search_intent=SearchIntent.BALANCED,
        selected_lanes=("content", "perceptual", "instance", "concept"),
        provider_ids=("opencv-phash", "instance-provider", "concept-provider"),
        embedding_spaces=(embedding_space(),),
        hash_spaces=(hash_space(),),
        index_generation_ids=("generation-instance-1", "generation-concept-1"),
        bundle_manifest_sha256s=(SHA_A, SHA_B),
        resolved_profile_ref="personal-lite",
        execution_profile=ExecutionProfile.CPU,
        activation_tier="preview",
    )


def test_reverse_image_contracts_are_deliberately_exported() -> None:
    assert canonical_contracts.SearchIntent is SearchIntent
    assert canonical_contracts.SearchFailureCode is SearchFailureCode
    assert canonical_contracts.SearchWarningCode is SearchWarningCode
    assert canonical_contracts.HashSpaceRef is HashSpaceRef
    assert canonical_contracts.IndexParametersRef is IndexParametersRef
    assert canonical_contracts.IndexGeneration is IndexGeneration
    assert canonical_contracts.SearchExecutionSummary is SearchExecutionSummary
    assert canonical_contracts.MAX_CURSOR_BYTES == MAX_CURSOR_BYTES
    assert canonical_contracts.MAX_FILTER_VALUES == MAX_FILTER_VALUES
    assert canonical_contracts.MAX_INLINE_IMAGE_BYTES == MAX_INLINE_IMAGE_BYTES
    assert canonical_contracts.MAX_LANGUAGE_BYTES == MAX_LANGUAGE_BYTES
    assert canonical_contracts.MAX_MEDIA_TYPE_BYTES == MAX_MEDIA_TYPE_BYTES
    assert canonical_contracts.MAX_PATH_BYTES == MAX_PATH_BYTES
    assert canonical_contracts.MAX_REF_BYTES == MAX_REF_BYTES
    assert canonical_contracts.MAX_TEMPORAL_MILLISECONDS == MAX_TEMPORAL_MILLISECONDS
    assert canonical_contracts.MAX_TEXT_BYTES == MAX_TEXT_BYTES


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


@pytest.mark.parametrize(
    ("constructor", "maximum_bytes"),
    [
        (ArtifactSource, MAX_REF_BYTES),
        (LocalPathSource, MAX_PATH_BYTES),
        (lambda value: MountSource(value, "relative"), MAX_REF_BYTES),
        (lambda value: MountSource("mount", value), MAX_PATH_BYTES),
    ],
)
def test_direct_media_source_strings_enforce_tool_byte_caps(
    constructor: Callable[[str], object],
    maximum_bytes: int,
) -> None:
    assert constructor("x" * maximum_bytes)

    with pytest.raises(ValueError, match="at most"):
        constructor("x" * (maximum_bytes + 1))


def test_direct_inline_source_enforces_payload_and_media_type_byte_caps() -> None:
    assert InlineBytesSource(b"x" * MAX_INLINE_IMAGE_BYTES, "x" * MAX_MEDIA_TYPE_BYTES)

    with pytest.raises(ValueError, match="data"):
        InlineBytesSource(b"x" * (MAX_INLINE_IMAGE_BYTES + 1), "image/png")
    with pytest.raises(ValueError, match="media_type"):
        InlineBytesSource(b"image", "x" * (MAX_MEDIA_TYPE_BYTES + 1))


@pytest.mark.parametrize("text", ["", "  "])
def test_text_query_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        TextQuery(text)


def test_direct_text_query_enforces_text_and_language_byte_caps() -> None:
    assert TextQuery("x" * MAX_TEXT_BYTES, language="x" * MAX_LANGUAGE_BYTES)

    with pytest.raises(ValueError, match="text"):
        TextQuery("x" * (MAX_TEXT_BYTES + 1))
    with pytest.raises(ValueError, match="language"):
        TextQuery("query", language="x" * (MAX_LANGUAGE_BYTES + 1))


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
    assert request.search_intent is SearchIntent.BALANCED


def test_search_request_accepts_explicit_intent_without_coercion() -> None:
    request = SearchRequest(
        query=ImageQuery(ArtifactSource("query-image")),
        targets=frozenset({TargetType.IMAGE}),
        search_intent=SearchIntent.INSTANCE,
    )

    assert request.search_intent is SearchIntent.INSTANCE

    with pytest.raises(TypeError, match="search_intent"):
        SearchRequest(
            query=ImageQuery(ArtifactSource("query-image")),
            targets=frozenset({TargetType.IMAGE}),
            search_intent="instance",  # type: ignore[arg-type]
        )


def test_search_request_preserves_complete_prechange_positional_constructor() -> None:
    filters = SearchFilters(corpus_refs=frozenset({"corpus-1"}))
    request = SearchRequest(
        TextQuery("rain"),
        frozenset({TargetType.IMAGE}),
        7,
        "cursor-1",
        DetailLevel.FULL,
        "profile-1",
        ExecutionProfile.CPU,
        "policy-1",
        filters,
    )

    assert request.top_k == 7
    assert request.cursor == "cursor-1"
    assert request.detail_level is DetailLevel.FULL
    assert request.retrieval_profile_ref == "profile-1"
    assert request.execution_profile is ExecutionProfile.CPU
    assert request.data_policy_ref == "policy-1"
    assert request.filters is filters
    assert request.search_intent is SearchIntent.BALANCED


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


@pytest.mark.parametrize(
    ("constructor", "maximum_bytes"),
    [
        (
            lambda value: SearchRequest(
                TextQuery("rain"), frozenset({TargetType.IMAGE}), cursor=value
            ),
            MAX_CURSOR_BYTES,
        ),
        (
            lambda value: SearchRequest(
                TextQuery("rain"),
                frozenset({TargetType.IMAGE}),
                retrieval_profile_ref=value,
            ),
            MAX_REF_BYTES,
        ),
        (
            lambda value: SearchRequest(
                TextQuery("rain"),
                frozenset({TargetType.IMAGE}),
                data_policy_ref=value,
            ),
            MAX_REF_BYTES,
        ),
    ],
)
def test_direct_search_request_enforces_tool_byte_caps(
    constructor: Callable[[str], SearchRequest],
    maximum_bytes: int,
) -> None:
    assert constructor("x" * maximum_bytes)

    with pytest.raises(ValueError, match="at most"):
        constructor("x" * (maximum_bytes + 1))


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


@pytest.mark.parametrize(
    ("constructor", "maximum_bytes"),
    [
        (lambda values: SearchFilters(corpus_refs=values), MAX_REF_BYTES),
        (lambda values: SearchFilters(artifact_refs=values), MAX_REF_BYTES),
        (lambda values: SearchFilters(media_types=values), MAX_MEDIA_TYPE_BYTES),
    ],
)
def test_direct_search_filters_enforce_tool_count_and_value_caps(
    constructor: Callable[[frozenset[str]], SearchFilters],
    maximum_bytes: int,
) -> None:
    accepted = frozenset(f"value-{index}" for index in range(MAX_FILTER_VALUES))
    assert constructor(accepted)
    assert constructor(frozenset({"x" * maximum_bytes}))

    with pytest.raises(ValueError, match="at most"):
        constructor(frozenset(f"value-{index}" for index in range(MAX_FILTER_VALUES + 1)))
    with pytest.raises(ValueError, match="at most"):
        constructor(frozenset({"x" * (maximum_bytes + 1)}))


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
    assert TemporalRange(
        start_ms=MAX_TEMPORAL_MILLISECONDS - 1,
        end_ms=MAX_TEMPORAL_MILLISECONDS,
    )

    with pytest.raises(ValueError, match="end_ms"):
        TemporalRange(start_ms=100, end_ms=100)


@pytest.mark.parametrize(("start_ms", "end_ms"), [(-1, 2), (2, 1), (True, 2)])
def test_temporal_range_rejects_invalid_coordinates(start_ms: int, end_ms: int) -> None:
    with pytest.raises(ValueError):
        TemporalRange(start_ms=start_ms, end_ms=end_ms)


@pytest.mark.parametrize(
    ("start_ms", "end_ms"),
    [
        (MAX_TEMPORAL_MILLISECONDS, MAX_TEMPORAL_MILLISECONDS + 1),
        (0, MAX_TEMPORAL_MILLISECONDS + 1),
    ],
)
def test_temporal_range_rejects_values_beyond_signed_64_bit_milliseconds(
    start_ms: int,
    end_ms: int,
) -> None:
    with pytest.raises(ValueError, match="milliseconds"):
        TemporalRange(start_ms=start_ms, end_ms=end_ms)


def test_embedding_space_checks_modality_compatibility() -> None:
    space = embedding_space()

    assert space.accepts(QueryModality.TEXT, TargetType.KEYFRAME)
    assert not space.accepts(QueryModality.VIDEO_CLIP, TargetType.KEYFRAME)
    assert not space.accepts(QueryModality.TEXT, TargetType.VIDEO_SEGMENT)


@pytest.mark.parametrize(
    ("field", "value"),
    [("dimension", True), ("schema_version", True)],
)
def test_embedding_space_rejects_invalid_integer_fields(field: str, value: object) -> None:
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


def test_hash_space_preserves_complete_immutable_identity() -> None:
    space = hash_space()

    assert space.target_modalities == frozenset({TargetType.IMAGE})
    with pytest.raises(FrozenInstanceError):
        space.space_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("space_id", " "),
        ("provider_id", ""),
        ("provider_version", " "),
        ("preprocessing_version", ""),
        ("algorithm", "dhash"),
        ("bit_width", 32),
        ("bit_width", True),
        ("hash_version", "phash-32-v1"),
        ("schema_version", 2),
        ("schema_version", True),
    ],
)
def test_hash_space_rejects_incompatible_version_one_identity(
    field: str,
    value: object,
) -> None:
    values = {
        "space_id": "opencv-phash-64-v1",
        "provider_id": "opencv-phash",
        "provider_version": "4.12.0",
        "preprocessing_version": "preprocess-v1",
        "algorithm": "phash",
        "bit_width": 64,
        "hash_version": "phash-64-v1",
        "target_modalities": frozenset({TargetType.IMAGE}),
        "schema_version": 1,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        HashSpaceRef(**values)  # type: ignore[arg-type]


def test_hash_space_rejects_empty_or_duplicate_target_modalities() -> None:
    with pytest.raises(ValueError, match="target_modalities"):
        HashSpaceRef(
            space_id="opencv-phash-64-v1",
            provider_id="opencv-phash",
            provider_version="4.12.0",
            preprocessing_version="preprocess-v1",
            algorithm="phash",
            bit_width=64,
            hash_version="phash-64-v1",
            target_modalities=frozenset(),
        )
    with pytest.raises(ValueError, match="duplicates"):
        HashSpaceRef(
            space_id="opencv-phash-64-v1",
            provider_id="opencv-phash",
            provider_version="4.12.0",
            preprocessing_version="preprocess-v1",
            algorithm="phash",
            bit_width=64,
            hash_version="phash-64-v1",
            target_modalities=(TargetType.IMAGE, TargetType.IMAGE),  # type: ignore[arg-type]
        )


def test_index_parameters_ref_is_content_addressed_and_round_trips() -> None:
    parameters = IndexParametersRef(
        parameter_schema="numpy-flat-f32-v1",
        payload_ref="payload:index-parameters",
        sha256=SHA_A,
    )

    assert IndexParametersRef(**asdict(parameters)) == parameters
    with pytest.raises(FrozenInstanceError):
        parameters.payload_ref = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parameter_schema", " "),
        ("payload_ref", ""),
        ("payload_ref", "/tmp/parameters.json"),
        ("payload_ref", "payload:../parameters"),
        ("payload_ref", "file:parameters.json"),
        ("sha256", "not-a-digest"),
        ("schema_version", 2),
        ("schema_version", True),
    ],
)
def test_index_parameters_ref_rejects_invalid_identity(field: str, value: object) -> None:
    values = {
        "parameter_schema": "numpy-flat-f32-v1",
        "payload_ref": "payload:index-parameters",
        "sha256": SHA_A,
        "schema_version": 1,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        IndexParametersRef(**values)  # type: ignore[arg-type]


def test_index_generation_preserves_complete_immutable_identity() -> None:
    generation = index_generation()

    assert generation.embedding_spaces == (embedding_space(),)
    assert generation.hash_spaces == ()
    assert generation.shard_refs == ("payload:instance-vectors",)
    assert "state" not in {item.name for item in fields(generation)}
    with pytest.raises(FrozenInstanceError):
        generation.generation_id = "changed"  # type: ignore[misc]


def test_index_generation_requires_at_least_one_unique_compatible_space() -> None:
    values = asdict(index_generation())
    values["embedding_spaces"] = ()
    values["hash_spaces"] = ()
    values["index_parameters_ref"] = index_generation().index_parameters_ref
    values["build_processing_fingerprint"] = processing_fingerprint()

    with pytest.raises(ValueError, match="embedding or hash"):
        IndexGeneration(**values)

    duplicate = embedding_space()
    values["embedding_spaces"] = (duplicate, duplicate)
    with pytest.raises(ValueError, match="embedding_spaces"):
        IndexGeneration(**values)

    values["embedding_spaces"] = ()
    duplicate_hash = hash_space()
    values["hash_spaces"] = (duplicate_hash, duplicate_hash)
    with pytest.raises(ValueError, match="hash_spaces"):
        IndexGeneration(**values)


@pytest.mark.parametrize(
    "shard_refs",
    [
        (),
        ("payload:one", "payload:one"),
        ("/tmp/index.npy",),
        ("payload:../index",),
        ("file:index.npy",),
    ],
)
def test_index_generation_rejects_invalid_shard_refs(shard_refs: tuple[str, ...]) -> None:
    values = asdict(index_generation())
    values["embedding_spaces"] = (embedding_space(),)
    values["index_parameters_ref"] = index_generation().index_parameters_ref
    values["build_processing_fingerprint"] = processing_fingerprint()
    values["shard_refs"] = shard_refs

    with pytest.raises((TypeError, ValueError), match="shard_ref"):
        IndexGeneration(**values)


def test_index_generation_requires_aware_timestamp() -> None:
    values = asdict(index_generation())
    values["embedding_spaces"] = (embedding_space(),)
    values["index_parameters_ref"] = index_generation().index_parameters_ref
    values["build_processing_fingerprint"] = processing_fingerprint()
    values["created_at"] = datetime(2026, 8, 21)

    with pytest.raises(ValueError, match="created_at"):
        IndexGeneration(**values)


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
    ("field", "value"),
    [
        ("unit_id", "界" * 43),
        ("evidence_ref", "界" * 86),
        ("artifact_ref", "界" * 342),
    ],
)
def test_searchable_unit_rejects_wire_incompatible_utf8_identity(
    field: str,
    value: str,
) -> None:
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


def test_searchable_unit_accepts_exact_wire_identity_byte_boundaries() -> None:
    unit = SearchableUnit(
        unit_id="a" * 128,
        unit_type=TargetType.IMAGE,
        source_hash=SHA_A,
        artifact_ref="a" * 1_024,
        evidence_ref="a" * 256,
        representation_refs=("a" * 1_024,),
        processing_run_ref="run-1",
        processing_version="v1",
        processing_fingerprint=processing_fingerprint(),
    )

    assert len(unit.unit_id.encode("utf-8")) == 128
    assert len(unit.evidence_ref.encode("utf-8")) == 256


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


def test_compact_keyframe_timestamp_enforces_signed_64_bit_millisecond_bound() -> None:
    assert CompactHit(
        "keyframe-max",
        TargetType.KEYFRAME,
        0.8,
        "evidence-video",
        timestamp_ms=MAX_TEMPORAL_MILLISECONDS,
    )

    with pytest.raises(ValueError, match="milliseconds"):
        CompactHit(
            "keyframe-overflow",
            TargetType.KEYFRAME,
            0.8,
            "evidence-video",
            timestamp_ms=MAX_TEMPORAL_MILLISECONDS + 1,
        )


@pytest.mark.parametrize("brief_score", [math.nan, math.inf, -0.1, 1.1])
def test_compact_hit_rejects_invalid_brief_score(brief_score: float) -> None:
    with pytest.raises(ValueError, match="brief_score"):
        CompactHit("image-1", TargetType.IMAGE, brief_score, "evidence-1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "界" * 43),
        ("evidence_id", "界" * 86),
    ],
)
def test_compact_hit_rejects_wire_incompatible_utf8_identity(
    field: str,
    value: str,
) -> None:
    values = {
        "id": "image-1",
        "target_type": TargetType.IMAGE,
        "brief_score": 0.8,
        "evidence_id": "evidence-1",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        CompactHit(**values)  # type: ignore[arg-type]


def test_compact_hit_accepts_exact_wire_identity_byte_boundaries() -> None:
    hit = CompactHit(
        id="a" * 128,
        target_type=TargetType.IMAGE,
        brief_score=0.8,
        evidence_id="a" * 256,
    )

    assert len(hit.id.encode("utf-8")) == 128
    assert len(hit.evidence_id.encode("utf-8")) == 256


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
                lane_id="concept",
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representation_refs", tuple(f"representation-{index}" for index in range(101))),
        (
            "lane_scores",
            (
                *(
                    LaneScore(lane, 0.5, 0.5, "score", "v1", 1)
                    for lane in ("content", "perceptual", "instance", "concept")
                ),
                LaneScore("content", 0.4, 0.4, "other", "v2", 2),
            ),
        ),
        ("index_generation_ids", tuple(f"generation-{index}" for index in range(5))),
        ("lineage_refs", tuple(f"lineage-{index}" for index in range(65))),
    ],
)
def test_full_hit_rejects_wire_incompatible_collection_bounds(
    field: str,
    value: object,
) -> None:
    values = {
        "id": "image-1",
        "target_type": TargetType.IMAGE,
        "brief_score": 0.8,
        "evidence_id": "evidence-1",
        field: value,
    }

    with pytest.raises(ValueError, match=field.removesuffix("s")):
        FullHit(**values)  # type: ignore[arg-type]


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
            lane_id="concept",
            raw_score=raw_score,
            normalized_score=normalized_score,
            score_semantics="distance",
            calibration_version="v1",
            rank=1,
        )


def test_lane_score_rejects_boolean_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        LaneScore(
            lane_id="concept",
            raw_score=0.5,
            normalized_score=0.5,
            score_semantics="distance",
            calibration_version="v1",
            rank=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lane_id", "界" * 43),
        ("score_semantics", "界" * 43),
        ("calibration_version", "界" * 43),
        ("rank", 100_001),
    ],
)
def test_lane_score_rejects_wire_incompatible_values(field: str, value: object) -> None:
    values = {
        "lane_id": "concept",
        "raw_score": 0.5,
        "normalized_score": 0.5,
        "score_semantics": "cosine",
        "calibration_version": "v1",
        "rank": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        LaneScore(**values)  # type: ignore[arg-type]


def test_search_result_allows_a_successful_empty_result() -> None:
    result = SearchResult(
        status=SearchStatus.SUCCEEDED,
        query_id="query-1",
        hits=(),
    )

    assert result.hits == ()
    assert not result.degraded
    assert result.next_cursor is None
    assert result.failure_code is None
    assert result.execution_summary is None


def test_search_result_preserves_complete_prechange_positional_constructor() -> None:
    result = SearchResult(
        SearchStatus.SUCCEEDED,
        "query-1",
        True,
        (SearchWarningCode.RETRIEVAL_PREVIEW,),
        (),
        "cursor-1",
    )

    assert result.degraded is True
    assert result.warning_codes == (SearchWarningCode.RETRIEVAL_PREVIEW,)
    assert result.hits == ()
    assert result.next_cursor == "cursor-1"
    assert result.failure_code is None
    assert result.execution_summary is None


def test_search_result_enforces_primary_failure_code_invariant() -> None:
    failed = SearchResult(
        status=SearchStatus.FAILED,
        query_id="query-failed",
        failure_code=SearchFailureCode.INVALID_QUERY,
    )

    assert failed.failure_code is SearchFailureCode.INVALID_QUERY

    with pytest.raises(ValueError, match="failure_code"):
        SearchResult(status=SearchStatus.FAILED, query_id="query-failed")
    with pytest.raises(ValueError, match="failure_code"):
        SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id="query-succeeded",
            failure_code=SearchFailureCode.INTERNAL,
        )
    with pytest.raises(TypeError, match="failure_code"):
        SearchResult(
            status=SearchStatus.FAILED,
            query_id="query-failed",
            failure_code="invalid_query",  # type: ignore[arg-type]
        )


def test_search_failure_codes_distinguish_capacity_limit_from_backpressure() -> None:
    assert SearchFailureCode.CAPACITY_EXCEEDED.value == "capacity_exceeded"
    assert SearchFailureCode.BACKPRESSURE.value == "backpressure"


def test_full_detail_zero_hit_result_carries_bounded_execution_summary() -> None:
    summary = execution_summary()
    result = SearchResult(
        status=SearchStatus.SUCCEEDED,
        query_id="query-1",
        warning_codes=(SearchWarningCode.RETRIEVAL_PREVIEW,),
        hits=(),
        execution_summary=summary,
    )

    assert result.execution_summary is summary
    assert summary.selected_lanes == ("content", "perceptual", "instance", "concept")
    assert not hasattr(summary, "source")
    assert not hasattr(summary, "path")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("search_intent", "balanced"),
        ("selected_lanes", ()),
        ("selected_lanes", ("content", "unknown")),
        ("selected_lanes", ("content", "content")),
        ("provider_ids", ("provider-1",) * 5),
        ("index_generation_ids", ("/tmp/generation",)),
        ("bundle_manifest_sha256s", (SHA_A, SHA_B, "c" * 64)),
        ("bundle_manifest_sha256s", ("not-a-digest",)),
        ("resolved_profile_ref", "file:profile.json"),
        ("execution_profile", "cpu"),
        ("activation_tier", "active"),
        ("schema_version", 2),
    ],
)
def test_execution_summary_rejects_unbounded_or_unsafe_values(
    field: str,
    value: object,
) -> None:
    values = {
        "search_intent": SearchIntent.BALANCED,
        "selected_lanes": ("content", "perceptual"),
        "provider_ids": ("opencv-phash",),
        "embedding_spaces": (embedding_space(),),
        "hash_spaces": (hash_space(),),
        "index_generation_ids": ("generation-1",),
        "bundle_manifest_sha256s": (SHA_A,),
        "resolved_profile_ref": "personal-lite",
        "execution_profile": ExecutionProfile.CPU,
        "activation_tier": "preview",
        "schema_version": 1,
    }
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        SearchExecutionSummary(**values)  # type: ignore[arg-type]


def test_search_result_rejects_unknown_or_duplicate_warning_codes() -> None:
    with pytest.raises(TypeError, match="warning_codes"):
        SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id="query-1",
            warning_codes=("provider_unavailable",),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="warning_codes"):
        SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id="query-1",
            warning_codes=(
                SearchWarningCode.PROVIDER_UNAVAILABLE,
                SearchWarningCode.PROVIDER_UNAVAILABLE,
            ),
        )


def test_search_result_requires_warning_when_degraded() -> None:
    with pytest.raises(ValueError, match="warning"):
        SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id="query-1",
            degraded=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_id", "界" * 342),
        ("next_cursor", "界" * 1_366),
        (
            "hits",
            tuple(
                CompactHit(f"image-{index}", TargetType.IMAGE, 0.5, f"evidence-{index}")
                for index in range(101)
            ),
        ),
    ],
)
def test_search_result_rejects_wire_incompatible_output_bounds(
    field: str,
    value: object,
) -> None:
    values = {
        "status": SearchStatus.SUCCEEDED,
        "query_id": "query-1",
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        SearchResult(**values)  # type: ignore[arg-type]
