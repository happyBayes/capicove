"""Tests for Elasticsearch text index mapping and document codecs."""

from __future__ import annotations

import math

import pytest
from capicove_retriever_elasticsearch.codec import (
    build_managed_text_index_mapping,
    build_text_index_mapping,
    decode_text_candidates,
    encode_text_index_entry,
    inspect_managed_text_index_mapping,
    physical_index_name,
    text_index_mapping_sha256,
    vector_field_name,
)

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    VectorNormalization,
)
from capicove.retrieval import (
    TextEmbedding,
    TextIndexEntry,
    TextIndexGenerationManifest,
)


SOURCE_HASH = "a" * 64
MAPPING_SHA256 = (
    "97fe07c89cff5a75c994bfd389b32b2e4d9050aa707eb458ec132e169ce5165d"  # pragma: allowlist secret
)
MANIFEST_SHA256 = (
    "773525821682a7f211d204b226f937a135fe3b73b5cbd32cb51409d366f1bcba"  # pragma: allowlist secret
)


def embedding_space(
    *,
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
) -> EmbeddingSpaceRef:
    """Return one compatible three-dimensional text space."""
    return EmbeddingSpaceRef(
        space_id="text-dense-v1",
        provider_id="embedding-provider",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=distance_metric,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def text_entry(locator: TextChunkLocator) -> TextIndexEntry:
    """Return one vector-bearing text index entry."""
    space = embedding_space()
    unit = SearchableUnit(
        unit_id="text-1",
        unit_type=TargetType.TEXT_CHUNK,
        source_hash=SOURCE_HASH,
        artifact_ref="artifact-1",
        evidence_ref="evidence-1",
        representation_refs=("representation-1",),
        processing_run_ref="run-1",
        processing_version="text-v1",
        processing_fingerprint=ProcessingFingerprint.from_manifest(
            {"source_hash": SOURCE_HASH, "processor_id": "capicove.text"}
        ),
        embedding_spaces=(space,),
        text_locator=locator,
    )
    return TextIndexEntry(
        unit=unit,
        text="A complete immutable matching passage.",
        embeddings=(TextEmbedding(space, (0.1, 0.2, 0.3)),),
        corpus_refs=("corpus-1",),
        media_type="text/plain",
        language="en",
    )


def generation_manifest() -> TextIndexGenerationManifest:
    """Return one generation bound to the accepted text index mapping."""
    return TextIndexGenerationManifest(
        generation_id="generation-42",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-1",),
        searchable_unit_manifest_sha256="a" * 64,
        content_sha256="b" * 64,
        mapping_sha256=MAPPING_SHA256,
        embedding_spaces=(embedding_space(),),
        retriever_id="capicove.retriever.elasticsearch",
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=2,
    )


def test_provider_names_hash_canonical_identities_into_safe_names() -> None:
    assert physical_index_name("capicove-text", "generation-42") == (
        "capicove-text-g-79091da8c3ddbe5b32099e4c"
    )
    assert vector_field_name("text-dense-v1") == "vector_77bad12a9fc0e553b1a845c3"

    with pytest.raises(ValueError, match="generation_id"):
        physical_index_name("capicove-text", " ")
    with pytest.raises(ValueError, match="space_id"):
        vector_field_name("")


@pytest.mark.parametrize(
    ("metric", "similarity"),
    [
        (DistanceMetric.COSINE, "cosine"),
        (DistanceMetric.INNER_PRODUCT, "dot_product"),
        (DistanceMetric.EUCLIDEAN, "l2_norm"),
    ],
)
def test_mapping_is_strict_and_preserves_vector_space_semantics(
    metric: DistanceMetric,
    similarity: str,
) -> None:
    mapping = build_text_index_mapping((embedding_space(distance_metric=metric),))
    properties = mapping["properties"]

    assert mapping["dynamic"] == "strict"
    assert properties["text"] == {"type": "text"}
    assert properties["preview"] == {
        "type": "keyword",
        "index": False,
        "doc_values": False,
    }
    assert properties["origin"] == {"type": "keyword"}
    assert properties["start_char"] == {"type": "long"}
    assert properties["page_number"] == {"type": "integer"}
    assert properties["start_ms"] == {"type": "long"}
    assert properties["vector_77bad12a9fc0e553b1a845c3"] == {
        "type": "dense_vector",
        "dims": 3,
        "index": True,
        "similarity": similarity,
        "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
    }


def test_mapping_rejects_duplicate_embedding_spaces() -> None:
    space = embedding_space()

    with pytest.raises(ValueError, match="duplicate"):
        build_text_index_mapping((space, space))


def test_vector_index_options_are_explicit_and_stable_across_server_defaults() -> None:
    mapping = build_text_index_mapping((embedding_space(),))
    vector = mapping["properties"][vector_field_name("text-dense-v1")]
    assert vector["index_options"] == {"type": "hnsw", "m": 16, "ef_construction": 100}


@pytest.mark.parametrize(
    "field,value", [("type", "int8_hnsw"), ("m", 32), ("ef_construction", 200)]
)
def test_changed_vector_index_options_fail_integrity_validation(field: str, value: object) -> None:
    manifest = generation_manifest()
    mapping = build_managed_text_index_mapping(manifest)
    mapping["properties"][vector_field_name("text-dense-v1")]["index_options"][field] = value
    assert inspect_managed_text_index_mapping(mapping, manifest) == ("mapping_digest_mismatch",)


def test_managed_mapping_has_deterministic_integrity_identity() -> None:
    manifest = generation_manifest()

    assert text_index_mapping_sha256(manifest.embedding_spaces) == MAPPING_SHA256
    mapping = build_managed_text_index_mapping(manifest)

    assert mapping["_meta"] == {
        "capicove": {
            "schema_version": 1,
            "generation_id": "generation-42",
            "manifest_sha256": MANIFEST_SHA256,
            "mapping_sha256": MAPPING_SHA256,
            "content_sha256": "b" * 64,
            "expected_unit_count": 2,
            "index_format_version": "elasticsearch-text-v1",
        }
    }
    assert inspect_managed_text_index_mapping(mapping, manifest) == ()


@pytest.mark.parametrize(
    ("field_name", "bad_value", "issue_code"),
    [
        ("schema_version", 2, "schema_mismatch"),
        ("generation_id", "different", "generation_id_mismatch"),
        ("manifest_sha256", "f" * 64, "manifest_digest_mismatch"),
        ("mapping_sha256", "f" * 64, "mapping_digest_mismatch"),
        ("content_sha256", "f" * 64, "content_digest_mismatch"),
        ("expected_unit_count", 3, "expected_unit_count_mismatch"),
        ("index_format_version", "different", "index_format_mismatch"),
    ],
)
def test_managed_mapping_reports_stable_metadata_mismatch_codes(
    field_name: str,
    bad_value: object,
    issue_code: str,
) -> None:
    manifest = generation_manifest()
    mapping = build_managed_text_index_mapping(manifest)
    metadata = mapping["_meta"]["capicove"]
    metadata[field_name] = bad_value

    assert issue_code in inspect_managed_text_index_mapping(mapping, manifest)


def test_managed_mapping_detects_unexpected_physical_schema() -> None:
    manifest = generation_manifest()
    mapping = build_managed_text_index_mapping(manifest)
    mapping["properties"]["text"] = {"type": "keyword"}

    assert inspect_managed_text_index_mapping(mapping, manifest) == ("mapping_digest_mismatch",)


@pytest.mark.parametrize(
    ("locator", "expected_locator_fields"),
    [
        (
            TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 38)),
            {"origin": "document", "start_char": 0, "end_char": 38},
        ),
        (
            TextChunkLocator(TextOrigin.OCR, CharacterRange(0, 38), page_number=2),
            {
                "origin": "ocr",
                "start_char": 0,
                "end_char": 38,
                "page_number": 2,
            },
        ),
        (
            TextChunkLocator(
                TextOrigin.ASR,
                CharacterRange(0, 38),
                time_range=TemporalRange(1_000, 2_000),
            ),
            {
                "origin": "asr",
                "start_char": 0,
                "end_char": 38,
                "start_ms": 1_000,
                "end_ms": 2_000,
            },
        ),
    ],
)
def test_document_encoding_preserves_lineage_filters_locator_and_vector(
    locator: TextChunkLocator,
    expected_locator_fields: dict[str, object],
) -> None:
    document = encode_text_index_entry(text_entry(locator))

    assert document == {
        "unit_id": "text-1",
        "evidence_ref": "evidence-1",
        "source_hash": SOURCE_HASH,
        "artifact_ref": "artifact-1",
        "representation_refs": ["representation-1"],
        "processing_run_ref": "run-1",
        "processing_version": "text-v1",
        "processing_fingerprint": text_entry(locator).unit.processing_fingerprint.sha256,
        "corpus_refs": ["corpus-1"],
        "media_type": "text/plain",
        "language": "en",
        "text": "A complete immutable matching passage.",
        "preview": "A complete immutable matching passage.",
        "vector_77bad12a9fc0e553b1a845c3": [0.1, 0.2, 0.3],
        **expected_locator_fields,
    }


def test_document_encoding_bounds_preview_without_truncating_indexed_text() -> None:
    original = text_entry(TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 3_000)))
    entry = TextIndexEntry(
        unit=original.unit,
        text="x" * 3_000,
        embeddings=original.embeddings,
        corpus_refs=original.corpus_refs,
        media_type=original.media_type,
        language=original.language,
    )

    document = encode_text_index_entry(entry)

    assert document["text"] == "x" * 3_000
    assert document["preview"] == "x" * 2_048


def test_hit_decoding_strips_highlights_and_assigns_stable_ranks() -> None:
    response = {
        "hits": {
            "hits": [
                {
                    "_id": "text-1",
                    "_score": 5.5,
                    "_source": {
                        "unit_id": "text-1",
                        "evidence_ref": "evidence-1",
                        "origin": "ocr",
                        "start_char": 10,
                        "end_char": 40,
                        "page_number": 2,
                        "preview": "Complete source text is not returned.",
                    },
                    "highlight": {"text": ["A <em>matching</em> passage."]},
                },
                {
                    "_id": "text-2",
                    "_score": 4.25,
                    "_source": {
                        "unit_id": "text-2",
                        "evidence_ref": "evidence-2",
                        "origin": "document",
                        "start_char": 50,
                        "end_char": 80,
                        "preview": "Second matching passage.",
                    },
                },
            ]
        }
    }

    candidates = decode_text_candidates(
        response,
        provider_id="capicove.retriever.elasticsearch",
        generation_id="generation-42",
        score_semantics="elasticsearch_bm25_score",
    )

    assert [(candidate.unit_id, candidate.rank) for candidate in candidates] == [
        ("text-1", 1),
        ("text-2", 2),
    ]
    assert candidates[0].snippet == "A matching passage."
    assert candidates[0].text_locator.page_number == 2
    assert candidates[1].snippet == "Second matching passage."
    assert candidates[1].index_generation_id == "generation-42"


def test_hit_decoding_bounds_fallback_asr_text() -> None:
    response = {
        "hits": {
            "hits": [
                {
                    "_id": "text-asr",
                    "_score": 0.75,
                    "_source": {
                        "unit_id": "text-asr",
                        "evidence_ref": "evidence-asr",
                        "origin": "asr",
                        "start_char": 0,
                        "end_char": 3_000,
                        "start_ms": 1_000,
                        "end_ms": 5_000,
                        "preview": "x" * 3_000,
                    },
                }
            ]
        }
    }

    candidate = decode_text_candidates(
        response,
        provider_id="capicove.retriever.elasticsearch",
        generation_id="generation-42",
        score_semantics="elasticsearch_knn_score",
    )[0]

    assert candidate.snippet == "x" * 2_048
    assert candidate.text_locator.time_range == TemporalRange(1_000, 5_000)


@pytest.mark.parametrize(
    "origin,coordinates",
    [
        ("ocr", {"page_number": 1, "start_ms": 0, "end_ms": 1}),
        ("asr", {"page_number": 1}),
        ("document", {"page_number": 1}),
        ("document", {"start_ms": 0, "end_ms": 1}),
        ("ocr", {"start_ms": 0}),
    ],
)
def test_decoding_rejects_ambiguous_or_malformed_temporal_coordinates(
    origin: str,
    coordinates: dict[str, int],
) -> None:
    source = {
        "unit_id": "text-1",
        "evidence_ref": "evidence-1",
        "origin": origin,
        "start_char": 0,
        "end_char": 10,
        "preview": "authored",
        **coordinates,
    }
    response = {"hits": {"hits": [{"_id": "text-1", "_score": 1.0, "_source": source}]}}
    with pytest.raises(ValueError):
        decode_text_candidates(
            response,
            provider_id="capicove.retriever.elasticsearch",
            generation_id="generation-42",
            score_semantics="bm25",
        )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"hits": {"hits": "not-a-list"}},
        {
            "hits": {
                "hits": [
                    {
                        "_id": "different-id",
                        "_score": 1.0,
                        "_source": {
                            "unit_id": "text-1",
                            "evidence_ref": "evidence-1",
                            "origin": "document",
                            "start_char": 0,
                            "end_char": 10,
                            "preview": "VERY_SECRET_CHUNK",
                        },
                    }
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_id": "text-1",
                        "_score": math.nan,
                        "_source": {"preview": "VERY_SECRET_CHUNK"},
                    }
                ]
            }
        },
    ],
)
def test_hit_decoding_rejects_malformed_responses_without_echoing_content(
    response: object,
) -> None:
    with pytest.raises(ValueError, match="malformed Elasticsearch response") as raised:
        decode_text_candidates(
            response,
            provider_id="capicove.retriever.elasticsearch",
            generation_id="generation-42",
            score_semantics="elasticsearch_bm25_score",
        )

    assert "VERY_SECRET_CHUNK" not in str(raised.value)
