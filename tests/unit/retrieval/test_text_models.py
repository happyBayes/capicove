"""Tests for provider-neutral text retrieval boundary values."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    TextChunkLocator,
    TextOrigin,
    VectorNormalization,
)
from capicove.retrieval import (
    TextCandidate,
    TextEmbedding,
    TextEvidence,
    TextIndexEntry,
)


SOURCE_HASH = "a" * 64


def text_unit(
    *,
    embedding_spaces: tuple[EmbeddingSpaceRef, ...] = (),
) -> SearchableUnit:
    """Return one canonical document-text unit."""
    return SearchableUnit(
        unit_id="text-1",
        unit_type=TargetType.TEXT_CHUNK,
        source_hash=SOURCE_HASH,
        artifact_ref="artifact-1",
        evidence_ref="evidence-1",
        representation_refs=("representation-1",),
        processing_run_ref="run-1",
        processing_version="text-chunk-v1",
        processing_fingerprint=ProcessingFingerprint.from_manifest(
            {"source_hash": SOURCE_HASH, "processor_id": "capicove.text"}
        ),
        embedding_spaces=embedding_spaces,
        text_locator=TextChunkLocator(
            TextOrigin.DOCUMENT,
            CharacterRange(0, 25),
        ),
    )


def text_embedding_space() -> EmbeddingSpaceRef:
    """Return one compatible text query and text target vector space."""
    return EmbeddingSpaceRef(
        space_id="text-dense-v1",
        provider_id="embedding-provider",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def image_unit() -> SearchableUnit:
    """Return one canonical non-text unit."""
    return SearchableUnit(
        unit_id="image-1",
        unit_type=TargetType.IMAGE,
        source_hash=SOURCE_HASH,
        artifact_ref="artifact-1",
        evidence_ref="evidence-1",
        representation_refs=("representation-1",),
        processing_run_ref="run-1",
        processing_version="image-v1",
        processing_fingerprint=ProcessingFingerprint.from_manifest(
            {"source_hash": SOURCE_HASH, "processor_id": "capicove.image"}
        ),
    )


def candidate_values() -> dict[str, object]:
    """Return valid literal values for one lexical candidate."""
    return {
        "unit_id": "text-1",
        "evidence_ref": "evidence-1",
        "text_locator": TextChunkLocator(
            TextOrigin.DOCUMENT,
            CharacterRange(0, 25),
        ),
        "snippet": "A bounded matching passage.",
        "raw_score": 7.25,
        "score_semantics": "bm25",
        "rank": 1,
        "provider_id": "capicove.retriever.elasticsearch",
        "index_generation_id": "generation-42",
    }


def test_text_index_entry_freezes_filter_values_and_preserves_lineage() -> None:
    entry = TextIndexEntry(
        unit=text_unit(),
        text="A complete immutable text chunk.",
        corpus_refs=["corpus-b", "corpus-a"],  # type: ignore[arg-type]
        media_type="text/plain",
        language="en",
    )

    assert entry.corpus_refs == ("corpus-b", "corpus-a")
    assert entry.unit.source_hash == SOURCE_HASH
    with pytest.raises(FrozenInstanceError):
        entry.text = "changed"  # type: ignore[misc]


def test_text_index_entry_preserves_existing_positional_corpus_refs() -> None:
    entry = TextIndexEntry(
        text_unit(),
        "A complete immutable text chunk.",
        ("corpus-1",),
    )

    assert entry.corpus_refs == ("corpus-1",)
    assert entry.embeddings == ()


def test_text_embedding_freezes_a_compatible_finite_vector() -> None:
    embedding = TextEmbedding(
        space=text_embedding_space(),
        vector=[0.1, 0.2, 0.3],  # type: ignore[arg-type]
    )

    assert embedding.vector == (0.1, 0.2, 0.3)
    with pytest.raises(FrozenInstanceError):
        embedding.vector = (1.0, 2.0, 3.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "vector",
    [
        (0.1, 0.2),
        (0.1, 0.2, 0.3, 0.4),
        (0.1, math.nan, 0.3),
        (0.1, math.inf, 0.3),
        (0.1, True, 0.3),
    ],
)
def test_text_embedding_rejects_invalid_vector_values(
    vector: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="vector"):
        TextEmbedding(space=text_embedding_space(), vector=vector)


@pytest.mark.parametrize(
    ("query_modalities", "target_modalities"),
    [
        (frozenset({QueryModality.IMAGE}), frozenset({TargetType.TEXT_CHUNK})),
        (frozenset({QueryModality.TEXT}), frozenset({TargetType.IMAGE})),
    ],
)
def test_text_embedding_rejects_an_incompatible_embedding_space(
    query_modalities: frozenset[QueryModality],
    target_modalities: frozenset[TargetType],
) -> None:
    incompatible = EmbeddingSpaceRef(
        space_id="incompatible-v1",
        provider_id="embedding-provider",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=query_modalities,
        target_modalities=target_modalities,
    )

    with pytest.raises(ValueError, match="text queries and text chunks"):
        TextEmbedding(space=incompatible, vector=(0.1, 0.2, 0.3))


def test_text_index_entry_requires_exact_embedding_space_parity() -> None:
    space = text_embedding_space()
    embedding = TextEmbedding(space=space, vector=(0.1, 0.2, 0.3))
    entry = TextIndexEntry(
        unit=text_unit(embedding_spaces=(space,)),
        text="A complete immutable text chunk.",
        embeddings=(embedding,),
    )

    assert entry.embeddings == (embedding,)

    with pytest.raises(ValueError, match="embedding spaces"):
        TextIndexEntry(
            unit=text_unit(embedding_spaces=(space,)),
            text="A complete immutable text chunk.",
        )
    with pytest.raises(ValueError, match="embedding spaces"):
        TextIndexEntry(
            unit=text_unit(),
            text="A complete immutable text chunk.",
            embeddings=(embedding,),
        )
    with pytest.raises(ValueError, match="duplicate"):
        TextIndexEntry(
            unit=text_unit(embedding_spaces=(space,)),
            text="A complete immutable text chunk.",
            embeddings=(embedding, embedding),
        )


@pytest.mark.parametrize("text", ["", " "])
def test_text_index_entry_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        TextIndexEntry(unit=text_unit(), text=text)


def test_text_index_entry_rejects_non_text_units() -> None:
    with pytest.raises(ValueError, match="text chunk"):
        TextIndexEntry(unit=image_unit(), text="image metadata")


@pytest.mark.parametrize("corpus_refs", [("corpus-1", "corpus-1"), (" ",)])
def test_text_index_entry_rejects_invalid_corpus_refs(
    corpus_refs: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="corpus_ref"):
        TextIndexEntry(
            unit=text_unit(),
            text="A complete immutable text chunk.",
            corpus_refs=corpus_refs,
        )


def test_text_evidence_contains_complete_text_at_the_evidence_boundary() -> None:
    evidence = TextEvidence(unit=text_unit(), text="Complete evidence text.")

    assert evidence.unit.evidence_ref == "evidence-1"
    assert evidence.text == "Complete evidence text."


@pytest.mark.parametrize(
    ("unit", "text"),
    [(image_unit(), "metadata"), (text_unit(), " ")],
)
def test_text_evidence_rejects_non_text_units_and_blank_text(
    unit: SearchableUnit,
    text: str,
) -> None:
    with pytest.raises(ValueError):
        TextEvidence(unit=unit, text=text)


def test_text_candidate_preserves_raw_lane_observations_and_generation() -> None:
    candidate = TextCandidate(**candidate_values())  # type: ignore[arg-type]

    assert candidate.raw_score == 7.25
    assert candidate.rank == 1
    assert candidate.index_generation_id == "generation-42"
    assert not hasattr(candidate, "text")
    with pytest.raises(FrozenInstanceError):
        candidate.rank = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("unit_id", " "),
        ("evidence_ref", ""),
        ("snippet", " "),
        ("snippet", "x" * 2_049),
        ("snippet", "<em>match</em>"),
        ("raw_score", math.nan),
        ("raw_score", math.inf),
        ("raw_score", True),
        ("rank", 0),
        ("rank", True),
        ("score_semantics", " "),
        ("provider_id", ""),
        ("index_generation_id", " "),
        ("text_locator", "not-a-locator"),
    ],
)
def test_text_candidate_rejects_invalid_boundary_values(
    field: str,
    value: object,
) -> None:
    values = candidate_values()
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=field):
        TextCandidate(**values)  # type: ignore[arg-type]
