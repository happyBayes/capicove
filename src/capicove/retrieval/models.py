"""Immutable values for multimodal retrieval inference."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from capicove.contracts import (
    EmbeddingSpaceRef,
    QueryModality,
    SearchableUnit,
    TargetType,
    TextChunkLocator,
)


_MAX_TEXT_SNIPPET_LENGTH = 2_048


@dataclass(frozen=True, slots=True)
class ProviderMatch:
    """Contain one model-native image-text comparison."""

    logit: float
    match_probability: float


@dataclass(frozen=True, slots=True)
class RankedMatch:
    """Contain one projected and ranked text candidate."""

    text: str
    logit: float
    match_probability: float
    candidate_probability: float
    rank: int


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Contain ranked candidates and model lineage."""

    model_id: str
    model_revision: str
    device: str
    matches: tuple[RankedMatch, ...]


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Contain normalized vectors and model lineage."""

    model_id: str
    model_revision: str
    device: str
    dimension: int
    embeddings: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class TextEmbedding:
    """Contain one text vector bound to its complete embedding-space identity."""

    space: EmbeddingSpaceRef
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise TypeError("space must be an EmbeddingSpaceRef")
        if not self.space.accepts(QueryModality.TEXT, TargetType.TEXT_CHUNK):
            raise ValueError("space must accept text queries and text chunks")
        vector = tuple(self.vector)
        if len(vector) != self.space.dimension:
            raise ValueError("vector dimension must match its embedding space")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("vector must contain only finite numbers")
        object.__setattr__(self, "vector", tuple(float(value) for value in vector))


@dataclass(frozen=True, slots=True)
class TextIndexEntry:
    """Contain one canonical text unit and its provider-indexable fields."""

    unit: SearchableUnit
    text: str
    corpus_refs: tuple[str, ...] = ()
    media_type: str | None = None
    language: str | None = None
    embeddings: tuple[TextEmbedding, ...] = ()

    def __post_init__(self) -> None:
        _require_text_unit(self.unit)
        _require_string("text", self.text)
        _require_non_blank("text", self.text)
        embeddings = tuple(self.embeddings)
        if any(not isinstance(embedding, TextEmbedding) for embedding in embeddings):
            raise TypeError("embeddings must contain TextEmbedding values")
        embedding_space_ids = tuple(embedding.space.space_id for embedding in embeddings)
        if len(set(embedding_space_ids)) != len(embedding_space_ids):
            raise ValueError("embeddings must not contain duplicate embedding spaces")
        unit_spaces = {space.space_id: space for space in self.unit.embedding_spaces}
        embedded_spaces = {embedding.space.space_id: embedding.space for embedding in embeddings}
        if embedded_spaces != unit_spaces:
            raise ValueError("embeddings must match the unit embedding spaces")
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(
            self,
            "corpus_refs",
            _freeze_strings("corpus_ref", self.corpus_refs),
        )
        for field_name in ("media_type", "language"):
            value = getattr(self, field_name)
            if value is not None:
                _require_string(field_name, value)
                _require_non_blank(field_name, value)


@dataclass(frozen=True, slots=True)
class TextCandidate:
    """Contain one bounded provider candidate from a pinned index generation."""

    unit_id: str
    evidence_ref: str
    text_locator: TextChunkLocator
    snippet: str
    raw_score: float
    score_semantics: str
    rank: int
    provider_id: str
    index_generation_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "unit_id",
            "evidence_ref",
            "score_semantics",
            "provider_id",
            "index_generation_id",
        ):
            value = getattr(self, field_name)
            _require_string(field_name, value)
            _require_non_blank(field_name, value)
        if not isinstance(self.text_locator, TextChunkLocator):
            raise TypeError("text_locator must be a TextChunkLocator")
        _require_string("snippet", self.snippet)
        _require_non_blank("snippet", self.snippet)
        if len(self.snippet) > _MAX_TEXT_SNIPPET_LENGTH:
            raise ValueError("snippet must contain at most 2048 characters")
        normalized_snippet = self.snippet.casefold()
        if "<em>" in normalized_snippet or "</em>" in normalized_snippet:
            raise ValueError("snippet must not contain provider highlight markup")
        if isinstance(self.raw_score, bool) or not isinstance(
            self.raw_score,
            (int, float),
        ):
            raise ValueError("raw_score must be finite")
        if not math.isfinite(self.raw_score):
            raise ValueError("raw_score must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True, slots=True)
class TextEvidence:
    """Contain complete immutable text at the evidence access boundary."""

    unit: SearchableUnit
    text: str

    def __post_init__(self) -> None:
        _require_text_unit(self.unit)
        _require_string("text", self.text)
        _require_non_blank("text", self.text)


def _require_text_unit(unit: object) -> None:
    if not isinstance(unit, SearchableUnit):
        raise TypeError("unit must be a SearchableUnit")
    if unit.unit_type is not TargetType.TEXT_CHUNK:
        raise ValueError("unit must be a text chunk")


def _require_string(field_name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")


def _require_non_blank(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _freeze_strings(field_name: str, values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} values must be an iterable of strings")
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} values must not contain duplicates")
    for value in materialized:
        _require_string(field_name, value)
        _require_non_blank(field_name, value)
    return materialized
