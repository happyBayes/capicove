"""Pure Elasticsearch mapping and document codecs for canonical text chunks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
)
from capicove.retrieval import (
    TextCandidate,
    TextIndexEntry,
    TextIndexGenerationManifest,
)


_IDENTITY_LIMIT = 500
_INDEX_PREFIX = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,39})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SNIPPET_LENGTH = 2_048
_SIMILARITY = {
    DistanceMetric.COSINE: "cosine",
    DistanceMetric.INNER_PRODUCT: "dot_product",
    DistanceMetric.EUCLIDEAN: "l2_norm",
}


def physical_index_name(prefix: str, generation_id: str) -> str:
    """Return a safe deterministic physical index name for one generation."""
    if not isinstance(prefix, str) or _INDEX_PREFIX.fullmatch(prefix) is None:
        raise ValueError("prefix must be a safe lowercase index prefix")
    _require_identity("generation_id", generation_id)
    digest = hashlib.sha256(generation_id.encode("utf-8")).hexdigest()
    return f"{prefix}-g-{digest[:24]}"


def vector_field_name(space_id: str) -> str:
    """Return a safe deterministic dense-vector field for one embedding space."""
    _require_identity("space_id", space_id)
    digest = hashlib.sha256(space_id.encode("utf-8")).hexdigest()
    return f"vector_{digest[:24]}"


def build_text_index_mapping(
    spaces: tuple[EmbeddingSpaceRef, ...],
) -> dict[str, Any]:
    """Build one strict mapping from reviewed fields and embedding spaces."""
    materialized = tuple(spaces)
    if any(not isinstance(space, EmbeddingSpaceRef) for space in materialized):
        raise TypeError("spaces must contain EmbeddingSpaceRef values")
    space_ids = tuple(space.space_id for space in materialized)
    if len(set(space_ids)) != len(space_ids):
        raise ValueError("spaces must not contain duplicate space IDs")
    properties: dict[str, Any] = {
        "unit_id": {"type": "keyword"},
        "evidence_ref": {"type": "keyword"},
        "source_hash": {"type": "keyword"},
        "artifact_ref": {"type": "keyword"},
        "representation_refs": {"type": "keyword"},
        "processing_run_ref": {"type": "keyword"},
        "processing_version": {"type": "keyword"},
        "processing_fingerprint": {"type": "keyword"},
        "corpus_refs": {"type": "keyword"},
        "media_type": {"type": "keyword"},
        "language": {"type": "keyword"},
        "text": {"type": "text"},
        "preview": {"type": "keyword", "index": False, "doc_values": False},
        "origin": {"type": "keyword"},
        "start_char": {"type": "long"},
        "end_char": {"type": "long"},
        "page_number": {"type": "integer"},
        "start_ms": {"type": "long"},
        "end_ms": {"type": "long"},
    }
    for space in materialized:
        if not space.accepts(QueryModality.TEXT, TargetType.TEXT_CHUNK):
            raise ValueError("spaces must accept text queries and text chunks")
        properties[vector_field_name(space.space_id)] = {
            "type": "dense_vector",
            "dims": space.dimension,
            "index": True,
            "similarity": _SIMILARITY[space.distance_metric],
            "index_options": {"type": "hnsw", "m": 16, "ef_construction": 100},
        }
    return {"dynamic": "strict", "properties": properties}


def text_index_mapping_sha256(
    spaces: tuple[EmbeddingSpaceRef, ...],
) -> str:
    """Return the canonical digest of one provider text index mapping."""
    return _mapping_digest(build_text_index_mapping(spaces))


def build_managed_text_index_mapping(
    manifest: TextIndexGenerationManifest,
) -> dict[str, Any]:
    """Build a strict mapping with bounded mirrored generation metadata."""
    if not isinstance(manifest, TextIndexGenerationManifest):
        raise TypeError("manifest must be a TextIndexGenerationManifest")
    mapping = build_text_index_mapping(manifest.embedding_spaces)
    actual_mapping_sha256 = _mapping_digest(mapping)
    if manifest.mapping_sha256 != actual_mapping_sha256:
        raise ValueError("manifest mapping_sha256 does not match the provider mapping")
    mapping["_meta"] = {
        "capicove": {
            "schema_version": 1,
            "generation_id": manifest.generation_id,
            "manifest_sha256": manifest.manifest_sha256,
            "mapping_sha256": manifest.mapping_sha256,
            "content_sha256": manifest.content_sha256,
            "expected_unit_count": manifest.expected_unit_count,
            "index_format_version": manifest.index_format_version,
        }
    }
    return mapping


def inspect_managed_text_index_mapping(
    value: object,
    manifest: TextIndexGenerationManifest,
) -> tuple[str, ...]:
    """Return stable integrity issues without exposing provider mapping data."""
    if not isinstance(manifest, TextIndexGenerationManifest):
        raise TypeError("manifest must be a TextIndexGenerationManifest")
    if not isinstance(value, Mapping):
        raise ValueError("malformed managed Elasticsearch mapping")
    mapping = dict(value)
    metadata_container = mapping.pop("_meta", None)
    metadata: Mapping[object, object] = {}
    if isinstance(metadata_container, Mapping):
        candidate = metadata_container.get("capicove")
        if isinstance(candidate, Mapping):
            metadata = candidate

    issues: list[str] = []
    try:
        actual_mapping_sha256 = _mapping_digest(mapping)
    except (TypeError, ValueError):
        actual_mapping_sha256 = ""
    if (
        actual_mapping_sha256 != manifest.mapping_sha256
        or metadata.get("mapping_sha256") != manifest.mapping_sha256
    ):
        issues.append("mapping_digest_mismatch")
    expected_metadata = (
        ("schema_version", 1, "schema_mismatch"),
        ("generation_id", manifest.generation_id, "generation_id_mismatch"),
        (
            "manifest_sha256",
            manifest.manifest_sha256,
            "manifest_digest_mismatch",
        ),
        ("content_sha256", manifest.content_sha256, "content_digest_mismatch"),
        (
            "expected_unit_count",
            manifest.expected_unit_count,
            "expected_unit_count_mismatch",
        ),
        (
            "index_format_version",
            manifest.index_format_version,
            "index_format_mismatch",
        ),
    )
    for field_name, expected, issue_code in expected_metadata:
        if metadata.get(field_name) != expected:
            issues.append(issue_code)
    return tuple(issues)


def decode_managed_text_index_identity(value: object) -> tuple[str, str]:
    """Decode a self-consistent generation identity from managed mapping data."""
    try:
        if not isinstance(value, Mapping):
            raise TypeError
        mapping = dict(value)
        metadata_container = _mapping(mapping.pop("_meta"))
        metadata = _mapping(metadata_container["capicove"])
        if set(metadata) != {
            "schema_version",
            "generation_id",
            "manifest_sha256",
            "mapping_sha256",
            "content_sha256",
            "expected_unit_count",
            "index_format_version",
        }:
            raise ValueError
        if metadata["schema_version"] != 1:
            raise ValueError
        generation_id = _string(metadata["generation_id"])
        _require_identity("generation_id", generation_id)
        manifest_sha256 = _string(metadata["manifest_sha256"])
        mapping_sha256 = _string(metadata["mapping_sha256"])
        content_sha256 = _string(metadata["content_sha256"])
        if any(
            _SHA256.fullmatch(digest) is None
            for digest in (manifest_sha256, mapping_sha256, content_sha256)
        ):
            raise ValueError
        expected_unit_count = metadata["expected_unit_count"]
        if (
            isinstance(expected_unit_count, bool)
            or not isinstance(expected_unit_count, int)
            or expected_unit_count < 0
        ):
            raise ValueError
        _require_identity("index_format_version", metadata["index_format_version"])
        if _mapping_digest(mapping) != mapping_sha256:
            raise ValueError
    except Exception:
        raise ValueError("malformed managed Elasticsearch mapping") from None
    return generation_id, manifest_sha256


def _mapping_digest(mapping: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_text_index_entry(entry: TextIndexEntry) -> dict[str, object]:
    """Encode one validated canonical text entry using only reviewed fields."""
    if not isinstance(entry, TextIndexEntry):
        raise TypeError("entry must be a TextIndexEntry")
    locator = entry.unit.text_locator
    if locator is None:
        raise ValueError("text index entries require a text locator")
    document: dict[str, object] = {
        "unit_id": entry.unit.unit_id,
        "evidence_ref": entry.unit.evidence_ref,
        "source_hash": entry.unit.source_hash,
        "artifact_ref": entry.unit.artifact_ref,
        "representation_refs": list(entry.unit.representation_refs),
        "processing_run_ref": entry.unit.processing_run_ref,
        "processing_version": entry.unit.processing_version,
        "processing_fingerprint": entry.unit.processing_fingerprint.sha256,
        "corpus_refs": list(entry.corpus_refs),
        "text": entry.text,
        "preview": entry.text[:_MAX_SNIPPET_LENGTH],
        "origin": locator.origin.value,
        "start_char": locator.character_range.start_char,
        "end_char": locator.character_range.end_char,
    }
    if entry.media_type is not None:
        document["media_type"] = entry.media_type
    if entry.language is not None:
        document["language"] = entry.language
    if locator.page_number is not None:
        document["page_number"] = locator.page_number
    if locator.time_range is not None:
        document["start_ms"] = locator.time_range.start_ms
        document["end_ms"] = locator.time_range.end_ms
    for embedding in entry.embeddings:
        document[vector_field_name(embedding.space.space_id)] = list(embedding.vector)
    return document


def decode_text_candidates(
    response: object,
    *,
    provider_id: str,
    generation_id: str,
    score_semantics: str,
) -> tuple[TextCandidate, ...]:
    """Decode bounded candidates without exposing a raw provider response."""
    try:
        root = _mapping(
            response if isinstance(response, Mapping) else getattr(response, "body", None)
        )
        hits_container = _mapping(root["hits"])
        hits = hits_container["hits"]
        if not isinstance(hits, list):
            raise TypeError
        candidates = tuple(
            _decode_hit(
                hit,
                rank=rank,
                provider_id=provider_id,
                generation_id=generation_id,
                score_semantics=score_semantics,
            )
            for rank, hit in enumerate(hits, start=1)
        )
    except Exception:
        raise ValueError("malformed Elasticsearch response") from None
    return candidates


def _decode_hit(
    value: object,
    *,
    rank: int,
    provider_id: str,
    generation_id: str,
    score_semantics: str,
) -> TextCandidate:
    hit = _mapping(value)
    unit_id = _string(hit["_id"])
    source = _mapping(hit["_source"])
    if _string(source["unit_id"]) != unit_id:
        raise ValueError
    evidence_ref = _string(source["evidence_ref"])
    origin = TextOrigin(_string(source["origin"]))
    start_char = _integer(source["start_char"])
    end_char = _integer(source["end_char"])
    character_range = CharacterRange(start_char, end_char)
    has_page = source.get("page_number") is not None
    has_time = source.get("start_ms") is not None or source.get("end_ms") is not None
    if (has_page and has_time) or (origin is TextOrigin.DOCUMENT and (has_page or has_time)):
        raise ValueError("ambiguous text locator")
    if origin is TextOrigin.ASR and has_page:
        raise ValueError("ASR cannot use page coordinates")
    if origin is TextOrigin.DOCUMENT:
        locator = TextChunkLocator(origin, character_range)
    elif origin is TextOrigin.OCR and source.get("page_number") is not None:
        locator = TextChunkLocator(
            origin,
            character_range,
            page_number=_integer(source["page_number"]),
        )
    else:
        locator = TextChunkLocator(
            origin,
            character_range,
            time_range=TemporalRange(
                _integer(source["start_ms"]),
                _integer(source["end_ms"]),
            ),
        )
    snippet = _snippet(hit, source)
    return TextCandidate(
        unit_id=unit_id,
        evidence_ref=evidence_ref,
        text_locator=locator,
        snippet=snippet,
        raw_score=_number(hit["_score"]),
        score_semantics=score_semantics,
        rank=rank,
        provider_id=provider_id,
        index_generation_id=generation_id,
    )


def _snippet(hit: Mapping[object, object], source: Mapping[object, object]) -> str:
    highlighted: str | None = None
    highlight_value = hit.get("highlight")
    if highlight_value is not None:
        highlight = _mapping(highlight_value)
        fragments = highlight.get("text")
        if not isinstance(fragments, list) or not fragments:
            raise TypeError
        highlighted = _string(fragments[0])
    text = highlighted if highlighted is not None else _string(source["preview"])
    plain_text = text.replace("<em>", "").replace("</em>", "")
    return plain_text[:_MAX_SNIPPET_LENGTH]


def _mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError
    return value


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError
    return result


def _require_identity(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > _IDENTITY_LIMIT:
        raise ValueError(f"{field_name} must be non-blank and bounded")


__all__ = [
    "build_managed_text_index_mapping",
    "build_text_index_mapping",
    "decode_managed_text_index_identity",
    "decode_text_candidates",
    "encode_text_index_entry",
    "inspect_managed_text_index_mapping",
    "physical_index_name",
    "text_index_mapping_sha256",
    "vector_field_name",
]
