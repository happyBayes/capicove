"""Pure managed-index codec for OpenSearch text retrieval."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Mapping

from capicove.contracts import CharacterRange, TemporalRange, TextChunkLocator, TextOrigin
from capicove.retrieval import (
    TextCandidate,
    TextIndexEntry,
    TextIndexGenerationManifest,
)


_TAG = re.compile(r"</?em>", re.IGNORECASE)


def vector_field_name(space_id: str) -> str:
    """Return one deterministic safe vector field name."""
    return f"embedding_{space_id}"


def physical_index_name(prefix: str, generation_id: str) -> str:
    """Return one bounded physical index name from immutable identity."""
    suffix = hashlib.sha256(generation_id.encode()).hexdigest()[:24]
    return f"{prefix}-g-{suffix}"


def build_managed_text_index_mapping(
    manifest: TextIndexGenerationManifest,
) -> dict[str, object]:
    """Build one strict OpenSearch 3.x KNN mapping."""
    properties: dict[str, object] = {
        "unit_id": {"type": "keyword"},
        "evidence_ref": {"type": "keyword"},
        "artifact_ref": {"type": "keyword"},
        "corpus_refs": {"type": "keyword"},
        "media_type": {"type": "keyword"},
        "language": {"type": "keyword"},
        "origin": {"type": "keyword"},
        "start_char": {"type": "long"},
        "end_char": {"type": "long"},
        "page_number": {"type": "long"},
        "start_ms": {"type": "long"},
        "end_ms": {"type": "long"},
        "text": {"type": "text"},
        "preview": {"type": "text", "index": False},
    }
    for space in manifest.embedding_spaces:
        properties[vector_field_name(space.space_id)] = {
            "type": "knn_vector",
            "dimension": space.dimension,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene",
            },
        }
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "dynamic": "strict",
            "_meta": {
                "capicove_generation_id": manifest.generation_id,
                "capicove_manifest_sha256": manifest.manifest_sha256,
                "capicove_mapping_sha256": manifest.mapping_sha256,
                "capicove_index_format_version": manifest.index_format_version,
            },
            "properties": properties,
        },
    }


def encode_text_index_entry(entry: TextIndexEntry) -> dict[str, object]:
    """Encode one canonical entry into the strict managed document shape."""
    locator = entry.unit.text_locator
    if locator is None:
        raise ValueError("text entry requires a text locator")
    value: dict[str, object] = {
        "unit_id": entry.unit.unit_id,
        "evidence_ref": entry.unit.evidence_ref,
        "artifact_ref": entry.unit.artifact_ref,
        "corpus_refs": list(entry.corpus_refs),
        "origin": locator.origin.value,
        "start_char": locator.character_range.start_char,
        "end_char": locator.character_range.end_char,
        "text": entry.text,
        "preview": entry.text[:2_048],
    }
    if entry.media_type is not None:
        value["media_type"] = entry.media_type
    if entry.language is not None:
        value["language"] = entry.language
    if locator.page_number is not None:
        value["page_number"] = locator.page_number
    if locator.time_range is not None:
        value["start_ms"] = locator.time_range.start_ms
        value["end_ms"] = locator.time_range.end_ms
    for embedding in entry.embeddings:
        value[vector_field_name(embedding.space.space_id)] = list(embedding.vector)
    return value


def decode_text_candidates(
    response: object,
    *,
    provider_id: str,
    generation_id: str,
    score_semantics: str,
) -> tuple[TextCandidate, ...]:
    """Decode a bounded OpenSearch response into canonical candidates."""
    if not isinstance(response, Mapping):
        raise ValueError("OpenSearch response must be a mapping")
    hits_wrapper = response.get("hits")
    if not isinstance(hits_wrapper, Mapping) or not isinstance(hits_wrapper.get("hits"), list):
        raise ValueError("OpenSearch response contains invalid hits")
    candidates: list[TextCandidate] = []
    for rank, hit in enumerate(hits_wrapper["hits"], start=1):
        if not isinstance(hit, Mapping) or not isinstance(hit.get("_source"), Mapping):
            raise ValueError("OpenSearch hit is invalid")
        source = hit["_source"]
        origin = TextOrigin(_string(source, "origin"))
        locator = TextChunkLocator(
            origin=origin,
            character_range=CharacterRange(
                _integer(source, "start_char"), _integer(source, "end_char")
            ),
            page_number=_optional_integer(source, "page_number"),
            time_range=_time_range(source),
        )
        snippet = _string(source, "preview")
        highlight = hit.get("highlight")
        if isinstance(highlight, Mapping):
            fragments = highlight.get("text")
            if isinstance(fragments, list) and fragments and isinstance(fragments[0], str):
                snippet = html.unescape(_TAG.sub("", fragments[0]))
        candidates.append(
            TextCandidate(
                unit_id=_string(source, "unit_id"),
                evidence_ref=_string(source, "evidence_ref"),
                text_locator=locator,
                snippet=snippet[:2_048],
                raw_score=_score(hit.get("_score")),
                score_semantics=score_semantics,
                rank=rank,
                provider_id=provider_id,
                index_generation_id=generation_id,
            )
        )
    return tuple(candidates)


def _time_range(source: Mapping[object, object]) -> TemporalRange | None:
    start = _optional_integer(source, "start_ms")
    end = _optional_integer(source, "end_ms")
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("OpenSearch hit contains an incomplete time range")
    return TemporalRange(start, end)


def _string(value: Mapping[object, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"OpenSearch hit contains invalid {key}")
    return result


def _integer(value: Mapping[object, object], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"OpenSearch hit contains invalid {key}")
    return result


def _optional_integer(value: Mapping[object, object], key: str) -> int | None:
    return None if key not in value or value[key] is None else _integer(value, key)


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("OpenSearch hit contains invalid score")
    return float(value)


__all__ = [
    "build_managed_text_index_mapping",
    "decode_text_candidates",
    "encode_text_index_entry",
    "physical_index_name",
    "vector_field_name",
]
