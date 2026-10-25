"""Pure codecs for immutable FAISS visual generation metadata."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchFilters,
    TargetType,
    TemporalRange,
    VectorNormalization,
)
from capicove.retrieval.video_models import (
    VisualIndexEntry,
    VisualIndexGenerationManifest,
)


_ORDINAL_KEYS = frozenset(
    {
        "ordinal",
        "unit_id",
        "target_type",
        "artifact_ref",
        "corpus_refs",
        "timestamp_ms",
        "start_ms",
        "end_ms",
    }
)


def encode_visual_manifest(manifest: VisualIndexGenerationManifest) -> dict[str, object]:
    """Encode one public visual manifest as canonical provider-owned JSON data."""
    if not isinstance(manifest, VisualIndexGenerationManifest):
        raise TypeError("manifest must be a VisualIndexGenerationManifest")
    return {
        "schema_version": manifest.schema_version,
        "generation_id": manifest.generation_id,
        "workspace_ref": manifest.workspace_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "target_type": manifest.target_type.value,
        "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "embedding_space": _encode_space(manifest.embedding_space),
        "retriever_id": manifest.retriever_id,
        "index_format_version": manifest.index_format_version,
        "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
        "expected_unit_count": manifest.expected_unit_count,
        "manifest_sha256": manifest.manifest_sha256,
    }


def decode_visual_manifest(value: object) -> VisualIndexGenerationManifest:
    """Decode and authenticate one stored visual manifest."""
    if not isinstance(value, dict):
        raise ValueError("stored visual manifest must be an object")
    expected_keys = {
        "schema_version",
        "generation_id",
        "workspace_ref",
        "corpus_refs",
        "target_type",
        "searchable_unit_manifest_sha256",
        "content_sha256",
        "embedding_space",
        "retriever_id",
        "index_format_version",
        "build_processing_fingerprint",
        "expected_unit_count",
        "manifest_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("stored visual manifest has unsupported fields")
    space_value = value["embedding_space"]
    if not isinstance(space_value, dict):
        raise ValueError("stored embedding space must be an object")
    manifest = VisualIndexGenerationManifest(
        generation_id=_string(value, "generation_id"),
        workspace_ref=_string(value, "workspace_ref"),
        corpus_refs=tuple(_strings(value, "corpus_refs")),
        target_type=TargetType(_string(value, "target_type")),
        searchable_unit_manifest_sha256=_string(value, "searchable_unit_manifest_sha256"),
        content_sha256=_string(value, "content_sha256"),
        embedding_space=_decode_space(space_value),
        retriever_id=_string(value, "retriever_id"),
        index_format_version=_string(value, "index_format_version"),
        build_processing_fingerprint=ProcessingFingerprint(
            _string(value, "build_processing_fingerprint")
        ),
        expected_unit_count=_integer(value, "expected_unit_count"),
        schema_version=_integer(value, "schema_version"),
    )
    if value["manifest_sha256"] != manifest.manifest_sha256:
        raise ValueError("stored visual manifest digest is invalid")
    return manifest


def encode_ordinal(
    ordinal: int,
    entry: VisualIndexEntry,
    corpus_refs: tuple[str, ...],
) -> dict[str, object]:
    """Encode one stable native ordinal and its closed filter metadata."""
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    if not isinstance(entry, VisualIndexEntry):
        raise TypeError("entry must be a VisualIndexEntry")
    unit = entry.unit
    time_range = unit.scene_range if unit.unit_type is TargetType.KEYFRAME else unit.time_range
    return {
        "ordinal": ordinal,
        "unit_id": unit.unit_id,
        "target_type": unit.unit_type.value,
        "artifact_ref": unit.artifact_ref,
        "corpus_refs": list(corpus_refs),
        "timestamp_ms": unit.timestamp_ms,
        "start_ms": None if time_range is None else time_range.start_ms,
        "end_ms": None if time_range is None else time_range.end_ms,
    }


def validate_ordinals(
    rows: Iterable[dict[str, object]],
    manifest: VisualIndexGenerationManifest,
) -> tuple[str, ...]:
    """Return stable integrity issues for provider ordinal metadata."""
    materialized = tuple(rows)
    unit_ids: list[str] = []
    issues: set[str] = set()
    try:
        for expected, row in enumerate(materialized):
            if set(row) != _ORDINAL_KEYS or _integer(row, "ordinal") != expected:
                issues.add("ordinal_mismatch")
                continue
            unit_ids.append(_string(row, "unit_id"))
            if TargetType(_string(row, "target_type")) is not manifest.target_type:
                issues.add("ordinal_target_mismatch")
            if tuple(_strings(row, "corpus_refs")) != manifest.corpus_refs:
                issues.add("ordinal_corpus_mismatch")
            _string(row, "artifact_ref")
            _validate_optional_integer(row, "timestamp_ms")
            _validate_optional_integer(row, "start_ms")
            _validate_optional_integer(row, "end_ms")
    except (KeyError, TypeError, ValueError):
        issues.add("ordinal_mismatch")
    if len(unit_ids) != len(set(unit_ids)):
        issues.add("duplicate_unit_id")
    if len(materialized) != manifest.expected_unit_count:
        issues.add("ordinal_count_mismatch")
    return tuple(sorted(issues))


def filter_ordinals(
    rows: tuple[dict[str, object], ...],
    manifest: VisualIndexGenerationManifest,
    filters: SearchFilters,
) -> tuple[int, ...]:
    """Apply the supported closed visual filters to native ordinals."""
    if not isinstance(filters, SearchFilters):
        raise TypeError("filters must be SearchFilters")
    requested_corpora = set(filters.corpus_refs)
    manifest_corpora = set(manifest.corpus_refs)
    if requested_corpora:
        if not requested_corpora.intersection(manifest_corpora):
            return ()
        if not manifest_corpora.issubset(requested_corpora) and len(manifest_corpora) > 1:
            raise ValueError("corpus subset filtering requires per-unit corpus metadata")
    if filters.media_types:
        raise ValueError("media-type filtering is unsupported by FAISS ordinals")
    if filters.text_origins:
        raise ValueError("text-origin filtering is unsupported by a visual index")
    selected: list[int] = []
    for row in rows:
        if filters.artifact_refs and _string(row, "artifact_ref") not in filters.artifact_refs:
            continue
        if filters.time_range is not None and not _overlaps(row, filters.time_range):
            continue
        selected.append(_integer(row, "ordinal"))
    return tuple(selected)


def reconstructed_content_sha256(
    rows: tuple[dict[str, object], ...],
    vectors: NDArray[np.float32],
) -> str:
    """Hash native reconstructed float32 vectors using the public contract codec."""
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise ValueError("reconstructed vectors do not match ordinal rows")
    return reconstructed_stream_sha256(rows, iter(vectors))


def reconstructed_stream_sha256(
    rows: tuple[dict[str, object], ...],
    vectors: Iterable[NDArray[np.float32]],
) -> str:
    """Preserve the canonical digest without retaining all reconstructed vectors."""
    digest = hashlib.sha256(b"[")
    for index, (row, vector) in enumerate(zip(rows, vectors, strict=True)):
        if index:
            digest.update(b",")
        payload = {
            "unit_id": _string(row, "unit_id"),
            "target_type": _string(row, "target_type"),
            "vector_float32_be": [
                struct.pack("!f", float(component)).hex() for component in vector
            ],
        }
        digest.update(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).encode()
        )
    digest.update(b"]")
    return digest.hexdigest()


def _encode_space(space: EmbeddingSpaceRef) -> dict[str, object]:
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": sorted(item.value for item in space.query_modalities),
        "target_modalities": sorted(item.value for item in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _decode_space(value: dict[str, object]) -> EmbeddingSpaceRef:
    expected_keys = {
        "space_id",
        "provider_id",
        "model_revision",
        "preprocessing_version",
        "dimension",
        "normalization",
        "distance_metric",
        "query_modalities",
        "target_modalities",
        "schema_version",
    }
    if set(value) != expected_keys:
        raise ValueError("stored embedding space has unsupported fields")
    return EmbeddingSpaceRef(
        space_id=_string(value, "space_id"),
        provider_id=_string(value, "provider_id"),
        model_revision=_string(value, "model_revision"),
        preprocessing_version=_string(value, "preprocessing_version"),
        dimension=_integer(value, "dimension"),
        normalization=VectorNormalization(_string(value, "normalization")),
        distance_metric=DistanceMetric(_string(value, "distance_metric")),
        query_modalities=frozenset(
            QueryModality(item) for item in _strings(value, "query_modalities")
        ),
        target_modalities=frozenset(
            TargetType(item) for item in _strings(value, "target_modalities")
        ),
        schema_version=_integer(value, "schema_version"),
    )


def _overlaps(row: dict[str, object], requested: TemporalRange) -> bool:
    timestamp = row.get("timestamp_ms")
    if isinstance(timestamp, int) and not isinstance(timestamp, bool):
        return requested.start_ms <= timestamp < requested.end_ms
    start = row.get("start_ms")
    end = row.get("end_ms")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    ):
        return max(start, requested.start_ms) < min(end, requested.end_ms)
    return False


def _validate_optional_integer(value: dict[str, object], key: str) -> None:
    item = value[key]
    if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
        raise ValueError(f"{key} must be null or a non-negative integer")


def _string(value: dict[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{key} must be a non-blank string")
    return item


def _integer(value: dict[str, object], key: str) -> int:
    item = value[key]
    if isinstance(item, bool) or not isinstance(item, int):
        raise TypeError(f"{key} must be an integer")
    return item


def _strings(value: dict[str, object], key: str) -> list[str]:
    item = value[key]
    if not isinstance(item, list) or any(not isinstance(element, str) for element in item):
        raise TypeError(f"{key} must be a list of strings")
    return item


__all__ = [
    "decode_visual_manifest",
    "encode_ordinal",
    "encode_visual_manifest",
    "filter_ordinals",
    "reconstructed_content_sha256",
    "validate_ordinals",
]
