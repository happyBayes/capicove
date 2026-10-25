"""Exact read-only NumPy reference backend for visual retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    PinnedVisualIndexGeneration,
    VisualCandidate,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    VisualIndexValidationReport,
    visual_entries_content_sha256,
)


_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_INDEX_FORMAT = "numpy-exact-v1"
_PROVIDER_ID = "capicove.retriever.numpy"
_MAX_TOP_K = 500
_NORM_TOLERANCE = 1e-4


class NumpyVisualIndexError(RuntimeError):
    """Report a bounded exact-index storage or integrity failure."""


@dataclass(frozen=True, slots=True)
class NumpyExactVisualConfiguration:
    """Configure the local exact reference index with a closed capacity bound."""

    index_root: Path
    maximum_units_per_target: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.index_root, Path) or not self.index_root.is_absolute():
            raise ValueError("index_root must be an absolute Path")
        if (
            isinstance(self.maximum_units_per_target, bool)
            or not isinstance(self.maximum_units_per_target, int)
            or not 1 <= self.maximum_units_per_target <= 100_000
        ):
            raise ValueError("maximum_units_per_target must be an integer from 1 to 100,000")
        _reject_symlink_components(self.index_root)


class NumpyExactVisualRetriever:
    """Build, validate, activate, and query exact immutable float32 indexes."""

    provider_id = _PROVIDER_ID

    def __init__(self, configuration: NumpyExactVisualConfiguration) -> None:
        if not isinstance(configuration, NumpyExactVisualConfiguration):
            raise TypeError("configuration must be NumpyExactVisualConfiguration")
        self._configuration = configuration
        configuration.index_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink_components(configuration.index_root)

    @property
    def index_writer(self) -> NumpyExactVisualRetriever:
        """Return this backend's writer role."""
        return self

    @property
    def candidate_retriever(self) -> NumpyExactVisualRetriever:
        """Return this backend's exact candidate role."""
        return self

    @property
    def generation_lifecycle(self) -> NumpyExactVisualRetriever:
        """Return this backend's generation lifecycle role."""
        return self

    def create(self, manifest: VisualIndexGenerationManifest) -> str:
        """Create one unique mutable build directory for a frozen manifest."""
        self._validate_manifest_policy(manifest)
        generation_path = self._generation_path(manifest.generation_id)
        build_path = generation_path / "build-manifest.json"
        if generation_path.exists():
            _require_regular_directory(generation_path)
            existing = _read_json_object(build_path)
            if existing != _manifest_dict(manifest):
                raise NumpyVisualIndexError(
                    "generation ID identifies a different immutable manifest"
                )
            return str(generation_path)
        try:
            generation_path.mkdir(mode=0o700)
            _write_json_exclusive(build_path, _manifest_dict(manifest))
            _fsync_directory(generation_path)
            _fsync_directory(generation_path.parent)
        except OSError as error:
            raise NumpyVisualIndexError("visual generation creation failed") from error
        return str(generation_path)

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[VisualIndexEntry, ...],
    ) -> None:
        """Write the complete ordered vector payload exactly once."""
        path = self._generation_path(index_generation_id)
        manifest = _manifest_from_dict(_read_json_object(path / "build-manifest.json"))
        self._validate_manifest_policy(manifest)
        materialized = tuple(entries)
        if len(materialized) != manifest.expected_unit_count:
            raise ValueError("entry count must match the generation manifest")
        if len(materialized) > self._configuration.maximum_units_per_target:
            raise ValueError("entry count exceeds exact-index capacity")
        if any(not isinstance(entry, VisualIndexEntry) for entry in materialized):
            raise TypeError("entries must contain VisualIndexEntry values")
        unit_ids = tuple(entry.unit.unit_id for entry in materialized)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("entries must not contain duplicate unit IDs")
        for entry in materialized:
            if entry.unit.unit_type is not manifest.target_type:
                raise ValueError("entry target does not match generation target")
            if len(entry.vector) != manifest.embedding_space.dimension:
                raise ValueError("entry vector dimension does not match embedding space")
            if manifest.embedding_space not in entry.unit.embedding_spaces:
                raise ValueError("entry unit does not declare the generation embedding space")
            _require_unit_vector(entry.vector)
        if visual_entries_content_sha256(materialized) != manifest.content_sha256:
            raise ValueError("entry content digest does not match generation manifest")

        payload_manifest_path = path / "manifest.json"
        if payload_manifest_path.exists():
            report = self.validate(manifest)
            if report.valid:
                return
            raise NumpyVisualIndexError("generation payload already exists but does not validate")
        vectors = np.asarray(
            tuple(entry.vector for entry in materialized),
            dtype=np.float32,
            order="C",
        )
        vectors = vectors.reshape((len(materialized), manifest.embedding_space.dimension))
        ordinal_rows = tuple(
            _ordinal_from_entry(index, entry, manifest.corpus_refs)
            for index, entry in enumerate(materialized)
        )
        temporary_paths = {
            name: path / f".{name}.{uuid.uuid4().hex}.tmp"
            for name in ("vectors.npy", "ordinals.jsonl", "manifest.json")
        }
        try:
            with temporary_paths["vectors.npy"].open("xb") as output:
                np.save(output, vectors, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            _write_jsonl_exclusive(temporary_paths["ordinals.jsonl"], ordinal_rows)
            payload = {
                "schema_version": 1,
                "manifest_sha256": manifest.manifest_sha256,
                "content_sha256": manifest.content_sha256,
                "vectors_sha256": _file_sha256(temporary_paths["vectors.npy"]),
                "ordinals_sha256": _file_sha256(temporary_paths["ordinals.jsonl"]),
            }
            _write_json_exclusive(temporary_paths["manifest.json"], payload)
            os.replace(temporary_paths["vectors.npy"], path / "vectors.npy")
            os.replace(temporary_paths["ordinals.jsonl"], path / "ordinals.jsonl")
            os.replace(temporary_paths["manifest.json"], payload_manifest_path)
            _fsync_directory(path)
        except OSError as error:
            raise NumpyVisualIndexError("exact visual payload write failed") from error
        finally:
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)

    def validate(self, manifest: VisualIndexGenerationManifest) -> VisualIndexValidationReport:
        """Reopen and fully validate one exact payload without trusting process state."""
        self._validate_manifest_policy(manifest)
        path = self._generation_path(manifest.generation_id)
        issues: set[str] = set()
        observed_count = 0
        dimension = manifest.embedding_space.dimension
        try:
            stored_build = _read_json_object(path / "build-manifest.json")
            if stored_build != _manifest_dict(manifest):
                issues.add("manifest_mismatch")
            payload = _read_json_object(path / "manifest.json")
            if payload.get("manifest_sha256") != manifest.manifest_sha256:
                issues.add("manifest_mismatch")
            if payload.get("content_sha256") != manifest.content_sha256:
                issues.add("content_mismatch")
            vectors_path = _validated_payload_file(path / "vectors.npy")
            ordinals_path = _validated_payload_file(path / "ordinals.jsonl")
            if payload.get("vectors_sha256") != _file_sha256(vectors_path):
                issues.add("checksum_mismatch")
            if payload.get("ordinals_sha256") != _file_sha256(ordinals_path):
                issues.add("checksum_mismatch")
            vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
            if vectors.ndim != 2:
                issues.add("shape_mismatch")
            else:
                observed_count = int(vectors.shape[0])
                if int(vectors.shape[1]) != dimension:
                    issues.add("dimension_mismatch")
            if vectors.dtype != np.dtype(np.float32):
                issues.add("dtype_mismatch")
            if not np.isfinite(vectors).all():
                issues.add("non_finite_vectors")
            elif observed_count:
                norms = np.linalg.norm(vectors, axis=1)
                if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE, rtol=0.0):
                    issues.add("normalization_mismatch")
            ordinals = _read_ordinals(ordinals_path)
            if len(ordinals) != observed_count:
                issues.add("ordinal_mismatch")
            if not _valid_ordinals(ordinals, manifest):
                issues.add("ordinal_mismatch")
            if vectors.ndim == 2 and len(ordinals) == observed_count:
                observed_content = _payload_content_sha256(ordinals, vectors)
                if observed_content != manifest.content_sha256:
                    issues.add("content_mismatch")
        except (NumpyVisualIndexError, OSError, ValueError, TypeError, json.JSONDecodeError):
            issues.add("payload_corrupt")
        if observed_count != manifest.expected_unit_count:
            issues.add("unit_count_mismatch")
        return VisualIndexValidationReport(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref=str(path),
            content_sha256=manifest.content_sha256,
            expected_unit_count=manifest.expected_unit_count,
            observed_unit_count=observed_count,
            vector_dimension=dimension,
            issue_codes=tuple(issues),
        )

    def activate(
        self,
        manifest: VisualIndexGenerationManifest,
        validation: VisualIndexValidationReport,
    ) -> PinnedVisualIndexGeneration:
        """Return a pin only after fresh exact validation of the immutable payload."""
        if not isinstance(validation, VisualIndexValidationReport):
            raise TypeError("validation must be a VisualIndexValidationReport")
        expected_path = str(self._generation_path(manifest.generation_id))
        if (
            validation.generation_id != manifest.generation_id
            or validation.manifest_sha256 != manifest.manifest_sha256
            or validation.physical_index_ref != expected_path
            or validation.content_sha256 != manifest.content_sha256
            or validation.expected_unit_count != manifest.expected_unit_count
            or validation.vector_dimension != manifest.embedding_space.dimension
        ):
            raise ValueError("validation identity does not match generation manifest")
        fresh = self.validate(manifest)
        if not validation.valid or not fresh.valid or fresh != validation:
            raise ValueError("cannot activate an invalid or stale visual generation")
        return PinnedVisualIndexGeneration(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref=expected_path,
            target_type=manifest.target_type,
            embedding_space=manifest.embedding_space,
        )

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedVisualIndexGeneration,
        top_k: int,
    ) -> tuple[VisualCandidate, ...]:
        """Return stable exact inner-product candidates from one explicit pin."""
        if not isinstance(embedding_space, EmbeddingSpaceRef):
            raise TypeError("embedding_space must be an EmbeddingSpaceRef")
        if not isinstance(filters, SearchFilters):
            raise TypeError("filters must be SearchFilters")
        if not isinstance(generation, PinnedVisualIndexGeneration):
            raise TypeError("generation must be a PinnedVisualIndexGeneration")
        if embedding_space != generation.embedding_space:
            raise ValueError("query embedding space does not match pinned generation")
        vector = tuple(query_vector)
        if len(vector) != embedding_space.dimension:
            raise ValueError("query vector dimension does not match embedding space")
        _require_query_vector(vector)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= _MAX_TOP_K:
            raise ValueError("top_k must be an integer from 1 to 500")
        path = self._pinned_path(generation)
        manifest = _manifest_from_dict(_read_json_object(path / "build-manifest.json"))
        if (
            generation.generation_id != manifest.generation_id
            or generation.manifest_sha256 != manifest.manifest_sha256
            or generation.target_type is not manifest.target_type
            or generation.embedding_space != manifest.embedding_space
        ):
            raise NumpyVisualIndexError("pinned generation identity is stale or invalid")
        report = self.validate(manifest)
        if not report.valid:
            raise NumpyVisualIndexError("pinned visual generation is corrupt")
        vectors = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
        ordinals = _read_ordinals(path / "ordinals.jsonl")
        selected = _filter_ordinals(ordinals, manifest, filters)
        if not selected:
            return ()
        selected_array = np.asarray(selected, dtype=np.int64)
        query = np.asarray(vector, dtype=np.float32)
        scores = np.asarray(vectors[selected_array] @ query, dtype=np.float32)
        ranked = sorted(
            (
                (
                    float(scores[local_index]),
                    str(ordinals[selected[local_index]]["unit_id"]),
                )
                for local_index in range(len(selected))
            ),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]
        lane_id = (
            "keyframe_visual" if manifest.target_type is TargetType.KEYFRAME else "segment_visual"
        )
        return tuple(
            VisualCandidate(
                unit_id=unit_id,
                target_type=manifest.target_type,
                lane_id=lane_id,
                raw_score=score,
                score_semantics="inner_product_similarity",
                rank=rank,
                provider_id=self.provider_id,
                generation_id=manifest.generation_id,
            )
            for rank, (score, unit_id) in enumerate(ranked, start=1)
        )

    def close(self) -> None:
        """Release backend resources; exact mmap handles are request-scoped."""

    def _validate_manifest_policy(self, manifest: VisualIndexGenerationManifest) -> None:
        if not isinstance(manifest, VisualIndexGenerationManifest):
            raise TypeError("manifest must be a VisualIndexGenerationManifest")
        if manifest.retriever_id != self.provider_id:
            raise ValueError("manifest retriever does not match the NumPy backend")
        if manifest.index_format_version != _INDEX_FORMAT:
            raise ValueError("manifest index format is not supported")
        if manifest.expected_unit_count > self._configuration.maximum_units_per_target:
            raise ValueError("manifest unit count exceeds exact-index capacity")

    def _generation_path(self, generation_id: str) -> Path:
        if not isinstance(generation_id, str) or _GENERATION_ID.fullmatch(generation_id) is None:
            raise ValueError("generation_id is not safe for a local index path")
        return self._configuration.index_root / generation_id

    def _pinned_path(self, generation: PinnedVisualIndexGeneration) -> Path:
        path = Path(generation.physical_index_ref)
        if not path.is_absolute() or path != self._generation_path(generation.generation_id):
            raise NumpyVisualIndexError("pinned physical index is outside index_root")
        _require_regular_directory(path)
        return path


def _manifest_dict(manifest: VisualIndexGenerationManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "generation_id": manifest.generation_id,
        "workspace_ref": manifest.workspace_ref,
        "corpus_refs": list(manifest.corpus_refs),
        "target_type": manifest.target_type.value,
        "searchable_unit_manifest_sha256": manifest.searchable_unit_manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "embedding_space": _space_dict(manifest.embedding_space),
        "retriever_id": manifest.retriever_id,
        "index_format_version": manifest.index_format_version,
        "build_processing_fingerprint": manifest.build_processing_fingerprint.sha256,
        "expected_unit_count": manifest.expected_unit_count,
        "manifest_sha256": manifest.manifest_sha256,
    }


def _manifest_from_dict(value: dict[str, object]) -> VisualIndexGenerationManifest:
    space_value = value.get("embedding_space")
    if not isinstance(space_value, dict):
        raise NumpyVisualIndexError("stored embedding space is invalid")
    manifest = VisualIndexGenerationManifest(
        generation_id=_string(value, "generation_id"),
        workspace_ref=_string(value, "workspace_ref"),
        corpus_refs=tuple(_strings(value, "corpus_refs")),
        target_type=TargetType(_string(value, "target_type")),
        searchable_unit_manifest_sha256=_string(value, "searchable_unit_manifest_sha256"),
        content_sha256=_string(value, "content_sha256"),
        embedding_space=_space_from_dict(space_value),
        retriever_id=_string(value, "retriever_id"),
        index_format_version=_string(value, "index_format_version"),
        build_processing_fingerprint=ProcessingFingerprint(
            _string(value, "build_processing_fingerprint")
        ),
        expected_unit_count=_integer(value, "expected_unit_count"),
        schema_version=_integer(value, "schema_version"),
    )
    if value.get("manifest_sha256") != manifest.manifest_sha256:
        raise NumpyVisualIndexError("stored build manifest digest is invalid")
    return manifest


def _space_dict(space: EmbeddingSpaceRef) -> dict[str, object]:
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


def _space_from_dict(value: dict[str, object]) -> EmbeddingSpaceRef:
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


def _ordinal_from_entry(
    ordinal: int,
    entry: VisualIndexEntry,
    corpus_refs: tuple[str, ...],
) -> dict[str, object]:
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


def _valid_ordinals(
    ordinals: tuple[dict[str, object], ...], manifest: VisualIndexGenerationManifest
) -> bool:
    unit_ids: list[str] = []
    try:
        for expected, row in enumerate(ordinals):
            if _integer(row, "ordinal") != expected:
                return False
            unit_ids.append(_string(row, "unit_id"))
            if TargetType(_string(row, "target_type")) is not manifest.target_type:
                return False
            if tuple(_strings(row, "corpus_refs")) != manifest.corpus_refs:
                return False
            _string(row, "artifact_ref")
        return len(set(unit_ids)) == len(unit_ids)
    except (KeyError, TypeError, ValueError):
        return False


def _filter_ordinals(
    ordinals: tuple[dict[str, object], ...],
    manifest: VisualIndexGenerationManifest,
    filters: SearchFilters,
) -> tuple[int, ...]:
    requested_corpora = set(filters.corpus_refs)
    manifest_corpora = set(manifest.corpus_refs)
    if requested_corpora:
        if not requested_corpora.intersection(manifest_corpora):
            return ()
        if not manifest_corpora.issubset(requested_corpora) and len(manifest_corpora) > 1:
            raise ValueError("corpus subset filtering requires per-unit corpus metadata")
    if filters.media_types:
        raise ValueError("media-type filtering is unsupported by visual ordinals")
    if filters.text_origins:
        raise ValueError("text-origin filtering is unsupported by a visual index")
    selected: list[int] = []
    for row in ordinals:
        if filters.artifact_refs and _string(row, "artifact_ref") not in filters.artifact_refs:
            continue
        if filters.time_range is not None and not _ordinal_overlaps(row, filters.time_range):
            continue
        selected.append(_integer(row, "ordinal"))
    return tuple(selected)


def _ordinal_overlaps(row: dict[str, object], requested: TemporalRange) -> bool:
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


def _payload_content_sha256(ordinals: tuple[dict[str, object], ...], vectors: np.ndarray) -> str:
    payload = [
        {
            "unit_id": _string(row, "unit_id"),
            "target_type": _string(row, "target_type"),
            "vector_float32_be": [
                struct.pack("!f", float(value)).hex() for value in vectors[index]
            ],
        }
        for index, row in enumerate(ordinals)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_unit_vector(vector: tuple[float, ...]) -> None:
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("entry vector must contain only finite values")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isclose(norm, 1.0, abs_tol=_NORM_TOLERANCE, rel_tol=0.0):
        raise ValueError("entry vector must be L2 normalized")


def _require_query_vector(vector: tuple[float, ...]) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("query vector must contain only finite values")
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not math.isclose(norm, 1.0, abs_tol=_NORM_TOLERANCE, rel_tol=0.0):
        raise ValueError("query vector must be L2 normalized")


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    with path.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())


def _write_jsonl_exclusive(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    with path.open("xb") as output:
        for row in rows:
            output.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            )
            output.write(b"\n")
        output.flush()
        os.fsync(output.fileno())


def _read_json_object(path: Path) -> dict[str, object]:
    path = _validated_payload_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NumpyVisualIndexError("stored JSON must be an object")
    return value


def _read_ordinals(path: Path) -> tuple[dict[str, object], ...]:
    path = _validated_payload_file(path)
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise NumpyVisualIndexError("ordinal row must be an object")
        rows.append(value)
    return tuple(rows)


def _validated_payload_file(path: Path) -> Path:
    _reject_symlink_components(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise NumpyVisualIndexError("visual payload file is missing") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise NumpyVisualIndexError("visual payload must be a regular non-symlink file")
    return path


def _require_regular_directory(path: Path) -> None:
    _reject_symlink_components(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise NumpyVisualIndexError("visual generation directory is missing") from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise NumpyVisualIndexError("visual generation path must be a regular directory")


def _reject_symlink_components(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("index path must not contain symlink components")
        current = current.parent


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    "NumpyExactVisualConfiguration",
    "NumpyExactVisualRetriever",
    "NumpyVisualIndexError",
]
