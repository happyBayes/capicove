"""Immutable local FAISS HNSW adapter for Capicove visual retrieval."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from threading import Lock
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from capicove.contracts import EmbeddingSpaceRef, SearchFilters, TargetType
from capicove.retrieval.video_models import (
    PinnedVisualIndexGeneration,
    VisualCandidate,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
    VisualIndexValidationReport,
    visual_entries_content_sha256,
)
from capicove_retriever_faiss.codec import (
    decode_visual_manifest,
    encode_ordinal,
    encode_visual_manifest,
    filter_ordinals,
    reconstructed_stream_sha256,
    validate_ordinals,
)
from capicove_retriever_faiss.config import FaissConfiguration


_GENERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_INDEX_FORMAT = "faiss-hnsw-flat-ip-v1"
_PROVIDER_ID = "capicove.retriever.faiss"
_MAX_TOP_K = 500
_MAX_NATIVE_CANDIDATES = 5_000
_FILTER_CANDIDATE_MULTIPLIER = 10
_NORM_TOLERANCE = 1e-4


class _NativeHnsw(Protocol):
    efConstruction: int
    efSearch: int

    def nb_neighbors(self, level: int) -> int: ...


class _NativeIndex(Protocol):
    d: int
    ntotal: int
    metric_type: int
    hnsw: _NativeHnsw

    def add(self, vectors: NDArray[np.float32]) -> None: ...

    def search(
        self, queries: NDArray[np.float32], top_k: int
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...

    def reconstruct_n(self, start: int, count: int) -> NDArray[np.float32]: ...


class _FaissModule(Protocol):
    METRIC_INNER_PRODUCT: int

    def IndexHNSWFlat(self, dimension: int, m: int, metric_type: int) -> _NativeIndex: ...

    def omp_set_num_threads(self, threads: int) -> None: ...

    def write_index(self, index: _NativeIndex, path: str) -> None: ...

    def read_index(self, path: str) -> _NativeIndex: ...


class FaissOperationError(RuntimeError):
    """Report a sanitized local FAISS storage or retrieval failure."""


class FaissVisualRetriever:
    """Build, validate, activate, and query immutable FAISS HNSW generations."""

    provider_id = _PROVIDER_ID

    def __init__(
        self,
        configuration: FaissConfiguration,
        *,
        faiss_module: object | None = None,
    ) -> None:
        if not isinstance(configuration, FaissConfiguration):
            raise TypeError("configuration must be a FaissConfiguration")
        self._configuration = configuration
        self._faiss = _load_faiss() if faiss_module is None else cast(_FaissModule, faiss_module)
        _reject_symlink_components(configuration.index_root)
        configuration.index_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_regular_directory(configuration.index_root)
        self._open_lock = Lock()
        self._opened: OrderedDict[
            str,
            tuple[
                tuple[tuple[int, ...], ...],
                VisualIndexGenerationManifest,
                tuple[dict[str, object], ...],
                _NativeIndex,
            ],
        ] = OrderedDict()

    @property
    def index_writer(self) -> FaissVisualRetriever:
        """Return this backend's visual writer role."""
        return self

    @property
    def candidate_retriever(self) -> FaissVisualRetriever:
        """Return this backend's visual candidate role."""
        return self

    @property
    def generation_lifecycle(self) -> FaissVisualRetriever:
        """Return this backend's immutable generation lifecycle role."""
        return self

    def create(self, manifest: VisualIndexGenerationManifest) -> str:
        """Create a unique build directory for one frozen public manifest."""
        self._validate_manifest_policy(manifest)
        generation_path = self._generation_path(manifest.generation_id)
        build_manifest_path = generation_path / "build-manifest.json"
        if generation_path.exists():
            _require_regular_directory(generation_path)
            if _read_json_object(build_manifest_path) != encode_visual_manifest(manifest):
                raise FaissOperationError("generation ID identifies a different immutable manifest")
            return str(generation_path)
        try:
            generation_path.mkdir(mode=0o700)
            _write_json_exclusive(build_manifest_path, encode_visual_manifest(manifest))
            _fsync_directory(generation_path)
            _fsync_directory(generation_path.parent)
        except OSError:
            raise FaissOperationError("FAISS generation creation failed") from None
        return str(generation_path)

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[VisualIndexEntry, ...],
    ) -> None:
        """Build and publish one complete HNSW payload exactly once."""
        path = self._generation_path(index_generation_id)
        manifest = decode_visual_manifest(_read_json_object(path / "build-manifest.json"))
        self._validate_manifest_policy(manifest)
        materialized = tuple(entries)
        self._validate_entries(manifest, materialized)
        self.add_stream(index_generation_id=index_generation_id, entries=iter(materialized))

    def add_stream(self, *, index_generation_id: str, entries: Iterable[VisualIndexEntry]) -> None:
        """Build in 512-vector batches without retaining the complete Python corpus.

        Existing valid immutable generations are idempotent by manifest identity.
        A fresh stream must exactly match its predeclared count/content digest.
        """
        path = self._generation_path(index_generation_id)
        manifest = decode_visual_manifest(_read_json_object(path / "build-manifest.json"))
        self._validate_manifest_policy(manifest)
        payload_manifest_path = path / "manifest.json"
        if payload_manifest_path.exists():
            report = self.validate(manifest)
            if report.valid:
                return
            raise FaissOperationError("generation payload already exists but does not validate")

        temporary_paths = {
            name: path / f".{name}.{uuid.uuid4().hex}.tmp"
            for name in ("index.faiss", "ordinals.jsonl", "manifest.json")
        }
        try:
            self._set_threads()
            index = self._faiss.IndexHNSWFlat(
                manifest.embedding_space.dimension,
                self._configuration.hnsw_m,
                self._faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = self._configuration.ef_construction
            index.hnsw.efSearch = self._configuration.ef_search
            self._write_stream(index, manifest, entries, temporary_paths["ordinals.jsonl"])
            self._faiss.write_index(index, str(temporary_paths["index.faiss"]))
            _fsync_regular_file(temporary_paths["index.faiss"])
            payload = {
                "schema_version": 1,
                "manifest_sha256": manifest.manifest_sha256,
                "content_sha256": manifest.content_sha256,
                "index_format_version": _INDEX_FORMAT,
                "target_type": manifest.target_type.value,
                "dimension": manifest.embedding_space.dimension,
                "expected_unit_count": manifest.expected_unit_count,
                "metric": "inner_product",
                "metric_type": self._faiss.METRIC_INNER_PRODUCT,
                "hnsw": {
                    "m": self._configuration.hnsw_m,
                    "ef_construction": self._configuration.ef_construction,
                },
                "ef_search": self._configuration.ef_search,
                "index_sha256": _file_sha256(temporary_paths["index.faiss"]),
                "ordinals_sha256": _file_sha256(temporary_paths["ordinals.jsonl"]),
            }
            _write_json_exclusive(temporary_paths["manifest.json"], payload)
            _publish_exclusive(temporary_paths["index.faiss"], path / "index.faiss")
            _publish_exclusive(temporary_paths["ordinals.jsonl"], path / "ordinals.jsonl")
            _publish_exclusive(temporary_paths["manifest.json"], payload_manifest_path)
            _fsync_directory(path)
        except (OSError, TypeError, ValueError, RuntimeError):
            raise FaissOperationError("FAISS visual payload build failed") from None
        finally:
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)

    def _write_stream(
        self,
        index: _NativeIndex,
        manifest: VisualIndexGenerationManifest,
        entries: Iterable[VisualIndexEntry],
        ordinal_path: Path,
    ) -> None:
        batch: list[tuple[float, ...]] = []
        observed = 0
        with ordinal_path.open("x", encoding="utf-8") as output:

            def consume() -> Iterator[VisualIndexEntry]:
                nonlocal observed
                for entry in entries:
                    if not isinstance(entry, VisualIndexEntry):
                        raise TypeError("invalid visual entry")
                    if observed >= manifest.expected_unit_count:
                        raise ValueError("entry count exceeds manifest")
                    if (
                        entry.unit.unit_type is not manifest.target_type
                        or manifest.embedding_space not in entry.unit.embedding_spaces
                    ):
                        raise ValueError("entry identity mismatch")
                    if len(entry.vector) != manifest.embedding_space.dimension:
                        raise ValueError("entry dimension mismatch")
                    _require_normalized_vector(entry.vector, label="entry vector")
                    output.write(
                        json.dumps(
                            encode_ordinal(observed, entry, manifest.corpus_refs),
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    observed += 1
                    batch.append(entry.vector)
                    if len(batch) == 512:
                        index.add(np.asarray(batch, dtype=np.float32))
                        batch.clear()
                    yield entry

            digest = visual_entries_content_sha256(consume())
            if observed != manifest.expected_unit_count or digest != manifest.content_sha256:
                raise ValueError("stream count or content mismatch")
            if batch:
                index.add(np.asarray(batch, dtype=np.float32))
            output.flush()
            os.fsync(output.fileno())

    def validate(self, manifest: VisualIndexGenerationManifest) -> VisualIndexValidationReport:
        """Reopen and validate native data, ordinals, checksums, and HNSW identity."""
        self._validate_manifest_policy(manifest)
        path = self._generation_path(manifest.generation_id)
        issues: set[str] = set()
        observed_count = 0
        dimension = manifest.embedding_space.dimension
        try:
            if _read_json_object(path / "build-manifest.json") != encode_visual_manifest(manifest):
                issues.add("manifest_mismatch")
            payload = _read_json_object(path / "manifest.json")
            _inspect_payload_identity(payload, manifest, self._configuration, issues)
            index_path = _validated_payload_file(path / "index.faiss")
            ordinals_path = _validated_payload_file(path / "ordinals.jsonl")
            if payload.get("index_sha256") != _file_sha256(index_path):
                issues.add("checksum_mismatch")
            if payload.get("ordinals_sha256") != _file_sha256(ordinals_path):
                issues.add("checksum_mismatch")
            ordinals = _read_ordinals(ordinals_path)
            issues.update(validate_ordinals(ordinals, manifest))
            self._set_threads()
            index = self._faiss.read_index(str(index_path))
            observed_count = _native_integer(index.ntotal, "ntotal")
            native_dimension = _native_integer(index.d, "dimension")
            if native_dimension != dimension:
                issues.add("dimension_mismatch")
            if index.metric_type != self._faiss.METRIC_INNER_PRODUCT:
                issues.add("metric_mismatch")
            if index.hnsw.efConstruction != self._configuration.ef_construction:
                issues.add("hnsw_ef_construction_mismatch")
            if index.hnsw.efSearch != payload.get("ef_search"):
                issues.add("hnsw_ef_search_mismatch")
            native_m = _native_hnsw_m(index)
            if native_m is not None and native_m != self._configuration.hnsw_m:
                issues.add("hnsw_m_mismatch")

            def vectors() -> Iterator[NDArray[np.float32]]:
                for start in range(0, observed_count, 512):
                    count = min(512, observed_count - start)
                    batch = np.asarray(index.reconstruct_n(start, count), dtype=np.float32)
                    if batch.shape != (count, dimension):
                        issues.add("shape_mismatch")
                        raise ValueError("native vector shape mismatch")
                    if not np.isfinite(batch).all():
                        issues.add("non_finite_vectors")
                        raise ValueError("native vector values invalid")
                    if not np.allclose(
                        np.linalg.norm(batch, axis=1), 1.0, atol=_NORM_TOLERANCE, rtol=0.0
                    ):
                        issues.add("normalization_mismatch")
                    yield from batch

            if (
                len(ordinals) == observed_count
                and reconstructed_stream_sha256(ordinals, vectors()) != manifest.content_sha256
            ):
                issues.add("content_mismatch")
        except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
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
            issue_codes=tuple(sorted(issues)),
        )

    def activate(
        self,
        manifest: VisualIndexGenerationManifest,
        validation: VisualIndexValidationReport,
    ) -> PinnedVisualIndexGeneration:
        """Pin a generation only after matching fresh native validation."""
        if not isinstance(validation, VisualIndexValidationReport):
            raise TypeError("validation must be a VisualIndexValidationReport")
        self._validate_manifest_policy(manifest)
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
        if not validation.valid:
            raise ValueError("cannot activate an invalid or stale FAISS generation")
        # Opening verifies immutable payloads (or the unchanged verified cache)
        # before activation returns. Online requests must not pay a cold scan.
        _, opened_manifest, _, _ = self._open_verified(Path(expected_path))
        if opened_manifest != manifest:
            raise ValueError("cannot activate an invalid or stale FAISS generation")
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
        """Search one explicit pin with bounded ANN work and stable tie ordering."""
        _validate_query_inputs(
            query_vector=query_vector,
            embedding_space=embedding_space,
            filters=filters,
            generation=generation,
            top_k=top_k,
        )
        path = self._pinned_path(generation)
        try:
            signature, manifest, ordinals, index = self._open_verified(path)
            if (
                generation.generation_id != manifest.generation_id
                or generation.manifest_sha256 != manifest.manifest_sha256
                or generation.target_type is not manifest.target_type
                or generation.embedding_space != manifest.embedding_space
            ):
                raise ValueError
            allowed = (
                None
                if filters == SearchFilters()
                else frozenset(filter_ordinals(ordinals, manifest, filters))
            )
            if allowed is not None and not allowed:
                return ()
            self._set_threads()
            native_count = top_k
            if allowed is not None and len(allowed) != len(ordinals):
                native_count = min(
                    len(ordinals),
                    _MAX_NATIVE_CANDIDATES,
                    max(top_k, top_k * _FILTER_CANDIDATE_MULTIPLIER),
                )
            query = np.asarray(query_vector, dtype=np.float32).reshape((1, -1))
            scores, native_ordinals = index.search(query, native_count)
            if scores.shape != (1, native_count) or native_ordinals.shape != (
                1,
                native_count,
            ):
                raise ValueError
            ranked: list[tuple[float, str]] = []
            seen: set[int] = set()
            for score_value, ordinal_value in zip(scores[0], native_ordinals[0], strict=True):
                ordinal = int(ordinal_value)
                if ordinal == -1:
                    continue
                if ordinal < 0 or ordinal >= len(ordinals) or ordinal in seen:
                    raise ValueError
                seen.add(ordinal)
                if allowed is not None and ordinal not in allowed:
                    continue
                score = float(score_value)
                if not math.isfinite(score):
                    raise ValueError
                unit_id = _row_string(ordinals[ordinal], "unit_id")
                ranked.append((score, unit_id))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            if _payload_signature(path) != signature:
                raise ValueError("generation changed during search")
        except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            raise FaissOperationError("FAISS visual retrieval failed") from None
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
            for rank, (score, unit_id) in enumerate(ranked[:top_k], start=1)
        )

    def close(self) -> None:
        """Drop cached native references; in-flight readers retain their own references."""
        with self._open_lock:
            self._opened.clear()

    def _open_verified(
        self, path: Path
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        VisualIndexGenerationManifest,
        tuple[dict[str, object], ...],
        _NativeIndex,
    ]:
        """Verify once per unchanged immutable payload, retaining at most two indexes."""
        with self._open_lock:
            signature = _payload_signature(path)
            cached = self._opened.get(str(path))
            if cached is not None and cached[0] == signature:
                self._opened.move_to_end(str(path))
                return cached
            self._opened.pop(str(path), None)
            manifest = decode_visual_manifest(_read_json_object(path / "build-manifest.json"))
            if not self.validate(manifest).valid:
                raise ValueError("invalid generation")
            ordinals = _read_ordinals(path / "ordinals.jsonl")
            index = self._faiss.read_index(str(path / "index.faiss"))
            # Search effort is startup query policy, not immutable graph topology.
            # Configure once before sharing this verified in-memory index.
            index.hnsw.efSearch = self._configuration.ef_search
            if _payload_signature(path) != signature:
                raise ValueError("generation changed during open")
            opened = (signature, manifest, ordinals, index)
            if len(self._opened) >= 2:
                self._opened.popitem(last=False)
            self._opened[str(path)] = opened
            return opened

    def _validate_manifest_policy(self, manifest: VisualIndexGenerationManifest) -> None:
        if not isinstance(manifest, VisualIndexGenerationManifest):
            raise TypeError("manifest must be a VisualIndexGenerationManifest")
        if manifest.retriever_id != self.provider_id:
            raise ValueError("manifest retriever does not match the FAISS backend")
        if manifest.index_format_version != _INDEX_FORMAT:
            raise ValueError("manifest index format is not supported")
        if manifest.expected_unit_count > self._configuration.capacity_for(manifest.target_type):
            raise ValueError("manifest unit count exceeds FAISS target capacity")

    def _validate_entries(
        self,
        manifest: VisualIndexGenerationManifest,
        entries: tuple[VisualIndexEntry, ...],
    ) -> None:
        if len(entries) != manifest.expected_unit_count:
            raise ValueError("entry count must match the generation manifest")
        if len(entries) > self._configuration.capacity_for(manifest.target_type):
            raise ValueError("entry count exceeds FAISS target capacity")
        if any(not isinstance(entry, VisualIndexEntry) for entry in entries):
            raise TypeError("entries must contain VisualIndexEntry values")
        unit_ids = tuple(entry.unit.unit_id for entry in entries)
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("entries must not contain duplicate unit IDs")
        for entry in entries:
            if entry.unit.unit_type is not manifest.target_type:
                raise ValueError("entry target does not match generation target")
            if len(entry.vector) != manifest.embedding_space.dimension:
                raise ValueError("entry vector dimension does not match embedding space")
            if manifest.embedding_space not in entry.unit.embedding_spaces:
                raise ValueError("entry unit does not declare the generation embedding space")
            _require_normalized_vector(entry.vector, label="entry vector")
        if visual_entries_content_sha256(entries) != manifest.content_sha256:
            raise ValueError("entry content digest does not match generation manifest")

    def _generation_path(self, generation_id: str) -> Path:
        if not isinstance(generation_id, str) or _GENERATION_ID.fullmatch(generation_id) is None:
            raise ValueError("generation_id is not safe for a local index path")
        return self._configuration.index_root / generation_id

    def _pinned_path(self, generation: PinnedVisualIndexGeneration) -> Path:
        path = Path(generation.physical_index_ref)
        if not path.is_absolute() or path != self._generation_path(generation.generation_id):
            raise FaissOperationError("pinned physical index is outside index_root")
        _require_regular_directory(path)
        return path

    def _set_threads(self) -> None:
        self._faiss.omp_set_num_threads(self._configuration.threads)


def _load_faiss() -> _FaissModule:
    try:
        module = importlib.import_module("faiss")
    except (ImportError, OSError):
        raise FaissOperationError("FAISS runtime is unavailable") from None
    return cast(_FaissModule, module)


def _inspect_payload_identity(
    payload: dict[str, object],
    manifest: VisualIndexGenerationManifest,
    configuration: FaissConfiguration,
    issues: set[str],
) -> None:
    expected = {
        "schema_version": 1,
        "manifest_sha256": manifest.manifest_sha256,
        "content_sha256": manifest.content_sha256,
        "index_format_version": _INDEX_FORMAT,
        "target_type": manifest.target_type.value,
        "dimension": manifest.embedding_space.dimension,
        "expected_unit_count": manifest.expected_unit_count,
        "metric": "inner_product",
    }
    for field_name, expected_value in expected.items():
        if payload.get(field_name) != expected_value:
            issues.add(f"{field_name}_mismatch")
    stored_effort = payload.get("ef_search")
    if (
        isinstance(stored_effort, bool)
        or not isinstance(stored_effort, int)
        or not 1 <= stored_effort <= 4096
    ):
        issues.add("ef_search_mismatch")
    hnsw = payload.get("hnsw")
    if not isinstance(hnsw, dict) or hnsw != {
        "m": configuration.hnsw_m,
        "ef_construction": configuration.ef_construction,
    }:
        issues.add("hnsw_metadata_mismatch")


def _validate_query_inputs(
    *,
    query_vector: tuple[float, ...],
    embedding_space: EmbeddingSpaceRef,
    filters: SearchFilters,
    generation: PinnedVisualIndexGeneration,
    top_k: int,
) -> None:
    if not isinstance(embedding_space, EmbeddingSpaceRef):
        raise TypeError("embedding_space must be an EmbeddingSpaceRef")
    if not isinstance(filters, SearchFilters):
        raise TypeError("filters must be SearchFilters")
    if not isinstance(generation, PinnedVisualIndexGeneration):
        raise TypeError("generation must be a PinnedVisualIndexGeneration")
    if embedding_space != generation.embedding_space:
        raise ValueError("query embedding space does not match pinned generation")
    if len(query_vector) != embedding_space.dimension:
        raise ValueError("query vector dimension does not match embedding space")
    _require_normalized_vector(query_vector, label="query vector")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= _MAX_TOP_K:
        raise ValueError("top_k must be an integer from 1 to 500")


def _require_normalized_vector(vector: tuple[float, ...], *, label: str) -> None:
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError(f"{label} must contain only finite numbers")
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if not math.isclose(norm, 1.0, abs_tol=_NORM_TOLERANCE, rel_tol=0.0):
        raise ValueError(f"{label} must be L2 normalized")


def _native_hnsw_m(index: _NativeIndex) -> int | None:
    direct = getattr(index.hnsw, "m", None)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    try:
        neighbors = index.hnsw.nb_neighbors(1)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return neighbors if isinstance(neighbors, int) and not isinstance(neighbors, bool) else None


def _native_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"native {label} is invalid")
    return value


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


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(_validated_payload_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("stored JSON must be an object")
    return value


def _read_ordinals(path: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for line in _validated_payload_file(path).read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("ordinal row must be an object")
        rows.append(value)
    return tuple(rows)


def _publish_exclusive(source: Path, destination: Path) -> None:
    os.link(source, destination, follow_symlinks=False)
    source.unlink()


def _fsync_regular_file(path: Path) -> None:
    validated = _validated_payload_file(path)
    with validated.open("rb") as input_file:
        os.fsync(input_file.fileno())


def _payload_signature(path: Path) -> tuple[tuple[int, ...], ...]:
    """Detect replacement, mutation and erasure of every verified member."""
    result = []
    for name in ("build-manifest.json", "manifest.json", "index.faiss", "ordinals.jsonl"):
        value = _validated_payload_file(path / name).stat()
        result.append(
            (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        )
    return tuple(result)


def _validated_payload_file(path: Path) -> Path:
    _reject_symlink_components(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError("FAISS payload file is missing") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ValueError("FAISS payload must be a regular non-symlink file")
    return path


def _require_regular_directory(path: Path) -> None:
    _reject_symlink_components(path)
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ValueError("FAISS generation directory is missing") from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise ValueError("FAISS generation path must be a regular directory")


def _reject_symlink_components(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("FAISS index path must not contain symlink components")
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


def _row_string(value: dict[str, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item.strip():
        raise TypeError(f"{key} must be a non-blank string")
    return item


__all__ = ["FaissOperationError", "FaissVisualRetriever"]
