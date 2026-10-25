"""Bounded immutable NumPy indexes for the personal reverse-image profile."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol, Self, cast

import numpy as np
import numpy.typing as npt

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    IndexGeneration,
    IndexParametersRef,
    MatchKind,
    QueryModality,
    SearchFailureCode,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval.search_models import (
    Candidate,
    IndexedCandidateProvenance,
    PerceptualHashQuery,
    QueryRepresentation,
    QueryVector,
    RetrievalLane,
    SearchFailure,
)


MAX_PERSONAL_UNITS = 10_000
MAX_CANDIDATES = 10_000
_SEMANTIC_FORMAT = "numpy-flat-f32-v1"
_PHASH_FORMAT = "phash-flat-u64-v1"
_FLOAT_DTYPE = np.dtype("<f4")
_PHASH_DTYPE = np.dtype("<u8")
_L2_TOLERANCE = 1e-5
_SCAN_BATCH_SIZE = 256
_MAX_VECTOR_DIMENSION = 16_384


class OrdinalResolver(Protocol):
    """Resolve bounded winning ordinals against the pinned unit catalog."""

    @property
    def unit_count(self) -> int: ...

    @property
    def unit_manifest_sha256(self) -> str: ...

    @property
    def activation_set_id(self) -> str: ...

    def unit_ids_for_ordinals(
        self,
        ordinals: tuple[int, ...],
        *,
        deadline: datetime,
    ) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class OrdinalVector:
    """Bind one target vector to its stable catalog ordinal and exact space."""

    ordinal: int
    space: EmbeddingSpaceRef
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_non_negative_int("ordinal", self.ordinal)
        if not isinstance(self.space, EmbeddingSpaceRef):
            raise TypeError("space must be an EmbeddingSpaceRef")
        if not isinstance(self.values, tuple):
            raise TypeError("values must be a tuple")
        normalized: list[float] = []
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("values must contain finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("values must contain finite numbers")
            normalized.append(number)
        object.__setattr__(self, "values", tuple(normalized))


@dataclass(frozen=True, slots=True)
class OrdinalPhash:
    """Bind one unsigned pHash to its stable catalog ordinal and exact space."""

    ordinal: int
    space: HashSpaceRef
    value: int

    def __post_init__(self) -> None:
        _require_non_negative_int("ordinal", self.ordinal)
        if not isinstance(self.space, HashSpaceRef):
            raise TypeError("space must be a HashSpaceRef")
        _require_uint64("value", self.value)


@dataclass(frozen=True, slots=True)
class PersonalIndexPayload:
    """Describe one sealed local index payload without treating a ref as a path."""

    path: Path
    sha256: str
    size_bytes: int
    unit_count: int
    shape: tuple[int, ...]
    dtype: str
    c_order: bool
    generation_id: str
    activation_set_id: str
    unit_manifest_sha256: str
    index_format: str
    index_parameters_ref: IndexParametersRef
    lane: RetrievalLane
    embedding_space: EmbeddingSpaceRef | None = None
    hash_space: HashSpaceRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be a Path")
        _require_sha256("sha256", self.sha256)
        _require_positive_int("size_bytes", self.size_bytes)
        _require_positive_int("unit_count", self.unit_count)
        if self.unit_count > MAX_PERSONAL_UNITS:
            raise ValueError("unit_count exceeds the 10,000-unit personal limit")
        if not self.shape or self.shape[0] != self.unit_count:
            raise ValueError("shape must begin with unit_count")
        if not isinstance(self.c_order, bool) or not self.c_order:
            raise ValueError("payload must use C order")
        for name in ("generation_id", "activation_set_id", "index_format", "dtype"):
            _require_non_blank(name, getattr(self, name))
        _require_sha256("unit_manifest_sha256", self.unit_manifest_sha256)
        if not isinstance(self.index_parameters_ref, IndexParametersRef):
            raise TypeError("index_parameters_ref must be an IndexParametersRef")
        if not isinstance(self.lane, RetrievalLane):
            raise TypeError("lane must be a RetrievalLane")
        if self.lane in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
            if not isinstance(self.embedding_space, EmbeddingSpaceRef):
                raise ValueError("semantic payload requires one embedding space")
            if self.hash_space is not None:
                raise ValueError("semantic payload must not carry a hash space")
        elif self.lane is RetrievalLane.PERCEPTUAL:
            if not isinstance(self.hash_space, HashSpaceRef):
                raise ValueError("pHash payload requires one hash space")
            if self.embedding_space is not None:
                raise ValueError("pHash payload must not carry an embedding space")
        else:
            raise ValueError("personal NumPy payload cannot represent the content lane")


@dataclass(slots=True)
class _OpenedPayload:
    path: Path
    handle: BinaryIO
    fingerprint: tuple[int, int, int, int, int]

    def close(self) -> None:
        self.handle.close()


class NumpyExactIndexWriter:
    """Write ordinal-aligned normalized float32 vectors to a sealed ``.npy`` file."""

    def __init__(self, max_units: int = MAX_PERSONAL_UNITS) -> None:
        self._max_units = _validated_capacity(max_units)

    def build_array(
        self,
        vectors: npt.NDArray[np.generic],
        generation: IndexGeneration,
        destination: Path,
        *,
        lane: RetrievalLane,
        deadline: datetime,
    ) -> PersonalIndexPayload:
        """Validate and persist one small in-memory test/ingest array."""
        _check_deadline(deadline)
        space = _semantic_generation_space(generation, lane)
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a NumPy array")
        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (unit_count, dimension)")
        unit_count, dimension = vectors.shape
        self._require_count_before_write(unit_count)
        if dimension != space.dimension:
            raise ValueError("vector dimension must match the complete embedding space")
        if vectors.dtype != _FLOAT_DTYPE:
            raise ValueError("vectors must use little-endian float32")
        if not vectors.flags.c_contiguous:
            raise ValueError("vectors must be C-contiguous")
        _validate_vector_array(vectors)
        _check_deadline(deadline)
        path = _resolve_destination(destination, generation.generation_id, lane.value)
        return self._write_array(
            vectors,
            generation,
            path,
            lane=lane,
            embedding_space=space,
            deadline=deadline,
        )

    def build(
        self,
        generation: IndexGeneration,
        rows: Iterable[OrdinalVector],
        destination: Path,
        *,
        expected_units: int,
        lane: RetrievalLane,
        deadline: datetime,
    ) -> PersonalIndexPayload:
        """Stream validated ordinal-contiguous rows directly into an on-disk array."""
        _check_deadline(deadline)
        space = _semantic_generation_space(generation, lane)
        self._require_count_before_write(expected_units)
        path = _resolve_destination(destination, generation.generation_id, lane.value)
        _require_new_destination(path)
        temporary = _temporary_sibling(path)
        array: np.memmap[Any, Any] | None = None
        try:
            array = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=_FLOAT_DTYPE,
                shape=(expected_units, space.dimension),
                fortran_order=False,
            )
            seen = 0
            for expected_ordinal, row in enumerate(rows):
                _check_deadline(deadline)
                if expected_ordinal >= expected_units:
                    raise ValueError("rows exceed expected_units")
                if not isinstance(row, OrdinalVector):
                    raise TypeError("rows must contain OrdinalVector values")
                if row.ordinal != expected_ordinal:
                    raise ValueError("vector ordinals must be contiguous from zero")
                if row.space != space:
                    raise ValueError("vector space must match the complete generation space")
                vector = np.asarray(row.values, dtype=_FLOAT_DTYPE)
                if vector.shape != (space.dimension,):
                    raise ValueError("vector dimension must match the complete embedding space")
                _validate_vector_array(vector.reshape(1, -1))
                array[expected_ordinal] = vector
                seen += 1
            if seen != expected_units:
                raise ValueError("row count must equal expected_units")
            array.flush()
            _close_memmap(array)
            del array
            array = None
            _check_deadline(deadline)
            sha256, size_bytes = _sealed_file_identity(temporary, deadline)
            _publish_new(temporary, path)
        except BaseException:
            if array is not None:
                _close_memmap(array)
                del array
            _remove_temporary_sibling(temporary)
            raise
        return _payload_for_file(
            path,
            generation,
            sha256=sha256,
            size_bytes=size_bytes,
            unit_count=expected_units,
            shape=(expected_units, space.dimension),
            dtype=_FLOAT_DTYPE,
            lane=lane,
            embedding_space=space,
            hash_space=None,
        )

    def _require_count_before_write(self, unit_count: int) -> None:
        _require_positive_int("unit_count", unit_count)
        if unit_count > self._max_units:
            raise ValueError(f"unit_count exceeds the {self._max_units:,}-unit limit")

    def _write_array(
        self,
        values: npt.NDArray[np.generic],
        generation: IndexGeneration,
        path: Path,
        *,
        lane: RetrievalLane,
        embedding_space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> PersonalIndexPayload:
        _require_new_destination(path)
        temporary = _temporary_sibling(path)
        array: np.memmap[Any, Any] | None = None
        try:
            array = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=_FLOAT_DTYPE,
                shape=values.shape,
                fortran_order=False,
            )
            array[:] = values
            array.flush()
            _close_memmap(array)
            del array
            array = None
            _check_deadline(deadline)
            sha256, size_bytes = _sealed_file_identity(temporary, deadline)
            _publish_new(temporary, path)
        except BaseException:
            if array is not None:
                _close_memmap(array)
                del array
            _remove_temporary_sibling(temporary)
            raise
        unit_count, dimension = values.shape
        return _payload_for_file(
            path,
            generation,
            sha256=sha256,
            size_bytes=size_bytes,
            unit_count=unit_count,
            shape=(unit_count, dimension),
            dtype=_FLOAT_DTYPE,
            lane=lane,
            embedding_space=embedding_space,
            hash_space=None,
        )


class NumpyExactIndexReader:
    """Search a checksum-verified read-only normalized float32 mmap exactly."""

    def __init__(
        self,
        generation: IndexGeneration,
        payload: PersonalIndexPayload,
        ordinal_resolver: OrdinalResolver,
        array: np.memmap[Any, Any],
        opened_payload: _OpenedPayload,
    ) -> None:
        self._generation = generation
        self._payload = payload
        self._ordinal_resolver = ordinal_resolver
        self._array = array
        self._opened_payload = opened_payload
        self._closed = False

    @classmethod
    def open_readonly(
        cls,
        generation: IndexGeneration,
        payload: PersonalIndexPayload,
        *,
        ordinal_resolver: OrdinalResolver,
        deadline: datetime,
    ) -> Self:
        """Verify immutable identities and native format before returning a reader."""
        _check_deadline(deadline)
        space = _semantic_generation_space(generation, payload.lane)
        _validate_payload_binding(generation, payload, ordinal_resolver)
        if payload.embedding_space != space:
            raise SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
        if payload.dtype != _FLOAT_DTYPE.str:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        opened_payload = _open_verified_payload(payload, deadline)
        array: np.memmap[Any, Any] | None = None
        try:
            array = _open_npy_memmap(opened_payload)
            if not isinstance(array, np.memmap):
                raise ValueError("NumPy payload did not produce a memory map")
            if (
                array.dtype != _FLOAT_DTYPE
                or array.shape != payload.shape
                or array.shape != (payload.unit_count, space.dimension)
                or not array.flags.c_contiguous
                or array.flags.writeable
            ):
                raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
            for start in range(0, payload.unit_count, _SCAN_BATCH_SIZE):
                _check_deadline(deadline)
                _validate_vector_array(array[start : start + _SCAN_BATCH_SIZE])
            _verify_opened_payload_unchanged(
                opened_payload,
                payload,
                deadline=deadline,
                verify_digest=True,
            )
        except (OSError, TypeError, ValueError, OverflowError) as error:
            if array is not None and isinstance(array, np.memmap):
                _close_memmap(array)
            opened_payload.close()
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error
        except BaseException:
            if array is not None and isinstance(array, np.memmap):
                _close_memmap(array)
            opened_payload.close()
            raise
        return cls(generation, payload, ordinal_resolver, array, opened_payload)

    @property
    def is_read_only(self) -> bool:
        return not self._array.flags.writeable

    @property
    def dtype(self) -> np.dtype[np.float32]:
        return cast("np.dtype[np.float32]", self._array.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    def search(
        self,
        query: QueryRepresentation,
        selection: object | None,
        *,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]:
        """Return exact descending inner-product results with ordinal tie-breaking."""
        self._require_open()
        _check_deadline(deadline)
        _verify_opened_payload_unchanged(
            self._opened_payload,
            self._payload,
            deadline=deadline,
            verify_digest=False,
        )
        if not isinstance(query, QueryVector):
            raise SearchFailure(SearchFailureCode.INVALID_QUERY)
        space = cast(EmbeddingSpaceRef, self._payload.embedding_space)
        if query.space != space:
            raise SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
        candidate_limit = _validated_limit(limit)
        ordinals = _selected_ordinals(selection, self._payload.unit_count)
        if not ordinals:
            return ()
        query_array = np.asarray(query.values, dtype=_FLOAT_DTYPE)
        scored: list[tuple[float, int]] = []
        for start in range(0, len(ordinals), _SCAN_BATCH_SIZE):
            _check_deadline(deadline)
            batch_ordinals = ordinals[start : start + _SCAN_BATCH_SIZE]
            indexes = np.asarray(batch_ordinals, dtype=np.intp)
            scores = self._array[indexes] @ query_array
            scored.extend(
                (float(score), ordinal)
                for score, ordinal in zip(scores.tolist(), batch_ordinals, strict=True)
            )
        winners = sorted(scored, key=lambda item: (-item[0], item[1]))[:candidate_limit]
        winning_ordinals = tuple(ordinal for _, ordinal in winners)
        _check_deadline(deadline)
        unit_ids = self._ordinal_resolver.unit_ids_for_ordinals(
            winning_ordinals,
            deadline=deadline,
        )
        _check_deadline(deadline)
        if len(unit_ids) != len(winning_ordinals) or len(set(unit_ids)) != len(unit_ids):
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        match_kind = (
            MatchKind.VISUAL if self._payload.lane is RetrievalLane.INSTANCE else MatchKind.SEMANTIC
        )
        provenance = IndexedCandidateProvenance(
            activation_set_id=self._generation.activation_set_id,
            generation_id=self._generation.generation_id,
            provider_id=space.provider_id,
        )
        results = tuple(
            Candidate(
                unit_id=unit_id,
                unit_ordinal=ordinal,
                lane=self._payload.lane,
                match_kind=match_kind,
                raw_score=score,
                score_semantics="normalized-inner-product",
                rank=rank,
                provenance=provenance,
                embedding_space=space,
            )
            for rank, (unit_id, (score, ordinal)) in enumerate(
                zip(unit_ids, winners, strict=True),
                start=1,
            )
        )
        _verify_opened_payload_unchanged(
            self._opened_payload,
            self._payload,
            deadline=deadline,
            verify_digest=False,
        )
        return results

    def close(self) -> None:
        if not self._closed:
            _close_memmap(self._array)
            self._opened_payload.close()
            self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("index reader is closed")


class PhashFlatIndexWriter:
    """Write ordinal-aligned unsigned 64-bit pHashes to a sealed ``.npy`` file."""

    def __init__(self, max_units: int = MAX_PERSONAL_UNITS) -> None:
        self._max_units = _validated_capacity(max_units)

    def build_values(
        self,
        values: Sequence[int] | npt.NDArray[np.generic],
        generation: IndexGeneration,
        destination: Path,
        *,
        deadline: datetime,
    ) -> PersonalIndexPayload:
        """Validate and persist unsigned pHashes in ordinal order."""
        _check_deadline(deadline)
        space = _phash_generation_space(generation)
        unit_count = len(values)
        self._require_count_before_write(unit_count)
        if isinstance(values, np.ndarray):
            if values.ndim != 1 or values.dtype != _PHASH_DTYPE:
                raise ValueError("pHash array must be one-dimensional little-endian uint64")
            normalized = values
        else:
            normalized = np.empty(unit_count, dtype=_PHASH_DTYPE)
            for ordinal, value in enumerate(values):
                _require_uint64("pHash value", value)
                normalized[ordinal] = value
        path = _resolve_destination(destination, generation.generation_id, "perceptual")
        return self._write_array(normalized, generation, path, space=space, deadline=deadline)

    def build(
        self,
        generation: IndexGeneration,
        rows: Iterable[OrdinalPhash],
        destination: Path,
        *,
        expected_units: int,
        deadline: datetime,
    ) -> PersonalIndexPayload:
        """Persist explicitly ordinal-bound pHash rows."""
        _check_deadline(deadline)
        space = _phash_generation_space(generation)
        self._require_count_before_write(expected_units)
        values = np.empty(expected_units, dtype=_PHASH_DTYPE)
        seen = 0
        for expected_ordinal, row in enumerate(rows):
            _check_deadline(deadline)
            if expected_ordinal >= expected_units:
                raise ValueError("rows exceed expected_units")
            if not isinstance(row, OrdinalPhash):
                raise TypeError("rows must contain OrdinalPhash values")
            if row.ordinal != expected_ordinal:
                raise ValueError("pHash ordinals must be contiguous from zero")
            if row.space != space:
                raise ValueError("pHash space must match the complete generation space")
            values[expected_ordinal] = row.value
            seen += 1
        if seen != expected_units:
            raise ValueError("row count must equal expected_units")
        path = _resolve_destination(destination, generation.generation_id, "perceptual")
        return self._write_array(values, generation, path, space=space, deadline=deadline)

    def _require_count_before_write(self, unit_count: int) -> None:
        _require_positive_int("unit_count", unit_count)
        if unit_count > self._max_units:
            raise ValueError(f"unit_count exceeds the {self._max_units:,}-unit limit")

    def _write_array(
        self,
        values: npt.NDArray[np.generic],
        generation: IndexGeneration,
        path: Path,
        *,
        space: HashSpaceRef,
        deadline: datetime,
    ) -> PersonalIndexPayload:
        _require_new_destination(path)
        temporary = _temporary_sibling(path)
        array: np.memmap[Any, Any] | None = None
        try:
            array = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=_PHASH_DTYPE,
                shape=values.shape,
                fortran_order=False,
            )
            array[:] = values
            array.flush()
            _close_memmap(array)
            del array
            array = None
            _check_deadline(deadline)
            sha256, size_bytes = _sealed_file_identity(temporary, deadline)
            _publish_new(temporary, path)
        except BaseException:
            if array is not None:
                _close_memmap(array)
                del array
            _remove_temporary_sibling(temporary)
            raise
        return _payload_for_file(
            path,
            generation,
            sha256=sha256,
            size_bytes=size_bytes,
            unit_count=len(values),
            shape=(len(values),),
            dtype=_PHASH_DTYPE,
            lane=RetrievalLane.PERCEPTUAL,
            embedding_space=None,
            hash_space=space,
        )


class PhashFlatIndexReader:
    """Search a checksum-verified read-only pHash mmap with exact Hamming distance."""

    def __init__(
        self,
        generation: IndexGeneration,
        payload: PersonalIndexPayload,
        ordinal_resolver: OrdinalResolver,
        array: np.memmap[Any, Any],
        opened_payload: _OpenedPayload,
    ) -> None:
        self._generation = generation
        self._payload = payload
        self._ordinal_resolver = ordinal_resolver
        self._array = array
        self._opened_payload = opened_payload
        self._closed = False

    @classmethod
    def open_readonly(
        cls,
        generation: IndexGeneration,
        payload: PersonalIndexPayload,
        *,
        ordinal_resolver: OrdinalResolver,
        deadline: datetime,
    ) -> Self:
        """Verify identity, digest, dtype, shape, and read-only mmap state."""
        _check_deadline(deadline)
        space = _phash_generation_space(generation)
        _validate_payload_binding(generation, payload, ordinal_resolver)
        if payload.lane is not RetrievalLane.PERCEPTUAL or payload.hash_space != space:
            raise SearchFailure(SearchFailureCode.HASH_SPACE_INCOMPATIBLE)
        if payload.dtype != _PHASH_DTYPE.str:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        opened_payload = _open_verified_payload(payload, deadline)
        array: np.memmap[Any, Any] | None = None
        try:
            array = _open_npy_memmap(opened_payload)
            if not isinstance(array, np.memmap):
                raise ValueError("NumPy payload did not produce a memory map")
            if (
                array.dtype != _PHASH_DTYPE
                or array.shape != payload.shape
                or array.shape != (payload.unit_count,)
                or not array.flags.c_contiguous
                or array.flags.writeable
            ):
                raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
            _verify_opened_payload_unchanged(
                opened_payload,
                payload,
                deadline=deadline,
                verify_digest=True,
            )
        except (OSError, TypeError, ValueError, OverflowError) as error:
            if array is not None and isinstance(array, np.memmap):
                _close_memmap(array)
            opened_payload.close()
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error
        except BaseException:
            if array is not None and isinstance(array, np.memmap):
                _close_memmap(array)
            opened_payload.close()
            raise
        return cls(generation, payload, ordinal_resolver, array, opened_payload)

    @property
    def is_read_only(self) -> bool:
        return not self._array.flags.writeable

    @property
    def dtype(self) -> np.dtype[np.uint64]:
        return cast("np.dtype[np.uint64]", self._array.dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._array.shape)

    def search(
        self,
        query: QueryRepresentation,
        selection: object | None,
        *,
        limit: int,
        deadline: datetime,
    ) -> tuple[Candidate, ...]:
        """Return radius-bounded Hamming matches in stable distance/ordinal order."""
        self._require_open()
        _check_deadline(deadline)
        _verify_opened_payload_unchanged(
            self._opened_payload,
            self._payload,
            deadline=deadline,
            verify_digest=False,
        )
        if not isinstance(query, PerceptualHashQuery):
            raise SearchFailure(SearchFailureCode.INVALID_QUERY)
        space = cast(HashSpaceRef, self._payload.hash_space)
        if query.space != space:
            raise SearchFailure(SearchFailureCode.HASH_SPACE_INCOMPATIBLE)
        candidate_limit = _validated_limit(limit)
        ordinals = _selected_ordinals(selection, self._payload.unit_count)
        matches: list[tuple[int, int]] = []
        for start in range(0, len(ordinals), _SCAN_BATCH_SIZE):
            _check_deadline(deadline)
            for ordinal in ordinals[start : start + _SCAN_BATCH_SIZE]:
                distance = (int(self._array[ordinal]) ^ query.value).bit_count()
                if distance <= query.max_hamming_distance and distance <= 2:
                    matches.append((distance, ordinal))
        winners = sorted(matches, key=lambda item: (item[0], item[1]))[:candidate_limit]
        winning_ordinals = tuple(ordinal for _, ordinal in winners)
        _check_deadline(deadline)
        unit_ids = self._ordinal_resolver.unit_ids_for_ordinals(
            winning_ordinals,
            deadline=deadline,
        )
        _check_deadline(deadline)
        if len(unit_ids) != len(winning_ordinals) or len(set(unit_ids)) != len(unit_ids):
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        provenance = IndexedCandidateProvenance(
            activation_set_id=self._generation.activation_set_id,
            generation_id=self._generation.generation_id,
            provider_id=space.provider_id,
        )
        results = tuple(
            Candidate(
                unit_id=unit_id,
                unit_ordinal=ordinal,
                lane=RetrievalLane.PERCEPTUAL,
                match_kind=MatchKind.NEAR_DUPLICATE,
                raw_score=float(distance),
                score_semantics="hamming-distance",
                rank=rank,
                provenance=provenance,
                hash_space=space,
            )
            for rank, (unit_id, (distance, ordinal)) in enumerate(
                zip(unit_ids, winners, strict=True),
                start=1,
            )
        )
        _verify_opened_payload_unchanged(
            self._opened_payload,
            self._payload,
            deadline=deadline,
            verify_digest=False,
        )
        return results

    def close(self) -> None:
        if not self._closed:
            _close_memmap(self._array)
            self._opened_payload.close()
            self._closed = True

    def __enter__(self) -> Self:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("index reader is closed")


def _semantic_generation_space(
    generation: IndexGeneration,
    lane: RetrievalLane,
) -> EmbeddingSpaceRef:
    if not isinstance(generation, IndexGeneration):
        raise TypeError("generation must be an IndexGeneration")
    if lane not in (RetrievalLane.INSTANCE, RetrievalLane.CONCEPT):
        raise ValueError("semantic index lane must be instance or concept")
    if (
        generation.target_type is not TargetType.IMAGE
        or generation.index_format != _SEMANTIC_FORMAT
        or generation.index_parameters_ref.parameter_schema != _SEMANTIC_FORMAT
        or len(generation.embedding_spaces) != 1
        or generation.hash_spaces
    ):
        raise ValueError("generation is not a compatible personal semantic generation")
    space = generation.embedding_spaces[0]
    if (
        space.dimension > _MAX_VECTOR_DIMENSION
        or space.normalization is not VectorNormalization.L2
        or space.distance_metric not in (DistanceMetric.COSINE, DistanceMetric.INNER_PRODUCT)
        or not space.accepts(QueryModality.IMAGE, TargetType.IMAGE)
    ):
        raise ValueError("semantic generation requires an L2 image embedding space")
    return space


def _phash_generation_space(generation: IndexGeneration) -> HashSpaceRef:
    if not isinstance(generation, IndexGeneration):
        raise TypeError("generation must be an IndexGeneration")
    if (
        generation.target_type is not TargetType.IMAGE
        or generation.index_format != _PHASH_FORMAT
        or generation.index_parameters_ref.parameter_schema != _PHASH_FORMAT
        or len(generation.hash_spaces) != 1
        or generation.embedding_spaces
    ):
        raise ValueError("generation is not a compatible personal pHash generation")
    return generation.hash_spaces[0]


def _validate_payload_binding(
    generation: IndexGeneration,
    payload: PersonalIndexPayload,
    resolver: OrdinalResolver,
) -> None:
    if not isinstance(payload, PersonalIndexPayload):
        raise TypeError("payload must be a PersonalIndexPayload")
    if (
        payload.generation_id != generation.generation_id
        or payload.activation_set_id != generation.activation_set_id
        or payload.index_format != generation.index_format
        or payload.index_parameters_ref != generation.index_parameters_ref
    ):
        raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)
    try:
        resolver_count = resolver.unit_count
        resolver_manifest = resolver.unit_manifest_sha256
        resolver_activation = resolver.activation_set_id
    except AttributeError as error:
        raise TypeError("ordinal_resolver must expose pinned catalog identity") from error
    if (
        payload.unit_manifest_sha256 != generation.unit_manifest_sha256
        or resolver_manifest != generation.unit_manifest_sha256
        or resolver_count != payload.unit_count
    ):
        raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
    if resolver_activation != generation.activation_set_id:
        raise SearchFailure(SearchFailureCode.INDEX_GENERATION_UNAVAILABLE)


def _payload_for_file(
    path: Path,
    generation: IndexGeneration,
    *,
    sha256: str,
    size_bytes: int,
    unit_count: int,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    lane: RetrievalLane,
    embedding_space: EmbeddingSpaceRef | None,
    hash_space: HashSpaceRef | None,
) -> PersonalIndexPayload:
    return PersonalIndexPayload(
        path=path,
        sha256=sha256,
        size_bytes=size_bytes,
        unit_count=unit_count,
        shape=shape,
        dtype=dtype.str,
        c_order=True,
        generation_id=generation.generation_id,
        activation_set_id=generation.activation_set_id,
        unit_manifest_sha256=generation.unit_manifest_sha256,
        index_format=generation.index_format,
        index_parameters_ref=generation.index_parameters_ref,
        lane=lane,
        embedding_space=embedding_space,
        hash_space=hash_space,
    )


def _open_verified_payload(
    payload: PersonalIndexPayload,
    deadline: datetime,
) -> _OpenedPayload:
    _check_deadline(deadline)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor: int | None = None
    handle: BinaryIO | None = None
    try:
        descriptor = os.open(payload.path, flags)
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = None
        file_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != payload.size_bytes:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        opened = _OpenedPayload(
            path=payload.path,
            handle=handle,
            fingerprint=_file_fingerprint(file_stat),
        )
        _verify_path_binding(opened)
        if _digest_file(handle, deadline=deadline) != payload.sha256:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        _verify_opened_payload_unchanged(
            opened,
            payload,
            deadline=deadline,
            verify_digest=False,
        )
        handle.seek(0)
        return opened
    except SearchFailure:
        if handle is not None:
            handle.close()
        raise
    except OSError as error:
        if handle is not None:
            handle.close()
        raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_npy_memmap(opened: _OpenedPayload) -> np.memmap[Any, Any]:
    handle = opened.handle
    handle.seek(0)
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError("NumPy payload version is unsupported")
    if dtype.hasobject:
        raise ValueError("NumPy object payloads are forbidden")
    offset = handle.tell()
    expected_size = offset + math.prod(shape) * dtype.itemsize
    if expected_size != opened.fingerprint[2]:
        raise ValueError("NumPy payload size does not match its native header")
    try:
        return np.memmap(
            handle,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )
    except (OSError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("NumPy payload could not be mapped read-only") from error


def _verify_opened_payload_unchanged(
    opened: _OpenedPayload,
    payload: PersonalIndexPayload,
    *,
    deadline: datetime,
    verify_digest: bool,
) -> None:
    _check_deadline(deadline)
    try:
        if _file_fingerprint(os.fstat(opened.handle.fileno())) != opened.fingerprint:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        _verify_path_binding(opened)
        if verify_digest and _digest_file(opened.handle, deadline=deadline) != payload.sha256:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        if _file_fingerprint(os.fstat(opened.handle.fileno())) != opened.fingerprint:
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        _verify_path_binding(opened)
    except OSError as error:
        raise SearchFailure(SearchFailureCode.INDEX_CORRUPT) from error
    _check_deadline(deadline)


def _verify_path_binding(opened: _OpenedPayload) -> None:
    path_stat = opened.path.lstat()
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
    if _file_fingerprint(path_stat) != opened.fingerprint:
        raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)


def _file_fingerprint(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _validate_vector_array(vectors: npt.NDArray[np.generic]) -> None:
    if not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.sum(vectors.astype(np.float64) ** 2, axis=1)
    if not np.all(np.isclose(norms, 1.0, rtol=0.0, atol=_L2_TOLERANCE)):
        raise ValueError("vectors must be L2-normalized")


def _selected_ordinals(selection: object | None, unit_count: int) -> tuple[int, ...]:
    if selection is None:
        return tuple(range(unit_count))
    if isinstance(selection, (tuple, list)):
        return _validate_ordinal_tuple(tuple(selection), unit_count)
    selection_count = getattr(selection, "unit_count", None)
    if selection_count != unit_count:
        raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
    raw_ordinals = getattr(selection, "ordinals", None)
    if raw_ordinals is not None:
        if not isinstance(raw_ordinals, tuple):
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        return _validate_ordinal_tuple(raw_ordinals, unit_count)
    bits = getattr(selection, "bits", None)
    if bits is not None:
        if not isinstance(bits, bytes) or len(bits) != math.ceil(unit_count / 8):
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        ordinals = tuple(
            ordinal for ordinal in range(unit_count) if bits[ordinal // 8] & (1 << (ordinal % 8))
        )
        matching_count = getattr(selection, "matching_count", len(ordinals))
        if matching_count != len(ordinals):
            raise SearchFailure(SearchFailureCode.INDEX_CORRUPT)
        return ordinals
    return tuple(range(unit_count))


def _validate_ordinal_tuple(values: tuple[object, ...], unit_count: int) -> tuple[int, ...]:
    ordinals: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("selected ordinals must be integers")
        if not 0 <= value < unit_count:
            raise ValueError("selected ordinal is outside the generation")
        ordinals.append(value)
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("selected ordinals must be unique")
    return tuple(ordinals)


def _resolve_destination(destination: Path, generation_id: str, lane: str) -> Path:
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path")
    if destination.exists() and destination.is_dir():
        return destination / f"{generation_id}-{lane}.npy"
    if destination.suffix != ".npy":
        raise ValueError("destination must be an existing directory or a .npy path")
    return destination


def _require_new_destination(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"index payload already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("destination parent must be an existing non-symlink directory")


def _temporary_sibling(path: Path) -> Path:
    staging_directory = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    staging_directory.chmod(0o700)
    return staging_directory / "payload.npy"


def _remove_temporary_sibling(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        path.parent.rmdir()
    except FileNotFoundError:
        pass


def _publish_new(temporary: Path, destination: Path) -> None:
    temporary.chmod(0o600)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"index payload already exists: {destination}") from None
    finally:
        _remove_temporary_sibling(temporary)
    directory_descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _sealed_file_identity(path: Path, deadline: datetime) -> tuple[str, int]:
    sha256 = _digest_file(path, deadline=deadline)
    _check_deadline(deadline)
    return sha256, path.stat().st_size


def _digest_file(source: Path | BinaryIO, *, deadline: datetime | None = None) -> str:
    if isinstance(source, Path):
        with source.open("rb") as handle:
            return _digest_file(handle, deadline=deadline)
    digest = hashlib.sha256()
    source.seek(0)
    while chunk := source.read(1024 * 1024):
        if deadline is not None:
            _check_deadline(deadline)
        digest.update(chunk)
    source.seek(0)
    return digest.hexdigest()


def _close_memmap(array: np.memmap[Any, Any]) -> None:
    backing = cast(Any, getattr(array, "_mmap", None))
    if backing is not None:
        backing.close()


def _validated_capacity(value: int) -> int:
    _require_positive_int("max_units", value)
    if value > MAX_PERSONAL_UNITS:
        raise ValueError("max_units must not exceed the 10,000-unit personal limit")
    return value


def _validated_limit(value: int) -> int:
    _require_positive_int("limit", value)
    if value > MAX_CANDIDATES:
        raise ValueError(f"limit must be at most {MAX_CANDIDATES}")
    return value


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be timezone-aware")
    if datetime.now(UTC) >= deadline:
        raise TimeoutError("personal index deadline exceeded")


def _require_uint64(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**64:
        raise ValueError(f"{name} must be an unsigned 64-bit integer")


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_non_blank(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-blank string")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


__all__ = [
    "MAX_PERSONAL_UNITS",
    "NumpyExactIndexReader",
    "NumpyExactIndexWriter",
    "OrdinalPhash",
    "OrdinalResolver",
    "OrdinalVector",
    "PersonalIndexPayload",
    "PhashFlatIndexReader",
    "PhashFlatIndexWriter",
]
