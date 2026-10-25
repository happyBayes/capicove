"""Tests for bounded personal NumPy retrieval indexes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    IndexGeneration,
    IndexParametersRef,
    ProcessingFingerprint,
    QueryModality,
    SearchFailureCode,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters import personal_numpy
from capicove.retrieval._adapters.personal_numpy import (
    NumpyExactIndexReader,
    NumpyExactIndexWriter,
    OrdinalPhash,
    OrdinalVector,
    PhashFlatIndexReader,
    PhashFlatIndexWriter,
)
from capicove.retrieval.search_models import (
    IndexedCandidateProvenance,
    PerceptualHashQuery,
    QueryVector,
    RetrievalLane,
    SearchFailure,
)
from capicove.retrieval.search_ports import IndexReader


FUTURE = datetime.now(UTC) + timedelta(hours=1)
PAST = datetime.now(UTC) - timedelta(seconds=1)
SHA_A = "a" * 64
SHA_B = "b" * 64


def test_numpy_staging_path_is_unpublished_inside_private_directory(tmp_path: Path) -> None:
    destination = tmp_path / "index.npy"

    staging_path = personal_numpy._temporary_sibling(destination)
    try:
        assert not staging_path.exists()
        assert staging_path.parent.parent == destination.parent
        assert staging_path.parent.stat().st_mode & 0o777 == 0o700
    finally:
        personal_numpy._remove_temporary_sibling(staging_path)


def _embedding_space(*, space_id: str = "image-space-v1") -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=space_id,
        provider_id="local-image-encoder",
        model_revision="revision-1",
        preprocessing_version="image-preprocess-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _hash_space(*, space_id: str = "opencv-phash-64-v1") -> HashSpaceRef:
    return HashSpaceRef(
        space_id=space_id,
        provider_id="opencv-phash",
        provider_version="4.12.0",
        preprocessing_version="opencv-phash-preprocess-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _generation(
    *,
    embedding_space: EmbeddingSpaceRef | None = None,
    hash_space: HashSpaceRef | None = None,
    generation_id: str = "generation-1",
    unit_manifest_sha256: str = SHA_A,
) -> IndexGeneration:
    is_phash = hash_space is not None
    index_format = "phash-flat-u64-v1" if is_phash else "numpy-flat-f32-v1"
    return IndexGeneration(
        generation_id=generation_id,
        activation_set_id="activation-1",
        workspace_id="workspace-1",
        corpus_ref="corpus-1",
        target_type=TargetType.IMAGE,
        unit_manifest_sha256=unit_manifest_sha256,
        embedding_spaces=() if embedding_space is None else (embedding_space,),
        hash_spaces=() if hash_space is None else (hash_space,),
        retriever_id="capicove.personal-numpy",
        index_format=index_format,
        index_parameters_ref=IndexParametersRef(
            parameter_schema=index_format,
            payload_ref=f"parameters:{generation_id}",
            sha256=SHA_B,
        ),
        shard_refs=(f"payload:{generation_id}",),
        calibration_profile_ref="exact-v1",
        build_processing_fingerprint=ProcessingFingerprint(SHA_A),
        created_at=datetime(2026, 8, 27, tzinfo=UTC),
        validation_report_ref=f"validation:{generation_id}",
    )


@dataclass(frozen=True, slots=True)
class _Resolver:
    unit_ids: tuple[str, ...]
    unit_manifest_sha256: str = SHA_A
    activation_set_id: str = "activation-1"

    @property
    def unit_count(self) -> int:
        return len(self.unit_ids)

    def unit_ids_for_ordinals(
        self,
        ordinals: tuple[int, ...],
        *,
        deadline: datetime,
    ) -> tuple[str, ...]:
        if datetime.now(UTC) >= deadline:
            raise TimeoutError("deadline exceeded")
        return tuple(self.unit_ids[ordinal] for ordinal in ordinals)


def test_semantic_index_is_exact_stable_read_only_and_generation_bound(tmp_path: Path) -> None:
    vectors = np.asarray(
        ((1.0, 0.0), (0.8, 0.6), (0.8, 0.6), (0.0, 1.0)),
        dtype=np.float32,
        order="C",
    )
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    payload = NumpyExactIndexWriter(max_units=10_000).build_array(
        vectors,
        generation,
        tmp_path / "instance.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )

    with NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=_Resolver(("a", "b", "c", "d")),
        deadline=FUTURE,
    ) as reader:
        reader_port: IndexReader = reader
        assert isinstance(reader_port, IndexReader)
        assert reader.is_read_only
        assert reader.dtype == np.dtype("<f4")
        assert reader.shape == (4, 2)
        hits = reader.search(
            QueryVector(space, (1.0, 0.0)),
            None,
            limit=4,
            deadline=FUTURE,
        )

    assert [hit.unit_id for hit in hits] == ["a", "b", "c", "d"]
    np.testing.assert_allclose([hit.raw_score for hit in hits], vectors[:, 0])
    assert [hit.unit_ordinal for hit in hits] == [0, 1, 2, 3]
    assert all(hit.lane is RetrievalLane.INSTANCE for hit in hits)
    assert all(
        isinstance(hit.provenance, IndexedCandidateProvenance)
        and hit.provenance.generation_id == generation.generation_id
        for hit in hits
    )
    assert all(
        isinstance(hit.provenance, IndexedCandidateProvenance)
        and hit.provenance.activation_set_id == generation.activation_set_id
        for hit in hits
    )
    assert all(
        isinstance(hit.provenance, IndexedCandidateProvenance)
        and hit.provenance.provider_id == space.provider_id
        for hit in hits
    )


def test_semantic_writer_supports_contiguous_ordinal_rows_and_concept_lane(
    tmp_path: Path,
) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space, generation_id="concept-1")
    rows = (
        OrdinalVector(0, space, (1.0, 0.0)),
        OrdinalVector(1, space, (0.0, 1.0)),
    )
    payload = NumpyExactIndexWriter().build(
        generation,
        rows,
        tmp_path / "concept.npy",
        expected_units=2,
        lane=RetrievalLane.CONCEPT,
        deadline=FUTURE,
    )

    with NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=_Resolver(("first", "second")),
        deadline=FUTURE,
    ) as reader:
        hits = reader.search(
            QueryVector(space, (0.0, 1.0)),
            (0, 1),
            limit=2,
            deadline=FUTURE,
        )

    assert [hit.unit_id for hit in hits] == ["second", "first"]
    assert all(hit.lane is RetrievalLane.CONCEPT for hit in hits)
    assert all(hit.match_kind.value == "semantic" for hit in hits)


def test_phash_index_uses_exact_radius_two_and_stable_ordinal_ties(tmp_path: Path) -> None:
    query_value = 0x0F0F0F0F0F0F0F0F
    values = (
        query_value ^ 0b11,
        query_value,
        query_value ^ 0b01,
        query_value ^ 0b1011,
        query_value ^ 0b101,
    )
    space = _hash_space()
    generation = _generation(hash_space=space, generation_id="phash-1")
    payload = PhashFlatIndexWriter().build_values(
        values,
        generation,
        tmp_path / "phash.npy",
        deadline=FUTURE,
    )

    with PhashFlatIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=_Resolver(("a", "b", "c", "d", "e")),
        deadline=FUTURE,
    ) as reader:
        assert reader.is_read_only
        assert reader.dtype == np.dtype("<u8")
        hits = reader.search(
            PerceptualHashQuery(SHA_A, space, query_value, 2),
            None,
            limit=10,
            deadline=FUTURE,
        )

    assert [(hit.unit_id, hit.raw_score) for hit in hits] == [
        ("b", 0.0),
        ("c", 1.0),
        ("a", 2.0),
        ("e", 2.0),
    ]
    assert all(hit.lane is RetrievalLane.PERCEPTUAL for hit in hits)
    assert all(
        isinstance(hit.provenance, IndexedCandidateProvenance)
        and hit.provenance.generation_id == "phash-1"
        for hit in hits
    )


def test_writer_rejects_capacity_and_invalid_arrays_before_publication(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    destination = tmp_path / "too-many.npy"

    with pytest.raises(ValueError, match="10,000"):
        NumpyExactIndexWriter().build_array(
            np.zeros((10_001, 2), dtype=np.float32),
            generation,
            destination,
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    assert not destination.exists()

    with pytest.raises(ValueError, match="float32"):
        NumpyExactIndexWriter().build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float64),
            generation,
            tmp_path / "float64.npy",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="L2-normalized"):
        NumpyExactIndexWriter().build_array(
            np.asarray(((0.5, 0.0),), dtype=np.float32),
            generation,
            tmp_path / "not-normalized.npy",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="finite"):
        NumpyExactIndexWriter().build_array(
            np.asarray(((np.nan, 0.0),), dtype=np.float32),
            generation,
            tmp_path / "nan.npy",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )

    phash_generation = _generation(hash_space=_hash_space())
    phash_destination = tmp_path / "too-many-phash.npy"
    with pytest.raises(ValueError, match="10,000"):
        PhashFlatIndexWriter().build_values(
            tuple(range(10_001)),
            phash_generation,
            phash_destination,
            deadline=FUTURE,
        )
    assert not phash_destination.exists()


def test_writers_accept_exact_personal_capacity(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    vectors = np.zeros((10_000, 2), dtype=np.float32)
    vectors[:, 0] = 1.0

    semantic_payload = NumpyExactIndexWriter().build_array(
        vectors,
        generation,
        tmp_path / "exact-capacity.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    phash_payload = PhashFlatIndexWriter().build_values(
        np.arange(10_000, dtype=np.uint64),
        _generation(hash_space=_hash_space(), generation_id="phash-capacity"),
        tmp_path / "exact-phash-capacity.npy",
        deadline=FUTURE,
    )

    assert semantic_payload.unit_count == 10_000
    assert phash_payload.unit_count == 10_000


def test_readers_fail_closed_for_corruption_space_count_and_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    writer = NumpyExactIndexWriter()
    payload = writer.build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "vectors.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )

    with pytest.raises(SearchFailure) as count_error:
        NumpyExactIndexReader.open_readonly(
            generation,
            payload,
            ordinal_resolver=_Resolver(("only-one",)),
            deadline=FUTURE,
        )
    assert count_error.value.code is SearchFailureCode.INDEX_CORRUPT

    with NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=_Resolver(("a", "b")),
        deadline=FUTURE,
    ) as reader:
        incompatible = replace(space, space_id="same-dimension-other-space")
        with pytest.raises(SearchFailure) as space_error:
            reader.search(
                QueryVector(incompatible, (1.0, 0.0)),
                None,
                limit=2,
                deadline=FUTURE,
            )
        assert space_error.value.code is SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE
        with pytest.raises(TimeoutError, match="deadline"):
            reader.search(QueryVector(space, (1.0, 0.0)), None, limit=2, deadline=PAST)

    contents = bytearray(payload.path.read_bytes())
    contents[-1] ^= 0x01
    payload.path.write_bytes(contents)

    def unexpected_load(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("np.load must not run before checksum validation")

    with monkeypatch.context() as scoped:
        scoped.setattr(np, "load", unexpected_load)
        with pytest.raises(SearchFailure) as corrupt_error:
            NumpyExactIndexReader.open_readonly(
                generation,
                payload,
                ordinal_resolver=_Resolver(("a", "b")),
                deadline=FUTURE,
            )
    assert corrupt_error.value.code is SearchFailureCode.INDEX_CORRUPT


def test_reader_rejects_native_dtype_and_shape_mismatch(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    valid_payload = NumpyExactIndexWriter().build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "valid.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )

    for name, array in (
        ("wrong-dtype", np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float64)),
        ("wrong-shape", np.asarray(((1.0, 0.0),), dtype=np.float32)),
    ):
        path = tmp_path / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        payload = replace(
            valid_payload,
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        with pytest.raises(SearchFailure) as error:
            NumpyExactIndexReader.open_readonly(
                generation,
                payload,
                ordinal_resolver=_Resolver(("a", "b")),
                deadline=FUTURE,
            )
        assert error.value.code is SearchFailureCode.INDEX_CORRUPT


def test_semantic_reader_rejects_payload_replacement_after_same_object_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    writer = NumpyExactIndexWriter()
    payload = writer.build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "semantic-original.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    replacement = writer.build_array(
        np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.float32),
        generation,
        tmp_path / "semantic-replacement.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    original_digest = personal_numpy._digest_file
    replaced = False

    def digest_then_replace(source: Any, *, deadline: datetime | None = None) -> str:
        nonlocal replaced
        digest = original_digest(source, deadline=deadline)
        if not replaced:
            replacement.path.replace(payload.path)
            replaced = True
        return digest

    monkeypatch.setattr(personal_numpy, "_digest_file", digest_then_replace)

    with pytest.raises(SearchFailure) as raised:
        NumpyExactIndexReader.open_readonly(
            generation,
            payload,
            ordinal_resolver=_Resolver(("a", "b")),
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.INDEX_CORRUPT


def test_semantic_reader_fails_closed_if_published_path_changes_after_open(
    tmp_path: Path,
) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    writer = NumpyExactIndexWriter()
    payload = writer.build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "semantic-open.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    replacement = writer.build_array(
        np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.float32),
        generation,
        tmp_path / "semantic-after-open.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    reader = NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=_Resolver(("a", "b")),
        deadline=FUTURE,
    )
    replacement.path.replace(payload.path)
    try:
        with pytest.raises(SearchFailure) as raised:
            reader.search(
                QueryVector(space, (1.0, 0.0)),
                None,
                limit=2,
                deadline=FUTURE,
            )
    finally:
        reader.close()

    assert raised.value.code is SearchFailureCode.INDEX_CORRUPT


def test_phash_reader_rejects_in_place_rewrite_after_same_object_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _hash_space()
    generation = _generation(hash_space=space, generation_id="phash-toctou")
    writer = PhashFlatIndexWriter()
    payload = writer.build_values(
        (1, 2), generation, tmp_path / "phash-original.npy", deadline=FUTURE
    )
    replacement = writer.build_values(
        (2, 1), generation, tmp_path / "phash-replacement.npy", deadline=FUTURE
    )
    original_digest = personal_numpy._digest_file
    replaced = False

    def digest_then_replace(source: Any, *, deadline: datetime | None = None) -> str:
        nonlocal replaced
        digest = original_digest(source, deadline=deadline)
        if not replaced:
            payload.path.write_bytes(replacement.path.read_bytes())
            replaced = True
        return digest

    monkeypatch.setattr(personal_numpy, "_digest_file", digest_then_replace)

    with pytest.raises(SearchFailure) as raised:
        PhashFlatIndexReader.open_readonly(
            generation,
            payload,
            ordinal_resolver=_Resolver(("a", "b")),
            deadline=FUTURE,
        )

    assert raised.value.code is SearchFailureCode.INDEX_CORRUPT


def test_expired_deadline_creates_no_payload(tmp_path: Path) -> None:
    destination = tmp_path / "expired.npy"
    with pytest.raises(TimeoutError, match="deadline"):
        NumpyExactIndexWriter().build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            _generation(embedding_space=_embedding_space()),
            destination,
            lane=RetrievalLane.INSTANCE,
            deadline=PAST,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    ("ordinal", "values", "message"),
    (
        (-1, (1.0, 0.0), "non-negative"),
        (0, [1.0, 0.0], "tuple"),
        (0, (True, 0.0), "finite"),
        (0, (float("inf"), 0.0), "finite"),
    ),
)
def test_ordinal_vector_rejects_invalid_identity_and_values(
    ordinal: Any,
    values: Any,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        OrdinalVector(ordinal, _embedding_space(), values)

    with pytest.raises(TypeError, match="EmbeddingSpaceRef"):
        OrdinalVector(0, cast(Any, _hash_space()), (1.0, 0.0))


@pytest.mark.parametrize("value", (-1, 2**64, True, 1.5))
def test_ordinal_phash_rejects_out_of_range_values(value: Any) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        OrdinalPhash(0, _hash_space(), value)

    with pytest.raises(TypeError, match="HashSpaceRef"):
        OrdinalPhash(0, cast(Any, _embedding_space()), 1)


def test_payload_descriptor_rejects_inconsistent_metadata(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    payload = NumpyExactIndexWriter().build_array(
        np.asarray(((1.0, 0.0),), dtype=np.float32),
        generation,
        tmp_path / "valid.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )

    invalid_changes = (
        ({"path": cast(Any, "not-a-path")}, "Path"),
        ({"sha256": "INVALID"}, "SHA-256"),
        ({"size_bytes": 0}, "positive"),
        ({"unit_count": 0, "shape": (0, 2)}, "positive"),
        ({"unit_count": 10_001, "shape": (10_001, 2)}, "10,000"),
        ({"shape": (2, 2)}, "unit_count"),
        ({"c_order": False}, "C order"),
        ({"generation_id": " "}, "non-blank"),
        ({"index_parameters_ref": cast(Any, object())}, "IndexParametersRef"),
        ({"lane": cast(Any, "instance")}, "RetrievalLane"),
        ({"embedding_space": None}, "embedding space"),
        ({"hash_space": _hash_space()}, "must not carry"),
        (
            {
                "lane": RetrievalLane.CONTENT,
                "embedding_space": None,
            },
            "content lane",
        ),
    )
    for changes, message in invalid_changes:
        with pytest.raises((TypeError, ValueError), match=message):
            replace(payload, **changes)

    phash_payload = PhashFlatIndexWriter().build_values(
        (1,),
        _generation(hash_space=_hash_space(), generation_id="phash-descriptor"),
        tmp_path / "phash.npy",
        deadline=FUTURE,
    )
    with pytest.raises(ValueError, match="hash space"):
        replace(phash_payload, hash_space=None)
    with pytest.raises(ValueError, match="must not carry"):
        replace(phash_payload, embedding_space=space)


def test_semantic_writer_rejects_shape_layout_and_destination_errors(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    writer = NumpyExactIndexWriter()

    cases = (
        (cast(Any, ((1.0, 0.0),)), tmp_path / "not-array.npy", "NumPy array"),
        (np.asarray((1.0, 0.0), dtype=np.float32), tmp_path / "one-d.npy", "shape"),
        (np.asarray(((1.0,),), dtype=np.float32), tmp_path / "dimension.npy", "dimension"),
        (
            np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32)[:, ::-1],
            tmp_path / "strided.npy",
            "C-contiguous",
        ),
    )
    for vectors, destination, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            writer.build_array(
                vectors,
                generation,
                destination,
                lane=RetrievalLane.INSTANCE,
                deadline=FUTURE,
            )

    with pytest.raises(TypeError, match="Path"):
        writer.build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            generation,
            cast(Any, "index.npy"),
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match=r"\.npy"):
        writer.build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            generation,
            tmp_path / "index.bin",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="parent"):
        writer.build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            generation,
            tmp_path / "missing" / "index.npy",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    existing = tmp_path / "existing.npy"
    existing.write_bytes(b"reserved")
    with pytest.raises(FileExistsError, match="already exists"):
        writer.build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            generation,
            existing,
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )


def test_stream_writers_fail_closed_for_bad_rows(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    semantic_cases = (
        ((OrdinalVector(0, space, (1.0, 0.0)), OrdinalVector(1, space, (0.0, 1.0))), 1, "exceed"),
        ((cast(Any, object()),), 1, "OrdinalVector"),
        ((OrdinalVector(1, space, (1.0, 0.0)),), 1, "contiguous"),
        ((OrdinalVector(0, replace(space, space_id="other"), (1.0, 0.0)),), 1, "space"),
        ((OrdinalVector(0, space, (1.0,)),), 1, "dimension"),
        ((), 1, "row count"),
    )
    for index, (rows, expected_units, message) in enumerate(semantic_cases):
        destination = tmp_path / f"bad-semantic-{index}.npy"
        with pytest.raises((TypeError, ValueError), match=message):
            NumpyExactIndexWriter().build(
                generation,
                rows,
                destination,
                expected_units=expected_units,
                lane=RetrievalLane.INSTANCE,
                deadline=FUTURE,
            )
        assert not destination.exists()

    hash_space = _hash_space()
    hash_generation = _generation(hash_space=hash_space, generation_id="phash-rows")
    phash_cases = (
        ((OrdinalPhash(0, hash_space, 1), OrdinalPhash(1, hash_space, 2)), 1, "exceed"),
        ((cast(Any, object()),), 1, "OrdinalPhash"),
        ((OrdinalPhash(1, hash_space, 1),), 1, "contiguous"),
        ((OrdinalPhash(0, replace(hash_space, space_id="other"), 1),), 1, "space"),
        ((), 1, "row count"),
    )
    for index, (rows, expected_units, message) in enumerate(phash_cases):
        with pytest.raises((TypeError, ValueError), match=message):
            PhashFlatIndexWriter().build(
                hash_generation,
                rows,
                tmp_path / f"bad-phash-{index}.npy",
                expected_units=expected_units,
                deadline=FUTURE,
            )


def test_phash_writer_accepts_bound_rows_and_rejects_invalid_native_arrays(tmp_path: Path) -> None:
    space = _hash_space()
    generation = _generation(hash_space=space, generation_id="phash-bound")
    payload = PhashFlatIndexWriter().build(
        generation,
        (OrdinalPhash(0, space, 1), OrdinalPhash(1, space, 2)),
        tmp_path,
        expected_units=2,
        deadline=FUTURE,
    )
    assert payload.path.name == "phash-bound-perceptual.npy"

    for array in (
        np.asarray(((1,),), dtype=np.uint64),
        np.asarray((1,), dtype=np.int64),
    ):
        with pytest.raises(ValueError, match="one-dimensional little-endian uint64"):
            PhashFlatIndexWriter().build_values(
                array,
                _generation(hash_space=space, generation_id=f"bad-native-{array.ndim}"),
                tmp_path / f"bad-native-{array.dtype}-{array.ndim}.npy",
                deadline=FUTURE,
            )
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        PhashFlatIndexWriter().build_values(
            (-1,),
            _generation(hash_space=space, generation_id="negative-phash"),
            tmp_path / "negative.npy",
            deadline=FUTURE,
        )


def test_generation_compatibility_and_capacities_fail_before_writing(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    vectors = np.asarray(((1.0, 0.0),), dtype=np.float32)

    with pytest.raises(ValueError, match="max_units"):
        NumpyExactIndexWriter(0)
    with pytest.raises(ValueError, match="10,000"):
        PhashFlatIndexWriter(10_001)
    with pytest.raises(TypeError, match="IndexGeneration"):
        NumpyExactIndexWriter().build_array(
            vectors,
            cast(Any, object()),
            tmp_path / "bad-generation.npy",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="instance or concept"):
        NumpyExactIndexWriter().build_array(
            vectors,
            generation,
            tmp_path / "bad-lane.npy",
            lane=RetrievalLane.PERCEPTUAL,
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="compatible personal pHash"):
        PhashFlatIndexWriter().build_values(
            (1,),
            generation,
            tmp_path / "semantic-as-phash.npy",
            deadline=FUTURE,
        )
    incompatible = replace(
        generation,
        embedding_spaces=(replace(space, normalization=VectorNormalization.NONE),),
    )
    with pytest.raises(ValueError, match="L2 image embedding"):
        NumpyExactIndexWriter().build_array(
            vectors,
            incompatible,
            tmp_path / "not-l2.npy",
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )


def test_semantic_reader_rejects_binding_native_load_and_query_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    payload = NumpyExactIndexWriter().build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "semantic.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    resolver = _Resolver(("a", "b"))

    binding_cases = (
        (
            replace(payload, generation_id="other"),
            resolver,
            SearchFailureCode.INDEX_GENERATION_UNAVAILABLE,
        ),
        (replace(payload, unit_manifest_sha256=SHA_B), resolver, SearchFailureCode.INDEX_CORRUPT),
        (
            payload,
            replace(resolver, activation_set_id="other"),
            SearchFailureCode.INDEX_GENERATION_UNAVAILABLE,
        ),
        (replace(payload, dtype="<f8"), resolver, SearchFailureCode.INDEX_CORRUPT),
        (
            replace(payload, embedding_space=replace(space, space_id="other")),
            resolver,
            SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE,
        ),
    )
    for candidate_payload, candidate_resolver, expected_code in binding_cases:
        with pytest.raises(SearchFailure) as caught:
            NumpyExactIndexReader.open_readonly(
                generation,
                candidate_payload,
                ordinal_resolver=candidate_resolver,
                deadline=FUTURE,
            )
        assert caught.value.code is expected_code

    with pytest.raises(TypeError, match="pinned catalog identity"):
        NumpyExactIndexReader.open_readonly(
            generation,
            payload,
            ordinal_resolver=cast(Any, object()),
            deadline=FUTURE,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            personal_numpy,
            "_open_npy_memmap",
            lambda *_args, **_kwargs: np.zeros((2, 2), dtype=np.float32),
        )
        with pytest.raises(SearchFailure) as not_mmap:
            NumpyExactIndexReader.open_readonly(
                generation,
                payload,
                ordinal_resolver=resolver,
                deadline=FUTURE,
            )
    assert not_mmap.value.code is SearchFailureCode.INDEX_CORRUPT

    with monkeypatch.context() as scoped:
        scoped.setattr(
            personal_numpy,
            "_open_npy_memmap",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bad")),
        )
        with pytest.raises(SearchFailure) as load_error:
            NumpyExactIndexReader.open_readonly(
                generation,
                payload,
                ordinal_resolver=resolver,
                deadline=FUTURE,
            )
    assert load_error.value.code is SearchFailureCode.INDEX_CORRUPT

    with NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=resolver,
        deadline=FUTURE,
    ) as reader:
        with pytest.raises(SearchFailure) as wrong_query:
            reader.search(
                PerceptualHashQuery(SHA_A, _hash_space(), 1, 2),
                None,
                limit=1,
                deadline=FUTURE,
            )
        assert wrong_query.value.code is SearchFailureCode.INVALID_QUERY
        for selection, message in (
            ((True,), "integers"),
            ((2,), "outside"),
            ((0, 0), "unique"),
        ):
            with pytest.raises(ValueError, match=message):
                reader.search(QueryVector(space, (1.0, 0.0)), selection, limit=1, deadline=FUTURE)
        with pytest.raises(ValueError, match="positive"):
            reader.search(QueryVector(space, (1.0, 0.0)), None, limit=0, deadline=FUTURE)
        with pytest.raises(ValueError, match="at most"):
            reader.search(QueryVector(space, (1.0, 0.0)), None, limit=10_001, deadline=FUTURE)

    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.search(QueryVector(space, (1.0, 0.0)), None, limit=1, deadline=FUTURE)
    with pytest.raises(RuntimeError, match="closed"):
        reader.__enter__()


def test_selection_identity_and_resolver_output_fail_closed(tmp_path: Path) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space)
    payload = NumpyExactIndexWriter().build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "selection.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    query = QueryVector(space, (1.0, 0.0))
    with NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=_Resolver(("a", "b")),
        deadline=FUTURE,
    ) as reader:
        selections = (
            SimpleNamespace(unit_count=1),
            SimpleNamespace(unit_count=2, ordinals=[0]),
            SimpleNamespace(unit_count=2, bits=b""),
            SimpleNamespace(unit_count=2, bits=b"\x03", matching_count=1),
        )
        for selection in selections:
            with pytest.raises(SearchFailure) as caught:
                reader.search(query, selection, limit=2, deadline=FUTURE)
            assert caught.value.code is SearchFailureCode.INDEX_CORRUPT
        assert [
            hit.unit_id
            for hit in reader.search(
                query,
                SimpleNamespace(unit_count=2, bits=b"\x01", matching_count=1),
                limit=2,
                deadline=FUTURE,
            )
        ] == ["a"]

    @dataclass(frozen=True, slots=True)
    class DuplicateResolver:
        unit_count: int = 2
        unit_manifest_sha256: str = SHA_A
        activation_set_id: str = "activation-1"

        def unit_ids_for_ordinals(
            self,
            ordinals: tuple[int, ...],
            *,
            deadline: datetime,
        ) -> tuple[str, ...]:
            return ("duplicate",) * len(ordinals)

    with NumpyExactIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=DuplicateResolver(),
        deadline=FUTURE,
    ) as reader:
        with pytest.raises(SearchFailure) as duplicate:
            reader.search(query, None, limit=2, deadline=FUTURE)
    assert duplicate.value.code is SearchFailureCode.INDEX_CORRUPT


@pytest.mark.parametrize("identity_failure", ("missing", "size", "symlink"))
def test_reader_verifies_regular_file_identity_before_loading(
    tmp_path: Path,
    identity_failure: str,
) -> None:
    space = _embedding_space()
    generation = _generation(embedding_space=space, generation_id=identity_failure)
    payload = NumpyExactIndexWriter().build_array(
        np.asarray(((1.0, 0.0),), dtype=np.float32),
        generation,
        tmp_path / f"{identity_failure}.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )
    if identity_failure == "missing":
        payload.path.unlink()
    elif identity_failure == "size":
        payload = replace(payload, size_bytes=payload.size_bytes + 1)
    else:
        target = tmp_path / "target.npy"
        payload.path.rename(target)
        payload.path.symlink_to(target)

    with pytest.raises(SearchFailure) as caught:
        NumpyExactIndexReader.open_readonly(
            generation,
            payload,
            ordinal_resolver=_Resolver(("a",)),
            deadline=FUTURE,
        )
    assert caught.value.code is SearchFailureCode.INDEX_CORRUPT


def test_writer_late_deadline_failure_removes_transient_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_check = personal_numpy._check_deadline

    def expire_after_materialization(deadline: datetime) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("injected late deadline")
        original_check(deadline)

    monkeypatch.setattr(personal_numpy, "_check_deadline", expire_after_materialization)
    destination = tmp_path / "late.npy"
    with pytest.raises(TimeoutError, match="late deadline"):
        NumpyExactIndexWriter().build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            _generation(embedding_space=_embedding_space()),
            destination,
            lane=RetrievalLane.INSTANCE,
            deadline=FUTURE,
        )
    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_phash_reader_rejects_binding_load_query_and_closed_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    space = _hash_space()
    generation = _generation(hash_space=space, generation_id="phash-failures")
    payload = PhashFlatIndexWriter().build_values(
        (1, 2),
        generation,
        tmp_path / "phash-failures.npy",
        deadline=FUTURE,
    )
    resolver = _Resolver(("a", "b"))

    for candidate_payload, expected_code in (
        (
            replace(payload, hash_space=replace(space, space_id="other")),
            SearchFailureCode.HASH_SPACE_INCOMPATIBLE,
        ),
        (replace(payload, dtype="<i8"), SearchFailureCode.INDEX_CORRUPT),
    ):
        with pytest.raises(SearchFailure) as caught:
            PhashFlatIndexReader.open_readonly(
                generation,
                candidate_payload,
                ordinal_resolver=resolver,
                deadline=FUTURE,
            )
        assert caught.value.code is expected_code

    with monkeypatch.context() as scoped:
        scoped.setattr(
            personal_numpy,
            "_open_npy_memmap",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
        )
        with pytest.raises(SearchFailure) as load_error:
            PhashFlatIndexReader.open_readonly(
                generation,
                payload,
                ordinal_resolver=resolver,
                deadline=FUTURE,
            )
    assert load_error.value.code is SearchFailureCode.INDEX_CORRUPT

    with monkeypatch.context() as scoped:
        scoped.setattr(
            personal_numpy,
            "_open_npy_memmap",
            lambda *_args, **_kwargs: np.asarray((1, 2), dtype=np.uint64),
        )
        with pytest.raises(SearchFailure) as not_mmap:
            PhashFlatIndexReader.open_readonly(
                generation,
                payload,
                ordinal_resolver=resolver,
                deadline=FUTURE,
            )
    assert not_mmap.value.code is SearchFailureCode.INDEX_CORRUPT

    with PhashFlatIndexReader.open_readonly(
        generation,
        payload,
        ordinal_resolver=resolver,
        deadline=FUTURE,
    ) as reader:
        with pytest.raises(SearchFailure) as wrong_query:
            reader.search(
                QueryVector(_embedding_space(), (1.0, 0.0)), None, limit=1, deadline=FUTURE
            )
        assert wrong_query.value.code is SearchFailureCode.INVALID_QUERY
        with pytest.raises(SearchFailure) as wrong_space:
            reader.search(
                PerceptualHashQuery(SHA_A, replace(space, space_id="other"), 1, 2),
                None,
                limit=1,
                deadline=FUTURE,
            )
        assert wrong_space.value.code is SearchFailureCode.HASH_SPACE_INCOMPATIBLE

    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.search(
            PerceptualHashQuery(SHA_A, space, 1, 2),
            None,
            limit=1,
            deadline=FUTURE,
        )


def test_writer_rejects_naive_deadline_before_touching_destination(tmp_path: Path) -> None:
    destination = tmp_path / "naive-deadline.npy"
    with pytest.raises(ValueError, match="timezone-aware"):
        NumpyExactIndexWriter().build_array(
            np.asarray(((1.0, 0.0),), dtype=np.float32),
            _generation(embedding_space=_embedding_space()),
            destination,
            lane=RetrievalLane.INSTANCE,
            deadline=datetime.now(),
        )
    assert not destination.exists()


def test_private_staging_cleanup_is_idempotent(tmp_path: Path) -> None:
    from capicove.retrieval._adapters.personal_numpy import (
        _remove_temporary_sibling,
        _temporary_sibling,
    )

    temporary = _temporary_sibling(tmp_path / "payload.npy")
    _remove_temporary_sibling(temporary)
    _remove_temporary_sibling(temporary)
    assert not temporary.parent.exists()


def test_published_payload_retains_owner_only_permissions(tmp_path: Path) -> None:
    from capicove.retrieval._adapters.personal_numpy import _publish_new, _temporary_sibling

    destination = tmp_path / "published.npy"
    temporary = _temporary_sibling(destination)
    temporary.write_bytes(b"synthetic test payload")
    temporary.chmod(0o644)
    _publish_new(temporary, destination)
    assert destination.stat().st_mode & 0o777 == 0o600
    assert destination.read_bytes() == b"synthetic test payload"
    assert not temporary.parent.exists()
