"""Tests for personal generation reader parameter composition."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    HashSpaceRef,
    IndexParametersRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval._adapters.personal_numpy import (
    NumpyExactIndexReader,
    PhashFlatIndexReader,
)
from capicove.retrieval._adapters.sqlite_image_generations import PublishedGenerationRecord
from capicove.retrieval._adapters.sqlite_unit_catalog import SqliteUnitCatalog
from capicove.retrieval.image_generations import PayloadFileRef
from capicove.retrieval.search_models import RetrievalLane
from capicove.runtime import retrieval_personal
from capicove.runtime.retrieval_personal import (
    MAX_PERSONAL_INDEX_PARAMETER_BYTES,
    PersonalGenerationHandleFactory,
    PersonalIndexParameters,
    decode_personal_index_parameters,
    encode_personal_index_parameters,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)
SHA_A = "a" * 64
SHA_B = "b" * 64


def _semantic_space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="image-space-v1",
        provider_id="local-image-encoder",
        model_revision="revision-1",
        preprocessing_version="image-preprocess-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _hash_space() -> HashSpaceRef:
    return HashSpaceRef(
        space_id="opencv-phash-64-v1",
        provider_id="opencv-phash",
        provider_version="4.12.0",
        preprocessing_version="opencv-phash-preprocess-v1",
        algorithm="phash",
        bit_width=64,
        hash_version="phash-64-v1",
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _payload_ref(root: Path, role: str, data: bytes) -> PayloadFileRef:
    path = root / f"{role}.bin"
    path.write_bytes(data)
    return PayloadFileRef(
        relative_path=PurePosixPath(path.name),
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        role=role,
    )


@pytest.mark.parametrize(
    "parameters",
    (
        PersonalIndexParameters(
            lane=RetrievalLane.PERCEPTUAL,
            index_format="phash-flat-u64-v1",
            unit_count=3,
            shape=(3,),
            dtype="<u8",
            c_order=True,
            shard_role="perceptual-shard",
        ),
        PersonalIndexParameters(
            lane=RetrievalLane.INSTANCE,
            index_format="numpy-flat-f32-v1",
            unit_count=3,
            shape=(3, 2),
            dtype="<f4",
            c_order=True,
            shard_role="instance-shard",
        ),
    ),
)
def test_personal_index_parameters_round_trip_canonically(
    parameters: PersonalIndexParameters,
) -> None:
    encoded = encode_personal_index_parameters(parameters)

    assert decode_personal_index_parameters(encoded) == parameters
    assert encode_personal_index_parameters(decode_personal_index_parameters(encoded)) == encoded
    assert encoded.endswith(b"\n")


def test_parameter_decoder_rejects_unknown_noncanonical_and_lane_mismatch() -> None:
    parameters = PersonalIndexParameters(
        lane=RetrievalLane.CONCEPT,
        index_format="numpy-flat-f32-v1",
        unit_count=2,
        shape=(2, 2),
        dtype="<f4",
        c_order=True,
        shard_role="concept-shard",
    )
    document = json.loads(encode_personal_index_parameters(parameters))
    document["unknown"] = True
    with pytest.raises(ValueError, match="fields"):
        decode_personal_index_parameters(json.dumps(document).encode())

    noncanonical = json.dumps(
        json.loads(encode_personal_index_parameters(parameters)),
        indent=2,
    ).encode()
    with pytest.raises(ValueError, match="canonical"):
        decode_personal_index_parameters(noncanonical)

    with pytest.raises(ValueError, match="dtype"):
        replace(parameters, dtype="<u8")
    with pytest.raises(ValueError, match="shape"):
        replace(parameters, shape=(2,))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"lane": RetrievalLane.CONTENT}, "indexed lane"),
        ({"index_format": "phash-flat-u64-v1"}, "index_format"),
        ({"unit_count": 0, "shape": (0, 2)}, "1 to 10,000"),
        ({"unit_count": True, "shape": (True, 2)}, "1 to 10,000"),
        ({"unit_count": 10_001, "shape": (10_001, 2)}, "1 to 10,000"),
        ({"shape": ()}, "begin with unit_count"),
        ({"shape": (3, 0)}, "positive integers"),
        ({"shape": (3, True)}, "positive integers"),
        ({"shape": (3,)}, "two-dimensional"),
        ({"dtype": "<u8"}, "dtype"),
        ({"c_order": False}, "C order"),
        ({"shard_role": " "}, "non-blank"),
        ({"schema_version": 2}, "unsupported"),
    ),
)
def test_personal_index_parameters_reject_invalid_closed_layout(
    changes: dict[str, Any],
    message: str,
) -> None:
    valid = PersonalIndexParameters(
        lane=RetrievalLane.INSTANCE,
        index_format="numpy-flat-f32-v1",
        unit_count=3,
        shape=(3, 2),
        dtype="<f4",
        c_order=True,
        shard_role="instance-shard",
    )
    with pytest.raises((TypeError, ValueError), match=message):
        replace(valid, **changes)


def test_phash_parameters_reject_multidimensional_shape() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        PersonalIndexParameters(
            lane=RetrievalLane.PERCEPTUAL,
            index_format="phash-flat-u64-v1",
            unit_count=3,
            shape=(3, 1),
            dtype="<u8",
            c_order=True,
            shard_role="phash-shard",
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (cast(Any, "not-bytes"), "invalid size"),
        (b"", "invalid size"),
        (b"x" * (MAX_PERSONAL_INDEX_PARAMETER_BYTES + 1), "invalid size"),
        (b"\xff", "invalid JSON"),
        (b"{", "invalid JSON"),
        (b"[]", "fields"),
    ),
)
def test_parameter_decoder_rejects_invalid_envelope(payload: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_personal_index_parameters(payload)


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    (
        ("shape", "not-an-array", ValueError, "array"),
        ("lane", 1, TypeError, "string"),
        ("index_format", 1, TypeError, "string"),
        ("unit_count", True, TypeError, "integer"),
        ("shape", [3, True], TypeError, "integer"),
        ("dtype", 1, TypeError, "string"),
        ("c_order", 1, TypeError, "boolean"),
        ("shard_role", 1, TypeError, "string"),
        ("schema_version", True, TypeError, "integer"),
    ),
)
def test_parameter_decoder_rejects_typed_schema_confusion(
    field: str,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    parameters = PersonalIndexParameters(
        lane=RetrievalLane.INSTANCE,
        index_format="numpy-flat-f32-v1",
        unit_count=3,
        shape=(3, 2),
        dtype="<f4",
        c_order=True,
        shard_role="instance-shard",
    )
    document = json.loads(encode_personal_index_parameters(parameters))
    document[field] = value
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    with pytest.raises(error_type, match=message):
        decode_personal_index_parameters(payload)


def test_parameter_encoder_requires_closed_parameter_type() -> None:
    with pytest.raises(TypeError, match="PersonalIndexParameters"):
        encode_personal_index_parameters(cast(Any, object()))


def test_verified_small_payload_requires_regular_stable_checksum_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    payload = _payload_ref(root, "parameters", b"trusted")

    assert (
        retrieval_personal._read_verified_small_payload(
            root,
            payload,
            maximum_bytes=32,
        )
        == b"trusted"
    )

    with pytest.raises(ValueError, match="size limit"):
        retrieval_personal._read_verified_small_payload(
            root,
            replace(payload, size_bytes=33),
            maximum_bytes=32,
        )
    with pytest.raises(ValueError, match="identity"):
        retrieval_personal._read_verified_small_payload(
            root,
            replace(payload, size_bytes=payload.size_bytes + 1),
            maximum_bytes=32,
        )
    with pytest.raises(ValueError, match="checksum"):
        retrieval_personal._read_verified_small_payload(
            root,
            replace(payload, sha256=SHA_A),
            maximum_bytes=32,
        )

    directory_payload = PayloadFileRef(
        relative_path=PurePosixPath("directory"),
        sha256=SHA_A,
        size_bytes=1,
        role="directory-role",
    )
    (root / "directory").mkdir()
    with pytest.raises(ValueError, match="identity"):
        retrieval_personal._read_verified_small_payload(
            root,
            directory_payload,
            maximum_bytes=32,
        )


def test_verified_small_payload_detects_short_read_and_identity_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    payload = _payload_ref(root, "parameters", b"trusted")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "read", lambda *_args: b"")
        with pytest.raises(ValueError, match="size changed"):
            retrieval_personal._read_verified_small_payload(root, payload, maximum_bytes=32)

    real_fstat = os.fstat
    calls = 0

    def changing_fstat(descriptor: int) -> Any:
        nonlocal calls
        calls += 1
        observed = real_fstat(descriptor)
        if calls == 2:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_size=observed.st_size,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino + 1,
            )
        return observed

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fstat", changing_fstat)
        with pytest.raises(ValueError, match="changed while reading"):
            retrieval_personal._read_verified_small_payload(root, payload, maximum_bytes=32)


def test_payload_resolution_rejects_missing_role_and_untrusted_root(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    payload = _payload_ref(root, "parameters", b"trusted")
    assert retrieval_personal._payload({payload.role: payload}, payload.role) is payload
    assert retrieval_personal._payload_path(root, payload) == root / payload.relative_path

    with pytest.raises(ValueError, match="missing payload role"):
        retrieval_personal._payload({}, "missing")
    with pytest.raises(ValueError, match="escapes"):
        retrieval_personal._payload_path(Path("relative-root"), payload)


class _Closable:
    def __init__(self, name: str, closed: list[str], *, unit_count: int = 2) -> None:
        self.name = name
        self.closed = closed
        self.unit_count = unit_count

    def close(self) -> None:
        self.closed.append(self.name)


def _published_record(root: Path) -> PublishedGenerationRecord:
    return PublishedGenerationRecord(
        activation_set_id="activation-1",
        manifest_sha256=SHA_A,
        published_root=root,
        workspace_id="workspace-1",
        corpus_ref="corpus-1",
        retrieval_profile_ref="personal-lite",
        published_at=datetime.now(UTC),
    )


def _fake_manifest(root: Path) -> SimpleNamespace:
    catalog_payload = _payload_ref(root, "catalog", b"catalog")
    lanes = tuple(
        SimpleNamespace(lane=lane)
        for lane in (
            RetrievalLane.PERCEPTUAL,
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        )
    )
    return SimpleNamespace(
        activation_set_id="activation-1",
        workspace_id="workspace-1",
        corpus_ref="corpus-1",
        retrieval_profile_ref="personal-lite",
        unit_manifest_sha256=SHA_B,
        payloads=(catalog_payload,),
        unit_catalog=SimpleNamespace(
            payload_ref=catalog_payload.role,
            catalog_format="sqlite-unit-catalog-v1",
            catalog_id="catalog-1",
            unit_count=2,
        ),
        lanes=lanes,
    )


def test_generation_handle_factory_opens_complete_set_and_closes_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemGenerationStore(tmp_path / "store")
    published_root = (tmp_path / "published").resolve()
    published_root.mkdir()
    record = _published_record(published_root)
    manifest = _fake_manifest(published_root)
    monkeypatch.setattr(
        store,
        "open_published",
        lambda *_args, **_kwargs: SimpleNamespace(root=published_root, manifest=manifest),
    )
    closed: list[str] = []
    catalog = _Closable("catalog", closed)
    monkeypatch.setattr(
        SqliteUnitCatalog,
        "open_readonly",
        lambda *_args, **_kwargs: catalog,
    )

    def open_lane(
        _root: Path,
        lane_generation: SimpleNamespace,
        _payloads: dict[str, PayloadFileRef],
        _catalog: _Closable,
        _deadline: datetime,
    ) -> _Closable:
        return _Closable(lane_generation.lane.value, closed)

    monkeypatch.setattr(PersonalGenerationHandleFactory, "_open_lane", staticmethod(open_lane))
    handles = PersonalGenerationHandleFactory(store).open(record, deadline=FUTURE)

    assert set(handles.readers) == {
        RetrievalLane.PERCEPTUAL,
        RetrievalLane.INSTANCE,
        RetrievalLane.CONCEPT,
    }
    handles.close()
    assert closed == ["perceptual", "instance", "concept", "catalog"]


def test_generation_handle_factory_closes_partial_open_on_lane_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemGenerationStore(tmp_path / "store")
    published_root = (tmp_path / "published").resolve()
    published_root.mkdir()
    record = _published_record(published_root)
    manifest = _fake_manifest(published_root)
    monkeypatch.setattr(
        store,
        "open_published",
        lambda *_args, **_kwargs: SimpleNamespace(root=published_root, manifest=manifest),
    )
    closed: list[str] = []
    catalog = _Closable("catalog", closed)
    monkeypatch.setattr(
        SqliteUnitCatalog,
        "open_readonly",
        lambda *_args, **_kwargs: catalog,
    )

    def fail_second_lane(
        _root: Path,
        lane_generation: SimpleNamespace,
        _payloads: dict[str, PayloadFileRef],
        _catalog: _Closable,
        _deadline: datetime,
    ) -> _Closable:
        if lane_generation.lane is RetrievalLane.INSTANCE:
            raise RuntimeError("lane open failed")
        return _Closable(lane_generation.lane.value, closed)

    monkeypatch.setattr(
        PersonalGenerationHandleFactory,
        "_open_lane",
        staticmethod(fail_second_lane),
    )
    with pytest.raises(RuntimeError, match="lane open failed"):
        PersonalGenerationHandleFactory(store).open(record, deadline=FUTURE)
    assert closed == ["perceptual", "catalog"]


def test_generation_handle_factory_rejects_registration_identity_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="FilesystemGenerationStore"):
        PersonalGenerationHandleFactory(cast(Any, object()))

    store = FilesystemGenerationStore(tmp_path / "store")
    published_root = (tmp_path / "published").resolve()
    published_root.mkdir()
    other_root = (tmp_path / "other").resolve()
    other_root.mkdir()
    record = _published_record(published_root)
    factory = PersonalGenerationHandleFactory(store)

    with pytest.raises(TypeError, match="PublishedGenerationRecord"):
        factory.open(cast(Any, object()), deadline=FUTURE)

    manifest = _fake_manifest(published_root)
    monkeypatch.setattr(
        store,
        "open_published",
        lambda *_args, **_kwargs: SimpleNamespace(root=other_root, manifest=manifest),
    )
    with pytest.raises(ValueError, match="publication root"):
        factory.open(record, deadline=FUTURE)

    manifest.workspace_id = "other-workspace"
    monkeypatch.setattr(
        store,
        "open_published",
        lambda *_args, **_kwargs: SimpleNamespace(root=published_root, manifest=manifest),
    )
    with pytest.raises(ValueError, match="publication identity"):
        factory.open(record, deadline=FUTURE)

    manifest.workspace_id = record.workspace_id
    manifest.unit_catalog.catalog_format = "unsupported"
    with pytest.raises(ValueError, match="SQLite unit catalog"):
        factory.open(record, deadline=FUTURE)

    manifest.unit_catalog.catalog_format = "sqlite-unit-catalog-v1"
    manifest.unit_catalog.payload_ref = "missing-catalog"
    with pytest.raises(ValueError, match="missing payload role"):
        factory.open(record, deadline=FUTURE)


def _lane_generation(
    root: Path,
    lane: RetrievalLane,
    *,
    unit_count: int = 2,
    shard_refs: tuple[str, ...] = ("lane-shard",),
    parameter_shard_role: str = "lane-shard",
) -> tuple[SimpleNamespace, dict[str, PayloadFileRef]]:
    is_phash = lane is RetrievalLane.PERCEPTUAL
    index_format = "phash-flat-u64-v1" if is_phash else "numpy-flat-f32-v1"
    parameters = PersonalIndexParameters(
        lane=lane,
        index_format=index_format,
        unit_count=unit_count,
        shape=(unit_count,) if is_phash else (unit_count, 2),
        dtype="<u8" if is_phash else "<f4",
        c_order=True,
        shard_role=parameter_shard_role,
    )
    parameter_payload = _payload_ref(
        root, "lane-parameters", encode_personal_index_parameters(parameters)
    )
    shard_payload = _payload_ref(root, "lane-shard", b"native-index")
    parameter_ref = IndexParametersRef(
        parameter_schema=index_format,
        payload_ref=parameter_payload.role,
        sha256=parameter_payload.sha256,
    )
    generation = SimpleNamespace(
        generation_id="generation-1",
        activation_set_id="activation-1",
        unit_manifest_sha256=SHA_A,
        index_format=index_format,
        index_parameters_ref=parameter_ref,
        shard_refs=shard_refs,
        embedding_spaces=() if is_phash else (_semantic_space(),),
        hash_spaces=(_hash_space(),) if is_phash else (),
    )
    return SimpleNamespace(lane=lane, generation=generation), {
        parameter_payload.role: parameter_payload,
        shard_payload.role: shard_payload,
    }


@pytest.mark.parametrize("lane", (RetrievalLane.PERCEPTUAL, RetrievalLane.INSTANCE))
def test_open_lane_reconstructs_exact_native_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: RetrievalLane,
) -> None:
    root = tmp_path.resolve()
    lane_generation, payloads = _lane_generation(root, lane)
    catalog = SimpleNamespace(unit_count=2)
    observed: dict[str, object] = {}

    def open_reader(
        generation: object,
        descriptor: object,
        *,
        ordinal_resolver: object,
        deadline: datetime,
    ) -> str:
        observed.update(
            generation=generation,
            descriptor=descriptor,
            ordinal_resolver=ordinal_resolver,
            deadline=deadline,
        )
        return "reader"

    reader_type = (
        PhashFlatIndexReader if lane is RetrievalLane.PERCEPTUAL else NumpyExactIndexReader
    )
    monkeypatch.setattr(reader_type, "open_readonly", open_reader)

    reader = cast(
        Any,
        PersonalGenerationHandleFactory._open_lane(
            root,
            cast(Any, lane_generation),
            payloads,
            cast(Any, catalog),
            FUTURE,
        ),
    )

    assert reader == "reader"
    descriptor = cast(Any, observed["descriptor"])
    assert descriptor.lane is lane
    assert descriptor.unit_count == 2
    assert descriptor.path == root / "lane-shard.bin"
    assert observed["ordinal_resolver"] is catalog


@pytest.mark.parametrize(
    ("unit_count", "shard_refs", "parameter_role"),
    (
        (3, ("lane-shard",), "lane-shard"),
        (2, ("other-shard",), "lane-shard"),
        (2, ("lane-shard", "second-shard"), "lane-shard"),
    ),
)
def test_open_lane_rejects_parameter_generation_mismatch(
    tmp_path: Path,
    unit_count: int,
    shard_refs: tuple[str, ...],
    parameter_role: str,
) -> None:
    root = tmp_path.resolve()
    lane_generation, payloads = _lane_generation(
        root,
        RetrievalLane.INSTANCE,
        unit_count=unit_count,
        shard_refs=shard_refs,
        parameter_shard_role=parameter_role,
    )
    catalog = cast(Any, SimpleNamespace(unit_count=2))

    with pytest.raises(ValueError, match="do not match"):
        PersonalGenerationHandleFactory._open_lane(
            root,
            cast(Any, lane_generation),
            payloads,
            catalog,
            FUTURE,
        )
