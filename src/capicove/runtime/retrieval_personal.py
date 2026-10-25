"""Concrete immutable personal-lite generation reader composition."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from capicove.retrieval._adapters.filesystem_generations import FilesystemGenerationStore
from capicove.retrieval._adapters.personal_numpy import (
    NumpyExactIndexReader,
    PersonalIndexPayload,
    PhashFlatIndexReader,
)
from capicove.retrieval._adapters.sqlite_image_generations import PublishedGenerationRecord
from capicove.retrieval._adapters.sqlite_unit_catalog import (
    SQLITE_UNIT_CATALOG_FORMAT,
    SqliteUnitCatalog,
    SqliteUnitCatalogDescriptor,
)
from capicove.retrieval.image_generations import LaneGeneration, PayloadFileRef
from capicove.retrieval.search_models import RetrievalLane
from capicove.runtime.retrieval import OpenedGenerationHandles


PERSONAL_INDEX_PARAMETER_SCHEMA_VERSION = 1
MAX_PERSONAL_INDEX_PARAMETER_BYTES = 65_536

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_FORMAT = "numpy-flat-f32-v1"
_PHASH_FORMAT = "phash-flat-u64-v1"
_EXPECTED_PARAMETER_FIELDS = frozenset(
    {
        "schema_version",
        "lane",
        "index_format",
        "unit_count",
        "shape",
        "dtype",
        "c_order",
        "shard_role",
    }
)


@dataclass(frozen=True, slots=True)
class PersonalIndexParameters:
    """Describe the closed native array layout required to reopen one lane."""

    lane: RetrievalLane
    index_format: str
    unit_count: int
    shape: tuple[int, ...]
    dtype: str
    c_order: bool
    shard_role: str
    schema_version: int = PERSONAL_INDEX_PARAMETER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.lane not in {
            RetrievalLane.PERCEPTUAL,
            RetrievalLane.INSTANCE,
            RetrievalLane.CONCEPT,
        }:
            raise ValueError("personal index parameters require an indexed lane")
        expected_format = (
            _PHASH_FORMAT if self.lane is RetrievalLane.PERCEPTUAL else _SEMANTIC_FORMAT
        )
        if self.index_format != expected_format:
            raise ValueError("index_format does not match the personal lane")
        if isinstance(self.unit_count, bool) or not 1 <= self.unit_count <= 10_000:
            raise ValueError("unit_count must be from 1 to 10,000")
        shape = tuple(self.shape)
        if not shape or shape[0] != self.unit_count:
            raise ValueError("shape must begin with unit_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape
        ):
            raise ValueError("shape dimensions must be positive integers")
        if self.lane is RetrievalLane.PERCEPTUAL and len(shape) != 1:
            raise ValueError("pHash shape must be one-dimensional")
        if self.lane is not RetrievalLane.PERCEPTUAL and len(shape) != 2:
            raise ValueError("semantic shape must be two-dimensional")
        object.__setattr__(self, "shape", shape)
        expected_dtype = "<u8" if self.lane is RetrievalLane.PERCEPTUAL else "<f4"
        if self.dtype != expected_dtype:
            raise ValueError("dtype does not match the personal lane")
        if self.c_order is not True:
            raise ValueError("personal arrays must use C order")
        _non_blank("shard_role", self.shard_role)
        if self.schema_version != PERSONAL_INDEX_PARAMETER_SCHEMA_VERSION:
            raise ValueError("unsupported personal index parameter schema")


def encode_personal_index_parameters(parameters: PersonalIndexParameters) -> bytes:
    """Encode one canonical newline-terminated parameter document."""
    if not isinstance(parameters, PersonalIndexParameters):
        raise TypeError("parameters must be PersonalIndexParameters")
    return (
        _canonical_json(
            {
                "schema_version": parameters.schema_version,
                "lane": parameters.lane.value,
                "index_format": parameters.index_format,
                "unit_count": parameters.unit_count,
                "shape": list(parameters.shape),
                "dtype": parameters.dtype,
                "c_order": parameters.c_order,
                "shard_role": parameters.shard_role,
            }
        )
        + b"\n"
    )


def decode_personal_index_parameters(payload: bytes) -> PersonalIndexParameters:
    """Strictly decode only canonical personal index parameter bytes."""
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_PERSONAL_INDEX_PARAMETER_BYTES:
        raise ValueError("personal index parameter payload has an invalid size")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("personal index parameter payload is invalid JSON") from error
    if not isinstance(document, dict) or set(document) != _EXPECTED_PARAMETER_FIELDS:
        raise ValueError("personal index parameter fields do not match the schema")
    shape_value = document["shape"]
    if not isinstance(shape_value, list):
        raise ValueError("shape must be an array")
    parameters = PersonalIndexParameters(
        lane=RetrievalLane(_string(document["lane"])),
        index_format=_string(document["index_format"]),
        unit_count=_integer(document["unit_count"]),
        shape=tuple(_integer(value) for value in shape_value),
        dtype=_string(document["dtype"]),
        c_order=_boolean(document["c_order"]),
        shard_role=_string(document["shard_role"]),
        schema_version=_integer(document["schema_version"]),
    )
    if encode_personal_index_parameters(parameters) != payload:
        raise ValueError("personal index parameter payload must use canonical encoding")
    return parameters


class PersonalGenerationHandleFactory:
    """Reopen one verified personal generation as a complete reader group."""

    def __init__(self, store: FilesystemGenerationStore) -> None:
        if not isinstance(store, FilesystemGenerationStore):
            raise TypeError("store must be a FilesystemGenerationStore")
        self._store = store

    def open(
        self,
        published: PublishedGenerationRecord,
        *,
        deadline: datetime,
    ) -> OpenedGenerationHandles:
        """Reverify the manifest/payloads and open catalog plus all three lanes."""
        if not isinstance(published, PublishedGenerationRecord):
            raise TypeError("published must be a PublishedGenerationRecord")
        reopened = self._store.open_published(
            published.activation_set_id,
            expected_manifest_sha256=published.manifest_sha256,
            deadline=deadline,
        )
        if reopened.root != published.published_root:
            raise ValueError("registered publication root does not match the trusted store")
        manifest = reopened.manifest
        if (
            manifest.workspace_id != published.workspace_id
            or manifest.corpus_ref != published.corpus_ref
            or manifest.retrieval_profile_ref != published.retrieval_profile_ref
        ):
            raise ValueError("registered publication identity does not match its manifest")
        payloads = {payload.role: payload for payload in manifest.payloads}
        catalog_payload = _payload(payloads, manifest.unit_catalog.payload_ref)
        if manifest.unit_catalog.catalog_format != SQLITE_UNIT_CATALOG_FORMAT:
            raise ValueError("personal generation must use the immutable SQLite unit catalog")
        catalog = SqliteUnitCatalog.open_readonly(
            SqliteUnitCatalogDescriptor(
                catalog_id=manifest.unit_catalog.catalog_id,
                activation_set_id=manifest.activation_set_id,
                corpus_ref=manifest.corpus_ref,
                unit_count=manifest.unit_catalog.unit_count,
                unit_manifest_sha256=manifest.unit_manifest_sha256,
                path=_payload_path(reopened.root, catalog_payload),
                sha256=catalog_payload.sha256,
                size_bytes=catalog_payload.size_bytes,
                payload_ref=catalog_payload.role,
                catalog_format=manifest.unit_catalog.catalog_format,
            ),
            deadline=deadline,
        )
        readers: dict[RetrievalLane, NumpyExactIndexReader | PhashFlatIndexReader] = {}
        try:
            for lane_generation in manifest.lanes:
                reader = self._open_lane(
                    reopened.root,
                    lane_generation,
                    payloads,
                    catalog,
                    deadline,
                )
                readers[lane_generation.lane] = reader
            return OpenedGenerationHandles(
                manifest=manifest,
                unit_catalog=catalog,
                readers=readers,
                close_callbacks=(
                    catalog.close,
                    *(readers[lane].close for lane in sorted(readers, key=lambda item: item.value)),
                ),
            )
        except BaseException:
            for reader in readers.values():
                reader.close()
            catalog.close()
            raise

    @staticmethod
    def _open_lane(
        root: Path,
        lane_generation: LaneGeneration,
        payloads: dict[str, PayloadFileRef],
        catalog: SqliteUnitCatalog,
        deadline: datetime,
    ) -> NumpyExactIndexReader | PhashFlatIndexReader:
        generation = lane_generation.generation
        parameter_payload = _payload(payloads, generation.index_parameters_ref.payload_ref)
        parameter_bytes = _read_verified_small_payload(
            root,
            parameter_payload,
            maximum_bytes=MAX_PERSONAL_INDEX_PARAMETER_BYTES,
        )
        parameters = decode_personal_index_parameters(parameter_bytes)
        if (
            parameters.lane is not lane_generation.lane
            or parameters.index_format != generation.index_format
            or parameters.unit_count != catalog.unit_count
            or parameters.shard_role not in generation.shard_refs
            or len(generation.shard_refs) != 1
        ):
            raise ValueError("personal index parameters do not match the generation")
        shard = _payload(payloads, parameters.shard_role)
        embedding_space = generation.embedding_spaces[0] if generation.embedding_spaces else None
        hash_space = generation.hash_spaces[0] if generation.hash_spaces else None
        descriptor = PersonalIndexPayload(
            path=_payload_path(root, shard),
            sha256=shard.sha256,
            size_bytes=shard.size_bytes,
            unit_count=parameters.unit_count,
            shape=parameters.shape,
            dtype=parameters.dtype,
            c_order=parameters.c_order,
            generation_id=generation.generation_id,
            activation_set_id=generation.activation_set_id,
            unit_manifest_sha256=generation.unit_manifest_sha256,
            index_format=generation.index_format,
            index_parameters_ref=generation.index_parameters_ref,
            lane=lane_generation.lane,
            embedding_space=embedding_space,
            hash_space=hash_space,
        )
        if lane_generation.lane is RetrievalLane.PERCEPTUAL:
            return PhashFlatIndexReader.open_readonly(
                generation,
                descriptor,
                ordinal_resolver=catalog,
                deadline=deadline,
            )
        return NumpyExactIndexReader.open_readonly(
            generation,
            descriptor,
            ordinal_resolver=catalog,
            deadline=deadline,
        )


def _payload(payloads: dict[str, PayloadFileRef], role: str) -> PayloadFileRef:
    try:
        return payloads[role]
    except KeyError as error:
        raise ValueError(f"published generation is missing payload role: {role}") from error


def _payload_path(root: Path, payload: PayloadFileRef) -> Path:
    path = root.joinpath(*payload.relative_path.parts)
    if not path.is_absolute() or root not in path.parents:
        raise ValueError("payload path escapes the published root")
    return path


def _read_verified_small_payload(
    root: Path,
    payload: PayloadFileRef,
    *,
    maximum_bytes: int,
) -> bytes:
    if payload.size_bytes > maximum_bytes:
        raise ValueError("parameter payload exceeds its size limit")
    path = _payload_path(root, payload)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != payload.size_bytes:
            raise ValueError("parameter payload identity is invalid")
        data = bytearray()
        while len(data) <= maximum_bytes:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(data) != payload.size_bytes or len(data) > maximum_bytes:
        raise ValueError("parameter payload size changed")
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise ValueError("parameter payload changed while reading")
    observed = hashlib.sha256(data).hexdigest()
    if observed != payload.sha256:
        raise ValueError("parameter payload checksum mismatch")
    return bytes(data)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected string")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected integer")
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("expected boolean")
    return value


def _non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")


__all__ = [
    "MAX_PERSONAL_INDEX_PARAMETER_BYTES",
    "PERSONAL_INDEX_PARAMETER_SCHEMA_VERSION",
    "PersonalGenerationHandleFactory",
    "PersonalIndexParameters",
    "decode_personal_index_parameters",
    "encode_personal_index_parameters",
]
