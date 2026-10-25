"""Explicit optional PyArrow adapter for closed graph tables."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from capicove.contracts.data import ComplexTypeKind, SchemaField, SchemaValueType
from capicove.graph_build._decode_reuse import GRAPH_TABLES
from capicove.graph_build.codec import fingerprint
from capicove.graph_build.datasets import codec_for_role


SERIALIZER_VERSION = "capicove.graph-parquet.v1"


def _safe_path(path: Path, *, writing: bool = False) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Parquet path must be absolute without traversal")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Parquet path must not contain a symlink")
    if writing:
        if path.exists():
            raise ValueError("Parquet destination already exists")
    elif not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Parquet part must be a regular file")


class ParquetTableCodec:
    """Read and write real typed Parquet; construction opts into PyArrow.

    Publication binds a scope/request digest in metadata. Standalone codec use
    leaves that digest empty and cannot impersonate a bound publication. Exact
    runtime version is included in serializer identity; replay across versions
    is deliberately not promised.
    """

    def __init__(self, *, publication_identity: str = "") -> None:
        import pyarrow  # type: ignore[import-untyped]
        import pyarrow.parquet  # type: ignore[import-untyped]

        self._arrow = pyarrow
        self._parquet = pyarrow.parquet
        self._publication_identity = publication_identity
        self.serializer_identity = f"{SERIALIZER_VERSION};pyarrow={pyarrow.__version__};zstd"

    def _type(self, value: SchemaValueType) -> Any:
        arrow = self._arrow
        if isinstance(value, str):
            return {
                "string": arrow.string,
                "boolean": arrow.bool_,
                "int64": arrow.int64,
                "float64": arrow.float64,
            }[value]()
        if value.kind is ComplexTypeKind.ARRAY:
            if value.element_type is None:
                raise ValueError("graph array element type is missing")
            return arrow.list_(
                arrow.field("element", self._type(value.element_type), nullable=False)
            )
        if value.kind is ComplexTypeKind.STRUCT:
            return arrow.struct([self._field(field) for field in value.fields])
        raise ValueError("graph tables do not use unrestricted maps")

    def _field(self, field: SchemaField) -> Any:
        return self._arrow.field(
            field.name, self._type(field.logical_type), nullable=field.nullable
        )

    def _schema(self, role: str, rows: tuple[object, ...]) -> Any:
        codec = codec_for_role(role)
        metadata = {
            "role": role,
            "serializer": self.serializer_identity,
            "schema_sha256": codec.schema.sha256,
            "row_count": str(len(rows)),
            "content_sha256": fingerprint([codec.encode(row) for row in rows]),
            "publication_identity": self._publication_identity,
        }
        return self._arrow.schema(
            [self._field(field) for field in codec.schema.fields],
            metadata={
                f"capicove.{key}".encode(): value.encode() for key, value in metadata.items()
            },
        )

    def write(self, path: Path, role: str, rows: Iterable[object]) -> None:
        """Create one exclusive Zstandard-compressed part with its full schema."""
        codec = codec_for_role(role)
        ordered = codec.snapshot(rows)
        table = self._arrow.Table.from_pylist(
            [codec.encode(row) for row in ordered], schema=self._schema(role, ordered)
        )
        _safe_path(path, writing=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            self._parquet.write_table(table, stream, compression="zstd", version="2.6")
            stream.flush()
            os.fsync(stream.fileno())

    def read(self, path: Path, role: str) -> tuple[object, ...]:
        """Validate exact Arrow schema, metadata, count, order and logical rows.

        The part is read once and decoded from the hashed buffer. Rows decoded
        earlier from byte-identical content under the same role, publication
        and serializer identity are reused; failures are never retained.
        """
        codec = codec_for_role(role)
        _safe_path(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Parquet part must be regular")
            payload = stream.read()
        key = (
            hashlib.sha256(payload).hexdigest(),
            role,
            self._publication_identity,
            self.serializer_identity,
        )
        reused = GRAPH_TABLES.get(key)
        if reused is not None:
            return tuple(reused)
        try:
            parquet = self._parquet.ParquetFile(self._arrow.BufferReader(payload))
            table = parquet.read()
        except (self._arrow.ArrowException, OSError) as error:
            raise ValueError("corrupt Parquet payload") from error
        rows = tuple(codec.decode(row) for row in table.to_pylist())
        ordered = codec.snapshot(rows)
        if rows != ordered:
            raise ValueError("Parquet rows are not in canonical identity order")
        if not table.schema.equals(self._schema(role, ordered), check_metadata=True):
            raise ValueError("Parquet schema or serializer metadata mismatch")
        if parquet.metadata.num_rows != len(rows):
            raise ValueError("Parquet row count mismatch")
        GRAPH_TABLES.put(key, rows, len(payload))
        return rows
