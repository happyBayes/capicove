"""Closed canonical wire encoding for immutable maintenance records."""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Literal, get_args, get_origin

from capicove.contracts.wiki_maintenance import (
    MAX_RECORD_BYTES,
    RECORD_TYPES,
    MaintenanceRecord,
    record_hints,
    valid_type,
)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _forbid_number(value: str) -> Any:
    raise ValueError("non-integer JSON number is not supported")


def _value(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is types.UnionType:
        for choice in args:
            try:
                return _value(value, choice)
            except ValueError:
                continue
        raise ValueError("invalid union value")
    if origin is tuple:
        if type(value) is not list or len(value) > 10_000:
            raise ValueError("expected bounded JSON array")
        return tuple(_value(item, args[0]) for item in value)
    if origin is Literal:
        if not valid_type(value, annotation):
            raise ValueError("invalid literal")
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        names = {item.name for item in fields(annotation)}
        if type(value) is not dict or set(value) != names:
            raise ValueError("record fields must exactly match the schema")
        hints = record_hints(annotation)
        return annotation(**{name: _value(value[name], hints[name]) for name in names})
    if not valid_type(value, annotation):
        raise ValueError("invalid scalar type")
    return value


def decode_record(raw: bytes) -> MaintenanceRecord:
    """Decode a closed bounded envelope without accepting caller authority."""
    if type(raw) is not bytes or len(raw) > MAX_RECORD_BYTES:
        raise ValueError("record exceeds 1 MiB or is not bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object,
            parse_float=_forbid_number,
            parse_constant=_forbid_number,
        )
        if type(value) is not dict or set(value) != {"schema_version", "record_type", "payload"}:
            raise ValueError("invalid maintenance envelope")
        version, name = value["schema_version"], value["record_type"]
        if (
            type(version) is not int
            or version != 1
            or type(name) is not str
            or name not in RECORD_TYPES
        ):
            raise ValueError("unsupported maintenance record")
        record = MaintenanceRecord(
            schema_version=version,
            record_type=name,
            payload=_value(value["payload"], RECORD_TYPES[name]),
        )
        encode_record(record)
        return record
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("invalid or excessively nested record") from error


def encode_record(record: MaintenanceRecord) -> bytes:
    """Encode exact versioned bytes; record identity is a separate SHA-256."""
    if type(record) is not MaintenanceRecord:
        raise ValueError("expected MaintenanceRecord")
    record.__post_init__()
    try:
        raw = json.dumps(
            asdict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid record text") from error
    limit = (
        65_536 if record.record_type in {"answer_submission", "answer_result"} else MAX_RECORD_BYTES
    )
    if len(raw) > limit:
        raise ValueError("record exceeds serialized byte limit")
    return raw


def record_digest(record: MaintenanceRecord) -> str:
    """Fingerprint canonical envelope bytes without a self-referential hash."""
    return hashlib.sha256(encode_record(record)).hexdigest()
