"""Immutable options, shard metadata, and manifest contracts for trace exports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, TypeVar

from capicove.contracts import ToolStatus
from capicove.traces import EpisodeStatus, TraceModality, TraceQuery


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_MAX_RECORDS_PER_SHARD = 10_000_000
_MAX_BYTES_PER_SHARD = 8 * 1024**3
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class MediaMode(StrEnum):
    """Choose whether an export retains references or copies local media."""

    REFERENCE = "reference"
    COPY = "copy"


@dataclass(frozen=True, slots=True)
class TraceExportOptions:
    """Control complete-Episode selection and bounded JSONL sharding."""

    query: TraceQuery = field(default_factory=TraceQuery)
    media_mode: MediaMode = MediaMode.REFERENCE
    max_records_per_shard: int = 100_000
    max_bytes_per_shard: int = 536_870_912
    max_record_bytes: int = 16_777_216

    def __post_init__(self) -> None:
        if not isinstance(self.query, TraceQuery):
            raise ValueError("query must be a TraceQuery")
        if not isinstance(self.media_mode, MediaMode):
            raise ValueError("media_mode must be a MediaMode")
        _require_bounded_int(
            "max_records_per_shard",
            self.max_records_per_shard,
            _MAX_RECORDS_PER_SHARD,
        )
        _require_bounded_int(
            "max_bytes_per_shard",
            self.max_bytes_per_shard,
            _MAX_BYTES_PER_SHARD,
        )
        _require_bounded_int("max_record_bytes", self.max_record_bytes, _MAX_BYTES_PER_SHARD)


@dataclass(frozen=True, slots=True)
class TraceExportShard:
    """Describe one immutable ordered JSONL shard in an export package."""

    path: PurePosixPath
    record_count: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_safe_path("shard path", self.path)
        _require_positive_int("record_count", self.record_count)
        _require_positive_int("size_bytes", self.size_bytes)
        _require_sha256("sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class TraceExportManifest:
    """Describe all integrity-relevant contents of one portable export package."""

    format_version: int
    schema_version: int
    export_id: str
    created_at: datetime
    options: TraceExportOptions
    shards: tuple[TraceExportShard, ...]
    record_count: int
    record_type_counts: Mapping[str, int]
    modality_counts: Mapping[str, int]
    tool_status_counts: Mapping[str, int]
    episode_status_counts: Mapping[str, int]
    included_episode_count: int
    incomplete_episode_count: int
    omitted_restricted_episode_count: int
    media_count: int
    media_size_bytes: int
    missing_media_count: int
    media_layout: str
    schema_path: PurePosixPath
    schema_sha256: str

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("format_version must be 1")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        _require_non_blank("export_id", self.export_id)
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if not isinstance(self.options, TraceExportOptions):
            raise ValueError("options must be TraceExportOptions")
        if not all(isinstance(shard, TraceExportShard) for shard in self.shards):
            raise ValueError("shards must contain TraceExportShard values")
        paths = tuple(shard.path for shard in self.shards)
        if len(set(paths)) != len(paths):
            raise ValueError("shard paths must be unique")
        expected_paths = tuple(
            PurePosixPath(f"records-{index:05d}.jsonl") for index in range(len(self.shards))
        )
        if paths != expected_paths:
            raise ValueError("shard paths must be contiguous canonical record names")
        for field_name in (
            "record_count",
            "included_episode_count",
            "incomplete_episode_count",
            "omitted_restricted_episode_count",
            "media_count",
            "media_size_bytes",
            "missing_media_count",
        ):
            _require_non_negative_int(field_name, getattr(self, field_name))
        count_fields = (
            "record_type_counts",
            "modality_counts",
            "tool_status_counts",
            "episode_status_counts",
        )
        for field_name in count_fields:
            object.__setattr__(
                self, field_name, _freeze_counts(field_name, getattr(self, field_name))
            )
        if self.record_count != sum(shard.record_count for shard in self.shards):
            raise ValueError("record_count must equal the sum of shard record counts")
        if self.record_count != sum(self.record_type_counts.values()):
            raise ValueError("record_count must equal the sum of record type counts")
        if self.incomplete_episode_count > self.included_episode_count:
            raise ValueError("incomplete Episode count cannot exceed included Episode count")
        if self.missing_media_count > self.media_count:
            raise ValueError("missing media count cannot exceed media count")
        if self.media_layout != "sha256-v1":
            raise ValueError("media_layout must be sha256-v1")
        _require_safe_path("schema_path", self.schema_path)
        _require_sha256("schema_sha256", self.schema_sha256)


def encode_trace_export_manifest(manifest: TraceExportManifest, /) -> dict[str, object]:
    """Encode one export manifest into its exact version 1 mapping shape."""
    return {
        "format_version": manifest.format_version,
        "schema_version": manifest.schema_version,
        "export_id": manifest.export_id,
        "created_at": _encode_timestamp(manifest.created_at),
        "options": _encode_options(manifest.options),
        "shards": [
            {
                "path": shard.path.as_posix(),
                "record_count": shard.record_count,
                "size_bytes": shard.size_bytes,
                "sha256": shard.sha256,
            }
            for shard in manifest.shards
        ],
        "record_count": manifest.record_count,
        "record_type_counts": dict(manifest.record_type_counts),
        "modality_counts": dict(manifest.modality_counts),
        "tool_status_counts": dict(manifest.tool_status_counts),
        "episode_status_counts": dict(manifest.episode_status_counts),
        "included_episode_count": manifest.included_episode_count,
        "incomplete_episode_count": manifest.incomplete_episode_count,
        "omitted_restricted_episode_count": manifest.omitted_restricted_episode_count,
        "media_count": manifest.media_count,
        "media_size_bytes": manifest.media_size_bytes,
        "missing_media_count": manifest.missing_media_count,
        "media_layout": manifest.media_layout,
        "schema_path": manifest.schema_path.as_posix(),
        "schema_sha256": manifest.schema_sha256,
    }


def decode_trace_export_manifest(value: Mapping[str, object], /) -> TraceExportManifest:
    """Strictly decode one exact version 1 export manifest."""
    manifest = _object(value, "manifest")
    _exact_keys(
        manifest,
        {
            "format_version",
            "schema_version",
            "export_id",
            "created_at",
            "options",
            "shards",
            "record_count",
            "record_type_counts",
            "modality_counts",
            "tool_status_counts",
            "episode_status_counts",
            "included_episode_count",
            "incomplete_episode_count",
            "omitted_restricted_episode_count",
            "media_count",
            "media_size_bytes",
            "missing_media_count",
            "media_layout",
            "schema_path",
            "schema_sha256",
        },
        "manifest",
    )
    try:
        return TraceExportManifest(
            format_version=_integer(manifest["format_version"], "format_version"),
            schema_version=_integer(manifest["schema_version"], "schema_version"),
            export_id=_string(manifest["export_id"], "export_id"),
            created_at=_decode_timestamp(manifest["created_at"]),
            options=_decode_options(manifest["options"]),
            shards=tuple(_decode_shard(item) for item in _array(manifest["shards"], "shards")),
            record_count=_integer(manifest["record_count"], "record_count"),
            record_type_counts=_decode_counts(manifest["record_type_counts"], "record_type_counts"),
            modality_counts=_decode_counts(manifest["modality_counts"], "modality_counts"),
            tool_status_counts=_decode_counts(manifest["tool_status_counts"], "tool_status_counts"),
            episode_status_counts=_decode_counts(
                manifest["episode_status_counts"], "episode_status_counts"
            ),
            included_episode_count=_integer(
                manifest["included_episode_count"], "included_episode_count"
            ),
            incomplete_episode_count=_integer(
                manifest["incomplete_episode_count"], "incomplete_episode_count"
            ),
            omitted_restricted_episode_count=_integer(
                manifest["omitted_restricted_episode_count"],
                "omitted_restricted_episode_count",
            ),
            media_count=_integer(manifest["media_count"], "media_count"),
            media_size_bytes=_integer(manifest["media_size_bytes"], "media_size_bytes"),
            missing_media_count=_integer(manifest["missing_media_count"], "missing_media_count"),
            media_layout=_string(manifest["media_layout"], "media_layout"),
            schema_path=PurePosixPath(_string(manifest["schema_path"], "schema_path")),
            schema_sha256=_string(manifest["schema_sha256"], "schema_sha256"),
        )
    except KeyError as error:
        raise ValueError("manifest is missing a required field") from error


def _encode_options(options: TraceExportOptions) -> dict[str, object]:
    query = options.query
    return {
        "query": {
            "episode_ids": _strings(query.episode_ids),
            "started_after": _optional_timestamp(query.started_after),
            "completed_before": _optional_timestamp(query.completed_before),
            "modalities": _enum_strings(query.modalities),
            "tool_ids": _strings(query.tool_ids),
            "tool_statuses": _enum_strings(query.tool_statuses),
            "episode_statuses": _enum_strings(query.episode_statuses),
            "include_incomplete": query.include_incomplete,
            "include_restricted": query.include_restricted,
        },
        "media_mode": options.media_mode.value,
        "max_records_per_shard": options.max_records_per_shard,
        "max_bytes_per_shard": options.max_bytes_per_shard,
        "max_record_bytes": options.max_record_bytes,
    }


def _decode_options(value: object) -> TraceExportOptions:
    options = _object(value, "options")
    _exact_keys(
        options,
        {
            "query",
            "media_mode",
            "max_records_per_shard",
            "max_bytes_per_shard",
            "max_record_bytes",
        },
        "options",
    )
    query = _object(options["query"], "query")
    _exact_keys(
        query,
        {
            "episode_ids",
            "started_after",
            "completed_before",
            "modalities",
            "tool_ids",
            "tool_statuses",
            "episode_statuses",
            "include_incomplete",
            "include_restricted",
        },
        "query",
    )
    return TraceExportOptions(
        query=TraceQuery(
            episode_ids=_optional_string_set(query["episode_ids"], "episode_ids"),
            started_after=_optional_decoded_timestamp(query["started_after"], "started_after"),
            completed_before=_optional_decoded_timestamp(
                query["completed_before"], "completed_before"
            ),
            modalities=_optional_enum_set(query["modalities"], "modalities", TraceModality),
            tool_ids=_optional_string_set(query["tool_ids"], "tool_ids"),
            tool_statuses=_optional_enum_set(query["tool_statuses"], "tool_statuses", ToolStatus),
            episode_statuses=_optional_enum_set(
                query["episode_statuses"], "episode_statuses", EpisodeStatus
            ),
            include_incomplete=_boolean(query["include_incomplete"], "include_incomplete"),
            include_restricted=_boolean(query["include_restricted"], "include_restricted"),
        ),
        media_mode=_enum(MediaMode, options["media_mode"], "media_mode"),
        max_records_per_shard=_integer(options["max_records_per_shard"], "max_records_per_shard"),
        max_bytes_per_shard=_integer(options["max_bytes_per_shard"], "max_bytes_per_shard"),
        max_record_bytes=_integer(options["max_record_bytes"], "max_record_bytes"),
    )


def _decode_shard(value: object) -> TraceExportShard:
    shard = _object(value, "shard")
    _exact_keys(shard, {"path", "record_count", "size_bytes", "sha256"}, "shard")
    return TraceExportShard(
        path=PurePosixPath(_string(shard["path"], "shard.path")),
        record_count=_integer(shard["record_count"], "shard.record_count"),
        size_bytes=_integer(shard["size_bytes"], "shard.size_bytes"),
        sha256=_string(shard["sha256"], "shard.sha256"),
    )


def _freeze_counts(field_name: str, value: Mapping[str, int]) -> Mapping[str, int]:
    frozen: dict[str, int] = {}
    for key, count in value.items():
        _require_non_blank(f"{field_name} key", key)
        _require_non_negative_int(f"{field_name} value", count)
        frozen[key] = count
    return MappingProxyType(frozen)


def _decode_counts(value: object, field_name: str) -> dict[str, int]:
    return {
        key: _integer(count, f"{field_name}.{key}")
        for key, count in _object(value, field_name).items()
    }


def _require_safe_path(field_name: str, value: object) -> None:
    if not isinstance(value, PurePosixPath):
        raise ValueError(f"{field_name} must be a PurePosixPath")
    if value.is_absolute() or value == PurePosixPath(".") or ".." in value.parts:
        raise ValueError(f"{field_name} must remain inside the export package")


def _require_bounded_int(field_name: str, value: object, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{field_name} must be between 1 and {maximum}")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_non_blank(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _encode_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _optional_timestamp(value: datetime | None) -> str | None:
    return None if value is None else _encode_timestamp(value)


def _decode_timestamp(value: object) -> datetime:
    text = _string(value, "timestamp")
    if _TIMESTAMP_PATTERN.fullmatch(text) is None:
        raise ValueError("timestamp must use six-digit UTC form")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("timestamp must be valid") from error


def _optional_decoded_timestamp(value: object, field_name: str) -> datetime | None:
    if value is None:
        return None
    try:
        return _decode_timestamp(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a UTC timestamp") from error


def _strings(value: frozenset[str] | None) -> list[str] | None:
    return None if value is None else sorted(value)


def _enum_strings(value: frozenset[StrEnum] | None) -> list[str] | None:
    return None if value is None else sorted(item.value for item in value)


def _optional_string_set(value: object, field_name: str) -> frozenset[str] | None:
    if value is None:
        return None
    return frozenset(_string(item, f"{field_name} item") for item in _array(value, field_name))


def _optional_enum_set(
    value: object,
    field_name: str,
    enum_type: type[StrEnum],
) -> frozenset[Any] | None:
    if value is None:
        return None
    return frozenset(
        _enum(enum_type, item, f"{field_name} item") for item in _array(value, field_name)
    )


def _object(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _array(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field_name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field_name} has unknown or missing fields")


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _enum(enum_type: type[_EnumT], value: object, field_name: str) -> _EnumT:
    try:
        return enum_type(_string(value, field_name))
    except ValueError as error:
        raise ValueError(f"{field_name} has an unsupported value") from error
