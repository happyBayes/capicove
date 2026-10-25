"""Canonical HMAC cursors for immutable reverse-image-search generations."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from capicove.contracts.retrieval import MAX_CURSOR_BYTES, MAX_REF_BYTES


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_FLOAT_HEX_BYTES = 64
_MAX_UNIT_ID_BYTES = 128
_CURSOR_FIELDS = frozenset(
    {
        "schema_version",
        "binding_sha256",
        "last_exact_content",
        "last_fused_score_hex",
        "last_best_lane_rank",
        "last_unit_id",
        "ranking_state_sha256",
        "consumed_count",
        "page_size",
        "issued_at",
        "expires_at",
    }
)


class StaleCursorError(ValueError):
    """Signal that a cursor is malformed, stale, expired, or context-mismatched."""


@dataclass(frozen=True, slots=True)
class SearchCursorBinding:
    """Bind pagination to one normalized query and immutable generation set."""

    query_sha256: str
    workspace_id: str
    targets: tuple[str, ...]
    filters_sha256: str
    search_intent: str
    retrieval_profile_ref: str
    data_policy_ref: str
    generation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256("query_sha256", self.query_sha256)
        _require_sha256("filters_sha256", self.filters_sha256)
        for name in (
            "workspace_id",
            "search_intent",
            "retrieval_profile_ref",
            "data_policy_ref",
        ):
            _require_bounded_string(name, getattr(self, name), MAX_REF_BYTES)
        object.__setattr__(
            self,
            "targets",
            _unique_strings("targets", self.targets, 3, MAX_REF_BYTES),
        )
        object.__setattr__(
            self,
            "generation_ids",
            _unique_strings("generation_ids", self.generation_ids, 4, MAX_REF_BYTES),
        )


@dataclass(frozen=True, slots=True)
class SearchCursorPosition:
    """Carry one stable result position without source bytes or local paths."""

    binding: SearchCursorBinding
    last_exact_content: bool
    last_fused_score_hex: str
    last_best_lane_rank: int
    last_unit_id: str
    ranking_state_sha256: str
    consumed_count: int
    page_size: int
    issued_at: datetime
    expires_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.binding, SearchCursorBinding):
            raise TypeError("binding must be a SearchCursorBinding")
        if not isinstance(self.last_exact_content, bool):
            raise TypeError("last_exact_content must be a boolean")
        _require_bounded_string(
            "last_fused_score_hex",
            self.last_fused_score_hex,
            _MAX_FLOAT_HEX_BYTES,
        )
        try:
            fused_score = float.fromhex(self.last_fused_score_hex)
        except ValueError as error:
            raise ValueError("last_fused_score_hex must be a finite positive float") from error
        if not math.isfinite(fused_score) or fused_score <= 0:
            raise ValueError("last_fused_score_hex must be a finite positive float")
        _require_positive_int("last_best_lane_rank", self.last_best_lane_rank)
        _require_bounded_string("last_unit_id", self.last_unit_id, _MAX_UNIT_ID_BYTES)
        _require_sha256("ranking_state_sha256", self.ranking_state_sha256)
        _require_non_negative_int("consumed_count", self.consumed_count)
        if self.consumed_count > 10_000:
            raise ValueError("consumed_count must not exceed 10,000")
        _require_positive_int("page_size", self.page_size)
        if self.page_size > 100:
            raise ValueError("page_size must not exceed 100")
        _require_aware("issued_at", self.issued_at)
        _require_aware("expires_at", self.expires_at)
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")


class HmacSearchCursorCodec:
    """Encode and verify bounded canonical cursors using a startup-owned key."""

    def __init__(self, *, signing_key: bytes, workspace_id: str) -> None:
        if not isinstance(signing_key, bytes) or len(signing_key) < 32:
            raise ValueError("signing_key must contain at least 32 bytes")
        _require_bounded_string("workspace_id", workspace_id, MAX_REF_BYTES)
        self._signing_key = signing_key
        self._workspace_id = workspace_id

    def encode(self, position: SearchCursorPosition) -> str:
        """Return one deterministic URL-safe signed cursor."""
        if not isinstance(position, SearchCursorPosition):
            raise TypeError("position must be a SearchCursorPosition")
        if position.binding.workspace_id != self._workspace_id:
            raise ValueError("cursor binding belongs to another workspace")
        body = _canonical_json(_encode_position(position))
        signature = hmac.digest(self._signing_key, body, "sha256")
        cursor = f"{_urlsafe_encode(body)}.{_urlsafe_encode(signature)}"
        if len(cursor.encode("ascii")) > MAX_CURSOR_BYTES:
            raise ValueError("encoded cursor exceeds the size limit")
        return cursor

    def decode(
        self,
        cursor: str,
        *,
        expected: SearchCursorBinding,
        now: datetime,
    ) -> SearchCursorPosition:
        """Verify one cursor against the exact current request and generation binding."""
        try:
            if not isinstance(cursor, str) or not cursor or len(cursor.encode()) > MAX_CURSOR_BYTES:
                raise ValueError
            if not isinstance(expected, SearchCursorBinding):
                raise TypeError
            _require_aware("now", now)
            body_part, signature_part = cursor.split(".")
            body = _urlsafe_decode(body_part)
            signature = _urlsafe_decode(signature_part)
            observed = hmac.digest(self._signing_key, body, "sha256")
            if len(signature) != 32 or not hmac.compare_digest(signature, observed):
                raise ValueError
            document = json.loads(body, object_pairs_hook=_closed_object)
            position = _decode_position(document, expected)
            if position.binding != expected:
                raise ValueError
            if position.binding.workspace_id != self._workspace_id:
                raise ValueError
            if now >= position.expires_at:
                raise ValueError
            return position
        except (KeyError, TypeError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise StaleCursorError("cursor is stale or invalid") from error


def _encode_position(position: SearchCursorPosition) -> dict[str, object]:
    return {
        "schema_version": position.schema_version,
        "binding_sha256": _binding_sha256(position.binding),
        "last_exact_content": position.last_exact_content,
        "last_fused_score_hex": position.last_fused_score_hex,
        "last_best_lane_rank": position.last_best_lane_rank,
        "last_unit_id": position.last_unit_id,
        "ranking_state_sha256": position.ranking_state_sha256,
        "consumed_count": position.consumed_count,
        "page_size": position.page_size,
        "issued_at": position.issued_at.isoformat(),
        "expires_at": position.expires_at.isoformat(),
    }


def _decode_position(
    value: object,
    expected_binding: SearchCursorBinding,
) -> SearchCursorPosition:
    document = _require_exact_mapping(value, _CURSOR_FIELDS)
    binding_sha256 = _string(document["binding_sha256"])
    _require_sha256("binding_sha256", binding_sha256)
    if not hmac.compare_digest(binding_sha256, _binding_sha256(expected_binding)):
        raise ValueError("cursor binding does not match the expected search")
    return SearchCursorPosition(
        binding=expected_binding,
        last_exact_content=_boolean(document["last_exact_content"]),
        last_fused_score_hex=_string(document["last_fused_score_hex"]),
        last_best_lane_rank=_integer(document["last_best_lane_rank"]),
        last_unit_id=_string(document["last_unit_id"]),
        ranking_state_sha256=_string(document["ranking_state_sha256"]),
        consumed_count=_integer(document["consumed_count"]),
        page_size=_integer(document["page_size"]),
        issued_at=_datetime(document["issued_at"]),
        expires_at=_datetime(document["expires_at"]),
        schema_version=_integer(document["schema_version"]),
    )


def _binding_sha256(binding: SearchCursorBinding) -> str:
    document = {
        "data_policy_ref": binding.data_policy_ref,
        "filters_sha256": binding.filters_sha256,
        "generation_ids": list(binding.generation_ids),
        "query_sha256": binding.query_sha256,
        "retrieval_profile_ref": binding.retrieval_profile_ref,
        "search_intent": binding.search_intent,
        "targets": list(binding.targets),
        "workspace_id": binding.workspace_id,
    }
    return hashlib.sha256(
        b"capicove-search-cursor-binding-v1\x00" + _canonical_json(document)
    ).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_decode(value: str) -> bytes:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("invalid base64url")
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _require_exact_mapping(value: object, fields: frozenset[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("cursor object fields do not match the schema")
    return value


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


def _datetime(value: object) -> datetime:
    parsed = datetime.fromisoformat(_string(value))
    _require_aware("datetime", parsed)
    return parsed


def _unique_strings(
    field_name: str,
    values: object,
    maximum: int,
    maximum_bytes: int,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values or len(values) > maximum:
        raise ValueError(f"{field_name} must contain from 1 to {maximum} values")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    for value in values:
        _require_bounded_string(field_name, value, maximum_bytes)
    return values


def _require_sha256(field_name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")


def _require_non_blank(field_name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value


def _require_bounded_string(field_name: str, value: object, maximum_bytes: int) -> None:
    normalized = _require_non_blank(field_name, value)
    try:
        size = len(normalized.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    if size > maximum_bytes:
        raise ValueError(f"{field_name} must be at most {maximum_bytes} UTF-8 bytes")


def _require_positive_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def _require_non_negative_int(field_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _require_aware(field_name: str, value: object) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


__all__ = [
    "HmacSearchCursorCodec",
    "SearchCursorBinding",
    "SearchCursorPosition",
    "StaleCursorError",
]
