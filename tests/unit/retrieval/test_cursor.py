"""Tests for privacy-bounded signed reverse-image-search cursors."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts.retrieval import MAX_CURSOR_BYTES, MAX_REF_BYTES
from capicove.retrieval.cursor import (
    HmacSearchCursorCodec,
    SearchCursorBinding,
    SearchCursorPosition,
    StaleCursorError,
)


NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64


def _binding() -> SearchCursorBinding:
    return SearchCursorBinding(
        query_sha256=SHA_A,
        workspace_id="workspace-1",
        targets=("image",),
        filters_sha256=SHA_B,
        search_intent="balanced",
        retrieval_profile_ref="personal-lite",
        data_policy_ref="local-only",
        generation_ids=("generation-a", "generation-b"),
    )


def _position() -> SearchCursorPosition:
    return SearchCursorPosition(
        binding=_binding(),
        last_exact_content=False,
        last_fused_score_hex=(0.125).hex(),
        last_best_lane_rank=3,
        last_unit_id="unit-9",
        ranking_state_sha256="c" * 64,
        consumed_count=20,
        page_size=20,
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )


def test_cursor_is_bound_to_query_generation_sort_key_and_expiry() -> None:
    codec = HmacSearchCursorCodec(signing_key=b"k" * 32, workspace_id="workspace-1")
    cursor = codec.encode(_position())

    assert codec.decode(cursor, expected=_binding(), now=NOW) == _position()

    body, signature = cursor.split(".")
    tampered = f"{body[:-1]}{'A' if body[-1] != 'A' else 'B'}.{signature}"
    with pytest.raises(StaleCursorError):
        codec.decode(tampered, expected=_binding(), now=NOW)
    with pytest.raises(StaleCursorError):
        codec.decode(
            cursor,
            expected=replace(_binding(), generation_ids=("generation-new",)),
            now=NOW,
        )
    with pytest.raises(StaleCursorError):
        codec.decode(cursor, expected=_binding(), now=NOW + timedelta(minutes=6))


def test_cursor_rejects_wrong_workspace_query_filter_and_policy() -> None:
    cursor = HmacSearchCursorCodec(
        signing_key=b"k" * 32,
        workspace_id="workspace-1",
    ).encode(_position())

    mismatches = (
        replace(_binding(), workspace_id="workspace-2"),
        replace(_binding(), query_sha256="c" * 64),
        replace(_binding(), targets=("keyframe",)),
        replace(_binding(), filters_sha256="d" * 64),
        replace(_binding(), search_intent="instance"),
        replace(_binding(), retrieval_profile_ref="another-profile"),
        replace(_binding(), data_policy_ref="another-policy"),
        replace(_binding(), generation_ids=("generation-new",)),
    )
    codec = HmacSearchCursorCodec(signing_key=b"k" * 32, workspace_id="workspace-1")
    for binding in mismatches:
        with pytest.raises(StaleCursorError):
            codec.decode(cursor, expected=binding, now=NOW)


def test_cursor_rejects_malformed_oversized_and_invalid_positions() -> None:
    codec = HmacSearchCursorCodec(signing_key=b"k" * 32, workspace_id="workspace-1")
    for cursor in (
        "",
        "missing-dot",
        ".",
        "a.b.c",
        "x" * (MAX_CURSOR_BYTES + 1),
    ):
        with pytest.raises(StaleCursorError):
            codec.decode(cursor, expected=_binding(), now=NOW)

    with pytest.raises(ValueError, match="32 bytes"):
        HmacSearchCursorCodec(signing_key=b"short", workspace_id="workspace-1")
    with pytest.raises(ValueError, match="consumed_count"):
        replace(_position(), consumed_count=-1)
    with pytest.raises(ValueError, match="later"):
        replace(_position(), expires_at=NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match="128 UTF-8 bytes"):
        replace(_position(), last_unit_id="u" * 129)
    with pytest.raises(ValueError, match="1024 UTF-8 bytes"):
        replace(_binding(), retrieval_profile_ref="r" * (MAX_REF_BYTES + 1))


def test_cursor_encoding_is_canonical_and_never_contains_signing_key() -> None:
    key = b"private-signing-key-material-1234"
    codec = HmacSearchCursorCodec(signing_key=key, workspace_id="workspace-1")

    first = codec.encode(_position())
    second = codec.encode(_position())

    assert first == second
    assert key.decode() not in first
    assert len(first.encode("ascii")) <= MAX_CURSOR_BYTES


def test_maximum_canonical_binding_encodes_below_the_result_cursor_limit() -> None:
    binding = replace(
        _binding(),
        workspace_id="w" * MAX_REF_BYTES,
        retrieval_profile_ref="r" * MAX_REF_BYTES,
        data_policy_ref="p" * MAX_REF_BYTES,
    )
    position = replace(
        _position(),
        binding=binding,
        last_unit_id="u" * 128,
    )
    codec = HmacSearchCursorCodec(
        signing_key=b"k" * 32,
        workspace_id=binding.workspace_id,
    )

    cursor = codec.encode(position)

    assert len(cursor.encode("ascii")) <= MAX_CURSOR_BYTES
    assert codec.decode(cursor, expected=binding, now=NOW) == position
