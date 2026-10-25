"""Closed wire codec and permission boundary of ``capicove.search.multimodal``."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts import InvocationContext, PrivacyClass, ToolStatus
from capicove.contracts.retrieval import (
    AudioClipQuery,
    CompactHit,
    DetailLevel,
    InlineBytesSource,
    LocalPathSource,
    SearchRequest,
    SearchResult,
    SearchStatus,
    TargetType,
    TemporalRange,
    TextQuery,
)
from capicove.tools.multimodal_search import (
    CAPICOVE_MULTIMODAL_EVIDENCE_SPEC,
    CAPICOVE_MULTIMODAL_SEARCH_SPEC,
    MultimodalSearchToolHandler,
    decode_multimodal_arguments,
    encode_multimodal_result,
)


def _context(*permissions: str) -> InvocationContext:
    return InvocationContext(
        request_id="r",
        trace_id="t",
        workspace_id="w",
        actor_id="a",
        environment_id="e",
        permissions=frozenset(permissions),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        deadline=datetime.now(UTC) + timedelta(seconds=30),
    )


def test_specs_are_read_only_offline_and_permissioned() -> None:
    assert CAPICOVE_MULTIMODAL_SEARCH_SPEC.tool_id == "capicove.search.multimodal"
    assert CAPICOVE_MULTIMODAL_SEARCH_SPEC.required_permissions == frozenset({"evidence:search"})
    assert CAPICOVE_MULTIMODAL_SEARCH_SPEC.network_access is False
    assert CAPICOVE_MULTIMODAL_EVIDENCE_SPEC.tool_id == "capicove.multimodal.get_evidence"
    assert CAPICOVE_MULTIMODAL_EVIDENCE_SPEC.required_permissions == frozenset({"evidence:read"})


def test_decodes_text_and_audio_queries() -> None:
    text = decode_multimodal_arguments(
        {"query": {"type": "text", "text": "dog"}, "targets": ["audio_segment"]},
        profile="audio-local-v1",
    )
    assert text.query == TextQuery("dog")
    assert text.retrieval_profile_ref == "audio-local-v1"
    audio = decode_multimodal_arguments(
        {
            "query": {
                "type": "audio_clip",
                "path": "/media/a.wav",
                "start_ms": 1000,
                "end_ms": 2000,
                "stream_index": 1,
            },
            "targets": ["audio_segment"],
            "detail_level": "full",
            "top_k": 5,
            "time_range": {"start_ms": 0, "end_ms": 10},
        },
        profile="audio-local-v1",
    )
    assert audio.query == AudioClipQuery(
        LocalPathSource("/media/a.wav"), TemporalRange(1000, 2000), 1
    )
    assert audio.detail_level is DetailLevel.FULL and audio.top_k == 5
    inline = decode_multimodal_arguments(
        {
            "query": {
                "type": "audio_clip",
                "data_base64": base64.b64encode(b"RIFF").decode(),
                "media_type": "audio/wav",
            },
            "targets": ["audio_segment"],
        },
        profile="audio-local-v1",
    )
    assert inline.query == AudioClipQuery(InlineBytesSource(b"RIFF", "audio/wav"))


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "dog", "targets": ["audio_segment"]},
        {"query": {"type": "text", "text": "dog"}},
        {"query": {"type": "text", "text": "dog"}, "targets": ["audio_segment"], "extra": 1},
        {"query": {"type": "text", "text": "dog", "model": "x"}, "targets": ["audio_segment"]},
        {
            "query": {"type": "audio_clip", "path": "/a.wav", "start_ms": 5},
            "targets": ["audio_segment"],
        },
        {"query": {"type": "audio_clip"}, "targets": ["audio_segment"]},
        {
            "query": {"type": "audio_clip", "path": "/a.wav", "data_base64": "UklGRg=="},
            "targets": ["audio_segment"],
        },
        {
            "query": {"type": "audio_clip", "data_base64": "%%%", "media_type": "audio/wav"},
            "targets": ["audio_segment"],
        },
        {"query": {"type": "video_clip", "path": "/a.mp4"}, "targets": ["audio_segment"]},
        {
            "query": {"type": "text", "text": "dog"},
            "targets": ["audio_segment"],
            "retrieval_profile_ref": "other",
        },
        {"query": {"type": "text", "text": "dog"}, "targets": ["audio_segment"], "top_k": True},
    ],
)
def test_rejects_unknown_fields_selection_and_partial_ranges(arguments: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        decode_multimodal_arguments(arguments, profile="audio-local-v1")


class _Recording:
    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResult:
        self.requests.append(request)
        return SearchResult(
            SearchStatus.SUCCEEDED,
            "q",
            hits=(
                CompactHit(
                    "u", TargetType.AUDIO_SEGMENT, 1.0, "mev-x", time_range=TemporalRange(0, 30)
                ),
            ),
        )


def test_handler_denies_without_search_permission_and_never_dispatches() -> None:
    operation = _Recording()
    handler = MultimodalSearchToolHandler(operation, profile="audio-local-v1")
    denied = handler(
        _context("evidence:read"),
        {"query": {"type": "text", "text": "q"}, "targets": ["audio_segment"]},
    )
    assert denied.status is ToolStatus.FAILED and operation.requests == []
    allowed = handler(
        _context("evidence:search"),
        {"query": {"type": "text", "text": "q"}, "targets": ["audio_segment"]},
    )
    assert allowed.status is ToolStatus.SUCCEEDED
    assert allowed.result["hits"] == [  # type: ignore[index]
        {
            "id": "u",
            "target_type": "audio_segment",
            "brief_score": 1.0,
            "evidence_id": "mev-x",
            "time_range": {"start_ms": 0, "end_ms": 30},
        }
    ]
    assert allowed.evidence_refs == ("mev-x",)


def test_failed_results_keep_canonical_failure_codes() -> None:
    from capicove.contracts.retrieval import SearchFailureCode

    payload = encode_multimodal_result(
        SearchResult(SearchStatus.FAILED, "q", failure_code=SearchFailureCode.INVALID_MEDIA)
    )
    assert payload == {
        "status": "failed",
        "query_id": "q",
        "degraded": False,
        "warning_codes": [],
        "hits": [],
        "failure_code": "invalid_media",
    }


def test_decodes_image_queries_by_path_or_inline_bytes() -> None:
    from capicove.contracts.retrieval import ImageQuery

    by_path = decode_multimodal_arguments(
        {"query": {"type": "image", "path": "/media/red.png"}, "targets": ["image"]},
        profile="unified-local-v1",
    )
    assert by_path.query == ImageQuery(LocalPathSource("/media/red.png"))
    inline = decode_multimodal_arguments(
        {
            "query": {
                "type": "image",
                "data_base64": base64.b64encode(b"\x89PNG").decode(),
                "media_type": "image/png",
            },
            "targets": ["image", "text_chunk"],
        },
        profile="unified-local-v1",
    )
    assert inline.query == ImageQuery(InlineBytesSource(b"\x89PNG", "image/png"))
    for bad in (
        {"type": "image"},
        {"type": "image", "path": "/a.png", "start_ms": 1, "end_ms": 2},
        {"type": "image", "path": "/a.png", "data_base64": "eA=="},
    ):
        with pytest.raises((TypeError, ValueError)):
            decode_multimodal_arguments(
                {"query": bad, "targets": ["image"]}, profile="unified-local-v1"
            )
