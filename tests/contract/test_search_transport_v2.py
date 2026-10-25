"""Video transport regressions, including the legacy text shape."""

import json

import pytest

from capicove.contracts import (
    CompactHit,
    ImageQuery,
    SearchResult,
    SearchStatus,
    TargetType,
    TextQuery,
)
from capicove.tools.search import decode_search_arguments, encode_search_result


def test_search_spec_declares_multimodal_video_capability() -> None:
    from capicove.tools.search import CAPICOVE_SEARCH_SPEC

    assert CAPICOVE_SEARCH_SPEC.contract_version == 3
    assert set(CAPICOVE_SEARCH_SPEC.modalities) == {"text", "image"}
    assert "video" in CAPICOVE_SEARCH_SPEC.description.lower()


def test_legacy_and_tagged_text() -> None:
    legacy = decode_search_arguments({"query": "query"})
    assert legacy.targets == frozenset({TargetType.TEXT_CHUNK})
    tagged = decode_search_arguments(
        {"query": {"type": "text", "text": "query"}, "targets": ["keyframe"]}
    )
    assert tagged.query == TextQuery("query")
    assert tagged.targets == frozenset({TargetType.KEYFRAME})


def test_image_source_is_closed() -> None:
    query = {"type": "image", "source": {"type": "artifact", "artifact_ref": "a"}}
    request = decode_search_arguments({"query": query, "targets": ["video_segment"]})
    assert isinstance(request.query, ImageQuery)
    query["source"]["provider"] = "unexpected"  # type: ignore[index]
    with pytest.raises(ValueError):
        decode_search_arguments({"query": query, "targets": ["video_segment"]})


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": {"type": "text", "text": "x"}},
        {"query": "x", "targets": ["keyframe"]},
        {"query": {"type": "text", "text": "x"}, "targets": ["text_chunk", "keyframe"]},
        {"query": {"type": "video_clip", "source": {}}, "targets": ["keyframe"]},
        {"query": {"type": "text", "text": "x", "model": "x"}, "targets": ["keyframe"]},
    ],
)
def test_rejects_ambiguous_or_unsupported_queries(arguments: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        decode_search_arguments(arguments)


def test_preview_required_at_v3_transport_but_not_v2_construction() -> None:
    legacy = CompactHit("k", TargetType.KEYFRAME, 0.5, "e", timestamp_ms=5)
    with pytest.raises(ValueError, match="preview"):
        encode_search_result(SearchResult(SearchStatus.SUCCEEDED, "q", hits=(legacy,)))
    hit = CompactHit(
        "k",
        TargetType.KEYFRAME,
        0.5,
        "e",
        timestamp_ms=5,
        preview_resource_ref="resource:preview:k",
    )
    payload = encode_search_result(SearchResult(SearchStatus.SUCCEEDED, "q", hits=(hit,)))
    assert payload["hits"] == [
        {
            "id": "k",
            "target_type": "keyframe",
            "brief_score": 0.5,
            "evidence_id": "e",
            "timestamp_ms": 5,
            "preview_resource_ref": "resource:preview:k",
        }
    ]
    assert len(json.dumps(payload).encode()) < 16384


def test_video_time_filter_round_trip() -> None:
    from capicove.contracts import TemporalRange

    request = decode_search_arguments(
        {
            "query": {"type": "text", "text": "slide"},
            "targets": ["video_segment"],
            "time_range": {"start_ms": 500, "end_ms": 1500},
        }
    )
    assert request.filters.time_range == TemporalRange(500, 1500)
