"""Tests for text-search CLI argument, execution, and output parity."""

import json

import pytest

from capicove.cli.app import main
from capicove.cli.commands.search import arguments_to_search, encode_search_tool_result
from capicove.contracts import (
    SearchRequest,
    SearchResult,
    TargetType,
    TemporalRange,
    TextOrigin,
    TextQuery,
)
from capicove.tools.search import SearchToolHandler, decode_search_arguments
from tests.unit.tools.test_search import SearchOperation, context


def test_cli_maps_text_arguments_and_preserves_tool_result() -> None:
    arguments = arguments_to_search("matching", top_k=5, profile="local-default")
    result = SearchToolHandler(SearchOperation())(context(), arguments)

    assert arguments == {
        "query": "matching",
        "top_k": 5,
        "retrieval_profile_ref": "local-default",
    }
    assert encode_search_tool_result(result)["result"]["query_id"] == "query-1"  # type: ignore[index]


def test_cli_executes_the_canonical_search_tool(capsys: object) -> None:
    handler = SearchToolHandler(SearchOperation())

    exit_code = main(
        ["search", "matching", "--top-k", "5", "--profile", "local-default"],
        search_handler=handler,
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["tool_id"] == "capicove.search"
    assert output["result"]["query_id"] == "query-1"


def test_cli_maps_image_video_targets_without_reading_the_file() -> None:
    from capicove.contracts import ImageQuery, LocalPathSource, TargetType
    from capicove.tools.search import decode_search_arguments

    values = arguments_to_search(
        None, top_k=3, profile=None, image="/not-opened/frame.png", targets=("keyframe",)
    )
    request = decode_search_arguments(values)
    assert request.query == ImageQuery(LocalPathSource("/not-opened/frame.png"))
    assert request.targets == frozenset({TargetType.KEYFRAME})
    assert request.retrieval_profile_ref == "video-local-exact"


def test_cli_maps_text_video_targets() -> None:
    values = arguments_to_search("slide", top_k=3, profile=None, targets=("video_segment",))
    assert decode_search_arguments(values).query == TextQuery("slide")


def test_cli_projects_canonical_search_filters_and_nested_language() -> None:
    values = arguments_to_search(
        "lecture",
        top_k=10,
        profile="video-local-exact",
        targets=("video_segment",),
        language="en",
        time_range=(1_000, 9_000),
        corpus_refs=("course-a", "course-a"),
        artifact_refs=("artifact-1",),
        media_types=("video/mp4",),
        text_origins=("asr", "ocr"),
    )

    assert values == {
        "query": {"type": "text", "text": "lecture", "language": "en"},
        "top_k": 10,
        "retrieval_profile_ref": "video-local-exact",
        "targets": ["video_segment"],
        "time_range": {"start_ms": 1_000, "end_ms": 9_000},
        "corpus_refs": ["course-a"],
        "artifact_refs": ["artifact-1"],
        "media_types": ["video/mp4"],
        "text_origins": ["asr", "ocr"],
    }
    request = decode_search_arguments(values)
    assert request.query == TextQuery("lecture", "en")
    assert request.targets == frozenset({TargetType.VIDEO_SEGMENT})
    assert request.filters.time_range == TemporalRange(1_000, 9_000)
    assert request.filters.text_origins == frozenset({TextOrigin.ASR, TextOrigin.OCR})


@pytest.mark.parametrize("time_range", [(1_000, None), (None, 9_000)])
def test_cli_rejects_incomplete_range(
    time_range: tuple[int | None, int | None],
) -> None:
    with pytest.raises(ValueError, match="--start-ms and --end-ms"):
        arguments_to_search("lecture", top_k=5, profile=None, time_range=time_range)


def test_cli_rejects_language_for_image_query() -> None:
    with pytest.raises(ValueError, match="language"):
        arguments_to_search(
            None,
            top_k=5,
            profile=None,
            image="/image.png",
            targets=("keyframe",),
            language="en",
        )


def test_cli_rejects_empty_tagged_language_and_image_text_target() -> None:
    with pytest.raises(ValueError, match="language"):
        arguments_to_search(
            "lecture", top_k=5, profile=None, targets=("video_segment",), language=""
        )
    with pytest.raises(ValueError, match="text_chunk"):
        arguments_to_search(
            None,
            top_k=5,
            profile=None,
            image="/image.png",
            targets=("text_chunk",),
        )


def test_cli_supports_text_chunk_and_legacy_language() -> None:
    tagged = arguments_to_search(
        "lecture", top_k=5, profile=None, targets=("text_chunk",), language="en"
    )
    legacy = arguments_to_search("lecture", top_k=5, profile=None, language="en")

    assert tagged["retrieval_profile_ref"] == "local-default"
    assert decode_search_arguments(tagged).targets == frozenset({TargetType.TEXT_CHUNK})
    assert decode_search_arguments(tagged).query == TextQuery("lecture", "en")
    assert legacy["language"] == "en"
    assert decode_search_arguments(legacy).query == TextQuery("lecture", "en")


def test_cli_rejects_mixed_text_and_video_targets() -> None:
    with pytest.raises(ValueError, match="text_chunk cannot be combined"):
        arguments_to_search(
            "lecture",
            top_k=5,
            profile=None,
            targets=("text_chunk", "video_segment"),
        )


def test_cli_forwards_filters_to_search_operation(capsys: pytest.CaptureFixture[str]) -> None:
    class RecordingOperation(SearchOperation):
        request: SearchRequest | None = None

        def search(self, request: object) -> SearchResult:
            assert isinstance(request, SearchRequest)
            self.request = request
            return super().search(request)

    operation = RecordingOperation()
    exit_code = main(
        [
            "search",
            "lecture",
            "--target",
            "video_segment",
            "--language",
            "en",
            "--start-ms",
            "1000",
            "--end-ms",
            "9000",
            "--corpus-ref",
            "course-a",
            "--text-origin",
            "asr",
        ],
        search_handler=SearchToolHandler(operation),
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"
    assert operation.request is not None
    assert operation.request.query == TextQuery("lecture", "en")
    assert operation.request.filters.time_range == TemporalRange(1_000, 9_000)
