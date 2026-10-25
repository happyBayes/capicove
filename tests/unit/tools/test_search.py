"""Tests for the canonical text search Tool boundary."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast

from capicove.contracts import (
    CharacterRange,
    CompactHit,
    InvocationContext,
    PrivacyClass,
    SearchResult,
    SearchStatus,
    TargetType,
    TextChunkLocator,
    TextOrigin,
    ToolStatus,
)
from capicove.retrieval import TextSearchError
from capicove.tools.search import CAPICOVE_SEARCH_SPEC, SearchToolHandler


class SearchOperation:
    """Return one configured result or stable error."""

    def __init__(self, failure: TextSearchError | None = None) -> None:
        self.failure = failure

    def search(self, request: object) -> SearchResult:
        if self.failure is not None:
            raise self.failure
        return SearchResult(
            status=SearchStatus.SUCCEEDED,
            query_id="query-1",
            hits=(
                CompactHit(
                    id="unit-1",
                    target_type=TargetType.TEXT_CHUNK,
                    brief_score=0.75,
                    evidence_id="evidence-1",
                    text_locator=TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, 8)),
                    snippet="matching",
                ),
            ),
        )


def context() -> InvocationContext:
    """Return an authorized search invocation."""
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        deadline=datetime.now(UTC) + timedelta(minutes=1),
        permissions=frozenset({"evidence:search"}),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def test_search_tool_propagates_absolute_deadline() -> None:
    class DeadlineOperation(SearchOperation):
        deadline: datetime | None = None

        def search_with_deadline(self, request: object, *, deadline: datetime) -> SearchResult:
            self.deadline = deadline
            return super().search(request)

    operation = DeadlineOperation()
    invocation = context()
    result = SearchToolHandler(operation)(invocation, {"query": "matching"})
    assert result.status is ToolStatus.SUCCEEDED
    assert operation.deadline == invocation.deadline


def test_search_tool_projects_canonical_text_hits_without_provider_knobs() -> None:
    handler = SearchToolHandler(SearchOperation())

    result = handler(
        context(),
        {"query": "matching", "top_k": 5, "retrieval_profile_ref": "local-default"},
    )

    assert CAPICOVE_SEARCH_SPEC.tool_id == "capicove.search"
    properties = cast(Mapping[str, object], CAPICOVE_SEARCH_SPEC.input_schema["properties"])
    assert "provider" not in properties
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result == {
        "status": "succeeded",
        "query_id": "query-1",
        "degraded": False,
        "warning_codes": [],
        "hits": [
            {
                "id": "unit-1",
                "target_type": "text_chunk",
                "brief_score": 0.75,
                "evidence_id": "evidence-1",
                "text_locator": {
                    "origin": "document",
                    "start_char": 0,
                    "end_char": 8,
                    "page_number": None,
                    "start_ms": None,
                    "end_ms": None,
                },
                "snippet": "matching",
            }
        ],
    }


def test_search_tool_maps_stable_unavailability_without_provider_detail() -> None:
    handler = SearchToolHandler(SearchOperation(TextSearchError("provider_unavailable")))

    result = handler(context(), {"query": "matching"})

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code.value == "unavailable"
    assert result.error.message == "Text retrieval is unavailable."
