"""Independent modality wrappers preserve their canonical Tool boundaries."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SearchRequest,
    SearchResult,
    SearchStatus,
    ToolErrorCode,
    ToolStatus,
)
from capicove.retrieval.search_models import InvocationAuthorizationContext
from capicove.search import search_image, search_text, search_video
from capicove.tools.image_search import StaticSearchAuthorizationFactory


class _TextOperation:
    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResult:
        self.requests.append(request)
        return SearchResult(SearchStatus.SUCCEEDED, "text-query")


def _context(*, permitted: bool = True) -> InvocationContext:
    return InvocationContext(
        request_id="request",
        trace_id="trace",
        workspace_id="workspace",
        actor_id="actor",
        environment_id="environment",
        deadline=datetime.now(UTC) + timedelta(seconds=30),
        permissions=frozenset({"evidence:search"}) if permitted else frozenset(),
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def test_text_wrapper_preserves_legacy_string_query_and_identity() -> None:
    operation = _TextOperation()
    result = search_text(_context(), {"query": "sample"}, operation=operation)
    assert result.status is ToolStatus.SUCCEEDED
    assert result.tool_id == "capicove.search"
    assert len(operation.requests) == 1


def test_text_wrapper_rejects_image_shape_without_calling_text_backend() -> None:
    operation = _TextOperation()
    result = search_text(_context(), {"query": {"modality": "image"}}, operation=operation)
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT
    assert operation.requests == []


def test_text_wrapper_does_not_grant_permissions() -> None:
    operation = _TextOperation()
    result = search_text(_context(permitted=False), {"query": "sample"}, operation=operation)
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert operation.requests == []


def test_video_wrapper_is_explicitly_unavailable_and_not_a_text_fallback() -> None:
    result = search_video(_context(), {"query": "sample"})
    assert result.status is ToolStatus.REJECTED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.UNAVAILABLE
    assert result.result is None
    assert result.request_id == "request"
    assert result.tool_id == "capicove.search.video"


class _ImageOperation:
    def __init__(self) -> None:
        self.authorizations: list[InvocationAuthorizationContext] = []

    def search(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        self.authorizations.append(authorization)
        return SearchResult(SearchStatus.SUCCEEDED, "image-query")


def test_image_wrapper_requires_image_permission_and_preserves_authorization() -> None:
    operation = _ImageOperation()
    factory = StaticSearchAuthorizationFactory(
        authorization_scope_ref="scope:images",
        permissions=frozenset({"retrieval:search"}),
        allowed_data_policy_refs=frozenset({"local-only"}),
        allowed_source_root_refs=frozenset(),
    )
    arguments: dict[str, object] = {
        "query": {"type": "image", "source": {"type": "artifact", "artifact_ref": "image:1"}},
        "targets": ["image"],
    }
    denied = search_image(_context(), arguments, service=operation, authorization_factory=factory)
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.PERMISSION_DENIED
    assert operation.authorizations == []
    context = replace(_context(), permissions=frozenset({"retrieval:search"}))
    accepted = search_image(context, arguments, service=operation, authorization_factory=factory)
    assert accepted.status is ToolStatus.SUCCEEDED
    assert accepted.tool_id == "capicove.search.image"
    assert operation.authorizations[0].operation == accepted.tool_id
    assert operation.authorizations[0].workspace_id == context.workspace_id


def test_text_and_image_tools_can_coexist_without_overwriting_registration() -> None:
    from capicove.tools import ToolRegistry
    from capicove.tools.image_search import SEARCH_SPEC
    from capicove.tools.image_search import SearchToolHandler as ImageHandler
    from capicove.tools.search import CAPICOVE_SEARCH_SPEC
    from capicove.tools.search import SearchToolHandler as TextHandler

    registry = ToolRegistry()
    text_handler = TextHandler(_TextOperation())
    image_handler = ImageHandler(
        _ImageOperation(),
        StaticSearchAuthorizationFactory(
            authorization_scope_ref="scope:images",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset(),
        ),
    )
    registry.register(CAPICOVE_SEARCH_SPEC, text_handler)
    registry.register(SEARCH_SPEC, image_handler)
    assert registry.resolve("capicove.search").handler is text_handler
    assert registry.resolve("capicove.search.image").handler is image_handler
