"""Tests for the canonical offline ``capicove.search`` Tool boundary."""

from __future__ import annotations

import base64
import copy
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

import capicove.tools.image_search as search_tool_module
from capicove.contracts import (
    MAX_CURSOR_BYTES,
    MAX_FILTER_VALUES,
    MAX_INLINE_IMAGE_BYTES,
    MAX_LANGUAGE_BYTES,
    MAX_MEDIA_TYPE_BYTES,
    MAX_PATH_BYTES,
    MAX_REF_BYTES,
    MAX_TEMPORAL_MILLISECONDS,
    MAX_TEXT_BYTES,
    CompactHit,
    DetailLevel,
    DistanceMetric,
    EmbeddingSpaceRef,
    ExecutionProfile,
    FullHit,
    HashSpaceRef,
    InvocationContext,
    LaneScore,
    MatchKind,
    PrivacyClass,
    ProcessingFingerprint,
    QueryModality,
    SearchExecutionSummary,
    SearchFailureCode,
    SearchIntent,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    SideEffect,
    TargetType,
    TemporalRange,
    ToolErrorCode,
    ToolResources,
    ToolStatus,
    VectorNormalization,
)
from capicove.retrieval.search_models import InvocationAuthorizationContext
from capicove.tools import ToolRegistry
from capicove.tools.image_search import (
    MAX_COMPACT_OUTPUT_BYTES,
    MAX_EVIDENCE_ID_BYTES,
    MAX_HIT_ID_BYTES,
    MAX_INLINE_BASE64_CHARS,
    SEARCH_INPUT_SCHEMA,
    SEARCH_OUTPUT_SCHEMA,
    SEARCH_SPEC,
    SearchAuthorizationFactory,
    SearchToolHandler,
    StaticSearchAuthorizationFactory,
    register_search_tool,
    validate_serialized_search_output,
)
from capicove.traces import TraceSanitizer


NOW = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(seconds=30)


def _serialized_compact_output() -> dict[str, object]:
    return {
        "status": "succeeded",
        "query_id": "query-1",
        "degraded": False,
        "warning_codes": [],
        "hits": [
            {
                "id": "image-1",
                "target_type": "image",
                "brief_score": 0.9,
                "evidence_id": "evidence-1",
                "timestamp_ms": None,
                "time_range": None,
            }
        ],
        "next_cursor": None,
        "failure_code": None,
        "execution_summary": None,
    }


def _serialized_full_output() -> dict[str, object]:
    embedding_space = {
        "space_id": "space-1",
        "provider_id": "provider-1",
        "model_revision": "revision-1",
        "preprocessing_version": "preprocess-v1",
        "dimension": 2,
        "normalization": "l2",
        "distance_metric": "cosine",
        "query_modalities": ["image"],
        "target_modalities": ["image"],
        "schema_version": 1,
    }
    hash_space = {
        "space_id": "opencv-phash-64-v1",
        "provider_id": "opencv-phash",
        "provider_version": "4.12.0",
        "preprocessing_version": "preprocess-v1",
        "algorithm": "phash",
        "bit_width": 64,
        "hash_version": "phash-64-v1",
        "target_modalities": ["image"],
        "schema_version": 1,
    }
    return {
        "status": "succeeded",
        "query_id": "query-1",
        "degraded": True,
        "warning_codes": ["retrieval_preview"],
        "hits": [
            {
                "id": "image-1",
                "target_type": "image",
                "brief_score": 0.9,
                "evidence_id": "evidence-1",
                "timestamp_ms": None,
                "time_range": None,
                "artifact_ref": "artifact-1",
                "searchable_unit_ref": "image-1",
                "representation_refs": ["representation-1"],
                "scene_range": None,
                "match_kinds": ["visual"],
                "lane_scores": [
                    {
                        "lane_id": "instance",
                        "raw_score": 0.8,
                        "normalized_score": 0.7,
                        "score_semantics": "cosine",
                        "calibration_version": "rrf-v1",
                        "rank": 1,
                    }
                ],
                "embedding_spaces": [embedding_space],
                "processing_fingerprint": {"sha256": "a" * 64},
                "index_generation_ids": ["generation-1"],
                "provider_ids": ["provider-1"],
                "execution_profile": "cpu",
                "warning_codes": ["retrieval_preview"],
                "lineage_refs": ["activation-1"],
            }
        ],
        "next_cursor": "cursor-1",
        "failure_code": None,
        "execution_summary": {
            "search_intent": "instance",
            "selected_lanes": ["content", "instance"],
            "provider_ids": ["provider-1"],
            "embedding_spaces": [embedding_space],
            "hash_spaces": [hash_space],
            "index_generation_ids": ["generation-1"],
            "bundle_manifest_sha256s": ["b" * 64],
            "resolved_profile_ref": "personal-lite",
            "execution_profile": "cpu",
            "activation_tier": "preview",
            "schema_version": 1,
        },
    }


def _maximum_compact_page() -> SearchResult:
    hits = tuple(
        CompactHit(
            id=f"{index:03d}-" + "i" * (MAX_HIT_ID_BYTES - 4),
            target_type=TargetType.IMAGE,
            brief_score=0.12345678901234568,
            evidence_id=f"{index:03d}-" + "e" * (MAX_EVIDENCE_ID_BYTES - 4),
        )
        for index in range(100)
    )
    return SearchResult(
        SearchStatus.SUCCEEDED,
        "q" * MAX_REF_BYTES,
        hits=hits,
        next_cursor="c" * MAX_CURSOR_BYTES,
    )


def _maximum_json_escaped_compact_page() -> SearchResult:
    escaped_ref = "\0" * MAX_REF_BYTES
    distinct_escaped_refs = tuple(chr(index) * MAX_REF_BYTES for index in range(1, 5))
    embedding_spaces = tuple(
        EmbeddingSpaceRef(
            space_id=space_id,
            provider_id=escaped_ref,
            model_revision=escaped_ref,
            preprocessing_version=escaped_ref,
            dimension=16_384,
            normalization=VectorNormalization.NONE,
            distance_metric=DistanceMetric.INNER_PRODUCT,
            # Every query modality the image-search output schema admits; version-4
            # audio queries belong to the shared-space Tool, not this contract.
            query_modalities=frozenset(
                {QueryModality.TEXT, QueryModality.IMAGE, QueryModality.VIDEO_CLIP}
            ),
            target_modalities=frozenset(
                {TargetType.IMAGE, TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}
            ),
        )
        for space_id in distinct_escaped_refs
    )
    hash_spaces = tuple(
        HashSpaceRef(
            space_id=space_id,
            provider_id=escaped_ref,
            provider_version=escaped_ref,
            preprocessing_version=escaped_ref,
            algorithm="phash",
            bit_width=64,
            hash_version="phash-64-v1",
            target_modalities=frozenset(
                {TargetType.IMAGE, TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}
            ),
        )
        for space_id in distinct_escaped_refs
    )
    summary = SearchExecutionSummary(
        search_intent=SearchIntent.BALANCED,
        selected_lanes=("content", "perceptual", "instance", "concept"),
        provider_ids=distinct_escaped_refs,
        embedding_spaces=embedding_spaces,
        hash_spaces=hash_spaces,
        index_generation_ids=distinct_escaped_refs,
        bundle_manifest_sha256s=("a" * 64, "b" * 64),
        resolved_profile_ref=distinct_escaped_refs[0],
        execution_profile=ExecutionProfile.AUTO,
        activation_tier="candidate_evaluation",
    )
    hits = tuple(
        CompactHit(
            id="\0" * MAX_HIT_ID_BYTES,
            target_type=TargetType.VIDEO_SEGMENT,
            brief_score=2.2250738585072014e-308,
            evidence_id="\0" * MAX_EVIDENCE_ID_BYTES,
            time_range=TemporalRange(
                MAX_TEMPORAL_MILLISECONDS - 1,
                MAX_TEMPORAL_MILLISECONDS,
            ),
        )
        for _ in range(100)
    )
    return SearchResult(
        SearchStatus.FAILED,
        "\0" * MAX_REF_BYTES,
        warning_codes=tuple(SearchWarningCode),
        hits=hits,
        next_cursor="\0" * MAX_CURSOR_BYTES,
        failure_code=SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE,
        execution_summary=summary,
    )


def test_tools_package_exposes_lazy_search_public_api() -> None:
    from capicove.tools import (
        SEARCH_SPEC as package_search_spec,
    )
    from capicove.tools import (
        ImageSearchToolHandler as package_search_handler,
    )
    from capicove.tools import (
        register_search_tool as package_register_search_tool,
    )

    assert package_search_spec is SEARCH_SPEC
    assert package_search_handler is SearchToolHandler
    assert package_register_search_tool is register_search_tool


class _Service:
    def __init__(self, result: SearchResult | BaseException | None = None) -> None:
        self.result = result or SearchResult(SearchStatus.SUCCEEDED, "query-1")
        self.calls: list[tuple[SearchRequest, InvocationAuthorizationContext, datetime]] = []

    def search(
        self,
        request: SearchRequest,
        *,
        authorization: InvocationAuthorizationContext,
        deadline: datetime,
    ) -> SearchResult:
        self.calls.append((request, authorization, deadline))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _AuthorizationFactory:
    def __init__(self) -> None:
        self.calls: list[InvocationContext] = []

    def create(self, context: InvocationContext) -> InvocationAuthorizationContext:
        self.calls.append(context)
        return InvocationAuthorizationContext(
            request_id=context.request_id,
            workspace_id=context.workspace_id,
            actor_id=context.actor_id,
            environment_id=context.environment_id,
            operation="capicove.search.image",
            authorization_scope_ref="scope:search-test",
            permissions=frozenset({"retrieval:search"}),
            allowed_data_policy_refs=frozenset({"local-only"}),
            allowed_source_root_refs=frozenset({"local:library"}),
            privacy_class=context.privacy_class,
        )


def _context(
    *,
    permissions: frozenset[str] = frozenset({"retrieval:search"}),
    deadline: datetime = FUTURE,
) -> InvocationContext:
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        deadline=deadline,
        permissions=permissions,
        privacy_class=PrivacyClass.CONFIDENTIAL,
    )


def _arguments(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "query": {
            "type": "image",
            "source": {"type": "artifact", "artifact_ref": "artifact:query-1"},
        },
        "targets": ["image"],
    }
    arguments.update(overrides)
    return arguments


def _handler(
    service: _Service,
    factory: SearchAuthorizationFactory | None = None,
) -> SearchToolHandler:
    return SearchToolHandler(service, factory or _AuthorizationFactory(), clock=lambda: NOW)


def test_search_spec_declares_one_offline_read_contract() -> None:
    assert SEARCH_SPEC.tool_id == "capicove.search.image"
    assert SEARCH_SPEC.contract_version == 1
    assert SEARCH_SPEC.required_permissions == frozenset({"retrieval:search"})
    assert SEARCH_SPEC.side_effect is SideEffect.READ
    assert SEARCH_SPEC.resources == ToolResources(filesystem_read=True)
    assert SEARCH_SPEC.network_access is False
    assert SEARCH_SPEC.idempotent is True
    assert SEARCH_SPEC.deterministic is False
    assert SEARCH_INPUT_SCHEMA["additionalProperties"] is False
    assert SEARCH_OUTPUT_SCHEMA["additionalProperties"] is False
    Draft202012Validator.check_schema(SEARCH_INPUT_SCHEMA)
    Draft202012Validator.check_schema(SEARCH_OUTPUT_SCHEMA)
    Draft202012Validator(SEARCH_INPUT_SCHEMA).validate(_arguments())


def test_search_spec_redacts_source_and_filter_identity_shapes_with_parity() -> None:
    assert SEARCH_SPEC.redacted_fields == (
        "query.text",
        "query.source.artifact_ref",
        "query.source.base64",
        "query.source.path",
        "query.source.mount_ref",
        "query.source.relative_path",
        "query.source.media_type",
        "filters.corpus_refs.*",
        "filters.artifact_refs.*",
        "filters.media_types.*",
    )

    sanitized = (
        TraceSanitizer()
        .sanitize(
            SEARCH_SPEC,
            {
                "query": {
                    "type": "image",
                    "source": {
                        "type": "mount",
                        "mount_ref": "mount:private-library",
                        "relative_path": "people/alice.png",
                    },
                },
                "filters": {
                    "corpus_refs": ["corpus:alice"],
                    "artifact_refs": ["artifact:alice"],
                    "media_types": ["image/x-alice-private"],
                },
            },
        )
        .arguments
    )

    assert sanitized["query"]["source"] == {  # type: ignore[index]
        "type": "mount",
        "mount_ref": "[REDACTED]",
        "relative_path": "[REDACTED]",
    }
    assert sanitized["filters"] == {
        "corpus_refs": ("[REDACTED]",),
        "artifact_refs": ("[REDACTED]",),
        "media_types": ("[REDACTED]",),
    }
    assert "alice" not in repr(sanitized)

    artifact = (
        TraceSanitizer()
        .sanitize(
            SEARCH_SPEC,
            {
                "query": {
                    "type": "image",
                    "source": {"type": "artifact", "artifact_ref": "artifact:private"},
                },
                "targets": ["image"],
            },
        )
        .arguments
    )
    inline = (
        TraceSanitizer()
        .sanitize(
            SEARCH_SPEC,
            {
                "query": {
                    "type": "image",
                    "source": {
                        "type": "inline_bytes",
                        "base64": "aGVsbG8=",
                        "media_type": "image/x-private",
                    },
                },
                "targets": ["image"],
            },
        )
        .arguments
    )

    assert artifact["query"]["source"]["artifact_ref"] == "[REDACTED]"  # type: ignore[index]
    assert inline["query"]["source"]["media_type"] == "[REDACTED]"  # type: ignore[index]
    assert "artifact:private" not in repr(artifact)
    assert "image/x-private" not in repr(inline)


def test_input_schema_uses_canonical_contract_caps() -> None:
    properties = SEARCH_INPUT_SCHEMA["properties"]
    assert isinstance(properties, Mapping)
    query = properties["query"]
    filters = properties["filters"]
    assert isinstance(query, Mapping)
    assert isinstance(filters, Mapping)
    variants = query["oneOf"]
    assert isinstance(variants, list)

    text_properties = variants[0]["properties"]
    source_schema = variants[1]["properties"]["source"]
    source_variants = source_schema["oneOf"]
    filter_properties = filters["properties"]

    assert text_properties["text"]["maxLength"] == MAX_TEXT_BYTES
    assert text_properties["language"]["maxLength"] == MAX_LANGUAGE_BYTES
    assert source_variants[0]["properties"]["artifact_ref"]["maxLength"] == MAX_REF_BYTES
    assert source_variants[1]["properties"]["path"]["maxLength"] == MAX_PATH_BYTES
    assert source_variants[2]["properties"]["mount_ref"]["maxLength"] == MAX_REF_BYTES
    assert source_variants[2]["properties"]["relative_path"]["maxLength"] == MAX_PATH_BYTES
    assert source_variants[3]["properties"]["media_type"]["maxLength"] == MAX_MEDIA_TYPE_BYTES
    assert properties["cursor"]["maxLength"] == MAX_CURSOR_BYTES
    assert properties["retrieval_profile_ref"]["maxLength"] == MAX_REF_BYTES
    assert properties["data_policy_ref"]["maxLength"] == MAX_REF_BYTES
    for name, maximum_bytes in (
        ("corpus_refs", MAX_REF_BYTES),
        ("artifact_refs", MAX_REF_BYTES),
        ("media_types", MAX_MEDIA_TYPE_BYTES),
    ):
        assert filter_properties[name]["maxItems"] == MAX_FILTER_VALUES
        assert filter_properties[name]["items"]["maxLength"] == maximum_bytes
    assert filter_properties["time_range"]["properties"]["start_ms"]["maximum"] == (
        MAX_TEMPORAL_MILLISECONDS
    )
    output_properties = cast(Mapping[str, Any], SEARCH_OUTPUT_SCHEMA["properties"])
    failure_code = cast(Mapping[str, Any], output_properties["failure_code"])
    assert set(failure_code["anyOf"][0]["enum"]) == {code.value for code in SearchFailureCode}
    compact_hit = output_properties["hits"]["items"]["oneOf"][0]
    assert compact_hit["properties"]["timestamp_ms"]["maximum"] == (MAX_TEMPORAL_MILLISECONDS)
    assert (
        compact_hit["properties"]["time_range"]["anyOf"][0]["properties"]["end_ms"]["maximum"]
        == MAX_TEMPORAL_MILLISECONDS
    )


@pytest.mark.parametrize(
    "output_factory",
    [_serialized_compact_output, _serialized_full_output],
)
def test_serialized_output_validator_accepts_every_closed_output_branch(
    output_factory: Callable[[], dict[str, object]],
) -> None:
    output = output_factory()

    validate_serialized_search_output(output)

    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(output)


def test_serialized_output_validator_accepts_exact_top_level_boundaries() -> None:
    output = _serialized_compact_output()
    output["query_id"] = "q" * MAX_REF_BYTES
    output["next_cursor"] = "c" * MAX_CURSOR_BYTES
    template = cast(list[dict[str, object]], output["hits"])[0]
    output["hits"] = [
        {
            **template,
            "id": f"{index:03d}-" + "i" * (MAX_HIT_ID_BYTES - 4),
            "evidence_id": f"{index:03d}-" + "e" * (MAX_EVIDENCE_ID_BYTES - 4),
        }
        for index in range(100)
    ]

    validate_serialized_search_output(output)

    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(output)


def test_serialized_output_validator_accepts_non_null_keyframe_timestamp() -> None:
    output = _serialized_compact_output()
    hit = cast(list[dict[str, object]], output["hits"])[0]
    hit["target_type"] = "keyframe"
    hit["timestamp_ms"] = 0

    canonical = validate_serialized_search_output(output)

    assert cast(list[dict[str, object]], canonical["hits"])[0]["timestamp_ms"] == 0
    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(canonical)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_top_field",
        "long_query_id",
        "non_boolean_degraded",
        "duplicate_warnings",
        "too_many_hits",
        "long_cursor",
        "failure_status_mismatch",
        "non_object_hit",
        "compact_long_id",
        "compact_open_shape",
        "compact_invalid_time_range",
        "non_numeric_brief_score",
        "out_of_range_brief_score",
        "too_many_representation_refs",
        "unknown_match_kind",
        "non_finite_lane_score",
        "lane_rank_overflow",
        "embedding_dimension_overflow",
        "invalid_hash_version",
        "invalid_processing_fingerprint",
        "duplicate_summary_lanes",
        "invalid_bundle_digest",
        "invalid_activation_tier",
    ],
)
def test_serialized_output_validator_rejects_nested_schema_and_canonical_violations(
    case: str,
) -> None:
    output = cast(dict[str, Any], copy.deepcopy(_serialized_full_output()))
    hits = cast(list[dict[str, Any]], output["hits"])
    hit = hits[0]
    summary = cast(dict[str, Any], output["execution_summary"])

    if case == "unknown_top_field":
        output["unknown"] = True
    elif case == "long_query_id":
        output["query_id"] = "界" * 342
    elif case == "non_boolean_degraded":
        output["degraded"] = 1
    elif case == "duplicate_warnings":
        output["warning_codes"] = ["retrieval_preview", "retrieval_preview"]
    elif case == "too_many_hits":
        output["hits"] = [copy.deepcopy(hit) for _ in range(101)]
    elif case == "long_cursor":
        output["next_cursor"] = "界" * 1_366
    elif case == "failure_status_mismatch":
        output["status"] = "failed"
    elif case == "non_object_hit":
        output["hits"] = ["not-an-object"]
    elif case == "compact_long_id":
        compact = _serialized_compact_output()
        compact_hit = cast(list[dict[str, Any]], compact["hits"])[0]
        compact_hit["id"] = "界" * 43
        output = cast(dict[str, Any], compact)
    elif case == "compact_open_shape":
        compact = _serialized_compact_output()
        compact_hit = cast(list[dict[str, Any]], compact["hits"])[0]
        compact_hit["unknown"] = True
        output = cast(dict[str, Any], compact)
    elif case == "compact_invalid_time_range":
        compact = _serialized_compact_output()
        compact_hit = cast(list[dict[str, Any]], compact["hits"])[0]
        compact_hit.update(
            target_type="video_segment",
            time_range={"start_ms": 2, "end_ms": 1},
        )
        output = cast(dict[str, Any], compact)
    elif case == "non_numeric_brief_score":
        hit["brief_score"] = True
    elif case == "out_of_range_brief_score":
        hit["brief_score"] = 1.01
    elif case == "too_many_representation_refs":
        hit["representation_refs"] = [f"representation-{index}" for index in range(101)]
    elif case == "unknown_match_kind":
        hit["match_kinds"] = ["unknown"]
    elif case == "non_finite_lane_score":
        cast(list[dict[str, Any]], hit["lane_scores"])[0]["raw_score"] = float("inf")
    elif case == "lane_rank_overflow":
        cast(list[dict[str, Any]], hit["lane_scores"])[0]["rank"] = 100_001
    elif case == "embedding_dimension_overflow":
        cast(list[dict[str, Any]], hit["embedding_spaces"])[0]["dimension"] = 16_385
    elif case == "invalid_hash_version":
        cast(list[dict[str, Any]], summary["hash_spaces"])[0]["hash_version"] = "phash-v2"
    elif case == "invalid_processing_fingerprint":
        hit["processing_fingerprint"] = {"sha256": "not-a-digest"}
    elif case == "duplicate_summary_lanes":
        summary["selected_lanes"] = ["content", "content"]
    elif case == "invalid_bundle_digest":
        summary["bundle_manifest_sha256s"] = ["not-a-digest"]
    elif case == "invalid_activation_tier":
        summary["activation_tier"] = "production"
    else:  # pragma: no cover - closed parametrization guard
        raise AssertionError(f"unexpected case: {case}")

    with pytest.raises((TypeError, ValueError)):
        validate_serialized_search_output(output)


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "query": {"type": "image", "source": {"type": "url", "url": "https://x"}},
            "targets": ["image"],
        },
        {**_arguments(), "provider_id": "attacker"},
        {**_arguments(), "top_k": 0},
        {**_arguments(), "targets": ["image", "image"]},
        {**_arguments(), "search_intent": "guess"},
        {**_arguments(), "filters": {"unknown": []}},
        {"query": {"type": "text", "text": "x", "model": "private"}, "targets": ["image"]},
    ],
)
def test_input_schema_rejects_aliases_knobs_and_open_ended_fields(
    arguments: Mapping[str, object],
) -> None:
    assert list(Draft202012Validator(SEARCH_INPUT_SCHEMA).iter_errors(arguments))


def test_permission_and_expired_deadline_precede_argument_parsing() -> None:
    service = _Service()
    factory = _AuthorizationFactory()
    handler = _handler(service, factory)

    denied = handler(_context(permissions=frozenset()), {"not": "valid"})
    expired = handler(_context(deadline=NOW), {"not": "valid"})

    assert denied.status is ToolStatus.REJECTED
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.PERMISSION_DENIED
    assert expired.status is ToolStatus.FAILED
    assert expired.error is not None
    assert expired.error.code is ToolErrorCode.TIMEOUT
    assert service.calls == []
    assert factory.calls == []


def test_handler_preserves_defaults_intent_deadline_and_trusted_authorization() -> None:
    service = _Service()
    factory = _AuthorizationFactory()
    handler = _handler(service, factory)

    result = handler(
        _context(),
        _arguments(search_intent="instance", data_policy_ref="request-controlled"),
    )

    request, authorization, deadline = service.calls[0]
    assert result.status is ToolStatus.SUCCEEDED
    assert request.search_intent is SearchIntent.INSTANCE
    assert request.top_k == 20
    assert request.detail_level is DetailLevel.COMPACT
    assert request.retrieval_profile_ref == "local-default"
    assert request.execution_profile is ExecutionProfile.AUTO
    assert request.data_policy_ref == "request-controlled"
    assert deadline is FUTURE
    assert authorization.allowed_data_policy_refs == frozenset({"local-only"})
    assert authorization.allowed_source_root_refs == frozenset({"local:library"})
    assert factory.calls == [_context()]


def test_static_authorization_factory_intersects_startup_and_invocation_permissions() -> None:
    factory = StaticSearchAuthorizationFactory(
        authorization_scope_ref="scope:startup-search",
        permissions=frozenset({"retrieval:search", "artifact:read"}),
        allowed_data_policy_refs=frozenset({"local-only", "team-private"}),
        allowed_source_root_refs=frozenset({"local:library", "mount:course-media"}),
    )
    context = _context(permissions=frozenset({"retrieval:search", "untrusted:extra"}))

    authorization = factory.create(context)

    assert authorization.permissions == frozenset({"retrieval:search"})
    assert authorization.allowed_data_policy_refs == frozenset({"local-only", "team-private"})
    assert authorization.allowed_source_root_refs == frozenset(
        {"local:library", "mount:course-media"}
    )
    assert authorization.operation == "capicove.search.image"


def test_parser_never_reads_local_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "private-query.png"
    service = _Service()

    def forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("Tool parser must not read a path")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    result = _handler(service)(
        _context(),
        _arguments(
            query={
                "type": "image",
                "source": {"type": "local_path", "path": str(source_path)},
            }
        ),
    )

    assert result.status is ToolStatus.SUCCEEDED
    request = service.calls[0][0]
    assert request.query.source.path == str(source_path)  # type: ignore[union-attr]


def test_inline_base64_is_strict_and_bounded_before_request_construction() -> None:
    service = _Service()
    handler = _handler(service)
    invalid = _arguments(
        query={
            "type": "image",
            "source": {"type": "inline_bytes", "base64": "secret%%%", "media_type": "image/png"},
        }
    )
    too_many_chars = _arguments(
        query={
            "type": "image",
            "source": {
                "type": "inline_bytes",
                "base64": "A" * (MAX_INLINE_BASE64_CHARS + 1),
                "media_type": "image/png",
            },
        }
    )
    oversized = base64.b64encode(b"x" * (MAX_INLINE_IMAGE_BYTES + 1)).decode("ascii")

    for arguments in (
        invalid,
        too_many_chars,
        _arguments(
            query={
                "type": "image",
                "source": {
                    "type": "inline_bytes",
                    "base64": oversized,
                    "media_type": "image/png",
                },
            }
        ),
    ):
        result = handler(_context(), arguments)
        assert result.status is ToolStatus.FAILED
        assert result.error is not None
        assert result.error.code is ToolErrorCode.INVALID_INPUT
    assert service.calls == []

    payload = b"synthetic-image"
    success = handler(
        _context(),
        _arguments(
            query={
                "type": "image",
                "source": {
                    "type": "inline_bytes",
                    "base64": base64.b64encode(payload).decode("ascii"),
                    "media_type": "image/png",
                },
            }
        ),
    )
    assert success.status is ToolStatus.SUCCEEDED
    assert service.calls[-1][0].query.source.data == payload  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "arguments",
    [
        _arguments(cursor=None),
        _arguments(query={"type": "text", "text": "query", "language": None}),
        _arguments(
            query={
                "type": "video_clip",
                "source": {"type": "artifact", "artifact_ref": "artifact:video"},
                "time_range": None,
            }
        ),
        _arguments(filters={"time_range": None}),
    ],
)
def test_manual_parser_rejects_explicit_null_for_non_nullable_fields(
    arguments: Mapping[str, object],
) -> None:
    service = _Service()
    factory = _AuthorizationFactory()

    result = _handler(service, factory)(_context(), arguments)

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INVALID_INPUT
    assert factory.calls == []
    assert service.calls == []


def test_success_serialization_preserves_hit_order_and_full_lineage() -> None:
    full_hit = FullHit(
        id="unit-1",
        target_type=TargetType.IMAGE,
        brief_score=0.9,
        evidence_id="evidence-1",
        artifact_ref="artifact:1",
        searchable_unit_ref="unit:1",
        representation_refs=("representation:1",),
        match_kinds=(MatchKind.VISUAL,),
        lane_scores=(LaneScore("instance", 0.9, 0.8, "inner-product", "cal-v1", 1),),
        processing_fingerprint=ProcessingFingerprint("a" * 64),
        index_generation_ids=("generation:1",),
        provider_ids=("provider:1",),
        execution_profile=ExecutionProfile.CPU,
        warning_codes=(SearchWarningCode.RETRIEVAL_PREVIEW.value,),
        lineage_refs=("activation:set-1",),
    )
    summary = SearchExecutionSummary(
        search_intent=SearchIntent.INSTANCE,
        selected_lanes=("content", "instance"),
        provider_ids=("provider:1",),
        embedding_spaces=(),
        hash_spaces=(),
        index_generation_ids=("generation:1",),
        bundle_manifest_sha256s=(),
        resolved_profile_ref="personal-lite",
        execution_profile=ExecutionProfile.CPU,
        activation_tier="preview",
    )
    canonical = SearchResult(
        SearchStatus.SUCCEEDED,
        "query-1",
        degraded=True,
        warning_codes=(SearchWarningCode.RETRIEVAL_PREVIEW,),
        hits=(full_hit,),
        execution_summary=summary,
    )
    result = _handler(_Service(canonical))(_context(), _arguments(detail_level="full"))

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(dict(result.result))
    serialized_hits = result.result["hits"]
    assert isinstance(serialized_hits, list)
    hit = serialized_hits[0]
    assert isinstance(hit, Mapping)
    assert hit["id"] == "unit-1"
    assert hit["lane_scores"][0]["lane_id"] == "instance"
    assert "values" not in json.dumps(dict(result.result))
    assert result.evidence_refs == ("evidence-1",)


def test_twenty_compact_hits_stay_within_compact_output_budget() -> None:
    hits = tuple(
        CompactHit(
            id=f"unit-{index}",
            target_type=TargetType.IMAGE,
            brief_score=0.5,
            evidence_id=f"evidence-{index}",
        )
        for index in range(20)
    )
    result = _handler(_Service(SearchResult(SearchStatus.SUCCEEDED, "query-1", hits=hits)))(
        _context(),
        _arguments(),
    )

    assert result.result is not None
    encoded = json.dumps(dict(result.result), separators=(",", ":")).encode("utf-8")
    assert len(encoded) < MAX_COMPACT_OUTPUT_BYTES
    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(dict(result.result))


def test_maximum_compact_page_preserves_all_one_hundred_hits_within_output_budget() -> None:
    canonical = _maximum_compact_page()

    result = _handler(_Service(canonical))(_context(), _arguments(top_k=100))

    assert MAX_COMPACT_OUTPUT_BYTES == 544 * 1_024
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    serialized_hits = cast(list[dict[str, object]], result.result["hits"])
    assert [hit["id"] for hit in serialized_hits] == [hit.id for hit in canonical.hits]
    encoded = json.dumps(
        dict(result.result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert 16 * 1_024 < len(encoded) <= MAX_COMPACT_OUTPUT_BYTES
    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(dict(result.result))


def test_maximum_json_escaped_compact_page_fits_tool_and_mcp_budget() -> None:
    from capicove.mcp.image_search import MCP_SEARCH_OUTPUT_SCHEMA, encode_search_mcp_result

    canonical = _maximum_json_escaped_compact_page()

    result = _handler(_Service(canonical))(_context(), _arguments(top_k=100))

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.UNAVAILABLE
    assert result.result is not None
    encoded_result = json.dumps(
        dict(result.result),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    # Shared v2 enums include text discriminators and degraded-lane warnings.
    # The unchanged upper bound remains the compatibility requirement.
    assert 512 * 1_024 < len(encoded_result) <= MAX_COMPACT_OUTPUT_BYTES
    assert len(cast(list[object], result.result["hits"])) == 100
    assert all(
        set(cast(Mapping[str, object], hit))
        == {
            "id",
            "target_type",
            "brief_score",
            "evidence_id",
            "timestamp_ms",
            "time_range",
        }
        for hit in cast(list[object], result.result["hits"])
    )
    Draft202012Validator(SEARCH_OUTPUT_SCHEMA).validate(dict(result.result))

    mcp_result = encode_search_mcp_result(result)

    assert mcp_result["result"] == dict(result.result)
    Draft202012Validator(MCP_SEARCH_OUTPUT_SCHEMA).validate(mcp_result)


def test_compact_output_budget_is_inclusive_and_fails_closed_above_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = SearchResult(
        SearchStatus.SUCCEEDED,
        "query-1",
        hits=(CompactHit("image-1", TargetType.IMAGE, 0.9, "evidence-1"),),
    )
    serialized = validate_serialized_search_output(_serialized_compact_output())
    exact_size = len(
        json.dumps(
            serialized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    monkeypatch.setattr(search_tool_module, "MAX_COMPACT_OUTPUT_BYTES", exact_size)

    exact = _handler(_Service(canonical))(_context(), _arguments())

    assert exact.status is ToolStatus.SUCCEEDED

    monkeypatch.setattr(search_tool_module, "MAX_COMPACT_OUTPUT_BYTES", exact_size - 1)
    oversized = _handler(_Service(canonical))(_context(), _arguments())

    assert oversized.status is ToolStatus.FAILED
    assert oversized.error is not None
    assert oversized.error.code is ToolErrorCode.INTERNAL


def test_compact_request_rejects_full_hit_projection_instead_of_bypassing_budget() -> None:
    canonical = SearchResult(
        SearchStatus.SUCCEEDED,
        "query-1",
        hits=(
            FullHit(
                id="image-1",
                target_type=TargetType.IMAGE,
                brief_score=0.9,
                evidence_id="evidence-1",
            ),
        ),
    )

    result = _handler(_Service(canonical))(_context(), _arguments())

    assert result.status is ToolStatus.FAILED
    assert result.result is None
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INTERNAL


def test_failure_mapping_is_total_and_degraded_success_remains_successful() -> None:
    expected = {
        SearchFailureCode.SOURCE_NOT_AUTHORIZED: (
            ToolStatus.REJECTED,
            ToolErrorCode.PERMISSION_DENIED,
            False,
        ),
        SearchFailureCode.PERMISSION_DENIED: (
            ToolStatus.REJECTED,
            ToolErrorCode.PERMISSION_DENIED,
            False,
        ),
        SearchFailureCode.CLIENT_INGRESS_DENIED: (
            ToolStatus.REJECTED,
            ToolErrorCode.POLICY_REJECTED,
            False,
        ),
        SearchFailureCode.INVALID_QUERY: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
        SearchFailureCode.STALE_CURSOR: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
        SearchFailureCode.SOURCE_CHANGED: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
        SearchFailureCode.INPUT_TOO_LARGE: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
        SearchFailureCode.INVALID_MEDIA: (ToolStatus.FAILED, ToolErrorCode.INVALID_INPUT, False),
        SearchFailureCode.PROVIDER_UNAVAILABLE: (
            ToolStatus.FAILED,
            ToolErrorCode.UNAVAILABLE,
            True,
        ),
        SearchFailureCode.INDEX_GENERATION_UNAVAILABLE: (
            ToolStatus.FAILED,
            ToolErrorCode.UNAVAILABLE,
            True,
        ),
        SearchFailureCode.PROFILE_UNAVAILABLE: (
            ToolStatus.FAILED,
            ToolErrorCode.UNAVAILABLE,
            False,
        ),
        SearchFailureCode.CAPACITY_EXCEEDED: (ToolStatus.FAILED, ToolErrorCode.UNAVAILABLE, False),
        SearchFailureCode.BACKPRESSURE: (ToolStatus.FAILED, ToolErrorCode.UNAVAILABLE, True),
        SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE: (
            ToolStatus.FAILED,
            ToolErrorCode.UNAVAILABLE,
            False,
        ),
        SearchFailureCode.HASH_SPACE_INCOMPATIBLE: (
            ToolStatus.FAILED,
            ToolErrorCode.UNAVAILABLE,
            False,
        ),
        SearchFailureCode.INDEX_CORRUPT: (ToolStatus.FAILED, ToolErrorCode.INTERNAL, False),
        SearchFailureCode.DEADLINE_EXCEEDED: (ToolStatus.FAILED, ToolErrorCode.TIMEOUT, True),
        SearchFailureCode.INTERNAL: (ToolStatus.FAILED, ToolErrorCode.INTERNAL, False),
    }
    assert set(expected) == set(SearchFailureCode)

    for code, (status, tool_code, retryable) in expected.items():
        result = _handler(
            _Service(SearchResult(SearchStatus.FAILED, "query-1", failure_code=code))
        )(_context(), _arguments())
        assert result.status is status
        assert result.error is not None
        assert (result.error.code, result.error.retryable) == (tool_code, retryable)
        if code is SearchFailureCode.BACKPRESSURE:
            assert result.error.message == "Search failed: backpressure"
        assert result.result is not None
        assert result.result["failure_code"] == code.value
        validate_serialized_search_output(result.result)

    degraded = SearchResult(
        SearchStatus.SUCCEEDED,
        "query-1",
        degraded=True,
        warning_codes=(SearchWarningCode.PROVIDER_UNAVAILABLE,),
    )
    result = _handler(_Service(degraded))(_context(), _arguments())
    assert result.status is ToolStatus.SUCCEEDED
    assert result.error is None


def test_unexpected_service_failure_is_sanitized() -> None:
    result = _handler(_Service(RuntimeError("secret /private/path")))(_context(), _arguments())

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INTERNAL
    assert "secret" not in result.error.message
    assert "/private" not in result.error.message


def test_service_validation_exception_is_internal_not_client_input() -> None:
    result = _handler(_Service(ValueError("internal provider invariant")))(
        _context(),
        _arguments(),
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.INTERNAL


def test_deadline_is_rechecked_after_service() -> None:
    instants = iter((NOW, FUTURE))
    handler = SearchToolHandler(
        _Service(),
        _AuthorizationFactory(),
        clock=lambda: next(instants),
    )

    result = handler(_context(), _arguments())

    assert result.status is ToolStatus.FAILED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TIMEOUT


def test_register_search_tool_adds_exactly_one_canonical_id() -> None:
    registry = ToolRegistry()
    handler = register_search_tool(
        registry,
        _Service(),
        _AuthorizationFactory(),
        clock=lambda: NOW,
    )

    assert isinstance(handler, SearchToolHandler)
    assert registry.list_specs() == (SEARCH_SPEC,)
    assert registry.resolve("capicove.search.image").handler is handler
