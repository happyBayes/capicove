"""Tests for immutable Elasticsearch text index generation lifecycle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from capicove_retriever_elasticsearch.adapter import (
    ElasticsearchOperationError,
    ElasticsearchTextRetriever,
)
from capicove_retriever_elasticsearch.codec import build_managed_text_index_mapping
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval import (
    TextIndexGenerationLifecycle,
    TextIndexGenerationManifest,
    TextIndexValidationReport,
)


PHYSICAL_INDEX = "capicove-text-g-79091da8c3ddbe5b32099e4c"
MAPPING_SHA256 = (
    "97fe07c89cff5a75c994bfd389b32b2e4d9050aa707eb458ec132e169ce5165d"  # pragma: allowlist secret
)


@dataclass(frozen=True)
class HeadResponse:
    """Mirror the boolean semantics of elastic_transport.HeadApiResponse."""

    value: bool

    def __bool__(self) -> bool:
        return self.value


def configuration() -> ElasticsearchConfiguration:
    """Return one closed test configuration."""
    return ElasticsearchConfiguration(
        hosts=("https://localhost:9200",),
        index_prefix="capicove-text",
        active_alias="capicove-text-active",
    )


def embedding_space() -> EmbeddingSpaceRef:
    """Return one canonical text embedding space."""
    return EmbeddingSpaceRef(
        space_id="text-dense-v1",
        provider_id="embedding-provider",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def manifest() -> TextIndexGenerationManifest:
    """Return one generation matching the provider mapping."""
    return TextIndexGenerationManifest(
        generation_id="generation-42",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-1",),
        searchable_unit_manifest_sha256="a" * 64,
        content_sha256="b" * 64,
        mapping_sha256=MAPPING_SHA256,
        embedding_spaces=(embedding_space(),),
        retriever_id="capicove.retriever.elasticsearch",
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=2,
    )


def mapping_response(
    value: TextIndexGenerationManifest,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return a complete get-mapping response for one physical index."""
    return {
        PHYSICAL_INDEX: {
            "mappings": build_managed_text_index_mapping(value),
        }
    }


@dataclass
class RecordingIndicesClient:
    """Record index administration calls and return controlled responses."""

    def get_settings(self, **kwargs: object) -> object:
        return {PHYSICAL_INDEX: {"settings": {"index.blocks.write": "true"}}}

    create_response: object = field(default_factory=lambda: {"acknowledged": True})
    refresh_response: object = field(
        default_factory=lambda: {
            "_shards": {"total": 1, "successful": 1, "failed": 0},
        }
    )
    get_mapping_response: object = field(default_factory=dict)
    update_aliases_response: object = field(
        default_factory=lambda: {"acknowledged": True, "errors": False}
    )
    get_alias_response: object = field(default_factory=dict)
    exists_alias_response: object = field(default_factory=lambda: HeadResponse(False))
    create_calls: list[dict[str, object]] = field(default_factory=list)
    refresh_calls: list[dict[str, object]] = field(default_factory=list)
    get_mapping_calls: list[dict[str, object]] = field(default_factory=list)
    update_aliases_calls: list[dict[str, object]] = field(default_factory=list)
    get_alias_calls: list[dict[str, object]] = field(default_factory=list)
    exists_alias_calls: list[dict[str, object]] = field(default_factory=list)
    add_block_calls: list[dict[str, object]] = field(default_factory=list)

    def create(self, **kwargs: object) -> object:
        self.create_calls.append(dict(kwargs))
        return self.create_response

    def refresh(self, **kwargs: object) -> object:
        self.refresh_calls.append(dict(kwargs))
        return self.refresh_response

    def get_mapping(self, **kwargs: object) -> object:
        self.get_mapping_calls.append(dict(kwargs))
        return self.get_mapping_response

    def update_aliases(self, **kwargs: object) -> object:
        self.update_aliases_calls.append(dict(kwargs))
        return self.update_aliases_response

    def get_alias(self, **kwargs: object) -> object:
        self.get_alias_calls.append(dict(kwargs))
        return self.get_alias_response

    def exists_alias(self, **kwargs: object) -> object:
        self.exists_alias_calls.append(dict(kwargs))
        return self.exists_alias_response

    def add_block(self, **kwargs: object) -> object:
        self.add_block_calls.append(dict(kwargs))
        return {
            "acknowledged": True,
            "shards_acknowledged": True,
            "indices": [{"name": PHYSICAL_INDEX, "blocked": True}],
        }


@dataclass
class RecordingClient:
    """Expose real adapter boundaries without a networked cluster."""

    indices: RecordingIndicesClient
    count_response: object = field(
        default_factory=lambda: {
            "count": 2,
            "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        }
    )
    count_calls: list[dict[str, object]] = field(default_factory=list)

    def count(self, **kwargs: object) -> object:
        self.count_calls.append(dict(kwargs))
        return self.count_response

    def bulk(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("unexpected bulk call")

    def search(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("unexpected search call")


def runtime_for(client: RecordingClient) -> ElasticsearchTextRetriever:
    """Return one lazy runtime over a controlled client."""
    return ElasticsearchTextRetriever(configuration(), client_factory=lambda _: client)


def ready_report(value: TextIndexGenerationManifest) -> TextIndexValidationReport:
    """Return one validation report that permits activation."""
    return TextIndexValidationReport(
        generation_id=value.generation_id,
        manifest_sha256=value.manifest_sha256,
        physical_index_ref=PHYSICAL_INDEX,
        mapping_sha256=value.mapping_sha256,
        expected_unit_count=value.expected_unit_count,
        observed_unit_count=value.expected_unit_count,
        issue_codes=(),
    )


def test_runtime_exposes_lifecycle_without_constructing_a_client() -> None:
    indices = RecordingIndicesClient()
    client = RecordingClient(indices)
    factory_calls: list[ElasticsearchConfiguration] = []

    def factory(value: ElasticsearchConfiguration) -> RecordingClient:
        factory_calls.append(value)
        return client

    runtime = ElasticsearchTextRetriever(configuration(), client_factory=factory)

    assert isinstance(runtime.generation_lifecycle, TextIndexGenerationLifecycle)
    assert runtime.generation_lifecycle.mapping_sha256((embedding_space(),)) == MAPPING_SHA256
    assert factory_calls == []


def test_create_builds_one_immutable_physical_index_without_an_alias() -> None:
    value = manifest()
    indices = RecordingIndicesClient()
    runtime = runtime_for(RecordingClient(indices))

    physical_ref = runtime.generation_lifecycle.create(value)

    assert physical_ref == PHYSICAL_INDEX
    assert indices.create_calls == [
        {
            "index": PHYSICAL_INDEX,
            "mappings": build_managed_text_index_mapping(value),
        }
    ]
    assert "aliases" not in indices.create_calls[0]


def test_create_rejects_provider_mapping_mismatch_before_opening_client() -> None:
    client_opened = False

    def factory(_: ElasticsearchConfiguration) -> RecordingClient:
        nonlocal client_opened
        client_opened = True
        return RecordingClient(RecordingIndicesClient())

    runtime = ElasticsearchTextRetriever(configuration(), client_factory=factory)

    with pytest.raises(ValueError, match="mapping_sha256"):
        runtime.generation_lifecycle.create(replace(manifest(), mapping_sha256="f" * 64))

    assert client_opened is False


def test_validate_refreshes_and_checks_mapping_metadata_and_visible_count() -> None:
    value = manifest()
    indices = RecordingIndicesClient(get_mapping_response=mapping_response(value))
    client = RecordingClient(indices)
    runtime = runtime_for(client)

    report = runtime.generation_lifecycle.validate(value)

    assert report.valid is True
    assert report.observed_unit_count == 2
    assert indices.add_block_calls == [{"index": PHYSICAL_INDEX, "block": "write"}]
    assert indices.refresh_calls == [{"index": PHYSICAL_INDEX}]
    assert indices.get_mapping_calls == [{"index": PHYSICAL_INDEX}]
    assert client.count_calls == [{"index": PHYSICAL_INDEX}]


def test_validate_returns_stable_corruption_issues_without_fallback() -> None:
    value = manifest()
    response = mapping_response(value)
    corrupted = deepcopy(response)
    corrupted[PHYSICAL_INDEX]["mappings"]["_meta"]["capicove"]["content_sha256"] = "f" * 64
    indices = RecordingIndicesClient(get_mapping_response=corrupted)
    client = RecordingClient(indices, count_response={"count": 1, "_shards": {"failed": 0}})
    runtime = runtime_for(client)

    report = runtime.generation_lifecycle.validate(value)

    assert report.valid is False
    assert report.issue_codes == (
        "content_digest_mismatch",
        "document_count_mismatch",
    )


def test_activate_atomically_replaces_alias_only_for_a_ready_generation() -> None:
    value = manifest()
    old_index = "capicove-text-g-aaaaaaaaaaaaaaaaaaaaaaaa"
    indices = RecordingIndicesClient(
        exists_alias_response=HeadResponse(True),
        get_alias_response={
            old_index: {"aliases": {"capicove-text-active": {}}},
        },
    )
    runtime = runtime_for(RecordingClient(indices))

    pinned = runtime.generation_lifecycle.activate(value, ready_report(value))

    assert pinned.generation_id == "generation-42"
    assert indices.update_aliases_calls == [
        {
            "actions": [
                {
                    "remove": {
                        "index": old_index,
                        "alias": "capicove-text-active",
                        "must_exist": True,
                    }
                },
                {
                    "add": {
                        "index": PHYSICAL_INDEX,
                        "alias": "capicove-text-active",
                        "is_write_index": False,
                    }
                },
            ]
        }
    ]
    assert indices.exists_alias_calls == [{"name": "capicove-text-active"}]
    assert indices.get_alias_calls == [{"name": "capicove-text-active"}]


def test_activate_rejects_corrupt_or_mismatched_reports_before_network_access() -> None:
    value = manifest()
    indices = RecordingIndicesClient()
    runtime = runtime_for(RecordingClient(indices))
    corrupt = replace(
        ready_report(value),
        observed_unit_count=1,
        issue_codes=("document_count_mismatch",),
    )
    mismatched = replace(ready_report(value), manifest_sha256="f" * 64)

    with pytest.raises(ValueError, match="ready"):
        runtime.generation_lifecycle.activate(value, corrupt)
    with pytest.raises(ValueError, match="manifest"):
        runtime.generation_lifecycle.activate(value, mismatched)

    assert indices.update_aliases_calls == []


def test_resolve_active_pins_exactly_one_verified_alias_target() -> None:
    value = manifest()
    indices = RecordingIndicesClient(
        get_alias_response={
            PHYSICAL_INDEX: {"aliases": {"capicove-text-active": {}}},
        },
        get_mapping_response=mapping_response(value),
    )
    runtime = runtime_for(RecordingClient(indices))

    pinned = runtime.generation_lifecycle.resolve_active()

    assert pinned.generation_id == "generation-42"
    assert pinned.manifest_sha256 == value.manifest_sha256
    assert pinned.physical_index_ref == PHYSICAL_INDEX
    assert indices.get_alias_calls == [{"name": "capicove-text-active"}]
    assert indices.get_mapping_calls == [{"index": PHYSICAL_INDEX}]


def test_resolve_active_rejects_a_corrupt_physical_mapping_without_fallback() -> None:
    value = manifest()
    response = mapping_response(value)
    response[PHYSICAL_INDEX]["mappings"]["properties"]["text"] = {"type": "keyword"}
    indices = RecordingIndicesClient(
        get_alias_response={
            PHYSICAL_INDEX: {"aliases": {"capicove-text-active": {}}},
        },
        get_mapping_response=response,
    )
    runtime = runtime_for(RecordingClient(indices))

    with pytest.raises(ElasticsearchOperationError, match="resolution failed"):
        runtime.generation_lifecycle.resolve_active()


@pytest.mark.parametrize(
    "alias_response",
    [
        {},
        {
            PHYSICAL_INDEX: {"aliases": {"capicove-text-active": {}}},
            "capicove-text-g-aaaaaaaaaaaaaaaaaaaaaaaa": {"aliases": {"capicove-text-active": {}}},
        },
        {"unrelated-index": {"aliases": {"capicove-text-active": {}}}},
    ],
)
def test_resolve_active_rejects_missing_multiple_or_mismatched_targets(
    alias_response: object,
) -> None:
    value = manifest()
    indices = RecordingIndicesClient(
        get_alias_response=alias_response,
        get_mapping_response=mapping_response(value),
    )
    runtime = runtime_for(RecordingClient(indices))

    with pytest.raises(ElasticsearchOperationError, match="resolution failed"):
        runtime.generation_lifecycle.resolve_active()


def test_lifecycle_provider_failures_are_sanitized() -> None:
    class RaisingIndicesClient(RecordingIndicesClient):
        def create(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("VERY_SECRET_PROVIDER_RESPONSE")

    runtime = runtime_for(RecordingClient(RaisingIndicesClient()))

    with pytest.raises(
        ElasticsearchOperationError,
        match="index creation failed",
    ) as raised:
        runtime.generation_lifecycle.create(manifest())

    assert "VERY_SECRET_PROVIDER_RESPONSE" not in str(raised.value)
    assert raised.value.__cause__ is None
