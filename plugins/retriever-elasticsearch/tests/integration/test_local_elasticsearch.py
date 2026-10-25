"""Opt-in authenticated TLS Elasticsearch acceptance without service downloads."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from uuid import uuid4

import pytest
from capicove_retriever_elasticsearch.adapter import (
    ElasticsearchOperationError,
    ElasticsearchTextRetriever,
)
from capicove_retriever_elasticsearch.codec import encode_text_index_entry
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    TargetType,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
    VectorNormalization,
)
from capicove.retrieval import TextEmbedding, TextIndexEntry, TextIndexGenerationManifest


pytestmark = pytest.mark.skipif(
    os.environ.get("CAPICOVE_TEST_REAL_ELASTICSEARCH") != "1",
    reason="explicit isolated TLS Elasticsearch opt-in required",
)


def _configuration() -> ElasticsearchConfiguration:
    suffix = uuid4().hex[:12]
    return ElasticsearchConfiguration(
        hosts=(os.environ["CAPICOVE_VIDEO_ES_ENDPOINT"],),
        ca_certs=os.environ["CAPICOVE_VIDEO_ES_CA"],
        api_key=os.environ.get("CAPICOVE_TEST_ES_API_KEY"),
        index_prefix=f"capicove-native-{suffix}",
        active_alias=f"capicove-native-active-{suffix}",
    )


def test_real_generation_bm25_knn_reopen_and_sealed_writes() -> None:
    configuration = _configuration()
    runtime = ElasticsearchTextRetriever(configuration)
    space = EmbeddingSpaceRef(
        "native-text-v1",
        "authored-vectors",
        "1",
        "1",
        3,
        VectorNormalization.L2,
        DistanceMetric.COSINE,
        frozenset({QueryModality.TEXT}),
        frozenset({TargetType.TEXT_CHUNK}),
    )
    entries = tuple(
        TextIndexEntry(
            unit=SearchableUnit(
                unit_id=name,
                unit_type=TargetType.TEXT_CHUNK,
                source_hash="a" * 64,
                artifact_ref=f"artifact:{name}",
                evidence_ref=f"evidence:{name}",
                representation_refs=(f"representation:{name}",),
                processing_run_ref="native-run",
                processing_version="1",
                processing_fingerprint=ProcessingFingerprint("b" * 64),
                embedding_spaces=(space,),
                text_locator=TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(0, len(text))),
            ),
            text=text,
            corpus_refs=("native-corpus",),
            media_type="text/plain",
            language="en",
            embeddings=(TextEmbedding(space, vector),),
        )
        for name, text, vector in (
            ("ocean", "whale migration ocean", (1.0, 0.0, 0.0)),
            ("mountain", "mountain hiking trail", (0.0, 1.0, 0.0)),
        )
    )
    content = json.dumps(
        [encode_text_index_entry(entry) for entry in entries],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    lifecycle = runtime.generation_lifecycle
    manifest = TextIndexGenerationManifest(
        generation_id=f"native-{uuid4().hex}",
        workspace_ref="native-acceptance",
        corpus_refs=("native-corpus",),
        searchable_unit_manifest_sha256="c" * 64,
        content_sha256=hashlib.sha256(content).hexdigest(),
        mapping_sha256=lifecycle.mapping_sha256((space,)),
        embedding_spaces=(space,),
        retriever_id=runtime.provider_id,
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=2,
    )
    physical = lifecycle.create(manifest)
    runtime.index_writer.add(index_generation_id=manifest.generation_id, entries=entries)
    validation = lifecycle.validate(manifest)
    assert validation.valid, validation.issue_codes
    assert lifecycle.validate(manifest).valid
    pin = lifecycle.activate(manifest, validation)
    assert pin.physical_index_ref == physical
    reopened = ElasticsearchTextRetriever(configuration)
    assert reopened.generation_lifecycle.resolve_active() == pin
    filters = SearchFilters(corpus_refs=frozenset({"native-corpus"}))
    lexical = reopened.lexical_retriever.retrieve(
        query=TextQuery("whale"), filters=filters, generation=pin, top_k=2
    )
    dense = reopened.dense_retriever.retrieve(
        query_vector=(1.0, 0.0, 0.0),
        embedding_space=space,
        filters=filters,
        generation=pin,
        top_k=2,
    )
    assert lexical[0].unit_id == dense[0].unit_id == "ocean"
    assert not reopened.dense_retriever.retrieve(
        query_vector=(1.0, 0.0, 0.0),
        embedding_space=space,
        filters=SearchFilters(corpus_refs=frozenset({"excluded-corpus"})),
        generation=pin,
        top_k=2,
    )
    with pytest.raises(ElasticsearchOperationError):
        runtime.index_writer.add(index_generation_id=manifest.generation_id, entries=entries)
    invalid_manifest = replace(manifest, generation_id=f"empty-{uuid4().hex}")
    lifecycle.create(invalid_manifest)
    invalid = lifecycle.validate(invalid_manifest)
    assert "document_count_mismatch" in invalid.issue_codes
    with pytest.raises(ValueError):
        lifecycle.activate(invalid_manifest, invalid)
    assert lifecycle.resolve_active() == pin


def test_real_authentication_failure_is_sanitized() -> None:
    if not os.environ.get("CAPICOVE_TEST_ES_API_KEY"):
        pytest.skip("authenticated cluster API key required")
    secret = uuid4().hex
    backend = ElasticsearchTextRetriever(replace(_configuration(), api_key=secret))
    with pytest.raises(ElasticsearchOperationError) as error:
        backend.generation_lifecycle.resolve_active()
    assert secret not in str(error.value)
    assert str(error.value) == "Elasticsearch active generation resolution failed"


def test_real_tls_rejects_untrusted_certificate() -> None:
    """The isolated node must use a private CA absent from the public trust store."""
    import certifi
    from elastic_transport import TlsError
    from elasticsearch import Elasticsearch

    configuration = _configuration()
    with Elasticsearch(
        configuration.hosts,
        ca_certs=certifi.where(),
        api_key=configuration.api_key,
        request_timeout=3,
        max_retries=0,
    ) as client:
        with pytest.raises(TlsError):
            client.info()
