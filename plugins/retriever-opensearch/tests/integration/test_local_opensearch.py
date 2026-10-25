"""Real OpenSearch 3.6.0 integration acceptance."""

from __future__ import annotations

import base64
import json
import os
import ssl
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
from capicove_retriever_opensearch import (
    OpenSearchConfiguration,
    OpenSearchTextRetriever,
)

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
from capicove.retrieval import (
    TextEmbedding,
    TextIndexEntry,
    TextIndexGenerationManifest,
)


IMAGE = "opensearchproject/opensearch@sha256:b5dd1512af2a99748c942cfbbd7f32162623336b210667d0fc6333c6321f171d"


def text_embedding_space() -> EmbeddingSpaceRef:
    """Return one real-compatible text embedding space."""
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


@contextmanager
def local_opensearch() -> Iterator[str]:
    """Run one isolated security-disabled OpenSearch node."""
    endpoint = os.environ.get("CAPICOVE_TEST_OPENSEARCH_URL")
    if endpoint:
        parsed = urllib.parse.urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname != "127.0.0.1"
            or not parsed.port
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
        ):
            raise ValueError("native OpenSearch tests require a credential-free loopback endpoint")
        headers = {}
        context = None
        if parsed.scheme == "https":
            context = ssl.create_default_context(cafile=os.environ["CAPICOVE_TEST_OS_CA"])
            pair = "capicove:" + os.environ["CAPICOVE_TEST_OS_PASSWORD"]
            headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        request = urllib.request.Request(endpoint, headers=headers)
        with urllib.request.urlopen(request, timeout=5, context=context) as response:
            assert json.load(response)["version"]["number"] == "3.6.0"
        yield endpoint
        return
    name = f"capicove-opensearch-test-{uuid.uuid4().hex[:12]}"
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--publish",
            "127.0.0.1::9200",
            "--env",
            "discovery.type=single-node",
            "--env",
            "DISABLE_SECURITY_PLUGIN=true",
            "--env",
            "DISABLE_INSTALL_DEMO_CONFIG=true",
            "--env",
            "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m",
            IMAGE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        port = (
            subprocess.run(
                ["docker", "port", name, "9200/tcp"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .rsplit(":", 1)[1]
        )
        endpoint = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90
        while True:
            try:
                with urllib.request.urlopen(endpoint, timeout=2) as response:
                    payload = json.load(response)
                if payload["version"]["number"] == "3.6.0":
                    break
            except Exception:
                if time.monotonic() >= deadline:
                    logs = subprocess.run(
                        ["docker", "logs", name], capture_output=True, text=True
                    ).stderr[-4_000:]
                    raise AssertionError(f"OpenSearch did not become ready: {logs}") from None
                time.sleep(1)
        yield endpoint
    finally:
        subprocess.run(
            ["docker", "stop", "--time", "5", name],
            check=False,
            capture_output=True,
        )


def entry(unit_id: str, text: str, vector: tuple[float, ...]) -> TextIndexEntry:
    """Return one real index entry."""
    space = text_embedding_space()
    unit = SearchableUnit(
        unit_id=unit_id,
        unit_type=TargetType.TEXT_CHUNK,
        source_hash="a" * 64,
        artifact_ref=f"artifact-{unit_id}",
        evidence_ref=f"evidence-{unit_id}",
        representation_refs=(f"representation-{unit_id}",),
        processing_run_ref="run-1",
        processing_version="text-v1",
        processing_fingerprint=ProcessingFingerprint.from_manifest({"unit": unit_id}),
        embedding_spaces=(space,),
        text_locator=TextChunkLocator(
            TextOrigin.DOCUMENT,
            CharacterRange(0, len(text)),
        ),
    )
    return TextIndexEntry(
        unit=unit,
        text=text,
        corpus_refs=("corpus-1",),
        media_type="text/plain",
        language="en",
        embeddings=(TextEmbedding(space, vector),),
    )


def configuration(endpoint: str) -> OpenSearchConfiguration:
    """Use verified TLS and environment credentials when the operator selects HTTPS."""
    secured = endpoint.startswith("https://")
    return OpenSearchConfiguration(
        hosts=(endpoint,),
        verify_certs=secured,
        username="capicove" if secured else None,
        password=os.environ["CAPICOVE_TEST_OS_PASSWORD"] if secured else None,
        ca_certs=os.environ["CAPICOVE_TEST_OS_CA"] if secured else None,
        index_prefix=f"capicove-{uuid.uuid4().hex[:8]}",
        active_alias=f"capicove-active-{uuid.uuid4().hex[:8]}",
    )


def test_real_opensearch_generation_bm25_and_knn() -> None:
    with local_opensearch() as endpoint:
        runtime = OpenSearchTextRetriever(configuration(endpoint))
        space = text_embedding_space()
        mapping_sha256 = runtime.generation_lifecycle.mapping_sha256((space,))
        manifest = TextIndexGenerationManifest(
            generation_id="generation-real-1",
            workspace_ref="workspace-1",
            corpus_refs=("corpus-1",),
            searchable_unit_manifest_sha256="b" * 64,
            content_sha256="c" * 64,
            mapping_sha256=mapping_sha256,
            embedding_spaces=(space,),
            retriever_id=runtime.provider_id,
            index_format_version="opensearch-text-v1",
            build_processing_fingerprint=ProcessingFingerprint.from_manifest({"build": 1}),
            expected_unit_count=2,
        )

        physical = runtime.generation_lifecycle.create(manifest)
        runtime.index_writer.add(
            index_generation_id=manifest.generation_id,
            entries=(
                entry("alpha", "ocean whale migration", (1.0, 0.0, 0.0)),
                entry("beta", "mountain trail hiking", (0.0, 1.0, 0.0)),
            ),
        )
        validation = runtime.generation_lifecycle.validate(manifest)
        assert validation.valid
        pin = runtime.generation_lifecycle.activate(manifest, validation)
        assert pin.physical_index_ref == physical
        assert runtime.generation_lifecycle.resolve_active() == pin

        lexical = runtime.lexical_retriever.retrieve(
            query=TextQuery("whale"),
            filters=SearchFilters(corpus_refs=frozenset({"corpus-1"})),
            generation=pin,
            top_k=2,
        )
        dense = runtime.dense_retriever.retrieve(
            query_vector=(1.0, 0.0, 0.0),
            embedding_space=space,
            filters=SearchFilters(corpus_refs=frozenset({"corpus-1"})),
            generation=pin,
            top_k=2,
        )

        assert lexical[0].unit_id == "alpha"
        assert dense[0].unit_id == "alpha"


@pytest.mark.parametrize("failure", ["credentials", "untrusted_ca"])
def test_real_tls_rejects_invalid_trust_and_authentication(failure: str) -> None:
    endpoint = os.environ.get("CAPICOVE_TEST_OPENSEARCH_URL", "")
    if not endpoint.startswith("https://"):
        pytest.skip("Select the private TLS OpenSearch qualification node")
    from capicove_retriever_opensearch.adapter import OpenSearchOperationError

    configured = configuration(endpoint)
    configured = (
        replace(configured, password="authored-wrong-password")  # pragma: allowlist secret
        if failure == "credentials"
        else replace(configured, ca_certs=None)
    )
    runtime = OpenSearchTextRetriever(configured)
    with pytest.raises(OpenSearchOperationError) as error:
        runtime.generation_lifecycle.resolve_active()
    assert os.environ["CAPICOVE_TEST_OS_PASSWORD"] not in str(error.value)
    assert "authored-wrong-password" not in str(error.value)
