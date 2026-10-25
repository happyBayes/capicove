"""Tests for side-effect-free execution backend discovery."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

import capicove.plugins as plugin_api
from capicove.contracts import EmbeddingSpaceRef, SearchFilters, TextQuery
from capicove.contracts.retrieval import RETRIEVAL_CONTRACT_VERSION
from capicove.documents import DocumentParserBackend
from capicove.plugins import (
    ConfigurableRetrieverPlugin,
    DiscoveredPlugin,
    DocumentParserModelInstaller,
    DocumentParserPlugin,
    ExecutionBackendPlugin,
    PluginAvailability,
    PluginMetadata,
    RetrieverPlugin,
    list_document_parser_plugins,
    list_execution_backend_plugins,
    load_document_parser_plugin,
    load_execution_backend_plugin,
)
from capicove.processing import ExecutionBackend
from capicove.processing.models import OperatorSpec
from capicove.retrieval import (
    DenseCandidateRetriever,
    LexicalCandidateRetriever,
    PinnedTextIndexGeneration,
    TextCandidate,
    TextIndexEntry,
    TextIndexGenerationLifecycle,
    TextIndexGenerationManifest,
    TextIndexValidationReport,
    TextIndexWriter,
    TextRetrieverBackend,
)


class _EntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        loaded: object,
        *,
        group: str = "capicove.execution_backends",
    ) -> None:
        self.name = name
        self.value = value
        self.group = group
        self._loaded = loaded
        self.load_calls = 0

    def load(self) -> object:
        self.load_calls += 1
        return self._loaded


class _Backend:
    backend_id = "spark"

    def submit(self, plan: object, context: object) -> object:
        return object()

    def poll(self, reference: object) -> object:
        return object()

    def cancel(self, reference: object) -> None:
        return None


class _Plugin:
    metadata = PluginMetadata(
        plugin_id="capicove.execution.spark",
        plugin_version="1.0.0",
        capability="execution_backend",
        contract_min=1,
        contract_max=1,
        optional_packages=("pyspark",),
        external_binaries=("java",),
        network_access=False,
    )

    def availability(self) -> PluginAvailability:
        return PluginAvailability(True)

    def operator_specs(self) -> tuple[OperatorSpec, ...]:
        return ()

    def create(self, configuration: Mapping[str, object]) -> ExecutionBackend:
        del configuration
        return cast(ExecutionBackend, _Backend())


def test_document_model_installer_is_a_separate_optional_protocol() -> None:
    class _Installer:
        def model_plan(self, profile_id: str, destination: Path) -> object:
            del profile_id, destination
            return object()

        def install_models(
            self,
            profile_id: str,
            destination: Path,
            *,
            confirmed: bool,
        ) -> object:
            del profile_id, destination, confirmed
            return object()

    assert isinstance(_Installer(), DocumentParserModelInstaller)


class _PluginWithoutOperatorSpecs:
    metadata = _Plugin.metadata

    def availability(self) -> PluginAvailability:
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> ExecutionBackend:
        del configuration
        return cast(ExecutionBackend, _Backend())


def _factory() -> ExecutionBackendPlugin:
    return _Plugin()


class _RetrieverPlugin:
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="capicove.retriever.elasticsearch",
            plugin_version="1.0.0",
            capability="retriever",
            contract_min=2,
            contract_max=2,
            optional_packages=("elasticsearch",),
            external_binaries=(),
            network_access=True,
        )

    def availability(self) -> PluginAvailability:
        return PluginAvailability(True)


class _IndexWriter:
    provider_id = "retriever-provider"

    def add(
        self,
        *,
        index_generation_id: str,
        entries: tuple[TextIndexEntry, ...],
    ) -> None:
        del index_generation_id, entries


class _LexicalRetriever:
    provider_id = "retriever-provider"

    def retrieve(
        self,
        *,
        query: TextQuery,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        del query, filters, generation, top_k
        return ()


class _DenseRetriever:
    provider_id = "retriever-provider"

    def retrieve(
        self,
        *,
        query_vector: tuple[float, ...],
        embedding_space: EmbeddingSpaceRef,
        filters: SearchFilters,
        generation: PinnedTextIndexGeneration,
        top_k: int,
    ) -> tuple[TextCandidate, ...]:
        del query_vector, embedding_space, filters, generation, top_k
        return ()


class _GenerationLifecycle:
    provider_id = "retriever-provider"

    def mapping_sha256(
        self,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...],
    ) -> str:
        del embedding_spaces
        return "f" * 64

    def create(self, manifest: TextIndexGenerationManifest) -> str:
        del manifest
        return "physical-index"

    def validate(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexValidationReport:
        raise NotImplementedError

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration:
        raise NotImplementedError

    def resolve_active(self) -> PinnedTextIndexGeneration:
        raise NotImplementedError


class _TextBackend:
    provider_id = "retriever-provider"
    index_writer: TextIndexWriter = _IndexWriter()
    lexical_retriever: LexicalCandidateRetriever = _LexicalRetriever()
    dense_retriever: DenseCandidateRetriever = _DenseRetriever()
    generation_lifecycle: TextIndexGenerationLifecycle = _GenerationLifecycle()


class _ConfigurableRetrieverPlugin(_RetrieverPlugin):
    def create(self, configuration: Mapping[str, object]) -> TextRetrieverBackend:
        del configuration
        return _TextBackend()


class _IncompatibleRetrieverPlugin(_RetrieverPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="capicove.retriever.elasticsearch",
            plugin_version="1.0.0",
            capability="retriever",
            contract_min=3,
            contract_max=3,
            optional_packages=("elasticsearch",),
            external_binaries=(),
            network_access=True,
        )


def _retriever_factory() -> object:
    return _RetrieverPlugin()


def _incompatible_retriever_factory() -> object:
    return _IncompatibleRetrieverPlugin()


class _DocumentBackend:
    parser_id = "paddleocr"

    def parse(self, request: object) -> object:
        raise NotImplementedError


class _DocumentParserPlugin:
    metadata = PluginMetadata(
        plugin_id="capicove.processor.paddleocr",
        plugin_version="1.0.0",
        capability="document_parser",
        contract_min=1,
        contract_max=1,
        optional_packages=("paddleocr",),
        external_binaries=(),
        network_access=False,
    )

    def availability(self) -> PluginAvailability:
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> DocumentParserBackend:
        del configuration
        return cast(DocumentParserBackend, _DocumentBackend())


def _document_parser_factory() -> DocumentParserPlugin:
    return _DocumentParserPlugin()


class _IncompatibleDocumentParserPlugin(_DocumentParserPlugin):
    metadata = PluginMetadata(
        plugin_id="capicove.processor.paddleocr",
        plugin_version="1.0.0",
        capability="document_parser",
        contract_min=2,
        contract_max=2,
        optional_packages=("paddleocr",),
        external_binaries=(),
        network_access=False,
    )


def test_configurable_retriever_protocol_preserves_discovery_only_descriptors() -> None:
    descriptor = _RetrieverPlugin()
    configurable = _ConfigurableRetrieverPlugin()

    assert isinstance(descriptor, RetrieverPlugin)
    assert not isinstance(descriptor, ConfigurableRetrieverPlugin)
    assert isinstance(configurable, ConfigurableRetrieverPlugin)
    backend = configurable.create({"hosts": ("https://localhost:9200",)})
    assert isinstance(backend, TextRetrieverBackend)
    assert isinstance(backend.index_writer, TextIndexWriter)
    assert isinstance(backend.lexical_retriever, LexicalCandidateRetriever)
    assert isinstance(backend.dense_retriever, DenseCandidateRetriever)
    assert isinstance(backend.generation_lifecycle, TextIndexGenerationLifecycle)


def test_listing_does_not_load_plugin_module() -> None:
    entry = _EntryPoint("spark", "spark_plugin:factory", _factory)

    listed = list_execution_backend_plugins(entry_points=(entry,))

    assert listed == (DiscoveredPlugin("spark", "spark_plugin:factory"),)
    assert entry.load_calls == 0


def test_importing_plugin_discovery_does_not_import_retrieval_runtime() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove.plugins; import sys; "
                "assert 'capicove.retrieval.ports' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_explicit_loading_validates_metadata_and_factory() -> None:
    entry = _EntryPoint("spark", "spark_plugin:factory", _factory)

    plugin = load_execution_backend_plugin("spark", entry_points=(entry,))

    assert plugin.metadata.plugin_id == "capicove.execution.spark"
    assert entry.load_calls == 1


def test_execution_plugin_protocol_requires_operator_catalog() -> None:
    assert not isinstance(_PluginWithoutOperatorSpecs(), ExecutionBackendPlugin)


def test_discovery_rejects_duplicate_entry_point_names() -> None:
    entries = (
        _EntryPoint("spark", "one:factory", _factory),
        _EntryPoint("spark", "two:factory", _factory),
    )

    with pytest.raises(ValueError, match="duplicate"):
        list_execution_backend_plugins(entry_points=entries)


def test_loading_rejects_unknown_backend() -> None:
    with pytest.raises(LookupError, match="unknown"):
        load_execution_backend_plugin("spark", entry_points=())


def test_listing_retrievers_does_not_load_plugin_module() -> None:
    entry = _EntryPoint(
        "elasticsearch",
        "elasticsearch_plugin:factory",
        _retriever_factory,
        group="capicove.retrievers",
    )

    listed = plugin_api.list_retriever_plugins(entry_points=(entry,))

    assert listed == (DiscoveredPlugin("elasticsearch", "elasticsearch_plugin:factory"),)
    assert entry.load_calls == 0


def test_explicit_retriever_loading_validates_v2_metadata() -> None:
    entry = _EntryPoint(
        "elasticsearch",
        "elasticsearch_plugin:factory",
        _retriever_factory,
        group="capicove.retrievers",
    )

    plugin = plugin_api.load_retriever_plugin(
        "elasticsearch",
        entry_points=(entry,),
    )

    assert plugin.metadata.capability == "retriever"
    assert plugin.metadata.contract_min == 2
    assert plugin.availability() == PluginAvailability(True)
    assert entry.load_calls == 1


def test_retriever_loading_rejects_incompatible_contract_range() -> None:
    entry = _EntryPoint(
        "elasticsearch",
        "elasticsearch_plugin:factory",
        _incompatible_retriever_factory,
        group="capicove.retrievers",
    )

    with pytest.raises(RuntimeError, match="contract range"):
        plugin_api.load_retriever_plugin(
            "elasticsearch",
            entry_points=(entry,),
        )


def test_explicit_v3_loading_does_not_change_legacy_default() -> None:
    entry = _EntryPoint(
        "visual", "visual:factory", _incompatible_retriever_factory, group="capicove.retrievers"
    )
    plugin = plugin_api.load_retriever_plugin("visual", entry_points=(entry,), contract_version=3)
    assert plugin.metadata.contract_min == 3
    with pytest.raises(RuntimeError, match="contract range"):
        plugin_api.load_retriever_plugin("visual", entry_points=(entry,))


@pytest.mark.parametrize("version", [True, 0, RETRIEVAL_CONTRACT_VERSION + 1])
def test_retriever_version_rejected_before_loading(version: int) -> None:
    entry = _EntryPoint(
        "visual", "visual:factory", _incompatible_retriever_factory, group="capicove.retrievers"
    )
    with pytest.raises(ValueError, match="contract version"):
        plugin_api.load_retriever_plugin("visual", entry_points=(entry,), contract_version=version)
    assert entry.load_calls == 0


def test_retriever_loading_rejects_execution_backend_metadata() -> None:
    entry = _EntryPoint(
        "elasticsearch",
        "elasticsearch_plugin:factory",
        _factory,
        group="capicove.retrievers",
    )

    with pytest.raises(TypeError, match="retriever capability"):
        plugin_api.load_retriever_plugin(
            "elasticsearch",
            entry_points=(entry,),
        )


def test_listing_document_parsers_does_not_load_plugin_module() -> None:
    entry = _EntryPoint(
        "paddleocr",
        "capicove_paddleocr:factory",
        _document_parser_factory,
        group="capicove.document_parsers",
    )

    assert list_document_parser_plugins(entry_points=(entry,)) == (
        DiscoveredPlugin("paddleocr", "capicove_paddleocr:factory"),
    )
    assert entry.load_calls == 0


def test_explicit_document_parser_loading_validates_contract_v1() -> None:
    entry = _EntryPoint(
        "paddleocr",
        "capicove_paddleocr:factory",
        _document_parser_factory,
        group="capicove.document_parsers",
    )

    plugin = load_document_parser_plugin("paddleocr", entry_points=(entry,))

    assert isinstance(plugin, DocumentParserPlugin)
    assert plugin.metadata.capability == "document_parser"
    assert entry.load_calls == 1


def test_document_parser_discovery_rejects_duplicates_and_unknown_names() -> None:
    entries = (
        _EntryPoint(
            "paddleocr",
            "one:factory",
            _document_parser_factory,
            group="capicove.document_parsers",
        ),
        _EntryPoint(
            "paddleocr",
            "two:factory",
            _document_parser_factory,
            group="capicove.document_parsers",
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        list_document_parser_plugins(entry_points=entries)
    with pytest.raises(LookupError, match="unknown"):
        load_document_parser_plugin("paddleocr", entry_points=())


def test_document_parser_loading_rejects_non_callable_and_wrong_factory() -> None:
    non_callable = _EntryPoint(
        "paddleocr",
        "broken:value",
        object(),
        group="capicove.document_parsers",
    )
    wrong_factory = _EntryPoint(
        "paddleocr",
        "wrong:factory",
        _factory,
        group="capicove.document_parsers",
    )

    with pytest.raises(TypeError, match="callable factory"):
        load_document_parser_plugin("paddleocr", entry_points=(non_callable,))
    with pytest.raises(TypeError, match="document parser"):
        load_document_parser_plugin("paddleocr", entry_points=(wrong_factory,))


def test_document_parser_loading_rejects_incompatible_contract_range() -> None:
    entry = _EntryPoint(
        "paddleocr",
        "incompatible:factory",
        lambda: _IncompatibleDocumentParserPlugin(),
        group="capicove.document_parsers",
    )

    with pytest.raises(RuntimeError, match="contract range"):
        load_document_parser_plugin("paddleocr", entry_points=(entry,))
