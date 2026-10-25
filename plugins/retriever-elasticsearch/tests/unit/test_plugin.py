"""Tests for the dependency-light Elasticsearch retriever descriptor."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType

import pytest

from capicove.plugins import (
    ConfigurableRetrieverPlugin,
    PluginAvailability,
    RetrieverPlugin,
)
from capicove.retrieval import TextRetrieverBackend


def _plugin_module() -> ModuleType:
    return importlib.import_module("capicove_retriever_elasticsearch.plugin")


def test_factory_returns_v2_retriever_metadata() -> None:
    module = _plugin_module()

    plugin = module.elasticsearch_retriever_plugin()

    assert isinstance(plugin, RetrieverPlugin)
    assert plugin.metadata.plugin_id == "capicove.retriever.elasticsearch"
    assert plugin.metadata.plugin_version == "0.1.0"
    assert plugin.metadata.capability == "retriever"
    assert plugin.metadata.contract_min == 2
    assert plugin.metadata.contract_max == 2
    assert plugin.metadata.optional_packages == ("elasticsearch",)
    assert plugin.metadata.external_binaries == ()
    assert plugin.metadata.network_access is True


def test_availability_reports_missing_elasticsearch_sdk(monkeypatch: object) -> None:
    module = _plugin_module()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)  # type: ignore[attr-defined]

    availability = module.elasticsearch_retriever_plugin().availability()

    assert availability == PluginAvailability(
        False,
        "install capicove-retriever-elasticsearch with Elasticsearch 8 support",
    )


def test_availability_checks_package_presence_without_importing_sdk(
    monkeypatch: object,
) -> None:
    module = _plugin_module()
    requested: list[str] = []

    def find_spec(name: str) -> object:
        requested.append(name)
        return object()

    monkeypatch.setattr(module.importlib.util, "find_spec", find_spec)  # type: ignore[attr-defined]

    availability = module.elasticsearch_retriever_plugin().availability()

    assert availability == PluginAvailability(True)
    assert requested == ["elasticsearch"]


def test_importing_plugin_package_does_not_import_elasticsearch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import capicove_retriever_elasticsearch; import sys; "
                "assert 'elasticsearch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_plugin_constructs_a_lazy_configured_runtime_without_importing_sdk() -> None:
    plugin = _plugin_module().elasticsearch_retriever_plugin()

    runtime = plugin.create({"hosts": ("https://localhost:9200",)})

    assert isinstance(plugin, ConfigurableRetrieverPlugin)
    assert isinstance(runtime, TextRetrieverBackend)
    assert "elasticsearch" not in sys.modules


def test_plugin_create_rejects_unreviewed_configuration() -> None:
    plugin = _plugin_module().elasticsearch_retriever_plugin()

    with pytest.raises(ValueError, match="configuration key"):
        plugin.create(
            {
                "hosts": ("https://localhost:9200",),
                "raw_dsl": {"script": "unreviewed"},
            }
        )
