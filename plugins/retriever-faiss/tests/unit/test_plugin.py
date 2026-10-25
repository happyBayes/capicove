"""Tests for dependency-light FAISS plugin discovery."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from capicove_retriever_faiss.plugin import faiss_retriever_plugin


def test_metadata_and_availability_do_not_import_faiss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("faiss", None)
    probes: list[str] = []

    def find_spec(name: str) -> None:
        probes.append(name)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    plugin = faiss_retriever_plugin()
    availability = plugin.availability()

    assert plugin.metadata.plugin_id == "capicove.retriever.faiss"
    assert plugin.metadata.contract_min == plugin.metadata.contract_max == 3
    assert plugin.metadata.optional_packages == ("faiss",)
    assert plugin.metadata.network_access is False
    assert availability.available is False
    assert probes == ["faiss"]
    assert "faiss" not in sys.modules


def test_create_fails_closed_before_import_when_faiss_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("faiss", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)

    with pytest.raises(RuntimeError, match="FAISS runtime is unavailable"):
        faiss_retriever_plugin().create({"index_root": tmp_path.resolve(), "threads": 1})

    assert "faiss" not in sys.modules
