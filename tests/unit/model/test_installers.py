"""Installer discovery is independent of document parser discovery."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from capicove.model.installers import ModelInstaller, load_model_installer


class Installer:
    def model_plan(self, profile_id: str, destination: Path) -> object:
        return object()

    def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> object:
        return object()


@dataclass
class Entry:
    name: str = "detector"
    group: str = "capicove.model_installers"

    def load(self) -> object:
        return Installer


def test_installer_does_not_require_a_document_parser() -> None:
    result = load_model_installer("detector", entry_points=(Entry(),))
    assert isinstance(result, ModelInstaller)


def test_duplicate_installer_never_falls_back() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_model_installer("detector", entry_points=(Entry(), Entry()))


def test_legacy_parser_installer_remains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import capicove.plugins

    monkeypatch.setattr(capicove.plugins, "load_document_parser_plugin", lambda name: Installer())
    assert isinstance(load_model_installer("legacy", entry_points=()), ModelInstaller)
