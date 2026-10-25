"""Only explicit compatible entry points can extend model composition."""

from dataclasses import dataclass

import pytest

from capicove.model import ModelCapability, ProviderDefinition, ProviderRegistry


@dataclass
class Entry:
    name: str = "fake"
    group: str = "capicove.model_providers"
    loaded: bool = False

    def was_loaded(self) -> bool:
        return self.loaded

    def load(self) -> object:
        self.loaded = True
        return lambda: ProviderDefinition(
            "fake", "1", ModelCapability.TEXT_EMBEDDING, lambda p: object()
        )


def test_plugin_loading_is_explicit_and_duplicates_fail_before_import() -> None:
    registry = ProviderRegistry()
    entry = Entry()
    assert not entry.was_loaded()
    registry.load_plugin("fake", entry_points=(entry,))
    assert entry.was_loaded() and registry.resolve("fake").version == "1"
    duplicate = Entry()
    with pytest.raises(ValueError, match="duplicate"):
        registry.load_plugin("fake", entry_points=(duplicate,))
    assert not duplicate.loaded


def test_unknown_and_ambiguous_plugins_are_rejected() -> None:
    registry = ProviderRegistry()
    with pytest.raises(ValueError):
        registry.resolve("unknown")
    with pytest.raises(ValueError):
        registry.load_plugin("missing", entry_points=())
    first, second = Entry(), Entry()
    with pytest.raises(ValueError):
        registry.load_plugin("fake", entry_points=(first, second))
    assert not first.loaded and not second.loaded


@pytest.mark.parametrize(
    "change",
    [
        {"provider_id": "package:Class"},
        {"version": ""},
        {"contract_version": 2},
        {"contract_version": True},
        {"create": None},
        {"capability": "arbitrary"},
    ],
)
def test_invalid_descriptors_are_rejected(change: dict[str, object]) -> None:
    values: dict[str, object] = {
        "provider_id": "fake",
        "version": "1",
        "capability": ModelCapability.TEXT_EMBEDDING,
        "create": lambda p: object(),
    }
    with pytest.raises(ValueError):
        ProviderDefinition(**{**values, **change})  # type: ignore[arg-type]
