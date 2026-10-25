"""Explicit host provider selection; never import model code for default memory."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from capicove.contracts.memory import MemoryQuery
from capicove.knowledge.ports import EntityFactExtractor
from capicove.runtime.memory_config import MemoryConfigurationError


@dataclass(frozen=True)
class MemoryProviders:
    extractor: EntityFactExtractor | None = None
    dense: Callable[[MemoryQuery], tuple[str, ...]] | None = None


def configured_providers(config: dict[str, Any]) -> MemoryProviders:
    factory = config.get("provider_factory")
    if factory is not None:
        if config.get("allow_provider_code") is not True:
            raise MemoryConfigurationError(
                "allow_provider_code",
                "Explicitly trust the configured local provider factory before loading its code.",
            )
        if not isinstance(factory, str) or factory.count(":") != 1:
            raise MemoryConfigurationError(
                "provider_factory", "Use an installed Python module:factory name."
            )
        if config.get("extractor", "none") != "none":
            raise MemoryConfigurationError(
                "extractor", "Choose the built-in extractor or the provider factory, not both."
            )
        module, name = factory.split(":")
        if not all(part.isidentifier() for part in (*module.split("."), name)):
            raise MemoryConfigurationError(
                "provider_factory", "Use an installed Python module:factory name."
            )
        try:
            creator = getattr(import_module(module), name)
        except (ImportError, AttributeError):
            raise MemoryConfigurationError(
                "provider_factory",
                "Install the selected host provider and check its module:factory name.",
            ) from None
        result = creator()
        if not isinstance(result, MemoryProviders):
            raise MemoryConfigurationError(
                "provider_factory", "Factory must return MemoryProviders."
            )
        return result
    if config.get("extractor") == "structured-json-v1":
        from capicove.knowledge._adapters.structured_json import StructuredJsonExtractor

        return MemoryProviders(extractor=StructuredJsonExtractor())
    return MemoryProviders()
