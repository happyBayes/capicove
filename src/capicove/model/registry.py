"""Explicit dependency-light model provider registration and discovery."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

from capicove.model._entry_points import select_entry_points
from capicove.model.configuration import ModelCapability, ModelProfile
from capicove.model.http_profile import HttpModelProfile


def _validate(profile: ModelProfile | HttpModelProfile) -> None:
    """Default validation leaves provider-specific policy to its creator."""


def _close(instance: object) -> None:
    """Release runtimes supporting explicit cleanup without importing any SDK."""
    close = getattr(instance, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    """Versioned trusted factory descriptor; SDK imports belong in create."""

    provider_id: str
    version: str
    capability: ModelCapability
    create: Callable[[ModelProfile], object] | Callable[[HttpModelProfile], object]
    validate: Callable[[ModelProfile], None] | Callable[[HttpModelProfile], None] = _validate
    dispose: Callable[[object], None] = _close
    contract_version: int = 1

    def validate_profile(self, profile: ModelProfile | HttpModelProfile) -> None:
        """Dispatch the capability's matching profile shape before construction."""
        if (self.capability == ModelCapability.MULTIMODAL_GENERATION) != isinstance(
            profile, HttpModelProfile
        ):
            raise ValueError("provider profile transport mismatch")
        cast(Callable[[ModelProfile | HttpModelProfile], None], self.validate)(profile)

    def create_instance(self, profile: ModelProfile | HttpModelProfile) -> object:
        """Construct an already validated local or HTTP provider lazily."""
        return cast(Callable[[ModelProfile | HttpModelProfile], object], self.create)(profile)

    def __post_init__(self) -> None:
        if type(self.contract_version) is not int or self.contract_version != 1:
            raise ValueError("incompatible model provider contract version")
        if not isinstance(self.provider_id, str) or not re.fullmatch(
            r"[a-z][a-z0-9.-]{0,63}", self.provider_id
        ):
            raise ValueError("invalid provider ID")
        if not isinstance(self.version, str) or not self.version.strip() or len(self.version) > 64:
            raise ValueError("provider requires a bounded version")
        if not isinstance(self.capability, ModelCapability) or any(
            not callable(value) for value in (self.create, self.validate, self.dispose)
        ):
            raise ValueError("invalid provider definition")


class ProviderRegistry:
    """Register trusted descriptors explicitly; never load arbitrary config paths."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderDefinition] = {}

    def register(self, provider: ProviderDefinition) -> None:
        """Reject duplicate identities instead of overriding trusted providers."""
        if not isinstance(provider, ProviderDefinition):
            raise TypeError("provider definition required")
        if provider.provider_id in self._providers:
            raise ValueError("duplicate model provider")
        self._providers[provider.provider_id] = provider

    def resolve(self, provider_id: str) -> ProviderDefinition:
        """Find only an already registered provider."""
        try:
            return self._providers[provider_id]
        except KeyError as error:
            raise ValueError("unknown model provider") from error

    def load_plugin(
        self, provider_id: str, *, entry_points: Iterable[object] | None = None
    ) -> None:
        """Explicitly load one installed capicove.model_providers entry point."""
        if provider_id in self._providers:
            raise ValueError("duplicate model provider")
        matches = select_entry_points("capicove.model_providers", provider_id, entry_points)
        if len(matches) != 1:
            raise ValueError("model provider entry point must exist exactly once")
        entry = matches[0]
        loader = entry.load
        descriptor = loader()
        if not callable(descriptor):
            raise TypeError("model provider entry point must load a callable")
        provider = descriptor()
        if not isinstance(provider, ProviderDefinition) or provider.provider_id != provider_id:
            raise TypeError("incompatible model provider descriptor")
        self.register(provider)
