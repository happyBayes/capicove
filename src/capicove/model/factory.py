"""Bounded typed model leases with deterministic offline startup selection."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, cast

from capicove.model.configuration import ModelCapability, ModelConfiguration, ModelProfile
from capicove.model.http_profile import HttpModelProfile
from capicove.model.registry import ProviderDefinition, ProviderRegistry


if TYPE_CHECKING:
    from capicove.documents import DocumentParserBackend
    from capicove.model.audio import AudioTranscriptionProvider
    from capicove.model.multimodal_embedding import SharedSpaceEncoder
    from capicove.model.ports import ConfiguredTextEncoder
    from capicove.model.vlm import VlmProvider
    from capicove.processing.image.ocr_configuration import ConfiguredImageOcrProvider
    from capicove.retrieval.ports import MultimodalEmbeddingProvider
    from capicove.retrieval.reranking import TextReranker


class ModelCapacityError(RuntimeError):
    """Another active profile occupies all explicitly permitted model slots."""


@dataclass
class _Instance:
    value: object
    provider: ProviderDefinition
    leases: int = 0
    use_lock: RLock = field(default_factory=RLock)


def _require_port(value: object, capability: ModelCapability) -> None:
    methods = {
        ModelCapability.AUDIO_TRANSCRIPTION: ("transcribe", "close"),
        ModelCapability.TEXT_EMBEDDING: ("encode",),
        ModelCapability.TEXT_RERANKING: ("rerank",),
        ModelCapability.IMAGE_TEXT_EMBEDDING: ("rank", "embed_images", "embed_texts"),
        ModelCapability.IMAGE_OCR: ("recognize",),
        ModelCapability.DOCUMENT_PARSER: ("parse",),
        ModelCapability.MULTIMODAL_GENERATION: ("generate", "close"),
        ModelCapability.MULTIMODAL_EMBEDDING: (
            "encode_text_queries",
            "encode_text_documents",
            "encode_images",
            "encode_audio",
            "close",
        ),
    }[capability]
    if any(not callable(getattr(value, name, None)) for name in methods):
        raise TypeError("model provider returned an incompatible inference port")
    attributes = {
        ModelCapability.AUDIO_TRANSCRIPTION: ("model_id",),
        ModelCapability.TEXT_EMBEDDING: ("space",),
        ModelCapability.TEXT_RERANKING: ("model_id", "model_revision"),
        ModelCapability.IMAGE_TEXT_EMBEDDING: ("model_id", "model_revision", "device"),
        ModelCapability.IMAGE_OCR: ("provider_id", "provider_version", "fingerprint"),
        ModelCapability.DOCUMENT_PARSER: ("parser_id",),
        ModelCapability.MULTIMODAL_GENERATION: ("model_id",),
        ModelCapability.MULTIMODAL_EMBEDDING: ("provider_id", "space"),
    }[capability]
    if any(not hasattr(value, name) for name in attributes):
        raise TypeError("model provider lacks required identity attributes")


class ModelFactory:
    """Reuse exact profiles, serialize leases and evict only idle instances.

    References borrowed from a lease must not escape its context. max_instances
    bounds live provider objects, not a promise of an exact memory quota. Optional
    runtime dependencies and asset hashing occur only during explicit acquisition.
    """

    def __init__(
        self, configuration: ModelConfiguration, *, registry: ProviderRegistry | None = None
    ) -> None:
        if not isinstance(configuration, ModelConfiguration):
            raise TypeError("model configuration required")
        if registry is None:
            from capicove.model.providers import builtin_registry

            registry = builtin_registry()
        self.configuration = configuration
        self._providers: dict[str, ProviderDefinition] = {}
        for profile in configuration.profiles.values():
            provider = registry.resolve(profile.provider)
            if provider.capability != profile.capability:
                raise ValueError("model provider capability mismatch")
            provider.validate_profile(profile)
            self._providers[profile.name] = provider
        self._instances: OrderedDict[tuple[str, str], _Instance] = OrderedDict()
        self._lock = RLock()
        self._closed = False

    def __enter__(self) -> ModelFactory:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def profile(self, binding: str, capability: ModelCapability) -> ModelProfile | HttpModelProfile:
        """Resolve a configured capability binding before any model construction."""
        try:
            profile = self.configuration.profiles[self.configuration.bindings[binding]]
        except KeyError as error:
            raise ValueError("unknown model binding") from error
        if profile.capability != capability:
            raise ValueError("model binding capability mismatch")
        return profile

    @contextmanager
    def acquire(self, binding: str, capability: ModelCapability) -> Iterator[object]:
        """Borrow one serialized instance; capacity failure never evicts active work."""
        profile = self.profile(binding, capability)
        provider = self._providers[profile.name]
        key = (profile.identity, provider.version)
        with self._lock:
            if self._closed:
                raise RuntimeError("model factory is closed")
            instance = self._instances.get(key)
            if instance is None:
                if len(self._instances) >= self.configuration.max_instances:
                    victim = next((k for k, v in self._instances.items() if v.leases == 0), None)
                    if victim is None:
                        raise ModelCapacityError("all model slots have active leases")
                    previous = self._instances.pop(victim)
                    previous.provider.dispose(previous.value)
                    del previous
                value = provider.create_instance(profile)
                try:
                    _require_port(value, capability)
                except BaseException:
                    provider.dispose(value)
                    raise
                instance = _Instance(value, provider)
                self._instances[key] = instance
            self._instances.move_to_end(key)
            instance.leases += 1
        try:
            with instance.use_lock:
                yield instance.value
        finally:
            with self._lock:
                instance.leases -= 1

    def text_encoder(self, binding: str) -> AbstractContextManager[ConfiguredTextEncoder]:
        """Borrow the existing ordered text batch encoder port."""
        return cast(
            "AbstractContextManager[ConfiguredTextEncoder]",
            self.acquire(binding, ModelCapability.TEXT_EMBEDDING),
        )

    def text_reranker(self, binding: str) -> AbstractContextManager[TextReranker]:
        """Borrow a resident text reranker using the configured retrieval binding."""
        return cast(
            "AbstractContextManager[TextReranker]",
            self.acquire(binding, ModelCapability.TEXT_RERANKING),
        )

    def image_text_encoder(
        self, binding: str
    ) -> AbstractContextManager[MultimodalEmbeddingProvider]:
        """Borrow an aligned image/text embedding provider."""
        return cast(
            "AbstractContextManager[MultimodalEmbeddingProvider]",
            self.acquire(binding, ModelCapability.IMAGE_TEXT_EMBEDDING),
        )

    def image_ocr(self, binding: str) -> AbstractContextManager[ConfiguredImageOcrProvider]:
        """Borrow the public OCR port with its processing fingerprint."""
        return cast(
            "AbstractContextManager[ConfiguredImageOcrProvider]",
            self.acquire(binding, ModelCapability.IMAGE_OCR),
        )

    def document_parser(self, binding: str) -> AbstractContextManager[DocumentParserBackend]:
        """Borrow a public document parser backend."""
        return cast(
            "AbstractContextManager[DocumentParserBackend]",
            self.acquire(binding, ModelCapability.DOCUMENT_PARSER),
        )

    def audio_transcriber(self, binding: str) -> AbstractContextManager[AudioTranscriptionProvider]:
        """Borrow the configured local audio transcription provider."""
        return cast(
            "AbstractContextManager[AudioTranscriptionProvider]",
            self.acquire(binding, ModelCapability.AUDIO_TRANSCRIPTION),
        )

    def vlm(self, binding: str) -> AbstractContextManager[VlmProvider]:
        """Borrow a configured multimodal generation provider."""
        return cast(
            "AbstractContextManager[VlmProvider]",
            self.acquire(binding, ModelCapability.MULTIMODAL_GENERATION),
        )

    def shared_space_encoder(self, binding: str) -> AbstractContextManager[SharedSpaceEncoder]:
        """Borrow a shared text/image/audio embedding provider (RFC-0032)."""
        return cast(
            "AbstractContextManager[SharedSpaceEncoder]",
            self.acquire(binding, ModelCapability.MULTIMODAL_EMBEDDING),
        )

    def close(self) -> None:
        """Release all idle models; reject closing a factory with active leases."""
        with self._lock:
            if any(instance.leases for instance in self._instances.values()):
                raise RuntimeError("model factory has active leases")
            self._closed = True
            instances = tuple(self._instances.values())
            self._instances.clear()
            errors: list[Exception] = []
            for instance in instances:
                try:
                    instance.provider.dispose(instance.value)
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("model cleanup failed", errors)
