"""Builtin descriptors; creating a registry never imports inference runtimes."""

from capicove.model.configuration import ModelCapability
from capicove.model.http_profile import HttpModelProfile
from capicove.model.providers.audio import create_audio, validate_audio
from capicove.model.providers.embeddinggemma import (
    create_embeddinggemma,
    create_embeddinggemma_code,
    validate_embeddinggemma,
    validate_embeddinggemma_code,
)
from capicove.model.providers.local import (
    create_bge,
    create_paddle_ocr,
    create_paddle_parser,
    create_reranker,
    create_siglip2,
    validate_bge,
    validate_paddle,
    validate_reranker,
    validate_siglip2,
)
from capicove.model.registry import ProviderDefinition, ProviderRegistry


def _create_vlm(profile: HttpModelProfile) -> object:
    from capicove.model.providers.openai_vlm import OpenAICompatibleVlm

    return OpenAICompatibleVlm(profile)


def builtin_registry() -> ProviderRegistry:
    """Register the locally qualified providers without loading model assets."""
    registry = ProviderRegistry()
    for provider in (
        ProviderDefinition(
            "whisper-cpp", "1", ModelCapability.AUDIO_TRANSCRIPTION, create_audio, validate_audio
        ),
        ProviderDefinition("openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, _create_vlm),
        ProviderDefinition("bge-m3", "1", ModelCapability.TEXT_EMBEDDING, create_bge, validate_bge),
        ProviderDefinition(
            "bge-reranker-v2-m3",
            "1",
            ModelCapability.TEXT_RERANKING,
            create_reranker,
            validate_reranker,
        ),
        ProviderDefinition(
            "siglip2", "1", ModelCapability.IMAGE_TEXT_EMBEDDING, create_siglip2, validate_siglip2
        ),
        ProviderDefinition(
            "paddleocr", "1", ModelCapability.IMAGE_OCR, create_paddle_ocr, validate_paddle
        ),
        ProviderDefinition(
            "embeddinggemma2-llamacpp",
            "1",
            ModelCapability.MULTIMODAL_EMBEDDING,
            create_embeddinggemma,
            validate_embeddinggemma,
        ),
        ProviderDefinition(
            "embeddinggemma2-llamacpp-code",
            "1",
            ModelCapability.TEXT_EMBEDDING,
            create_embeddinggemma_code,
            validate_embeddinggemma_code,
        ),
        ProviderDefinition(
            "paddle-document",
            "1",
            ModelCapability.DOCUMENT_PARSER,
            create_paddle_parser,
            validate_paddle,
        ),
    ):
        registry.register(provider)
    return registry
