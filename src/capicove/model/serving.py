"""Dependency-light catalog and explicit model service selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from capicove.model.configuration import ModelCapability, ModelConfiguration, ModelProfile
from capicove.model.http_profile import HttpModelProfile


ROUTES = {
    ModelCapability.TEXT_EMBEDDING: ("/v1/embeddings",),
    ModelCapability.TEXT_RERANKING: ("/v1/rerank",),
    ModelCapability.AUDIO_TRANSCRIPTION: ("/v1/audio/transcriptions",),
    ModelCapability.MULTIMODAL_GENERATION: ("/v1/chat/completions", "/v1/count_tokens"),
    ModelCapability.IMAGE_TEXT_EMBEDDING: (
        "/v1/embeddings/image",
        "/v1/embeddings/text",
        "/v1/similarity",
    ),
    ModelCapability.IMAGE_OCR: ("/v1/ocr",),
    ModelCapability.DOCUMENT_PARSER: ("/v1/documents/parse",),
    # Shared-space embeddings are consumed in-process by unified retrieval only.
    ModelCapability.MULTIMODAL_EMBEDDING: (),
}


def select_models(
    configuration: ModelConfiguration, models: tuple[str, ...] | None
) -> ModelConfiguration:
    """Serve only explicitly bound profiles, enforcing the configured resident budget."""
    names = (
        tuple(
            name
            for name in dict.fromkeys(configuration.bindings.values())
            if ROUTES[configuration.profiles[name].capability]
        )
        if models is None
        else models
    )
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in configuration.bindings.values() for name in names)
    ):
        raise ValueError("select distinct bound model names")
    if len(names) > configuration.max_instances:
        raise ValueError("selected models exceed max_instances")
    if any(
        name not in configuration.profiles or not ROUTES[configuration.profiles[name].capability]
        for name in names
    ):
        raise ValueError("selected model capability has no served endpoints")
    return ModelConfiguration(
        {name: configuration.profiles[name] for name in names},
        {key: name for key, name in configuration.bindings.items() if name in names},
        configuration.max_instances,
    )


def catalog(configuration: ModelConfiguration) -> list[dict[str, object]]:
    """Describe routes and identities without importing model or web dependencies."""
    result = []
    for name, profile in configuration.profiles.items():
        item: dict[str, object] = {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "capicove",
            "capability": profile.capability.value,
            "provider": profile.provider,
            "profile_identity": profile.identity,
            "endpoints": list(ROUTES[profile.capability]),
        }
        if isinstance(profile, HttpModelProfile):
            from capicove.model.vlm_service import model_info

            item.update(model_info(profile))
        result.append(item)
    return result


def document_binding(
    profile: ModelProfile, backend: DocumentParserBackend
) -> DocumentParserBinding:
    """Bind parser receipts to the verified installer's actual profile version."""
    from capicove.documents import DocumentParserBinding
    from capicove.model.installers import load_model_installer

    if profile.provider != "paddle-document":
        raise ValueError("document HTTP binding requires a qualified installer")
    plan = load_model_installer("paddleocr").model_plan("book-balanced", profile.root)
    if getattr(plan, "manifest_sha256", None) != profile.manifest_sha256:
        raise ValueError("document manifest mismatch")
    identity, version = getattr(plan, "profile_id", None), getattr(plan, "profile_version", None)
    if not isinstance(identity, str) or type(version) is not int:
        raise ValueError("invalid document installation identity")
    return DocumentParserBinding(backend, identity, version)


if TYPE_CHECKING:
    from capicove.documents import DocumentParserBackend, DocumentParserBinding
