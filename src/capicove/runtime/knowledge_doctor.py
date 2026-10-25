"""Read-only readiness checks for explicitly selected native knowledge runtimes."""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from capicove.runtime.knowledge_config import KnowledgeConfiguration, read_configuration
from capicove.runtime.media_handoff import verify_asset


def doctor(config: KnowledgeConfiguration) -> dict[str, Any]:
    """Validate configuration and inventories without sending evidence or downloading."""
    result: dict[str, Any] = {
        "lexical": "ready",
        "generation": "disabled",
        "embedding": "disabled",
        "documents": "disabled",
        "audio": "disabled",
        "graph": "disabled",
    }
    if config.models:
        from capicove.model import ModelCapability, ModelFactory, load_configuration

        with ModelFactory(load_configuration(config.models)) as factory:
            factory.profile("knowledge.embedding", ModelCapability.TEXT_EMBEDDING)
            result["embedding"] = "configured_runtime_unprobed"
            for binding, capability in (
                ("knowledge.ocr", ModelCapability.IMAGE_OCR),
                ("knowledge.parser", ModelCapability.DOCUMENT_PARSER),
            ):
                try:
                    factory.profile(binding, capability)
                    result[binding] = "configured_runtime_unprobed"
                except ValueError:
                    result[binding] = "not_configured"
            result["documents"] = (
                "configured_runtime_unprobed"
                if any(
                    result[binding] == "configured_runtime_unprobed"
                    for binding in ("knowledge.ocr", "knowledge.parser")
                )
                else "not_configured"
            )
    if config.generation:
        from capicove.graph_build.remote_generation import load_remote_generation_profile

        profile = load_remote_generation_profile(config.generation)
        result["generation"] = (
            "configured_runtime_unprobed"
            if profile.api_key_env is None or os.environ.get(profile.api_key_env)
            else "credential_missing"
        )
    if config.audio:
        from pathlib import Path

        raw = read_configuration(config.audio)
        for name in ("model", "vad_model", "whisper_cli", "whisper_vad_cli", "ffmpeg", "ffprobe"):
            record = raw["files"][name]
            verify_asset(Path(record["path"]), record["sha256"])
        result["audio"] = "assets_verified_runtime_unprobed"
    if config.graph:
        result["graph"] = (
            "configured_runtime_unprobed"
            if all(
                importlib.util.find_spec(name) is not None
                for name in ("pyarrow", "networkx", "transformers")
            )
            else "dependencies_missing"
        )
    return result
