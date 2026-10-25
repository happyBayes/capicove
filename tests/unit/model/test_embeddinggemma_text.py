"""Code-retrieval text_embedding adapter over the pinned llama.cpp EmbeddingGemma 2 runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.retrieval import QueryModality, TargetType
from capicove.model.configuration import ModelCapability, ModelProfile
from capicove.model.providers.embeddinggemma import (
    CODE_PROVIDER_ID,
    create_embeddinggemma_code,
    manifest_digest,
    validate_embeddinggemma_code,
)
from tests.unit.model.test_multimodal_embedding import _profile as _multimodal_profile


pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires Unix sockets")


def _profile(tmp_path: Path, mode: str = "ok") -> tuple[ModelProfile, Path]:
    base, log = _multimodal_profile(tmp_path, mode)
    return (
        ModelProfile(
            "code",
            CODE_PROVIDER_ID,
            ModelCapability.TEXT_EMBEDDING,
            base.root,
            manifest_digest(base.options),
            "cpu",
            dict(base.options),
        ),
        log,
    )


def _requests(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text().splitlines() if "items" in line]


def test_code_space_is_text_only_and_distinct_from_the_multimodal_space(tmp_path: Path) -> None:
    from capicove.model.providers.embeddinggemma import space_for

    profile, _ = _profile(tmp_path)
    validate_embeddinggemma_code(profile)
    encoder = create_embeddinggemma_code(profile)
    try:
        space = encoder.space  # type: ignore[attr-defined]
        assert space.query_modalities == frozenset({QueryModality.TEXT})
        assert space.target_modalities == frozenset({TargetType.TEXT_CHUNK})
        assert space.space_id != space_for(profile).space_id
        assert space.space_id.startswith("embeddinggemma2-llamacpp-code:")
    finally:
        encoder.close()  # type: ignore[attr-defined]


def test_roles_apply_code_query_and_document_prompts_in_order(tmp_path: Path) -> None:
    profile, log = _profile(tmp_path)
    encoder = create_embeddinggemma_code(profile)
    try:
        space = encoder.space  # type: ignore[attr-defined]
        queries = encoder.encode(("parse args",), space=space, role="query")  # type: ignore[attr-defined]
        documents = encoder.encode(  # type: ignore[attr-defined]
            ("src/a.py\ndef a(): pass", "src/b.py\ndef b(): pass"), space=space, role="document"
        )
        assert len(queries) == 1 and len(documents) == 2
        assert all(item.space == space and len(item.vector) == 768 for item in queries + documents)
        sent = _requests(log)
        assert sent[0]["items"] == [{"text": "task: code retrieval | query: parse args"}]
        assert sent[1]["items"] == [
            {"text": "title: none | text: src/a.py\ndef a(): pass"},
            {"text": "title: none | text: src/b.py\ndef b(): pass"},
        ]
        with pytest.raises(ValueError):
            encoder.encode(("x",), space=space, role="passage")  # type: ignore[attr-defined]
        from capicove.model.providers.embeddinggemma import space_for

        with pytest.raises(ValueError):
            encoder.encode(("x",), space=space_for(profile), role="query")  # type: ignore[attr-defined]
    finally:
        encoder.close()  # type: ignore[attr-defined]


def test_factory_exposes_it_as_a_text_encoder(tmp_path: Path) -> None:
    from capicove.model.configuration import ModelConfiguration
    from capicove.model.factory import ModelFactory

    profile, _ = _profile(tmp_path)
    factory = ModelFactory(ModelConfiguration({"code": profile}, {"code": "code"}))
    with factory.text_encoder("code") as encoder:
        assert encoder.space.dimension == 768
    factory.close()
