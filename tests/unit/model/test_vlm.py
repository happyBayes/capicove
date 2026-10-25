"""Generic endpoint registration and immutable multimodal request contracts."""

import base64
from dataclasses import replace
from typing import Any

import pytest

from capicove.model import HttpModelProfile, ModelCapability, ModelConfiguration, ModelFactory
from capicove.model.vlm import VlmMedia, VlmMessage, VlmRequest, VlmText, parse_chat_request


def configuration(model: str = "example/vision-model") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "profiles": {
            "vision": {
                "transport": "http",
                "provider": "openai-vlm",
                "capability": "multimodal_generation",
                "endpoint": "http://127.0.0.1:8000/v1",
                "model": model,
                "allow_network": True,
                "modalities": ["image", "video"],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
        "bindings": {"assistant.vision": "vision"},
    }


def test_generic_model_registers_without_local_assets_or_model_sdk() -> None:
    config = ModelConfiguration.from_mapping(configuration())
    with ModelFactory(config) as factory:
        with factory.vlm("assistant.vision") as provider:
            assert provider.model_id == "example/vision-model"
        with factory.vlm("assistant.vision") as same:
            assert same is provider
    assert config.profiles["vision"].capability == ModelCapability.MULTIMODAL_GENERATION


@pytest.mark.parametrize(
    "change",
    [
        {"allow_network": False},
        # Synthetic invalid credentials exercise configuration boundaries.
        {"endpoint": "http://user:password@localhost:8000/v1"},  # pragma: allowlist secret
        {"endpoint": "http://example.com/v1"},
        {"endpoint": "http://localhost:8000/v1?secret=yes"},
        {"api_key_env": "OPENAI_API_KEY"},  # pragma: allowlist secret
        {"api_key": "literal"},  # pragma: allowlist secret
        {"modalities": ["audio"]},
        {"timeout_seconds": 0},
        {"max_images": True},
        {"chat_template_kwargs": {"nested": {"a": 1}}},
    ],
)
def test_endpoint_configuration_rejects_unsafe_or_unsupported_policy(
    change: dict[str, Any],
) -> None:
    raw = configuration()
    raw["profiles"]["vision"].update(change)
    with pytest.raises(ValueError):
        ModelConfiguration.from_mapping(raw)


def test_profile_identity_is_immutable_alias_independent_and_model_specific() -> None:
    raw = configuration()
    profile = ModelConfiguration.from_mapping(raw).profiles["vision"]
    assert isinstance(profile, HttpModelProfile)
    assert replace(profile, name="another").identity == profile.identity
    assert (
        ModelConfiguration.from_mapping(configuration("another/model")).profiles["vision"].identity
        != profile.identity
    )
    raw["profiles"]["vision"]["chat_template_kwargs"]["enable_thinking"] = True
    assert profile.chat_template_kwargs["enable_thinking"] is False


def test_media_order_hashes_and_openai_request_roundtrip() -> None:
    image = VlmMedia(b"\x89PNG\r\n\x1a\nfixture", "image/png")
    request = VlmRequest((VlmMessage("user", (VlmText("compare"), image, image)),))
    raw = request.to_openai("example/model")
    assert parse_chat_request(raw) == request
    assert len(request.media_sha256) == 2
    assert request.media_sha256[0] == image.sha256
    assert "data:image/png;base64," in str(raw)


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"messages": [{"role": "user", "content": "hi"}], "stream": True},
        {"messages": [{"role": "user", "content": "hi"}], "n": 2},
        {"messages": [{"role": "user", "content": "hi"}], "tools": [{}]},
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": True},
        {"messages": [{"role": "user", "content": "hi"}], "temperature": float("nan")},
    ],
)
def test_invalid_requests_are_rejected(body: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_chat_request(body)


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "https://example.com/a.png", "data:image/png;base64,!!!"]
)
def test_media_url_cannot_fetch_remote_or_local_resources(url: str) -> None:
    with pytest.raises(ValueError):
        parse_chat_request(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
                ]
            }
        )


def test_media_type_must_match_signature() -> None:
    with pytest.raises(ValueError):
        VlmMedia(b"not-an-image", "image/png")
    uri = "data:video/mp4;base64," + base64.b64encode(b"\0\0\0\x18ftypisomfixture").decode()
    with pytest.raises(ValueError):
        parse_chat_request(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}
                ]
            }
        )


def test_aggregate_limit_rejects_before_encoding_media(monkeypatch: pytest.MonkeyPatch) -> None:
    image = VlmMedia(b"\x89PNG\r\n\x1a\n" + b"x" * (10 * 1024 * 1024), "image/png")

    def encode(self: VlmMedia) -> str:
        raise AssertionError("oversized request encoded before checking aggregate size")

    monkeypatch.setattr(VlmMedia, "data_uri", encode)
    with pytest.raises(ValueError, match="size limit"):
        VlmRequest((VlmMessage("user", (image,) * 4),))


def test_openai_image_detail_is_preserved() -> None:
    image = VlmMedia(b"\x89PNG\r\n\x1a\nfixture", "image/png", detail="high")
    request = VlmRequest((VlmMessage("user", (image,)),))
    assert parse_chat_request(request.to_openai("model")) == request
