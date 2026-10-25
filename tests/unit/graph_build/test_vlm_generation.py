"""Knowledge generation selects the same registered VLM as HTTP and CLI."""

import json
from pathlib import Path

import pytest

from capicove.graph_build.remote_generation import (
    OpenAICompatibleTokenizer,
    load_remote_generation_profile,
)
from tests.unit.model.test_vlm import configuration


def test_registered_generation_identity_and_tokenizer_requirement(tmp_path: Path) -> None:
    raw = configuration()
    name = next(iter(raw["profiles"]))
    raw["profiles"][name]["token_counting"] = "vllm"
    raw["profiles"][name]["model_revision"] = "revision-one"
    models = tmp_path / "models.json"
    models.write_text(json.dumps(raw))
    path = tmp_path / "generation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "models": str(models),
                "generation_model": name,
                "context_limit": 4096,
            }
        )
    )
    profile = load_remote_generation_profile(path)
    assert profile.model == raw["profiles"][name]["model"]
    assert OpenAICompatibleTokenizer(profile).tokenizer_id.startswith("vllm:")
    identity = profile.provider_id
    raw["profiles"][name]["model_revision"] = "revision-two"
    models.write_text(json.dumps(raw))
    assert load_remote_generation_profile(path).provider_id != identity
    raw["profiles"][name].pop("token_counting")
    models.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="token"):
        load_remote_generation_profile(path)


def test_registered_generation_uses_shared_port_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time
    from dataclasses import replace

    from capicove.graph_build.extraction import extraction_request
    from capicove.graph_build.inference import DispatchControl
    from capicove.graph_build.remote_generation import OpenAICompatibleGenerator
    from capicove.graph_build.vlm_generation import RegisteredGenerationProfile
    from capicove.model import (
        ModelCapability,
        ModelConfiguration,
        ModelFactory,
        ProviderDefinition,
        ProviderRegistry,
    )
    from capicove.model.http_profile import HttpModelProfile
    from capicove.model.vlm import VlmRequest, VlmResult, VlmTokenCount, VlmUsage
    from tests.unit.graph_build.test_extraction import build_profile, unit

    raw = configuration()
    profile = ModelConfiguration.from_mapping(raw).profiles["vision"]
    assert isinstance(profile, HttpModelProfile)
    selected = RegisteredGenerationProfile(
        replace(profile, token_counting="vllm", model_revision="test"), 32768
    )
    tokenizer = OpenAICompatibleTokenizer(selected)
    policy = replace(
        build_profile().inference,
        provider_id=selected.provider_id,
        model_id=selected.model,
        model_revision=selected.model_revision,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        context_window_tokens=32768,
        max_output_tokens=128,
    )
    request = extraction_request(unit(), replace(build_profile(), inference=policy))

    class Provider:
        model_id = selected.model
        input_tokens = 42

        def __init__(self) -> None:
            self.calls: list[VlmRequest] = []

        def count_tokens(self, request: VlmRequest) -> VlmTokenCount:
            self.calls.append(request)
            return VlmTokenCount(self.model_id, 42, 32768, "a" * 64, "b" * 64)

        def generate(self, request: VlmRequest) -> VlmResult:
            self.calls.append(request)
            return VlmResult(
                self.model_id,
                '{"entities":[],"relationships":[]}',
                "stop",
                VlmUsage(self.input_tokens, 10),
                "a" * 64,
                "b" * 64,
                (),
            )

        def close(self) -> None:
            pass

    provider = Provider()
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, lambda p: provider
        )
    )
    monkeypatch.setattr(
        "capicove.graph_build.vlm_generation.ModelFactory",
        lambda config: ModelFactory(config, registry=registry),
    )
    control = DispatchControl(time.monotonic() + 10, time.monotonic, lambda: False)
    result = OpenAICompatibleGenerator(selected).generate(request, control=control)
    assert result.input_tokens == 42
    assert provider.calls[0] == provider.calls[1]
    assert provider.calls[0].json_object
    provider.input_tokens = 43
    with pytest.raises(ValueError, match="drift"):
        OpenAICompatibleGenerator(selected).generate(request, control=control)
    before = len(provider.calls)
    with pytest.raises(TimeoutError):
        OpenAICompatibleGenerator(selected).generate(request, control=replace(control, deadline=0))
    assert len(provider.calls) == before


@pytest.mark.parametrize("status", [429, 504])
def test_registered_token_count_maps_retryable_errors(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    from dataclasses import replace

    from capicove.graph_build.extraction import extraction_request
    from capicove.graph_build.inference import DispatchControl, RateLimitError
    from capicove.graph_build.vlm_generation import (
        RegisteredGenerationProfile,
        RegisteredVlmTokenizer,
    )
    from capicove.model import HttpModelProfile, ModelConfiguration
    from capicove.model.providers.openai_vlm import OpenAICompatibleVlm
    from capicove.model.vlm import VlmError
    from tests.unit.graph_build.test_extraction import build_profile, unit

    profile = ModelConfiguration.from_mapping(configuration()).profiles["vision"]
    assert isinstance(profile, HttpModelProfile)
    selected = RegisteredGenerationProfile(
        replace(profile, token_counting="vllm", model_revision="test"), 32768
    )
    tokenizer = RegisteredVlmTokenizer(selected)
    policy = replace(
        build_profile().inference,
        provider_id=selected.provider_id,
        model_id=selected.model,
        model_revision=selected.model_revision,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        context_window_tokens=32768,
    )
    request = extraction_request(unit(), replace(build_profile(), inference=policy))

    def fail(*args: object, **kwargs: object) -> None:
        raise VlmError("upstream_failure", status)

    monkeypatch.setattr(OpenAICompatibleVlm, "count_tokens", fail)
    with pytest.raises(RateLimitError if status == 429 else TimeoutError):
        tokenizer.count_request_controlled(
            request, control=DispatchControl(10, lambda: 0, lambda: False)
        )
