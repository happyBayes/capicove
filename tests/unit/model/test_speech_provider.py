"""Local speech profiles verify exact assets without inference or downloads."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from capicove.model import ModelCapability, ModelConfiguration, ModelFactory, ModelProfile
from capicove.model.providers.speech import validate_speech
from capicove.processing.audio.synthesis import SynthesisError


def speech_profile(root: Path) -> ModelProfile:
    """Create tiny authored asset placeholders, never real model weights."""
    options: dict[str, str | int | float | bool] = {"backend_version": "authored-test"}
    for key, name in (
        ("executable", "authored-tts"),
        ("model", "model.gguf"),
        ("mmproj", "mmproj.gguf"),
    ):
        path = root.resolve() / name
        path.write_bytes(("authored placeholder: " + name).encode())
        options[key] = str(path) if key == "executable" else name
        options[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = sorted((str(options[k]), str(options[k + "_sha256"])) for k in ("model", "mmproj"))
    manifest = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
    return ModelProfile(
        "tts",
        "llama-cpp-tts",
        ModelCapability.AUDIO_SYNTHESIS,
        root.resolve(),
        manifest,
        options=options,
    )


def test_factory_verifies_assets_only_when_speech_lease_is_acquired(tmp_path: Path) -> None:
    profile = speech_profile(tmp_path)
    factory = ModelFactory(ModelConfiguration({"tts": profile}, {"tts": "tts"}))
    (tmp_path / "model.gguf").write_bytes(b"tampered")
    try:
        with pytest.raises(SynthesisError, match="asset_integrity"):
            with factory.speech():
                pytest.fail("changed weights must not be leased")
    finally:
        factory.close()


def test_valid_profile_builds_an_idle_offline_provider(tmp_path: Path) -> None:
    profile = speech_profile(tmp_path)
    factory = ModelFactory(ModelConfiguration({"tts": profile}, {"tts": "tts"}))
    with factory.speech() as provider:
        assert provider.features.reference_audio
        assert not provider.features.reference_text
        assert not provider.features.preset_voice
        assert provider.lineage["provider"] == "llama-cpp-tts"
    factory.close()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("threads", True),
        ("threads", 17),
        ("max_frames", 751),
        ("timeout_seconds", 601),
        ("executable", "relative"),
        ("model", "../foreign.gguf"),
        ("model_sha256", "invalid"),
        ("shell", True),
    ],
)
def test_invalid_provider_configuration_fails_before_asset_use(
    tmp_path: Path,
    key: str,
    value: str | int | float | bool,
) -> None:
    profile = speech_profile(tmp_path)
    with pytest.raises(ValueError):
        validate_speech(replace(profile, options={**profile.options, key: value}))


def test_bad_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest"):
        validate_speech(replace(speech_profile(tmp_path), manifest_sha256="0" * 64))


def test_cpu_provider_does_not_accept_implicit_cuda_fallback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="device"):
        validate_speech(replace(speech_profile(tmp_path), device="cuda"))
