"""Authored audio and provider fixtures; never evidence of real-model quality."""

from __future__ import annotations

import hashlib
import json
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from capicove.model import ModelCapability, ModelConfiguration, ModelFactory
from capicove.model.registry import ProviderDefinition, ProviderRegistry
from capicove.processing.audio.synthesis import SpeechFeatures, SpeechRequest


def write_wave(path: Path, *, seconds: int = 1, silent: bool = False) -> None:
    """Write a small, deterministic PCM fixture with no recorded user content."""
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes((b"\0\0" if silent else b"\x01\0") * 24000 * seconds)


class AuthoredSpeechProvider:
    """Exercise contracts without a model, native binary, or network access."""

    model_id = "authored-contract-fixture"
    features = SpeechFeatures(("zh", "en"), reference_audio=True)

    def __init__(self) -> None:
        self.lineage: Mapping[str, object] = {"provider": "authored-test-provider"}
        self.requests: list[SpeechRequest] = []
        self.output_mode = "valid"
        self.after_generation: Callable[[], None] | None = None
        self.closed = False

    def synthesize(self, request: SpeechRequest, output: Path, *, deadline: datetime) -> None:
        self.requests.append(request)
        if self.output_mode == "malformed":
            output.write_bytes(b"authored invalid WAV")
        elif self.output_mode == "empty":
            output.touch()
        else:
            write_wave(output, silent=self.output_mode == "silent")
        if self.after_generation is not None:
            self.after_generation()

    def close(self) -> None:
        self.closed = True


@dataclass
class SpeechFixture:
    """Trusted local configuration for transport and authorization tests."""

    config_path: Path
    output_root: Path
    reference: Path
    provider: AuthoredSpeechProvider
    key: bytes


def speech_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SpeechFixture:
    """Use the real factory and application with an explicitly marked fake port."""
    from capicove.runtime import speech

    root = tmp_path.resolve()
    inputs = root / "inputs"
    inputs.mkdir()
    reference = inputs / "reference.wav"
    write_wave(reference, seconds=3)
    model_path = root / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bindings": {"tts": "authored"},
                "profiles": {
                    "authored": {
                        "provider": "authored-speech",
                        "capability": "audio_synthesis",
                        "root": str(root),
                        "manifest_sha256": "a" * 64,
                    }
                },
            }
        )
    )
    provider = AuthoredSpeechProvider()
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "authored-speech", "1", ModelCapability.AUDIO_SYNTHESIS, lambda _: provider
        )
    )

    def factory(config: ModelConfiguration) -> ModelFactory:
        return ModelFactory(config, registry=registry)

    monkeypatch.setattr(speech, "ModelFactory", factory)
    key = b"authored-test-key-for-provenance-32"
    monkeypatch.setenv("CAPICOVE_TEST_SPEECH_KEY", key.hex())
    config_path = root / "speech.json"
    output_root = root / "outputs"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_config": str(model_path),
                "binding": "tts",
                "output_root": str(output_root),
                "read_roots": [str(inputs)],
                "authorizations": {
                    hashlib.sha256(reference.read_bytes()).hexdigest(): "authored-grant"
                },
                "signing_key_env": "CAPICOVE_TEST_SPEECH_KEY",
            }
        )
    )
    config_path.chmod(0o600)
    return SpeechFixture(config_path, output_root, reference, provider, key)
