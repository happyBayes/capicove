"""Explicit LLM readiness sends only an authored probe and returns bounded metadata."""

import json
from pathlib import Path

import pytest

from capicove.cli.app import main
from capicove.graph_build.inference import DispatchControl, RawGeneration
from capicove.graph_build.profiles import StructuredRequest
from capicove.graph_build.remote_generation import OpenAICompatibleGenerator
from tests.unit.graph_build.test_remote_generation import profile_at


def test_explicit_generation_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = profile_at(tmp_path)
    path = tmp_path / "generation.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "endpoint": profile.endpoint,
                "model": profile.model,
                "model_revision": profile.model_revision,
                "context_limit": profile.context_limit,
                "network_enabled": True,
                "api_key_env": profile.api_key_env,
                "tokenizer": {
                    "root": str(profile.tokenizer_assets.root),
                    "files": dict(profile.tokenizer_assets.files),
                    "manifest_sha256": profile.tokenizer_assets.manifest_sha256,
                },
            }
        )
    )
    requests = []

    def generate(
        self: OpenAICompatibleGenerator, request: StructuredRequest, *, control: DispatchControl
    ) -> RawGeneration:
        requests.append(request)
        return RawGeneration(b'{"ready":true}', "stop", 50, 5)

    monkeypatch.setattr(OpenAICompatibleGenerator, "generate", generate)
    assert main(["setup", "check-llm", "--config", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is True
    assert result["revision_verification"] == "operator_attested"
    assert requests[0].input_segments == ()
    assert "api_key" not in result


def test_probe_rejects_unknown_secret_configuration_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"api_key":"private-test-value"}')  # pragma: allowlist secret
    assert main(["setup", "check-llm", "--config", str(path)]) == 2
    output = capsys.readouterr()
    assert "private-test-value" not in output.out + output.err
