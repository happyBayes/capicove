"""Trusted speech startup configuration rejects malformed or unsafe authority."""

import json
from pathlib import Path

import pytest

from capicove.runtime.speech import SpeechRuntime
from tests.speech_support import speech_fixture


@pytest.mark.parametrize("field", ["model_config", "output_root", "signing_key_env"])
def test_missing_runtime_fields_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    body = json.loads(fixture.config_path.read_text())
    del body[field]
    fixture.config_path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="configuration"):
        SpeechRuntime(fixture.config_path)
    assert fixture.provider.requests == []


def test_configuration_cannot_be_under_a_readable_media_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    body = json.loads(fixture.config_path.read_text())
    body["read_roots"] = [str(tmp_path.resolve())]
    fixture.config_path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="outside"):
        SpeechRuntime(fixture.config_path)


@pytest.mark.parametrize("value", ["", "00", "invalid-hex"])
def test_invalid_provenance_key_prevents_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("CAPICOVE_TEST_SPEECH_KEY", value)
    with pytest.raises(ValueError):
        SpeechRuntime(fixture.config_path)
    assert fixture.provider.requests == []


def test_runtime_schema_version_rejects_boolean_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    body = json.loads(fixture.config_path.read_text())
    body["schema_version"] = True
    fixture.config_path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="configuration"):
        SpeechRuntime(fixture.config_path)
