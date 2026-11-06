"""Real CLI and MCP composition with an authored provider, not real-model QA."""

import json
from pathlib import Path
from typing import Any

import pytest

from capicove.cli.app import main
from capicove.mcp.bindings import invoke_binding
from capicove.runtime.mcp_config import load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host
from tests.speech_support import SpeechFixture, speech_fixture
from tests.unit.runtime.test_mcp_config import workspace_config_body


def host_config(fixture: SpeechFixture) -> Path:
    root = fixture.config_path.parent
    root.chmod(0o700)
    body = workspace_config_body(root)
    body.update(
        {
            "profile": "write",
            "permissions": ["audio:read", "artifact:write", "subprocess:execute"],
            "enabled_tools": ["capicove.audio.synthesize", "capicove.audio.clone"],
            "capabilities": {
                "audio_synthesis": {
                    "schema_version": 1,
                    "runtime_config": str(fixture.config_path),
                }
            },
        }
    )
    path = root / "host.json"
    path.write_text(json.dumps(body))
    return path


@pytest.mark.parametrize("operation", ["synthesize", "clone"])
def test_cli_and_mcp_preserve_artifact_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    operation: str,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    arguments = [
        "audio",
        operation,
        "--runtime-config",
        str(fixture.config_path),
        "--text",
        "Hello",
    ]
    values: dict[str, object] = {"text": "Hello"}
    if operation == "clone":
        values.update(reference=str(fixture.reference), authorization="authored-grant")
        arguments += ["--reference", str(fixture.reference), "--authorization", "authored-grant"]
    assert main(arguments) == 0
    cli = json.loads(capsys.readouterr().out)
    fixture.provider.closed = False
    with compose_mcp_host(load_mcp_host_config(host_config(fixture))) as bindings:
        selected = next(item for item in bindings if item.spec.tool_id.endswith("." + operation))
        mcp: dict[str, Any] = invoke_binding(selected, values)
    assert cli["tool_id"] == mcp["tool_id"] == "capicove.audio." + operation
    assert cli["status"] == mcp["status"] == "succeeded"
    for field in ("duration_seconds", "sample_rate", "channels", "warnings"):
        assert cli["result"][field] == mcp["result"][field]
    assert cli["result"]["path"] != mcp["result"]["path"]
    assert cli["result"]["lineage"]["output_sha256"] == mcp["result"]["lineage"]["output_sha256"]
    assert fixture.provider.closed


def test_catalog_discovery_does_not_generate_audio_or_create_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    with compose_mcp_host(
        load_mcp_host_config(host_config(fixture)), catalog_only=True
    ) as bindings:
        assert {item.name for item in bindings} == {
            "capicove_audio_synthesize",
            "capicove_audio_clone",
        }
    assert fixture.provider.requests == []
    assert not fixture.output_root.exists()


def test_read_profile_cannot_expose_speech_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    path = host_config(fixture)
    body = json.loads(path.read_text())
    body["profile"] = "read"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        with compose_mcp_host(load_mcp_host_config(path)):
            pytest.fail("read profiles cannot advertise speech artifact writes")
    assert fixture.provider.requests == []


def test_mcp_denies_forged_reference_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    with compose_mcp_host(load_mcp_host_config(host_config(fixture))) as bindings:
        clone = next(item for item in bindings if item.name == "capicove_audio_clone")
        result: dict[str, Any] = invoke_binding(
            clone,
            {
                "text": "Hello",
                "reference": str(fixture.reference),
                "authorization": "forged",
            },
        )
    assert result["status"] == "rejected"
    assert result["error"]["message"] == "voice_unauthorized"
    assert fixture.provider.requests == []
