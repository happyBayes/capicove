"""Speech CLI dispatch must execute the shared runtime and preserve failures."""

import json
import sqlite3
from pathlib import Path

import pytest

from capicove.cli.app import main
from tests.speech_support import speech_fixture


def test_capabilities_dispatches_without_generating_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    code = main(["audio", "speech-capabilities", "--runtime-config", str(fixture.config_path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["available"] is True
    assert payload["reference_audio"] is True
    assert payload["reference_text"] is False
    assert fixture.provider.closed
    assert fixture.provider.requests == []
    assert not fixture.output_root.exists()


def test_capabilities_reports_missing_configuration_as_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["audio", "speech-capabilities", "--runtime-config", str(tmp_path / "missing.json")])
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unavailable" in captured.err
    assert str(tmp_path) not in captured.err


@pytest.mark.parametrize("operation", ["synthesize", "clone"])
def test_generation_dispatches_and_publishes_provenance(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    if operation == "clone":
        arguments += ["--reference", str(fixture.reference), "--authorization", "authored-grant"]
    assert main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_id"] == "capicove.audio." + operation
    assert payload["status"] == "succeeded"
    assert Path(payload["result"]["path"]).is_file()
    assert Path(payload["result"]["provenance_path"]).is_file()
    assert fixture.provider.closed


def test_clone_without_authorization_never_invokes_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "audio",
                "clone",
                "--runtime-config",
                str(fixture.config_path),
                "--text",
                "Hello",
                "--reference",
                str(fixture.reference),
            ]
        )
    assert error.value.code == 2
    assert fixture.provider.requests == []
    capsys.readouterr()


def test_explicit_trace_store_records_speech_without_raw_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    trace = tmp_path / "trace.db"
    assert (
        main(
            [
                "--trace-store",
                str(trace),
                "audio",
                "synthesize",
                "--runtime-config",
                str(fixture.config_path),
                "--text",
                "Private authored narration",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert trace.is_file()
    with sqlite3.connect(trace) as connection:
        dump = "\n".join(connection.iterdump())
    assert "capicove.audio.synthesize" in dump
    assert "Private authored narration" not in dump
