"""RFC-0022: audio import and transcription in the workspace host.

Whisper.cpp and its VAD are replaced by small fake executables. These tests cover
authority, staging, integrity, and error paths only; they do not qualify ASR.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from capicove.cli.app import main
from capicove.mcp.bindings import invoke_binding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="host media handoff is POSIX-only (ADR-0026)"),
    pytest.mark.skipif(FFMPEG is None or FFPROBE is None, reason="needs local FFmpeg"),
]
IMPORT, TRANSCRIBE = "capicove.audio.import", "capicove.audio.transcribe"
PERMISSIONS = ["artifact:write", "audio:read", "audio:write", "subprocess:execute"]

FAKE_WHISPER = """#!{python}
import json, pathlib, sys
arguments = sys.argv[1:]
if arguments == ["--version"]:
    print("whisper.cpp fake 1.0")
    sys.exit(0)
pathlib.Path({calls!r}).open("a").write("whisper " + " ".join(arguments) + "\\n")
prefix = arguments[arguments.index("--output-file") + 1]
payload = {{
    "result": {{"language": "en"}},
    "transcription": [{{"offsets": {{"from": 0, "to": 900}}, "text": " hello world"}}],
}}
pathlib.Path(prefix + ".json").write_text(json.dumps(payload))
"""

FAKE_VAD = """#!{python}
import pathlib, sys
pathlib.Path({calls!r}).open("a").write("vad " + " ".join(sys.argv[1:]) + "\\n")
print("Detected 1 speech segments:")
print("Speech segment 0: start = 0.00, end = 100.00")
"""


def _ffmpeg(*arguments: str) -> None:
    assert FFMPEG is not None
    subprocess.run([FFMPEG, "-v", "error", "-y", *arguments], check=True, timeout=60)


@pytest.fixture(scope="session")
def recordings(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("audio-fixtures")
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2", str(directory / "tone.wav"))
    _ffmpeg(
        *("-f", "lavfi", "-i", "sine=frequency=440:duration=2"),
        *("-f", "lavfi", "-i", "sine=frequency=880:duration=2"),
        *("-map", "0:a", "-map", "1:a", "-c:a", "aac", str(directory / "two.m4a")),
    )
    return directory


@dataclass(frozen=True)
class AudioSpace:
    root: Path
    clips: Path
    tools: Path
    config: Path
    calls: Path

    def model(self, name: str) -> dict[str, str]:
        path = self.tools / name
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def body(self, **overrides: object) -> dict[str, object]:
        audio: dict[str, object] = {
            "schema_version": 1,
            "read_roots": [str(self.clips)],
            "staging_root": str(self.root / "staging"),
            "evidence_root": str(self.root / "evidence"),
            "ffmpeg": FFMPEG,
            "ffprobe": FFPROBE,
            "whisper_cli": str(self.tools / "whisper-cli"),
            "whisper_vad_cli": str(self.tools / "whisper-vad"),
            "model": self.model("ggml-tiny.bin"),
            "vad_model": self.model("ggml-silero.bin"),
            "max_input_bytes": 16 * 1024 * 1024,
            "max_duration_seconds": 60,
        }
        for key in [key for key in overrides if key.startswith("audio_")]:
            value = overrides.pop(key)
            name = key.removeprefix("audio_")
            if value is None:
                audio.pop(name, None)
            else:
                audio[name] = value
        body: dict[str, object] = {
            "schema_version": 2,
            "server_name": "capicove-local",
            "workspace_id": "workspace",
            "actor_id": "owner",
            "environment_id": "local",
            "workspace_root": str(self.root),
            "profile": "write",
            "permissions": PERMISSIONS,
            "enabled_tools": [IMPORT, TRANSCRIBE],
            "trace_store": None,
            "capabilities": {"audio": audio},
            "network": {"enabled": False, "source_hosts": []},
        }
        body.update(overrides)
        return body

    def write(self, **overrides: object) -> Path:
        self.config.write_text(json.dumps(self.body(**overrides)))
        self.config.chmod(0o600)
        return self.config

    def load(self) -> WorkspaceHostConfiguration:
        config = load_mcp_host_config(self.config)
        assert isinstance(config, WorkspaceHostConfiguration)
        return config

    def call_log(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.exists() else []


@pytest.fixture
def space(tmp_path: Path, recordings: Path) -> AudioSpace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("tone.wav", "two.m4a"):
        shutil.copyfile(recordings / name, clips / name)
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls.log"
    for name, template in (("whisper-cli", FAKE_WHISPER), ("whisper-vad", FAKE_VAD)):
        script = tools / name
        script.write_text(textwrap.dedent(template).format(python=sys.executable, calls=str(calls)))
        script.chmod(0o755)
    (tools / "ggml-tiny.bin").write_bytes(b"fake whisper model" * 64)
    (tools / "ggml-silero.bin").write_bytes(b"fake vad model" * 64)
    result = AudioSpace(root, clips, tools, root / "host.json", calls)
    result.write()
    return result


@pytest.fixture
def bindings(space: AudioSpace) -> Iterator[dict[str, object]]:
    with compose_mcp_host(space.load()) as composed:
        yield {item.spec.tool_id: item for item in composed}


def call(
    bindings: dict[str, object], tool_id: str, arguments: dict[str, object]
) -> dict[str, object]:
    return invoke_binding(bindings[tool_id], arguments)  # type: ignore[arg-type]


def evidence_files(space: AudioSpace) -> list[str]:
    root = space.root / "evidence"
    return sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []


# Configuration and authority


def test_catalog_needs_no_subprocess_and_creates_nothing(
    space: AudioSpace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("startup must not start a subprocess")

    monkeypatch.setattr(subprocess, "Popen", forbidden)

    assert main(["mcp", "catalog", "--config", str(space.config)]) == 0

    tools = {item["tool_id"]: item for item in json.loads(capsys.readouterr().out)["tools"]}
    assert sorted(tools) == [IMPORT, TRANSCRIBE]
    assert tools[TRANSCRIBE]["tool_version"] == 2
    assert space.call_log() == []
    assert not (space.root / "staging").exists() and not (space.root / "evidence").exists()


@pytest.mark.parametrize("tools", [[IMPORT], [TRANSCRIBE]])
def test_read_profile_rejects_audio_writes(
    space: AudioSpace, capsys: pytest.CaptureFixture[str], tools: list[str]
) -> None:
    space.write(profile="read", permissions=["subprocess:execute"], enabled_tools=tools)

    assert main(["mcp", "catalog", "--config", str(space.config)]) == 2
    assert capsys.readouterr().out == ""


def test_import_only_selection_needs_no_whisper_assets(space: AudioSpace) -> None:
    space.write(
        enabled_tools=[IMPORT],
        audio_whisper_cli=None,
        audio_whisper_vad_cli=None,
        audio_model=None,
        audio_vad_model=None,
    )

    with compose_mcp_host(space.load(), catalog_only=True) as composed:
        assert [item.spec.tool_id for item in composed] == [IMPORT]


@pytest.mark.parametrize(
    "override",
    [
        {"audio_extra": 1},
        {"audio_schema_version": 3},
        {"audio_evidence_root": "/tmp/outside-evidence"},
        {"audio_whisper_cli": "whisper-cli"},
        {"audio_max_duration_seconds": 14401},
        {"audio_max_input_bytes": 0},
        {"audio_model": {"path": "relative.bin", "sha256": "a" * 64}},
        {"audio_model": None},
    ],
)
def test_audio_config_is_closed_and_selection_checked(
    space: AudioSpace, override: dict[str, object]
) -> None:
    space.write(**override)

    with pytest.raises((ValueError, ImportError)):
        with compose_mcp_host(space.load(), catalog_only=True):
            pass


def test_doctor_classifies_model_integrity_and_missing_executables(
    space: AudioSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup", "doctor", "--config", str(space.config)]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True

    (space.tools / "ggml-tiny.bin").write_bytes(b"tampered")
    assert main(["setup", "doctor", "--config", str(space.config)]) == 2
    statuses = {
        item["tool_id"]: item["status"] for item in json.loads(capsys.readouterr().out)["tools"]
    }
    assert statuses == {IMPORT: "ready", TRANSCRIBE: "asset_integrity"}

    space.write(audio_whisper_cli=str(space.tools / "missing-cli"))
    assert main(["setup", "doctor", "--config", str(space.config)]) == 2
    statuses = {
        item["tool_id"]: item["status"] for item in json.loads(capsys.readouterr().out)["tools"]
    }
    assert statuses[TRANSCRIBE] == "missing_dependency"
    assert space.call_log() == []


# Import


def test_import_preserves_evidence_and_reuses_identical_bytes(
    space: AudioSpace, bindings: dict[str, object]
) -> None:
    source = space.clips / "tone.wav"
    original = source.read_bytes()
    arguments: dict[str, object] = {"source": {"type": "path", "path": str(source)}}

    first = call(bindings, IMPORT, arguments)
    files = evidence_files(space)
    second = call(bindings, IMPORT, arguments)

    assert first["status"] == "succeeded", first
    result = first["result"]
    assert isinstance(result, dict)
    digest = hashlib.sha256(original).hexdigest()
    assert result["evidence"]["evidence_id"] == f"sha256:{digest}"
    assert result["evidence"]["source_name"] == "tone.wav"
    assert result["reused"] is False
    assert second["result"]["reused"] is True  # type: ignore[index]
    assert second["result"]["evidence"] == result["evidence"]  # type: ignore[index]
    assert evidence_files(space) == files
    assert source.read_bytes() == original
    assert list((space.root / "staging").iterdir()) == []
    assert str(space.root / "staging") not in json.dumps([first, second])


def test_multi_stream_import_requires_explicit_selection(
    space: AudioSpace, bindings: dict[str, object]
) -> None:
    source = {"type": "path", "path": str(space.clips / "two.m4a")}

    omitted = call(bindings, IMPORT, {"source": source})
    selected = call(bindings, IMPORT, {"source": source, "audio_stream_index": 1})

    assert omitted["status"] == "rejected"
    failure = omitted["result"]["failure"]  # type: ignore[index]
    assert failure["code"] == "AUDIO_STREAM_SELECTION_REQUIRED"
    assert failure["preflight_audio_stream_count"] == 2
    assert selected["status"] == "succeeded", selected
    selection = selected["result"]["audio_stream_selection"]  # type: ignore[index]
    assert selection["selected_audio_stream_index"] == 1
    assert selection["preflight_audio_stream_count"] == 2


def test_size_and_duration_bounds_reject_before_evidence(space: AudioSpace) -> None:
    for override in ({"audio_max_input_bytes": 1024}, {"audio_max_duration_seconds": 1}):
        space.write(**override)
        with compose_mcp_host(space.load()) as composed:
            selected = {item.spec.tool_id: item for item in composed}
            payload = call(
                selected,  # type: ignore[arg-type]
                IMPORT,
                {"source": {"type": "path", "path": str(space.clips / "tone.wav")}},
            )
        assert payload["status"] == "failed", override
        assert payload["error"]["code"] == "invalid_input"  # type: ignore[index]
        assert evidence_files(space) == []


def test_paths_outside_roots_and_symlinks_are_denied(
    space: AudioSpace, bindings: dict[str, object], tmp_path: Path
) -> None:
    outside = tmp_path / "outside.wav"
    shutil.copyfile(space.clips / "tone.wav", outside)
    (space.clips / "linked.wav").symlink_to(outside)

    for path in (outside, space.clips / "linked.wav"):
        payload = call(bindings, IMPORT, {"source": {"type": "path", "path": str(path)}})
        assert payload["error"]["code"] == "permission_denied", path  # type: ignore[index]
    assert evidence_files(space) == []


# Transcription with fake executables


def test_transcription_uses_vad_and_reports_original_identity(
    space: AudioSpace, bindings: dict[str, object]
) -> None:
    payload = call(
        bindings,
        TRANSCRIBE,
        {"source": {"type": "path", "path": str(space.clips / "tone.wav")}, "language": "auto"},
    )

    assert payload["status"] == "succeeded", payload
    result = payload["result"]
    assert isinstance(result, dict)
    assert [segment["text"] for segment in result["segments"]] == ["hello world"]
    starts = [segment["start_seconds"] for segment in result["segments"]]
    assert starts == sorted(starts)
    assert result["evidence"]["source_name"] == "tone.wav"
    assert result["lineage"]["model"] == "ggml-tiny.bin"
    assert result["lineage"]["model_sha256"] == space.model("ggml-tiny.bin")["sha256"]
    assert result["lineage"]["vad_model_sha256"] == space.model("ggml-silero.bin")["sha256"]
    log = space.call_log()
    assert any(line.startswith("vad ") for line in log)
    assert any(line.startswith("whisper ") for line in log)
    assert str(space.tools / "ggml-tiny.bin") not in "\n".join(log), (
        "Whisper must read the staged copy"
    )
    assert str(space.root / "staging") not in json.dumps(payload)
    assert list((space.root / "staging").iterdir()) == []


def test_model_tampered_after_startup_rejects_before_asr(
    space: AudioSpace, bindings: dict[str, object]
) -> None:
    (space.tools / "ggml-tiny.bin").write_bytes(b"swapped after startup")

    payload = call(
        bindings,
        TRANSCRIBE,
        {"source": {"type": "path", "path": str(space.clips / "tone.wav")}, "language": "en"},
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "unavailable"  # type: ignore[index]
    assert space.call_log() == []


# CLI


def test_configured_cli_import_matches_binding_and_rejects_overrides(
    space: AudioSpace, bindings: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    source = str(space.clips / "tone.wav")
    assert main(["audio", "import", "--runtime-config", str(space.config), "--input", source]) == 0
    cli = json.loads(capsys.readouterr().out)
    host = call(bindings, IMPORT, {"source": {"type": "path", "path": source}})
    assert cli["result"]["evidence"] == host["result"]["evidence"]  # type: ignore[index]

    code = main(
        [
            "audio",
            "transcribe",
            "--runtime-config",
            str(space.config),
            "--input",
            source,
            "--model",
            str(space.tools / "ggml-tiny.bin"),
        ]
    )
    assert code == 2
    assert "--runtime-config" in capsys.readouterr().err

    assert (
        main(
            [
                "audio",
                "transcribe",
                "--runtime-config",
                str(space.config),
                "--input",
                source,
                "--output",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result"]["segments"][0]["text"] == "hello world"


# Review findings (2026-10-06)


class ReaderDetector:
    """Report whether any process opens a FIFO for reading while it is watched."""

    def __init__(self, path: Path) -> None:
        import threading

        os.mkfifo(path, 0o600)
        self.path, self.opened = path, False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        import time

        while not self._stop.is_set():
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                time.sleep(0.002)
                continue
            self.opened = True
            os.close(descriptor)

    def __enter__(self) -> ReaderDetector:
        self._thread.start()
        return self

    def __exit__(self, *details: object) -> None:
        self._stop.set()
        self._thread.join(5)


@pytest.mark.parametrize("name", ["list.mp3", "list.m3u8"])
def test_playlists_cannot_reach_files_outside_the_roots(
    space: AudioSpace, bindings: dict[str, object], tmp_path: Path, name: str
) -> None:
    external = tmp_path / "external.wav"
    playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXTINF:3,\nfile:" + str(external).encode()
    playlist += b"\n#EXT-X-ENDLIST\n"
    (space.clips / name).write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00" + playlist)

    with ReaderDetector(external) as detector:
        payload = call(
            bindings, IMPORT, {"source": {"type": "path", "path": str(space.clips / name)}}
        )

    assert payload["status"] != "succeeded"
    assert detector.opened is False, "a media tool opened a file outside the read roots"
    assert evidence_files(space) == []


def test_replaced_executable_is_never_run(space: AudioSpace, bindings: dict[str, object]) -> None:
    marker = space.root.parent / "replacement-ran"
    replacement = space.tools / "whisper-cli.new"
    replacement.write_text(f"#!/bin/sh\ntouch {marker}\n")
    replacement.chmod(0o755)
    os.replace(replacement, space.tools / "whisper-cli")

    payload = call(
        bindings,
        TRANSCRIBE,
        {"source": {"type": "path", "path": str(space.clips / "tone.wav")}, "language": "en"},
    )

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "unavailable"  # type: ignore[index]
    assert not marker.exists()


def test_configured_cli_validates_the_complete_selected_catalog(
    space: AudioSpace, capsys: pytest.CaptureFixture[str]
) -> None:
    (space.tools / "ggml-tiny.bin").write_bytes(b"tampered before the CLI call")

    code = main(
        [
            "audio",
            "import",
            "--runtime-config",
            str(space.config),
            "--input",
            str(space.clips / "tone.wav"),
        ]
    )

    assert code == 2
    assert evidence_files(space) == []
