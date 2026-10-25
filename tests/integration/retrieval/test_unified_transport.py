"""CLI and MCP composition agree on the audio-local loop (RFC-0032 transport slice).

The model is a deterministic fixture registered through the public registry; this
proves composition, rejection and identical results, not semantic quality.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from capicove.contracts import ToolStatus
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config
from tests.integration.retrieval.test_unified_generation import _models, _tone


FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(
    os.name != "posix" or FFMPEG is None or FFPROBE is None, reason="requires POSIX and FFmpeg"
)


def _config(tmp_path: Path, *, enabled: list[str], profile: str = "write") -> Path:
    root = (tmp_path / "workspace").resolve()
    (root / "media").mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    path = tmp_path / f"host-{profile}-{len(enabled)}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "server_name": "capicove-local",
                "workspace_id": "workspace",
                "actor_id": "owner",
                "environment_id": "local",
                "workspace_root": str(root),
                "profile": profile,
                "permissions": ["evidence:read", "evidence:search"],
                "enabled_tools": enabled,
                "trace_store": None,
                "capabilities": {
                    "multimodal": {
                        "schema_version": 1,
                        "profile": "audio-local-v1",
                        "source_roots": {"media": str(root / "media")},
                        "index_root": str(root / "index"),
                        "staging_root": str(root / "staging"),
                        "corpus_ref": "corpus:recordings",
                        "model_config": None,
                        "model_binding": "unified",
                        "ffmpeg": FFMPEG,
                        "ffprobe": FFPROBE,
                        "max_source_bytes": 64 * 1024 * 1024,
                    }
                },
                "network": {"enabled": False, "source_hosts": []},
            }
        )
    )
    path.chmod(0o600)
    return path


def _host(path: Path) -> WorkspaceHostConfiguration:
    parsed = parse_mcp_host_config(path.read_bytes())
    assert isinstance(parsed, WorkspaceHostConfiguration)
    return parsed


TOOLS = ["capicove.multimodal.get_evidence", "capicove.search.multimodal"]


def test_mcp_catalog_binds_exact_read_only_tools(tmp_path: Path) -> None:
    from capicove.runtime.mcp_host import compose_mcp_host

    with compose_mcp_host(_host(_config(tmp_path, enabled=TOOLS)), catalog_only=True) as bindings:
        assert sorted(item.name for item in bindings) == [
            "capicove_multimodal_get_evidence",
            "capicove_search_multimodal",
        ]
        assert all(not item.spec.network_access for item in bindings)


def test_unknown_tool_or_missing_permission_fails_before_listening(tmp_path: Path) -> None:
    from capicove.runtime.mcp_host import compose_mcp_host

    body = json.loads(_config(tmp_path, enabled=TOOLS).read_text())
    body["permissions"] = ["evidence:read"]
    with pytest.raises(ValueError):
        with compose_mcp_host(_host_from(body, tmp_path), catalog_only=True):
            pass
    body["permissions"] = ["evidence:read", "evidence:search"]
    body["enabled_tools"] = ["capicove.multimodal.build"]
    with pytest.raises(ValueError):
        with compose_mcp_host(_host_from(body, tmp_path), catalog_only=True):
            pass


def _host_from(body: dict[str, object], tmp_path: Path) -> WorkspaceHostConfiguration:
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(body))
    return _host(path)


def test_cli_operator_loop_and_mcp_handlers_agree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.cli.app import main
    from capicove.runtime import unified_retrieval

    path = _config(tmp_path, enabled=TOOLS)
    host = _host(path)
    # Every owner (build, CLI search, MCP host) receives and closes its own factory.
    monkeypatch.setattr(
        unified_retrieval, "_models", lambda settings, given: given or _models(tmp_path)
    )
    _tone(host.workspace_root / "media" / "dog.wav", 31, 440)
    assert (
        main(
            [
                "retrieval",
                "import",
                "--runtime-config",
                str(path),
                "--root",
                "media",
                "--relative-path",
                "dog.wav",
            ]
        )
        == 0
    )
    imported = json.loads(capsys.readouterr().out)
    assert imported["source_ref"] == "media:dog.wav" and imported["imported"] is True
    assert main(["retrieval", "build", "--runtime-config", str(path)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["units"] == 2
    assert (
        main(
            [
                "retrieval",
                "activate",
                "--runtime-config",
                str(path),
                "--generation",
                built["generation_id"],
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "retrieval",
                "search",
                "--runtime-config",
                str(path),
                "--query",
                "dog barking",
                "--top-k",
                "2",
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["status"] == "succeeded" and len(cli_result["hits"]) == 2
    evidence_id = cli_result["hits"][0]["evidence_id"]
    assert (
        main(
            [
                "retrieval",
                "get-evidence",
                "--runtime-config",
                str(path),
                "--evidence-id",
                evidence_id,
            ]
        )
        == 0
    )
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["metadata"]["source_ref"] == "media:dog.wav"
    assert str(host.workspace_root) not in json.dumps(evidence)

    from capicove.runtime.mcp_host import compose_mcp_host

    with compose_mcp_host(host) as bindings:
        by_name = {item.name: item for item in bindings}
        search = by_name["capicove_search_multimodal"]
        result = search.handler(
            search.context_factory(search.spec),
            {
                "query": {"type": "text", "text": "dog barking"},
                "targets": ["audio_segment"],
                "top_k": 2,
            },
        )
        assert result.status is ToolStatus.SUCCEEDED
        assert result.result is not None
        mcp_hits = cast(list[dict[str, Any]], result.result["hits"])
        assert [hit["id"] for hit in mcp_hits] == [hit["id"] for hit in cli_result["hits"]]
        reader = by_name["capicove_multimodal_get_evidence"]
        read = reader.handler(reader.context_factory(reader.spec), {"evidence_id": evidence_id})
        assert read.status is ToolStatus.SUCCEEDED
        assert read.result["metadata"] == evidence["metadata"]  # type: ignore[index]


def test_cli_rejects_conflicting_query_flags_and_read_profile_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    path = _config(tmp_path, enabled=TOOLS)
    with pytest.raises(SystemExit):
        main(
            [
                "retrieval",
                "search",
                "--runtime-config",
                str(path),
                "--query",
                "q",
                "--audio-query",
                "/x.wav",
            ]
        )
    assert (
        main(
            [
                "retrieval",
                "search",
                "--runtime-config",
                str(path),
                "--audio-query",
                "/x.wav",
                "--start-ms",
                "5",
            ]
        )
        == 2
    )
    read = _config(tmp_path, enabled=TOOLS, profile="read")
    _tone(_host(read).workspace_root / "media" / "a.wav", 1, 300)
    assert (
        main(
            [
                "retrieval",
                "import",
                "--runtime-config",
                str(read),
                "--root",
                "media",
                "--relative-path",
                "a.wav",
            ]
        )
        == 2
    )
    assert "write or admin" in capsys.readouterr().err
