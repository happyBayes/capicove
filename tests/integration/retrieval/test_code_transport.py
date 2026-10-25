"""S1 code search over the existing Search/Evidence Tools, MCP and CLI (RFC-0033)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from capicove.contracts import ToolStatus
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config
from tests.integration.retrieval.test_code_lexical import _repository


GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(os.name != "posix" or GIT is None, reason="requires POSIX and Git")


def _config(tmp_path: Path, *, profile: str = "write") -> Path:
    root = (tmp_path / "workspace").resolve()
    root.mkdir(exist_ok=True)
    root.chmod(0o700)
    repo = root / "repo" if (root / "repo").exists() else _repository(root)
    path = tmp_path / f"code-{profile}.json"
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
                "enabled_tools": ["capicove.get_evidence", "capicove.search"],
                "trace_store": None,
                "capabilities": {
                    "retrieval": {
                        "schema_version": 2,
                        "source_roots": {"repo": str(repo)},
                        "index_root": str(root / "index"),
                        "corpus_ref": "corpus:code",
                        "retrieval_profile_ref": "local-code-lexical-v1",
                        "max_source_bytes": 1024 * 1024,
                        "model_config_path": None,
                        "model_config_sha256": None,
                        "model_binding": None,
                        "max_units": 10_000,
                        "max_vector_bytes": 0,
                        "code_policy_ref": "code-symbols-v1",
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


def test_cli_import_build_search_and_mcp_agree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main
    from capicove.runtime.mcp_host import compose_mcp_host

    assert GIT
    path = _config(tmp_path)
    assert (
        main(["code", "import", "--runtime-config", str(path), "--root", "repo", "--git", GIT]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert len(report["imported"]) == 4
    assert main(["index", "build", "--runtime-config", str(path)]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["activated"] is True and built["generation_id"].startswith("cg-")
    assert main(["code", "search", "fetch_user_profile", "--runtime-config", str(path)]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli["status"] == "succeeded" and cli["hits"]
    assert main(["get-evidence", cli["hits"][0]["evidence_id"], "--runtime-config", str(path)]) == 0
    evidence_cli = json.loads(capsys.readouterr().out)
    assert "relative_path" in json.dumps(evidence_cli)
    with compose_mcp_host(_host(path)) as bindings:
        by_name = {item.name: item for item in bindings}
        search = by_name["capicove_search"]
        result = search.handler(
            search.context_factory(search.spec), {"query": "fetch_user_profile"}
        )
        assert result.status is ToolStatus.SUCCEEDED and result.result is not None
        hits = cast(list[dict[str, Any]], result.result["hits"])
        assert [hit["id"] for hit in hits] == [hit["id"] for hit in cli["hits"]]
        evidence = by_name["capicove_get_evidence"]
        read = evidence.handler(
            evidence.context_factory(evidence.spec), {"evidence_id": hits[0]["evidence_id"]}
        )
        assert read.status is ToolStatus.SUCCEEDED and read.result is not None
        metadata = cast(dict[str, Any], read.result["metadata"])
        assert metadata["relative_path"] == "src/app/client.py"
        assert str(tmp_path) not in json.dumps(dict(read.result), default=dict)


def test_single_source_import_policy_and_read_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.cli.app import main

    assert GIT
    path = _config(tmp_path)
    assert (
        main(
            [
                "source",
                "import",
                "--runtime-config",
                str(path),
                "--root",
                "repo",
                "--relative-path",
                "keys.py",
            ]
        )
        == 2
    )
    assert "secret_detected" in capsys.readouterr().err
    assert (
        main(
            [
                "source",
                "import",
                "--runtime-config",
                str(path),
                "--root",
                "repo",
                "--relative-path",
                "src/app/client.py",
            ]
        )
        == 0
    )
    capsys.readouterr()
    read = _config(tmp_path, profile="read")
    assert (
        main(["code", "import", "--runtime-config", str(read), "--root", "repo", "--git", GIT]) == 2
    )
    assert "write or admin" in capsys.readouterr().err
    assert main(["code", "search", "x", "--runtime-config", str(read)]) == 2


def test_catalog_validation_spawns_no_git_and_creates_no_index(tmp_path: Path) -> None:
    from capicove.runtime.mcp_host import compose_mcp_host

    path = _config(tmp_path)
    with compose_mcp_host(_host(path), catalog_only=True) as bindings:
        assert sorted(item.name for item in bindings) == [
            "capicove_get_evidence",
            "capicove_search",
        ]
    assert not (tmp_path / "workspace" / "index").exists()
