"""Retrieval capability config, chunking, host composition, and transport parity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from capicove.cli.app import main
from capicove.runtime.local_retrieval import (
    MAX_CHUNK_JSON_BYTES,
    RetrievalError,
    chunk_markdown,
    retrieval_settings,
)
from capicove.runtime.mcp_config import (
    WorkspaceHostConfiguration,
    load_mcp_host_config,
    tool_capability,
)
from capicove.runtime.mcp_host import compose_mcp_host
from tests.unit.runtime.test_local_retrieval import Workspace, retrieval_config_body
from tests.unit.runtime.test_local_retrieval import (
    workspace as workspace,  # re-exported pytest fixture
)


posix_only = pytest.mark.skipif(os.name != "posix", reason="retrieval is POSIX-only (ADR-0024)")


def load(space: Workspace) -> WorkspaceHostConfiguration:
    config = load_mcp_host_config(space.config)
    assert isinstance(config, WorkspaceHostConfiguration)
    return config


def test_search_and_evidence_tools_belong_to_retrieval() -> None:
    assert tool_capability("capicove.search") == "retrieval"
    assert tool_capability("capicove.get_evidence") == "retrieval"
    assert tool_capability("capicove.image.filter.shape") == "image"
    assert tool_capability("capicove.search.image") == "search"


def test_retrieval_is_not_a_version_one_capability(tmp_path: Path) -> None:
    from tests.unit.runtime.test_mcp_config import config_body

    body = config_body(tmp_path)
    body["capability"] = "retrieval"
    body["enabled_tools"] = ["capicove.search"]
    body["permissions"] = ["evidence:search"]
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body))
    config = load_mcp_host_config(path)

    with pytest.raises(ValueError, match="select image or memory"):
        with compose_mcp_host(config, catalog_only=True):
            pass


@posix_only
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"retrieval_extra": 1}, "exactly the documented fields"),
        ({"retrieval_schema_version": 2}, "version"),
        ({"retrieval_index_root": "relative/index"}, "absolute"),
        ({"retrieval_retrieval_profile_ref": "remote"}, "local-default"),
        ({"retrieval_max_source_bytes": 0}, "max_source_bytes"),
        ({"retrieval_max_source_bytes": 16 * 1024 * 1024 + 1}, "max_source_bytes"),
        ({"retrieval_corpus_ref": " "}, "corpus_ref"),
        ({"retrieval_source_roots": {}}, "source_roots"),
        ({"retrieval_source_roots": {"Bad": "/tmp"}}, "lowercase"),
    ],
)
def test_retrieval_config_is_closed_and_bounded(
    workspace: Workspace, override: dict[str, object], message: str
) -> None:
    workspace.write_config(**override)

    with pytest.raises((RetrievalError, ValueError), match=message):
        retrieval_settings(load(workspace))


@posix_only
def test_index_root_must_be_inside_workspace_and_disjoint_from_sources(
    workspace: Workspace, tmp_path: Path
) -> None:
    workspace.write_config(retrieval_index_root=str(tmp_path / "elsewhere"))
    with pytest.raises(RetrievalError, match="inside workspace_root"):
        retrieval_settings(load(workspace))

    workspace.write_config(retrieval_source_roots={"notes": str(workspace.root)})
    with pytest.raises(RetrievalError, match="disjoint"):
        retrieval_settings(load(workspace))


@posix_only
def test_source_root_symlink_is_rejected(workspace: Workspace, tmp_path: Path) -> None:
    link = tmp_path / "linked-notes"
    link.symlink_to(workspace.notes)
    workspace.write_config(retrieval_source_roots={"notes": str(link)})

    with pytest.raises(RetrievalError, match="symlinks"):
        retrieval_settings(load(workspace))


@posix_only
def test_shared_index_root_is_rejected(workspace: Workspace) -> None:
    workspace.index.mkdir(mode=0o755)
    workspace.index.chmod(0o755)

    with pytest.raises(RetrievalError, match="owner-private"):
        retrieval_settings(load(workspace))


def test_chunks_are_exact_trimmed_and_bounded() -> None:
    text = "# Title\n\nFirst paragraph.\n\n\n  Second paragraph.  \n" + ("界" * 3000) + "\n"

    chunks = chunk_markdown(text)

    assert chunks
    for start, end in chunks:
        segment = text[start:end]
        assert segment == segment.strip() and segment
        assert len(json.dumps(segment, ensure_ascii=False).encode()) <= MAX_CHUNK_JSON_BYTES
    covered = "".join(text[start:end] for start, end in chunks)
    assert "First paragraph." in covered and "Second paragraph." in covered
    assert covered.count("界") == 3000
    assert chunk_markdown(text) == chunks
    assert chunk_markdown("  \n\n \n") == ()


@posix_only
def test_catalog_advertises_offline_read_only_tools(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["mcp", "catalog", "--config", str(workspace.config)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["capabilities"] == ["retrieval"]
    tools = {item["name"]: item for item in payload["tools"]}
    assert set(tools) == {"capicove_get_evidence", "capicove_search"}
    for item in tools.values():
        assert item["annotations"]["readOnlyHint"] is True
        assert item["annotations"]["openWorldHint"] is False
    assert not workspace.index.exists()


@posix_only
def test_network_tools_stay_disabled_without_network_authority(workspace: Workspace) -> None:
    with compose_mcp_host(load(workspace), catalog_only=True) as bindings:
        assert all(not item.spec.network_access for item in bindings)


@posix_only
def test_mcp_binding_and_cli_return_identical_results(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    for argv in (
        [
            "source",
            "import",
            "--runtime-config",
            str(workspace.config),
            "--root",
            "notes",
            "--relative-path",
            "paris.md",
        ],
        ["index", "build", "--runtime-config", str(workspace.config)],
        ["search", "Paris", "--runtime-config", str(workspace.config)],
    ):
        assert main(argv) == 0
    cli = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    with compose_mcp_host(load(workspace)) as bindings:
        search = next(item for item in bindings if item.name == "capicove_search")
        mcp = search.encode_result(
            search.handler(search.context_factory(search.spec), {"query": "Paris"})
        )

    def stable(payload: dict[str, object]) -> object:
        result = payload["result"]
        assert isinstance(result, dict)
        result = dict(result)
        result.pop("query_id")
        return payload["status"], payload["evidence_refs"], result

    assert stable(mcp) == stable(cli)


@posix_only
def test_setup_doctor_reports_retrieval_ready(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["setup", "doctor", "--config", str(workspace.config)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is True
    assert {item["tool_id"] for item in report["tools"]} == {
        "capicove.get_evidence",
        "capicove.search",
    }
    assert not workspace.index.exists()


@posix_only
def test_setup_doctor_reports_missing_fts5(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("capicove.retrieval._adapters.sqlite_fts.fts5_available", lambda: False)

    assert main(["setup", "doctor", "--config", str(workspace.config)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert {item["status"] for item in report["tools"]} == {"missing_dependency"}


def test_profile_ceiling_allows_evidence_permissions(tmp_path: Path) -> None:
    body = retrieval_config_body(tmp_path, tmp_path / "notes", tmp_path / "index")
    body["profile"] = "read"
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body))

    config = load_mcp_host_config(path)

    assert isinstance(config, WorkspaceHostConfiguration)
    assert config.permissions == frozenset({"evidence:read", "evidence:search"})


@posix_only
def test_setup_doctor_classifies_unsafe_retrieval_paths(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.index.mkdir(mode=0o755)
    workspace.index.chmod(0o755)

    assert main(["setup", "doctor", "--config", str(workspace.config)]) == 2

    report = json.loads(capsys.readouterr().out)
    assert {item["status"] for item in report["tools"]} == {"unsafe_path"}
    assert report["diagnostics"][0]["capability"] == "retrieval"


def test_trailing_spaces_still_end_a_paragraph() -> None:
    paragraph = "Lorem ipsum dolor sit amet. " * 60 + " \n\n"
    text = paragraph * 4

    chunks = chunk_markdown(text)

    assert len(chunks) == 4
    assert all(text[start:end].endswith("amet.") for start, end in chunks)


def test_oversized_paragraphs_split_quickly_at_whitespace() -> None:
    import time

    words = ("word " * 1000).strip()
    text = (words + "\n\n") * 400

    started = time.perf_counter()
    chunks = chunk_markdown(text)

    assert time.perf_counter() - started < 2.0
    for start, end in chunks:
        assert not text[start:end].startswith("ord")
