"""Offline workspace retrieval: import, build, Search, and exact Evidence (RFC-0018)."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.cli.app import main
from capicove.contracts import InvocationContext, PrivacyClass


pytestmark = pytest.mark.skipif(os.name != "posix", reason="retrieval is POSIX-only (ADR-0024)")

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass(frozen=True)
class Workspace:
    root: Path
    notes: Path
    index: Path
    config: Path

    def write_config(self, **overrides: object) -> Path:
        body = retrieval_config_body(self.root, self.notes, self.index)
        retrieval = dict(body["capabilities"]["retrieval"])  # type: ignore[index]
        for key in list(overrides):
            if key in retrieval or key.startswith("retrieval_"):
                name = key.removeprefix("retrieval_")
                value = overrides.pop(key)
                if value is None:
                    retrieval.pop(name, None)
                else:
                    retrieval[name] = value
        body["capabilities"] = {"retrieval": retrieval}
        body.update(overrides)
        self.config.write_text(json.dumps(body))
        self.config.chmod(0o600)
        return self.config


def retrieval_config_body(root: Path, notes: Path, index: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.get_evidence", "capicove.search"],
        "trace_store": None,
        "capabilities": {
            "retrieval": {
                "schema_version": 1,
                "source_roots": {"notes": str(notes)},
                "index_root": str(index),
                "corpus_ref": "corpus:personal-notes",
                "retrieval_profile_ref": "local-default",
                "max_source_bytes": 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "paris.md").write_text(
        "# Travel\n\nAlice moved to Paris on January 1.\n\nShe works at a bakery.\n",
        encoding="utf-8",
    )
    (notes / "pixel.png").write_bytes(PNG_BYTES)
    space = Workspace(root, notes, root / "index", root / "host.json")
    space.write_config()
    return space


def context_for(workspace_id: str) -> InvocationContext:
    return InvocationContext(
        "request",
        "trace",
        workspace_id,
        "owner",
        "local",
        datetime.now(UTC) + timedelta(seconds=30),
        frozenset({"evidence:read", "evidence:search"}),
        PrivacyClass.CONFIDENTIAL,
    )


_last_stderr: list[str] = []


def run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, object]:
    code = main(argv)
    captured = capsys.readouterr()
    _last_stderr[:] = [captured.err]
    return code, json.loads(captured.out) if captured.out.strip() else None


def last_stderr() -> str:
    return _last_stderr[0] if _last_stderr else ""


def import_source(
    capsys: pytest.CaptureFixture[str], space: Workspace, relative: str
) -> tuple[int, object]:
    return run_json(
        capsys,
        [
            "source",
            "import",
            "--runtime-config",
            str(space.config),
            "--root",
            "notes",
            "--relative-path",
            relative,
        ],
    )


def build(capsys: pytest.CaptureFixture[str], space: Workspace) -> tuple[int, object]:
    return run_json(capsys, ["index", "build", "--runtime-config", str(space.config)])


def search(
    capsys: pytest.CaptureFixture[str], space: Workspace, query: str, *extra: str
) -> tuple[int, dict[str, object]]:
    code, payload = run_json(
        capsys, ["search", query, "--runtime-config", str(space.config), *extra]
    )
    assert isinstance(payload, dict)
    return code, payload


def evidence(
    capsys: pytest.CaptureFixture[str], space: Workspace, evidence_id: str, *sections: str
) -> tuple[int, dict[str, object]]:
    argv = ["get-evidence", evidence_id, "--runtime-config", str(space.config)]
    for section in sections:
        argv += ["--section", section]
    code, payload = run_json(capsys, argv)
    assert isinstance(payload, dict)
    return code, payload


def first_hit(payload: dict[str, object]) -> dict[str, object]:
    result = payload["result"]
    assert isinstance(result, dict)
    hits = result["hits"]
    assert isinstance(hits, list) and hits
    hit = hits[0]
    assert isinstance(hit, dict)
    return hit


def test_import_build_search_and_open_exact_evidence(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    before = {path.name: path.read_bytes() for path in workspace.notes.iterdir()}

    code, imported = import_source(capsys, workspace, "paris.md")
    assert code == 0
    assert isinstance(imported, dict)
    assert imported["source_ref"] == "notes/paris.md"
    assert imported["media_type"] == "text/markdown"
    assert imported["changed"] is True
    code, image = import_source(capsys, workspace, "pixel.png")
    assert code == 0
    assert isinstance(image, dict) and image["media_type"] == "image/png"

    code, built = build(capsys, workspace)
    assert code == 0
    assert isinstance(built, dict)
    assert built["activated"] is True
    assert built["sources"] == 2
    assert built["text_units"] >= 1

    code, found = search(capsys, workspace, "Paris bakery")
    assert code == 0, found
    hit = first_hit(found)
    assert hit["target_type"] == "text_chunk"
    assert "Paris" in str(hit["snippet"])

    code, opened = evidence(capsys, workspace, str(hit["evidence_id"]), "metadata", "lineage")
    assert code == 0, opened
    result = opened["result"]
    assert isinstance(result, dict)
    metadata = result["metadata"]
    assert isinstance(metadata, dict)
    source_text = (workspace.notes / "paris.md").read_text(encoding="utf-8")
    start, end = metadata["start_char"], metadata["end_char"]
    assert isinstance(start, int) and isinstance(end, int)
    assert metadata["text"] == source_text[start:end]
    assert "Alice moved to Paris" in str(metadata["text"])
    assert metadata["source_ref"] == "notes/paris.md"
    assert metadata["source_sha256"] == imported["sha256"]
    lineage = result["lineage"]
    assert isinstance(lineage, dict)
    assert lineage["generation_id"] == built["generation_id"]

    image_evidence = [
        item for item in built["source_evidence"] if item["media_type"] == "image/png"
    ]
    assert len(image_evidence) == 1
    code, opened_image = evidence(capsys, workspace, image_evidence[0]["evidence_id"])
    assert code == 0
    image_metadata = opened_image["result"]["metadata"]  # type: ignore[index]
    assert image_metadata["media_type"] == "image/png"
    assert image_metadata["size_bytes"] == len(PNG_BYTES)
    assert "text" not in image_metadata

    assert {path.name: path.read_bytes() for path in workspace.notes.iterdir()} == before


def test_image_import_is_never_a_lexical_hit(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "pixel.png")
    code, built = build(capsys, workspace)
    assert code == 0
    assert isinstance(built, dict) and built["text_units"] == 0

    code, found = search(capsys, workspace, "pixel")
    assert code == 0
    assert found["result"]["hits"] == []  # type: ignore[index]


def test_changed_source_requires_reimport_and_keeps_active_generation(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    _, generation_a = build(capsys, workspace)
    assert isinstance(generation_a, dict)
    _, found_a = search(capsys, workspace, "Paris")
    evidence_a = str(first_hit(found_a)["evidence_id"])

    (workspace.notes / "paris.md").write_text(
        "# Travel\n\nAlice moved to Lyon in March.\n", encoding="utf-8"
    )
    code, failed = build(capsys, workspace)
    assert code == 2
    assert failed is None
    code, still_a = search(capsys, workspace, "Paris")
    assert code == 0
    assert first_hit(still_a)["evidence_id"] == evidence_a

    code, reimported = import_source(capsys, workspace, "paris.md")
    assert code == 0 and isinstance(reimported, dict) and reimported["changed"] is True
    code, generation_b = build(capsys, workspace)
    assert code == 0 and isinstance(generation_b, dict)
    assert generation_b["generation_id"] != generation_a["generation_id"]

    code, stale = evidence(capsys, workspace, evidence_a)
    assert code == 1
    assert stale["status"] == "failed"
    code, found_b = search(capsys, workspace, "Lyon")
    assert code == 0
    code, opened_b = evidence(capsys, workspace, str(first_hit(found_b)["evidence_id"]))
    assert code == 0
    assert "Lyon" in opened_b["result"]["metadata"]["text"]  # type: ignore[index]


def test_failed_build_reports_reason_on_stderr(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    (workspace.notes / "paris.md").write_text("changed\n", encoding="utf-8")

    assert main(["index", "build", "--runtime-config", str(workspace.config)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "changed since import" in captured.err


def test_identical_rebuild_reuses_generation_identity(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    _, first = build(capsys, workspace)
    code, unchanged = import_source(capsys, workspace, "paris.md")
    assert code == 0 and isinstance(unchanged, dict) and unchanged["changed"] is False
    _, second = build(capsys, workspace)

    assert isinstance(first, dict) and isinstance(second, dict)
    assert second["generation_id"] == first["generation_id"]


def test_search_without_build_is_unavailable_and_creates_nothing(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = search(capsys, workspace, "Paris")

    assert code == 1
    assert payload["error"]["code"] == "unavailable"  # type: ignore[index]
    assert not workspace.index.exists()


def test_read_profile_cannot_import_or_build(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write_config(profile="read")

    code, _ = import_source(capsys, workspace, "paris.md")
    assert code == 2
    assert "write profile" in last_stderr()
    code, _ = build(capsys, workspace)
    assert code == 2
    assert not workspace.index.exists()


@pytest.mark.parametrize(
    "relative",
    ["../escape.md", "/etc/passwd", "sub/../paris.md", "", "a\\b.md", "missing.md"],
)
def test_import_rejects_unsafe_or_missing_paths(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], relative: str
) -> None:
    (workspace.root.parent / "escape.md").write_text("secret\n", encoding="utf-8")

    code, payload = import_source(capsys, workspace, relative)

    assert code == 2
    assert payload is None


def test_import_rejects_symlinked_source_components(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = workspace.root.parent / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("secret\n", encoding="utf-8")
    (workspace.notes / "linked").symlink_to(outside)
    (workspace.notes / "alias.md").symlink_to(outside / "secret.md")

    for relative in ("linked/secret.md", "alias.md"):
        code, payload = import_source(capsys, workspace, relative)
        assert code == 2
        assert payload is None


def test_import_rejects_unconfigured_alias(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "source",
            "import",
            "--runtime-config",
            str(workspace.config),
            "--root",
            "other",
            "--relative-path",
            "paris.md",
        ]
    )

    assert code == 2
    assert "not configured" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "payload"),
    [("notes.txt", b"plain text\n"), ("fake.png", b"not an image"), ("bad.md", b"\xff\xfe")],
)
def test_import_rejects_unsupported_media(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], name: str, payload: bytes
) -> None:
    (workspace.notes / name).write_bytes(payload)

    code, result = import_source(capsys, workspace, name)

    assert code == 2
    assert result is None


def test_import_rejects_oversized_source(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write_config(retrieval_max_source_bytes=16)

    code, result = import_source(capsys, workspace, "paris.md")

    assert code == 2
    assert result is None
    assert "max_source_bytes" in last_stderr()


def test_tampered_index_text_fails_closed(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)
    _, found = search(capsys, workspace, "Paris")
    evidence_id = str(first_hit(found)["evidence_id"])
    databases = list(workspace.index.glob("generations/*/index.sqlite3"))
    assert len(databases) == 1
    with sqlite3.connect(databases[0]) as connection:
        connection.execute("UPDATE units SET text = 'forged text'")

    code, opened = evidence(capsys, workspace, evidence_id)

    assert code == 1
    assert opened["status"] == "failed"


def test_cursor_is_rejected_as_invalid_input(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)

    code, payload = search(capsys, workspace, "Paris", "--cursor", "forged")

    assert code == 1
    assert payload["error"]["code"] == "invalid_input"  # type: ignore[index]


def test_search_filters_by_corpus_and_media_type(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)

    _, matching = search(
        capsys,
        workspace,
        "Paris",
        "--corpus-ref",
        "corpus:personal-notes",
        "--media-type",
        "text/markdown",
    )
    _, foreign = search(capsys, workspace, "Paris", "--corpus-ref", "corpus:other")

    assert matching["result"]["hits"]  # type: ignore[index]
    assert foreign["result"]["hits"] == []  # type: ignore[index]


def test_search_and_evidence_require_selected_tools(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.write_config(enabled_tools=["capicove.get_evidence"])

    code = main(["search", "Paris", "--runtime-config", str(workspace.config)])

    assert code == 2
    assert "capicove.search" in capsys.readouterr().err


def test_evidence_reader_rejects_foreign_workspace_context(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.runtime.local_retrieval import open_local_retrieval
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config

    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)
    _, found = search(capsys, workspace, "Paris")
    evidence_id = str(first_hit(found)["evidence_id"])
    config = load_mcp_host_config(workspace.config)
    assert isinstance(config, WorkspaceHostConfiguration)

    runtime = open_local_retrieval(config)
    own = runtime.evidence_handler(context_for("workspace"), {"evidence_id": evidence_id})
    foreign = runtime.evidence_handler(context_for("other"), {"evidence_id": evidence_id})

    assert own.status.value == "succeeded"
    assert foreign.status.value == "failed"
    assert foreign.error is not None and foreign.error.code.value == "permission_denied"
