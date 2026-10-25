import json
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import CaptureFixture

from capicove.cli import main


def _json_output(capsys: CaptureFixture[str]) -> dict[str, Any]:
    value = json.loads(capsys.readouterr().out)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_wiki_cli_initializes_scans_and_lints_json(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert main(["wiki", "init", str(tmp_path), "--date", "2026-09-01"]) == 0
    initialized = _json_output(capsys)
    assert initialized == {
        "authority": "standalone",
        "facts": 0,
        "identity_events": 0,
        "pages": 0,
        "schema_version": 1,
        "sources": 0,
        "wiki": str((tmp_path / "wiki").resolve()),
    }

    assert main(["wiki", "scan", str(tmp_path)]) == 0
    assert _json_output(capsys)["pages"] == 0
    assert main(["wiki", "lint", str(tmp_path)]) == 0
    assert _json_output(capsys) == {"findings": [], "valid": True}


def test_wiki_cli_ingests_builds_and_queries(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    source = tmp_path / "source.md"
    source.write_text("# Local Wiki\n\nCapicove keeps durable knowledge local.\n", encoding="utf-8")
    assert main(["wiki", "init", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()

    assert main(["wiki", "ingest", str(tmp_path), str(source), "--date", "2026-09-01"]) == 0
    ingested = _json_output(capsys)
    assert ingested["created"] is True
    assert ingested["page_path"] == "pages/local-wiki.md"

    assert main(["wiki", "build", str(tmp_path), "--date", "2026-09-02"]) == 0
    assert _json_output(capsys)["pages"] == 1
    assert main(["wiki", "query", str(tmp_path), "durable knowledge", "--top-k", "3"]) == 0
    queried = _json_output(capsys)
    assert queried["query"] == "durable knowledge"
    assert len(queried["hits"]) == 1
    assert queried["hits"][0]["page_path"] == "pages/local-wiki.md"


def test_wiki_cli_reports_a_stable_missing_index_error(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert main(["wiki", "init", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()

    assert main(["wiki", "query", str(tmp_path), "missing"]) == 2

    assert _json_output(capsys) == {
        "error": {
            "code": "wiki_index_unavailable",
            "message": "Wiki search index is unavailable; run wiki build",
        }
    }


def test_wiki_doctor_reports_invalid_content_with_exit_one(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert main(["wiki", "init", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()
    (tmp_path / "wiki" / "pages" / "bad.md").write_text("No frontmatter.\n", encoding="utf-8")

    assert main(["wiki", "doctor", str(tmp_path)]) == 1

    report = _json_output(capsys)
    assert report["healthy"] is False
    assert report["index_built"] is False
    assert report["lint"]["valid"] is False


def test_wiki_doctor_rejects_corrupt_and_stale_indexes(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nIndexed content.\n", encoding="utf-8")
    assert main(["wiki", "init", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()
    assert main(["wiki", "ingest", str(tmp_path), str(source), "--date", "2026-09-01"]) == 0
    capsys.readouterr()
    assert main(["wiki", "build", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()

    (tmp_path / "wiki" / "_derived" / "search.sqlite3").write_bytes(b"corrupt")
    assert main(["wiki", "doctor", str(tmp_path)]) == 1
    corrupt = _json_output(capsys)
    assert corrupt["index_built"] is True
    assert corrupt["index_healthy"] is False

    assert main(["wiki", "build", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()
    second = tmp_path / "second.md"
    second.write_text("# Second\n\nNew content.\n", encoding="utf-8")
    assert main(["wiki", "ingest", str(tmp_path), str(second), "--date", "2026-09-02"]) == 0
    capsys.readouterr()
    assert main(["wiki", "doctor", str(tmp_path)]) == 1
    stale = _json_output(capsys)
    assert stale["index_built"] is True
    assert stale["index_healthy"] is False


def test_wiki_cli_reports_a_workspace_symlink_loop_as_json(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    loop = tmp_path / "loop"
    loop.symlink_to(loop, target_is_directory=True)

    assert main(["wiki", "scan", str(loop)]) == 2

    assert _json_output(capsys) == {
        "error": {
            "code": "wiki_unsafe_path",
            "message": "Wiki workspace path cannot be resolved safely",
        }
    }


def test_wiki_cli_rejects_a_compact_date(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["wiki", "init", str(tmp_path), "--date", "20260901"])

    assert caught.value.code == 2
    assert "date must use YYYY-MM-DD" in capsys.readouterr().err


def test_wiki_cli_previews_then_explicitly_applies_v2_migration(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert main(["wiki", "init", str(tmp_path), "--date", "2026-09-01"]) == 0
    capsys.readouterr()

    assert main(["wiki", "migrate-v2", str(tmp_path)]) == 0
    assert _json_output(capsys)["applied"] is False
    assert not (tmp_path / "wiki" / "representations").exists()

    assert main(["wiki", "migrate-v2", str(tmp_path), "--apply"]) == 0
    assert _json_output(capsys)["applied"] is True
    assert (tmp_path / "wiki" / "representations").is_dir()
