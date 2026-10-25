import json
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from capicove.books import BookService, LocalBookWorkspace
from capicove.cli import main
from tests.unit.cli.test_book import _Parser


def test_chunk_cli_previews_persists_and_reads_context(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-book")
    workspace = LocalBookWorkspace.create(tmp_path / "workspace")
    book = BookService(workspace).prepare(source, _Parser())
    config = tmp_path / "chunks.toml"
    config.write_text('[chunking]\npreset = "technical-code"\ntarget_core_tokens = 500\n')
    args = [
        "book",
        "chunk",
        str(workspace.root),
        book.book_id,
        "--chunk-config",
        str(config),
        "--target-core-tokens",
        "700",
    ]
    assert main([*args, "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["resolved_config"]["target_core_tokens"] == 700
    assert preview["dry_run"] is True
    assert preview["resolved_config"]["front_matter_policy"] == "navigation-v1"
    assert main([*args, "--front-matter-policy", "preserve", "--dry-run"]) == 0
    legacy = json.loads(capsys.readouterr().out)
    assert legacy["run_id"] != preview["run_id"]
    assert main(args) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["run_id"] == preview["run_id"]
    assert (
        main(
            ["book", "chunks", str(workspace.root), book.book_id, created["run_id"], "--limit", "1"]
        )
        == 0
    )
    page = json.loads(capsys.readouterr().out)
    assert len(page["items"]) == 1
    assert (
        main(
            [
                "book",
                "context",
                str(workspace.root),
                book.book_id,
                created["run_id"],
                page["items"][0]["chunk_id"],
                "--max-context-read-tokens",
                "30",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["tokens"] <= 30


def test_invalid_chunk_config_fails_before_opening_workspace(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "book",
                "chunk",
                str(tmp_path / "absent"),
                "book:" + "1" * 24,
                "--target-core-tokens",
                "5000",
                "--dry-run",
            ]
        )
        == 2
    )
    assert "target_core_tokens" in json.loads(capsys.readouterr().out)["error"]["message"]
    assert not (tmp_path / "absent").exists()


def test_chunk_mcp_cli_requires_explicit_result_write_selection(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The startup command must pass only the operator-selected write Tool IDs."""
    from capicove.mcp import book_chunks as book_mcp

    selected: list[frozenset[str]] = []

    def serve(_workspace: Path, *, enabled_result_tools: frozenset[str]) -> None:
        selected.append(enabled_result_tools)

    monkeypatch.setattr(book_mcp, "run_stdio", serve)

    assert main(["book", "chunk-mcp", str(tmp_path)]) == 0
    assert selected == [frozenset()]
    assert (
        main(
            [
                "book",
                "chunk-mcp",
                str(tmp_path),
                "--enable-result-tool",
                "capicove.book.chunk-submit",
            ]
        )
        == 0
    )
    assert selected[-1] == frozenset({"capicove.book.chunk-submit"})
