"""``capicove eval run|score|compare`` offline CLI (experimental)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capicove.cli.app import main


ROOT = Path(__file__).resolve().parents[3]
VIDEO = ROOT / "tests" / "evaluation" / "video_retrieval" / "fixtures"


def test_run_score_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = tmp_path / "run.json"
    assert (
        main(
            [
                "eval",
                "run",
                "--stack",
                "video",
                "--dataset",
                str(VIDEO),
                "--k",
                "20",
                "--output",
                str(run),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["model"]["real_model_validation"] == "not_run"
    assert summary["metrics"]["mrr"] == 1.0
    assert main(["eval", "score", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "capicove.retrieval-eval-score.v1"
    score = tmp_path / "score.json"
    assert main(["eval", "score", str(run), "--output", str(score)]) == 0
    assert json.loads(score.read_text())["answerable_queries"] == 30
    comparison = tmp_path / "comparison.json"
    assert main(["eval", "compare", str(run), str(run), "--output", str(comparison)]) == 0
    assert json.loads(comparison.read_text())["rankings"]["results_identical"] is True
    assert main(["eval", "compare", str(run), str(run)]) == 0
    capsys.readouterr()


def test_failures_exit_two_without_overwriting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    assert (
        main(
            ["eval", "run", "--stack", "video", "--dataset", str(VIDEO), "--output", str(existing)]
        )
        == 2
    )
    assert "already exists" in capsys.readouterr().err
    assert existing.read_text() == "{}"
    assert main(["eval", "score", str(existing)]) == 2
    assert "unsupported evaluation run schema" in capsys.readouterr().err
    assert main(["eval", "score", str(tmp_path / "missing.json")]) == 2
    assert "evaluation failed" in capsys.readouterr().err
    assert (
        main(
            [
                "eval",
                "run",
                "--stack",
                "video",
                "--dataset",
                str(VIDEO),
                "--k",
                "0",
                "--output",
                str(tmp_path / "x.json"),
            ]
        )
        == 2
    )
    capsys.readouterr()
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    notes = ROOT / "docs" / "development" / "eval" / "personal-notes-eval-v1.json"
    assert (
        main(
            [
                "eval",
                "run",
                "--stack",
                "kb",
                "--dataset",
                str(notes),
                "--profile",
                str(profile),
                "--output",
                str(tmp_path / "y.json"),
            ]
        )
        == 2
    )
    assert "profile" in capsys.readouterr().err
