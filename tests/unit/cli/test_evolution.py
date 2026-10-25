"""Operator-facing supervised Skill lifecycle CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from capicove.cli import main


def test_evolution_init_and_status(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    root = tmp_path / "evolution"
    assert main(["evolution", "init", "--root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "initialized"
    assert main(["evolution", "status", "--root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidates"] == []


def test_missing_skill_is_reported_without_traceback(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    root = tmp_path / "evolution"
    assert main(["evolution", "init", "--root", str(root)]) == 0
    capsys.readouterr()
    assert main(["evolution", "active", "writing", "--root", str(root)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert "Traceback" not in capsys.readouterr().err


def test_cli_learns_evaluates_approves_and_rejects_stale_approval(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    from datetime import UTC, datetime

    from capicove.traces import EpisodeFeedback, EpisodeStatus, TraceRecorder, TraceSanitizer
    from capicove.traces._adapters.sqlite import SqliteTraceStore

    traces = tmp_path / "private" / "traces.sqlite3"
    recorder = TraceRecorder(
        SqliteTraceStore(traces), sanitizer=TraceSanitizer(), clock=lambda: datetime.now(UTC)
    )
    for index in range(3):
        session = recorder.start_session(
            episode_id=f"source-{index}",
            trace_id=f"source-{index}",
            goal="Format text",
            workspace_id="test",
            actor_id="operator",
            environment_id="fixture",
            environment_version=1,
            tool_catalog_sha256="a" * 64,
            active_asset_versions={},
        )
        session.complete(status=EpisodeStatus.SUCCEEDED)
        session.record_feedback(
            feedback_id="correction",
            payload=EpisodeFeedback(
                "correction",
                {
                    "task_signature": "formatting",
                    "lesson": "Convert text to uppercase.",
                    "applicability": "When formatting text",
                    "exclusion": "Other tasks",
                    "attribution": "skill_gap",
                    "confidence": 0.95,
                },
                "user",
            ),
        )
    root = tmp_path / "evolution"
    assert main(["evolution", "distill", "--root", str(root), "--store", str(traces)]) == 0
    batch_id = json.loads(capsys.readouterr().out)["batches"][0]
    assert (
        main(["evolution", "generate", batch_id, "--name", "text-formatting", "--root", str(root)])
        == 0
    )
    candidate_id = json.loads(capsys.readouterr().out)["candidate_id"]
    assert main(["evolution", "active", "text-formatting", "--root", str(root)]) == 1
    capsys.readouterr()
    args = [
        "evolution",
        "review",
        "approve",
        candidate_id,
        "--root",
        str(root),
        "--reviewer",
        "owner",
        "--reason",
        "Reviewed paired cases",
        "--expect-active",
        "none",
        "--expect-state",
        "awaiting_approval",
    ]
    assert main(args) == 1
    capsys.readouterr()
    campaign = tmp_path / "campaign.json"
    campaign.write_text(
        json.dumps(
            {
                split: [
                    {
                        "episode_id": f"{split}-{i}",
                        "input_text": text,
                        "expected_output": text.upper(),
                        "operation": "uppercase",
                    }
                    for i, text in enumerate(("first", "second"))
                ]
                for split in ("replay", "held_out")
            }
        )
    )
    assert (
        main(
            [
                "evolution",
                "evaluate",
                candidate_id,
                "--campaign",
                str(campaign),
                "--root",
                str(root),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "active"
    assert main(args) == 1
    capsys.readouterr()
    assert main(["evolution", "active", "text-formatting", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["candidate_id"] == candidate_id
    assert (
        main(
            [
                "evolution",
                "run",
                "mixed Case",
                "--operation",
                "uppercase",
                "--skill",
                "text-formatting",
                "--episode-id",
                "runtime-one",
                "--store",
                str(traces),
                "--root",
                str(root),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["output"] == "MIXED CASE"
    assert output["skills"][0]["candidate_id"] == candidate_id
    assert (
        main(
            [
                "evolution",
                "feedback",
                "runtime-one",
                "--feedback-id",
                "correction-2",
                "--store",
                str(traces),
                "--root",
                str(root),
                "--task-signature",
                "formatting",
                "--lesson",
                "Also strip whitespace.",
                "--applicability",
                "When formatting text",
                "--exclusion",
                "Other tasks",
                "--attribution",
                "skill_gap",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["evolution", "distill", "--root", str(root), "--store", str(traces)]) == 0
    next_batch = json.loads(capsys.readouterr().out)["batches"][0]
    assert (
        main(
            ["evolution", "generate", next_batch, "--name", "text-formatting", "--root", str(root)]
        )
        == 0
    )
    next_candidate = json.loads(capsys.readouterr().out)
    assert next_candidate["version"] == 2
    assert next_candidate["parent_id"] == candidate_id
    assert "strip whitespace" in next_candidate["content"]


def test_model_execution_requires_explicit_network_permission(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    result = main(
        [
            "evolution",
            "run",
            "hello",
            "--root",
            str(tmp_path / "registry"),
            "--store",
            str(tmp_path / "trace" / "traces.sqlite3"),
            "--episode-id",
            "one",
            "--operation",
            "summarize",
            "--generation-profile",
            str(tmp_path / "model.json"),
            "--expected-output",
            "summary",
        ]
    )
    assert result == 1
    assert "explicit_network_opt_in" in json.loads(capsys.readouterr().out)["error"]


def test_registry_failure_does_not_expose_storage_details(
    tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    from capicove.evolution import registry

    def broken(root: Path) -> None:
        raise sqlite3.OperationalError("private storage location")

    monkeypatch.setattr(registry, "EvolutionRegistry", broken)
    assert main(["evolution", "status", "--root", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "evolution_storage_unavailable"
