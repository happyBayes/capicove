"""Cross-connection transaction and persisted-integrity tests."""

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from capicove.evolution.models import EvaluationReport, PairedResult
from capicove.evolution.registry import EvolutionRegistry


_DOCUMENT = "---\nname: editing\ndescription: Edit paragraphs\n---\nUse short paragraphs.\n"


def _report(candidate_id: str, digest: str) -> EvaluationReport:
    return EvaluationReport(
        "report-" + candidate_id,
        candidate_id,
        digest,
        None,
        "integration-v1",
        ("source",),
        ("r1", "r2"),
        ("h1", "h2"),
        tuple(PairedResult(item, False, True) for item in ("r1", "r2", "h1", "h2")),
    )


def test_two_operators_cannot_activate_divergent_stale_drafts(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as first, EvolutionRegistry(tmp_path) as second:
        a = first.create_candidate("editing", _DOCUMENT, ("source",), expected_active=None)
        b = second.create_candidate(
            "editing", _DOCUMENT + "Use headings.\n", ("source",), expected_active=None
        )
        first.record_evaluation(_report(a.candidate_id, a.content_hash))
        second.record_evaluation(_report(b.candidate_id, b.content_hash))
        first.approve(a.candidate_id, "alice", "accepted", expected_active=None)
        with pytest.raises(ValueError, match="stale"):
            second.approve(b.candidate_id, "bob", "accepted", expected_active=None)
        assert second.active("editing") == first.active("editing")
        assert len(first.history("editing")) == 1


def test_corrupt_content_fails_closed_after_reopen(tmp_path: Path) -> None:
    with EvolutionRegistry(tmp_path) as registry:
        candidate = registry.create_candidate(
            "editing", _DOCUMENT, ("source",), expected_active=None
        )
        path = registry.path
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER immutable_candidate")
        db.execute("UPDATE candidates SET content=content || 'tampering'")
    with EvolutionRegistry(tmp_path) as registry:
        with pytest.raises(ValueError, match="hash"):
            registry.get_candidate(candidate.candidate_id)


def test_report_gate_rejects_small_and_large_regressions(tmp_path: Path) -> None:
    report = _report("candidate", "0" * 64)
    rows = tuple(replace(row, baseline_score=0.8, candidate_score=0.81) for row in report.results)
    assert not replace(report, results=rows).passed
    rows = tuple(replace(row, baseline_score=0.8, candidate_score=0.82) for row in report.results)
    assert replace(report, results=rows).passed
    rows = (replace(rows[0], candidate_score=0.5), *rows[1:])
    assert not replace(report, results=rows).passed


def test_reopen_closes_schema_probe_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only schema validation must not leak a second database connection."""
    root = tmp_path / "registry"
    with EvolutionRegistry(root):
        pass
    connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    def tracked(
        database: str | Path,
        *,
        uri: bool = False,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = None,
        timeout: float = 5.0,
    ) -> sqlite3.Connection:
        connection = connect(database, uri=uri, isolation_level=isolation_level, timeout=timeout)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", tracked)
    with EvolutionRegistry(root):
        pass
    assert len(opened) == 2
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
