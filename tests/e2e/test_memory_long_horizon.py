"""Run the fixed offline acceptance workload in independent writer/reader processes."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.e2e
@pytest.mark.evaluation
def test_offline_long_horizon_public_tools(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts/benchmarks/temporal_memory_long_horizon.py"
    )
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(report)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(report.read_text())
    assert result["passed"] == result["total"] == 13
    assert result["process_restart"]
    assert result["qualification"] == "authored_offline_retrieval_not_model_answer_accuracy"


@pytest.mark.slow
@pytest.mark.e2e
def test_small_capacity_workload_cannot_claim_full_scale_targets(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/benchmarks/temporal_memory.py"
    directory = tmp_path / "small"
    completed = subprocess.run(
        [sys.executable, str(script), "--episodes", "2", "--output", str(directory)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((directory / "report.json").read_text())
    assert report["all_initial_targets_met"] is False
    assert report["qualification"] == "smoke_only"
