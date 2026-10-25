"""Ensure the independent oracle catches corruption and extra writes."""

import importlib
import subprocess
import sys
from typing import Any

import pytest


def test_harness_is_available() -> None:
    from pathlib import Path

    assert Path("tests/wiki_maintenance_support/oracle.py").exists()


def test_snapshot_detects_corruption_and_unexpected_files(tmp_path: Any) -> None:
    oracle = importlib.import_module("tests.wiki_maintenance_support.oracle")
    path = tmp_path / "source.md"
    path.write_bytes(b"original")
    before = oracle.snapshot(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(AssertionError):
        oracle.assert_unchanged(tmp_path, before)
    path.write_bytes(b"original")
    (tmp_path / "extra").write_bytes(b"bad")
    with pytest.raises(AssertionError):
        oracle.assert_unchanged(tmp_path, before)


def test_crash_barrier_survives_real_child_exit(tmp_path: Any) -> None:
    script = "from pathlib import Path; from tests.wiki_maintenance_support.faults import FaultBarriers; FaultBarriers(Path(__import__('sys').argv[1])).stop_at('after_state')"
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)])
    assert result.returncode == 73
    assert (tmp_path / "after_state").read_bytes() == b"durable\n"


def test_fixture_sources_have_independent_expected_offsets(tmp_path: Any) -> None:
    fixtures = importlib.import_module("tests.wiki_maintenance_support.fixtures")
    case = fixtures.make_case(tmp_path)
    assert case.source_bytes == b"# Sample\r\nAlpha evidence.\r\n"
    assert case.expected_text[9:14] == "Alpha"
    before = case.snapshot()
    assert case.manual_path.read_text().endswith("Manual note.\n")
    assert len(case.page_bytes()) == 3
    assert before == case.snapshot()


def test_fixture_is_a_valid_legacy_wiki(tmp_path: Any) -> None:
    from capicove.wiki import WikiService
    from tests.wiki_maintenance_support.fixtures import make_case

    make_case(tmp_path)
    assert WikiService(tmp_path).lint().valid


def test_effects_and_unknown_barrier(tmp_path: Any) -> None:
    from tests.wiki_maintenance_support.faults import Effects, crash_at

    effect = Effects()
    assert effect.snapshot() == (0, 0, 0)
    effect.egress_bytes = 2
    assert effect.snapshot() != (0, 0, 0)
    with pytest.raises(ValueError):
        crash_at(tmp_path, "unknown")
    assert list(tmp_path.iterdir()) == []
