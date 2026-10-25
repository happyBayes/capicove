"""Compact report output cannot choose task identity or foreign citations."""

import json

import pytest

from capicove.graph_build.native_reports import validate_native_report
from capicove.graph_build.reports import report_tasks
from tests.unit.graph_build.test_reports import evidence


def test_native_report_binds_host_task_and_exact_allowlist() -> None:
    task = report_tasks((), (evidence(),), ())[0]
    raw = json.dumps(
        {
            "title": "Evidence overview",
            "statements": [{"text": "A source claim is available.", "references": ["a0"]}],
        }
    ).encode()
    report = validate_native_report(raw, task, "receipt:test")
    assert report.report_kind == task.report_kind
    assert report.sections.summary[0].assertion_refs == (task.source_assertions[0].assertion_id,)
    for ref in ("a999", "foreign", True):
        invalid = json.dumps(
            {"title": "Bad", "statements": [{"text": "Unsupported", "references": [ref]}]}
        ).encode()
        with pytest.raises(ValueError):
            validate_native_report(invalid, task, "receipt:test")
