"""Tests for immutable trace export validation reports and structural issues."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import PurePosixPath

import pytest

from capicove.export.validation import (
    TraceValidationIssue,
    TraceValidationReport,
    ValidationSeverity,
    _bounded_lines,
)


def test_bounded_line_iterator_discards_oversized_line_tail_while_hashing() -> None:
    payload = b"x" * 100_000 + b"\n" + b"next\n"
    digest = hashlib.sha256()

    lines = list(_bounded_lines(BytesIO(payload), 64, digest))

    assert lines[0][0] == 1
    assert len(lines[0][1]) <= 65
    assert lines[0][2:] == (100_001, True)
    assert lines[1] == (2, b"next\n", 5, False)
    assert digest.hexdigest() == hashlib.sha256(payload).hexdigest()


def test_validation_issue_and_report_are_immutable_and_consistent() -> None:
    issue = TraceValidationIssue(
        ValidationSeverity.ERROR,
        "INVALID_JSON",
        "record is not valid JSON",
        shard=PurePosixPath("records-00000.jsonl"),
        line=2,
    )
    report = TraceValidationReport(False, 1, 1, 0, (issue,))

    assert report.issues == (issue,)
    with pytest.raises(FrozenInstanceError):
        issue.code = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TraceValidationIssue(ValidationSeverity.ERROR, "", "message"),
        lambda: TraceValidationIssue(ValidationSeverity.ERROR, "CODE", "", line=1),
        lambda: TraceValidationIssue(ValidationSeverity.ERROR, "CODE", "message", line=0),
        lambda: TraceValidationReport(True, -1, 0, 0, ()),
        lambda: TraceValidationReport(
            True, 0, 0, 0, (TraceValidationIssue(ValidationSeverity.ERROR, "E", "bad"),)
        ),
    ],
)
def test_validation_values_reject_invalid_identity_counts_and_valid_flag(
    factory: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]
