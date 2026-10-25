"""Only wire failures are repairable; evidence and identity failures stay terminal."""

import hashlib
import json

import pytest

from capicove.graph_build.diagnostics import validate_findings
from capicove.graph_build.extraction import validate_extraction
from capicove.graph_build.reports import report_tasks, validate_report
from capicove.graph_build.validation import StructuralOutputError
from tests.unit.graph_build.test_diagnostics import bundle, record
from tests.unit.graph_build.test_diagnostics import payload as diagnostic_payload
from tests.unit.graph_build.test_extraction import unit
from tests.unit.graph_build.test_reports import LEAF, evidence
from tests.unit.graph_build.test_reports import payload as report_payload


@pytest.mark.parametrize("stage", ["extraction", "diagnostic", "report"])
@pytest.mark.parametrize("payload", [b"{", b"{}", b"[]", b"\xff", b'{"x":1,"x":2}'])
def test_wire_failure_is_explicitly_repairable(stage: str, payload: bytes) -> None:
    with pytest.raises(StructuralOutputError):
        if stage == "extraction":
            validate_extraction(
                payload, unit(), "personal-growth-v1", hashlib.sha256(payload).hexdigest()
            )
        elif stage == "diagnostic":
            validate_findings(payload, bundle(), "receipt:new")
        else:
            (task,) = report_tasks((LEAF,), (evidence(),), ())
            validate_report(payload, task, "receipt:new")


@pytest.mark.parametrize("stage", ["diagnostic", "report"])
def test_invalid_evidence_or_task_is_not_repairable(stage: str) -> None:
    with pytest.raises(ValueError) as caught:
        if stage == "diagnostic":
            item = bundle()
            validate_findings(
                diagnostic_payload(record(item, support_refs=["unrelated"])), item, "receipt:new"
            )
        else:
            (task,) = report_tasks((LEAF,), (evidence(),), ())
            value = json.loads(report_payload(task))
            value["task_key"] = "unrelated"
            validate_report(json.dumps(value).encode(), task, "receipt:new")
    assert not isinstance(caught.value, StructuralOutputError)
