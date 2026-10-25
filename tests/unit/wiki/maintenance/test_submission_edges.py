"""W2.01/W2.03 edge cases: every rejected intake or review leaves the Wiki unchanged."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from capicove.wiki.maintenance.service import MaintenanceService
from tests.unit.wiki.maintenance.test_capture import experimental_workspace
from tests.unit.wiki.maintenance.test_submissions import submission, wire
from tests.wiki_maintenance_support.oracle import snapshot


def _task(tmp_path: Path, text: str = "Plants use light.") -> tuple[MaintenanceService, object]:
    service = experimental_workspace(tmp_path)
    source = tmp_path / "a.md"
    source.write_text(text)
    captured = service.capture(source)
    service.build_discovery()
    return service, service.create_task((captured.source_id,), query="Plants")


def _object(raw: bytes) -> dict[str, object]:
    value = json.loads(raw)
    assert isinstance(value, dict)
    return value


def test_duplicate_wire_keys_and_unknown_fields_are_rejected(tmp_path: Path) -> None:
    service, task = _task(tmp_path)
    raw = wire(submission(task))  # type: ignore[arg-type]
    before = snapshot(tmp_path / "wiki")
    duplicated = raw.replace(
        b'"idempotency_key":', b'"idempotency_key":"other","idempotency_key":', 1
    )
    assert duplicated != raw
    document = _object(raw)
    payload = document["payload"]
    assert isinstance(payload, dict)
    for changed in (
        duplicated,
        json.dumps({**document, "payload": {**payload, "unexpected": True}}).encode(),
        json.dumps({**document, "payload": {**payload, "reviewer": "operator:forged"}}).encode(),
        json.dumps(
            {
                **document,
                "payload": {
                    **payload,
                    "provenance": {
                        "provider_id": "agent",
                        "model_revision": "r1",
                        "run_id": "run",
                        "reviewer_id": "operator:forged",
                    },
                },
            }
        ).encode(),
    ):
        with pytest.raises(ValueError):
            service.submit(changed)
        assert snapshot(tmp_path / "wiki") == before


@pytest.mark.parametrize(
    "dispositions",
    [("no_material", "new"), ("new", "new"), ("update",), ("disputed",)],
)
def test_disposition_misclassification_has_zero_effect(
    tmp_path: Path, dispositions: tuple[str, ...]
) -> None:
    service, task = _task(tmp_path)
    value = submission(task)  # type: ignore[arg-type]
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError):
        service.submit(wire(replace(value, dispositions=dispositions)))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.submit(
            wire(replace(value, dispositions=("no_material",), rationale="Not relevant."))
        )
    disputed_claim = replace(value.pages[0].claims[0], status="disputed")
    with pytest.raises(ValueError, match="disputed"):
        service.submit(
            wire(
                replace(
                    value,
                    pages=(replace(value.pages[0], claims=(disputed_claim,)),),
                )
            )
        )
    assert snapshot(tmp_path / "wiki") == before


def test_superseded_task_versions_and_aliases_are_out_of_scope(tmp_path: Path) -> None:
    service = experimental_workspace(tmp_path)
    first_source = tmp_path / "a.md"
    first_source.write_text("Plants use light.")
    second_source = tmp_path / "b.md"
    second_source.write_text("Plants also need water.")
    first = service.capture(first_source)
    second = service.capture(second_source)
    service.build_discovery()
    original = service.create_task((first.source_id,), query="Plants")
    expanded = service.create_task(
        (first.source_id, second.source_id), query="Plants", previous_task_id=original.task_id
    )
    assert expanded.task_version == original.task_version + 1
    assert {alias.alias for alias in expanded.evidence} > {
        alias.alias for alias in original.evidence
    }
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match="superseded"):
        service.submit(wire(submission(original)))
    new_alias = sorted(
        {alias.alias for alias in expanded.evidence} - {alias.alias for alias in original.evidence}
    )[0]
    stale = submission(original)
    claim = replace(stale.pages[0].claims[0], evidence_aliases=(new_alias,))
    with pytest.raises(ValueError):
        service.submit(wire(replace(stale, pages=(replace(stale.pages[0], claims=(claim,)),))))
    assert snapshot(tmp_path / "wiki") == before
    accepted = service.submit(wire(submission(expanded)))
    assert accepted.submission.task_version == expanded.task_version


def test_review_is_bound_to_the_previewed_policy_and_proposal(tmp_path: Path) -> None:
    from capicove.contracts.wiki_maintenance import ReviewDecision
    from capicove.wiki.maintenance.review import ReviewAuthority

    service, task = _task(tmp_path)
    proposal = service.submit(wire(submission(task)))  # type: ignore[arg-type]
    preview = service.preview(proposal.proposal_id)
    decisions = tuple(
        ReviewDecision(subject_id=item, verdict="supported", reason="Checked the evidence.")
        for item in preview.review_subjects
    )
    reviewer = ReviewAuthority("operator:test", frozenset({"wiki:review"}))
    before = snapshot(tmp_path / "wiki")
    forged_subjects = (
        ReviewDecision(subject_id="claim:forged", verdict="supported", reason="Forged."),
        *decisions[1:],
    )
    with pytest.raises(ValueError):
        service.review(proposal.proposal_id, forged_subjects, authority=reviewer, accept=True)
    with pytest.raises(PermissionError):
        service.review(
            proposal.proposal_id,
            decisions,
            authority=ReviewAuthority("operator:test", frozenset({"wiki:write", "wiki:admin"})),
            accept=True,
        )
    assert snapshot(tmp_path / "wiki") == before
    policy = tmp_path / "wiki" / "maintenance" / "source-policy.json"
    withdrawn = [task.sources[0].source_id]  # type: ignore[attr-defined]
    policy.write_text(
        json.dumps({"schema_version": 1, "policy_revision": 1, "withdrawn": withdrawn})
    )
    before = snapshot(tmp_path / "wiki")
    with pytest.raises(ValueError, match=r"policy|stale|withdrawn|unavailable"):
        service.review(proposal.proposal_id, decisions, authority=reviewer, accept=True)
    assert snapshot(tmp_path / "wiki") == before
