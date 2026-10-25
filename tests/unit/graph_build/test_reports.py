"""Evidence selection, bottom-up reporting, and source-closed citations."""

import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    ChildStatementRef,
    Community,
    DiagnosticFinding,
    EvidenceBundle,
    GraphScope,
    RelationshipAssertion,
    ReportSections,
)
from capicove.graph_build.reports import (
    CompletedReport,
    ReportEvidence,
    ReportProfile,
    ReportTask,
    report_cache_key,
    report_tasks,
    validate_report,
)


SCOPE = GraphScope("workspace", "private")
LEAF = Community("community:leaf", None, 0, ("skill",), ())


def evidence(target: str = "skill", suffix: str = "1", counter: bool = False) -> ReportEvidence:
    assertion = RelationshipAssertion(
        "assertion:" + suffix,
        "person",
        "HAS_SKILL",
        target,
        "explicit",
        "positive",
        None,
        (),
        ("source",),
        "support:" + suffix,
        None,
        (),
    )
    rows: tuple[RelationshipAssertion, ...] = (assertion,)
    counters: tuple[str, ...] = ()
    if counter:
        counters = ("counter:" + suffix,)
        rows += (
            replace(
                assertion,
                assertion_id=counters[0],
                polarity="negative",
                counter_assertion_ids=(assertion.assertion_id,),
            ),
        )
    bundle = EvidenceBundle(
        "bundle:" + suffix,
        "person",
        target,
        (assertion.assertion_id,),
        counters,
        ("event:" + suffix,),
        0,
    )
    return ReportEvidence(SCOPE, bundle, rows)


def finding(item: ReportEvidence, **changes: Any) -> DiagnosticFinding:
    return replace(
        DiagnosticFinding(
            "finding:1",
            "DIFFICULTY",
            "person",
            item.bundle.target_ref,
            "hypothesized",
            item.bundle.assertion_ids,
            item.bundle.counter_assertion_ids,
            "Check this tentative difficulty.",
            ("eligibility_not_diagnosis",),
            "Inspect a further observation.",
            "receipt:finding",
        ),
        **changes,
    )


def statement(kind: str = "fact", **changes: Any) -> dict[str, Any]:
    result = dict(
        local_id="s1",
        text="A source records an observation.",
        kind=kind,
        assertion_refs=["assertion:1"],
        finding_refs=[],
        child_statement_refs=[],
    )
    result.update(changes)
    return result


def payload(task: ReportTask, **sections: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "version": 1,
            "task_key": task.cache_key,
            "report_kind": task.report_kind,
            "community_id": task.community_id,
            "title": "Evidence inventory",
            "sections": {
                field.name: sections.get(field.name, []) for field in fields(ReportSections)
            },
        }
    ).encode()


def finished(task: ReportTask, **sections: list[dict[str, Any]]) -> CompletedReport:
    if not task.requires_generation:
        from capicove.graph_build.reports import materialize_inventory_report

        return CompletedReport(materialize_inventory_report(task), task)
    return CompletedReport(validate_report(payload(task, **sections), task, "receipt:report"), task)


def test_same_membership_with_changed_evidence_invalidates_report() -> None:
    old = report_cache_key("community:1", "a" * 64, (), "f" * 64)
    new = report_cache_key("community:1", "b" * 64, (), "f" * 64)
    assert old != new


def test_leaf_inventory_is_frozen_and_changed_source_content_invalidates() -> None:
    item = evidence()
    (task,) = report_tasks((LEAF,), (item,), ())
    assert task.community_id == LEAF.community_id
    assert task.allowed_assertion_ids == ("assertion:1",)
    assert task.source_refs == ("source",)
    changed = replace(item, assertions=(replace(item.assertions[0], evidence_refs=("source:v2",)),))
    assert report_tasks((LEAF,), (changed,), ())[0].cache_key != task.cache_key
    with pytest.raises(ValueError):
        replace(task, cache_key="a" * 64)


def test_bottom_up_waves_wait_for_explicit_terminal_children_and_overview() -> None:
    root = Community("root", None, 0, ("skill", "other"), ())
    left = Community("left", "root", 1, ("skill",), ())
    right = Community("right", "root", 1, ("other",), ())
    rows = (root, right, left)
    first = report_tasks(rows, (evidence(),), ())
    assert [task.community_id for task in first] == ["left", "right"]
    done = finished(first[0], summary=[statement()])
    pending = report_tasks(rows, (evidence(),), (), completed_reports=(done,))
    assert [task.community_id for task in pending] == ["right"]
    (parent,) = report_tasks(
        rows, (evidence(),), (), completed_reports=(done,), failed_community_ids=("right",)
    )
    assert parent.community_id == "root"
    assert (parent.coverage.declared, parent.coverage.accepted, parent.coverage.failed) == (2, 1, 1)
    child_ref = ChildStatementRef(
        done.report.report_id, done.report.sections.summary[0].statement_id
    )
    assert parent.allowed_child_statement_refs == (child_ref,)
    parent_done = finished(
        parent,
        summary=[statement(assertion_refs=[], child_statement_refs=[asdict(child_ref)])],
        limitations=[
            statement(
                "limitation", local_id="partial", text="partial_child_coverage", assertion_refs=[]
            )
        ],
    )
    (overview,) = report_tasks(
        rows,
        (evidence(),),
        (),
        completed_reports=(done, parent_done),
        failed_community_ids=("right",),
    )
    assert overview.community_id is None
    assert overview.report_kind == "overview"
    assert overview.source_refs == ("source",)
    assert "partial_child_coverage" in overview.required_limitations


@pytest.mark.parametrize("mutation", ["cycle", "missing_parent", "overlap", "subset", "depth"])
def test_invalid_hierarchies_fail_closed(mutation: str) -> None:
    root = Community("root", None, 0, ("a", "b"), ())
    left, right = (
        Community("left", "root", 1, ("a",), ()),
        Community("right", "root", 1, ("b",), ()),
    )
    if mutation == "cycle":
        root = replace(root, parent_id="left", depth=2)
    elif mutation == "missing_parent":
        left = replace(left, parent_id="absent")
    elif mutation == "overlap":
        right = replace(right, member_ids=("a",))
    elif mutation == "subset":
        right = replace(right, member_ids=("foreign",))
    else:
        right = replace(right, depth=2)
    with pytest.raises(ValueError):
        report_tasks((root, left, right), (), ())


def test_evidence_uses_target_before_shared_person_and_reserves_counters() -> None:
    other = Community("other", None, 0, ("other-skill", "person"), ())
    relevant = evidence(counter=True)
    irrelevant = evidence("other-skill", "2")
    tasks = report_tasks(
        (other, LEAF), (irrelevant, relevant), (), profile=ReportProfile(max_assertions=2)
    )
    task = next(task for task in tasks if task.community_id == LEAF.community_id)
    assert task.allowed_assertion_ids == ("assertion:1", "counter:1")
    assert task.bundle.counter_assertion_ids == ("counter:1",)
    crowded = report_tasks(
        (LEAF,), (evidence(suffix="0"), relevant), (), profile=ReportProfile(max_assertions=2)
    )[0]
    assert crowded.bundle.counter_assertion_ids == ("counter:1",)
    assert crowded.omitted_count == 1


def test_sparse_community_requires_insufficiency_notice_without_fabrication() -> None:
    (task,) = report_tasks((LEAF,), (), ())
    assert task.allowed_assertion_ids == ()
    with pytest.raises(ValueError):
        validate_report(payload(task), task, "receipt:report")
    report = finished(
        task, limitations=[statement("limitation", text="insufficient_evidence", assertion_refs=[])]
    ).report
    assert report.sections.summary == ()


def test_findings_require_supplied_matching_original_evidence() -> None:
    item = evidence()
    for bad in (finding(item, support_refs=("alien",)), finding(item, target_ref="alien")):
        with pytest.raises(ValueError):
            report_tasks((LEAF,), (item,), (bad,))
    with pytest.raises(TypeError):
        report_tasks((LEAF,), (item.bundle,), ())  # type: ignore[arg-type]


@pytest.mark.parametrize("where", ["id", "source", "derivation", "kind"])
def test_generated_text_never_becomes_independent_evidence(where: str) -> None:
    item = evidence()
    changes: dict[str, Any] = {}
    if where == "id":
        changes["assertion_id"] = "community-report:old"
    elif where == "source":
        changes["evidence_refs"] = ("generated-report:old",)
    elif where == "derivation":
        changes["derivation_ref"] = "receipt:report"
    with pytest.raises(ValueError):
        if where == "kind":
            replace(item, source_kind="generated_report")
        else:
            replace(item, assertions=(replace(item.assertions[0], **changes),))


@pytest.mark.parametrize(
    "mutation",
    [
        "alien_assertion",
        "alien_finding",
        "finding_fact",
        "self_child",
        "missing_section",
        "extra",
        "wrong_task",
        "duplicate_id",
        "bool_version",
        "oversized",
        "recommendation_fact",
    ],
)
def test_closed_report_rejects_unsafe_structured_output(mutation: str) -> None:
    item = evidence()
    (task,) = report_tasks((LEAF,), (item,), (finding(item),))
    value = json.loads(payload(task, summary=[statement()]))
    row = value["sections"]["summary"][0]
    if mutation == "alien_assertion":
        row["assertion_refs"] = ["alien"]
    elif mutation == "alien_finding":
        row["finding_refs"] = ["alien"]
    elif mutation == "finding_fact":
        row["finding_refs"] = ["finding:1"]
    elif mutation == "self_child":
        row["child_statement_refs"] = [{"report_id": "receipt:report", "statement_id": "s1"}]
    elif mutation == "missing_section":
        del value["sections"]["strengths"]
    elif mutation == "extra":
        row["executed"] = True
    elif mutation == "wrong_task":
        value["task_key"] = "a" * 64
    elif mutation == "duplicate_id":
        value["sections"]["strengths"] = [row.copy()]
    elif mutation == "bool_version":
        value["version"] = True
    elif mutation == "oversized":
        row["text"] = "x" * 2001
    else:
        value["sections"]["suggested_next_steps"] = [row]
        value["sections"]["summary"] = []
    with pytest.raises(ValueError):
        validate_report(json.dumps(value).encode(), task, "receipt:report")


def test_counter_coverage_is_required_and_cannot_be_hidden_in_summary() -> None:
    item = evidence(counter=True)
    (task,) = report_tasks((LEAF,), (item,), ())
    counter = statement(local_id="c", assertion_refs=["counter:1"])
    with pytest.raises(ValueError):
        validate_report(payload(task, summary=[counter]), task, "receipt:report")
    report = finished(task, summary=[statement()], counter_evidence=[counter]).report
    assert report.sections.counter_evidence[0].assertion_refs == ("counter:1",)


def test_recommendations_are_labeled_proposals_with_explicit_limitations() -> None:
    (task,) = report_tasks((LEAF,), (evidence(),), ())
    recommendation = statement("recommendation")
    with pytest.raises(ValueError):
        validate_report(
            payload(task, suggested_next_steps=[recommendation]), task, "receipt:report"
        )
    result = finished(
        task,
        suggested_next_steps=[recommendation],
        limitations=[
            statement(
                "limitation", local_id="limit", text="not_executed_or_verified", assertion_refs=[]
            )
        ],
    )
    assert result.report.sections.suggested_next_steps[0].kind == "recommendation"


def test_deterministic_permutations_and_receipt_binding() -> None:
    a, b = evidence(), evidence(suffix="2")
    original = report_tasks((LEAF,), (a, b), ())[0]
    assert report_tasks((LEAF,), (b, a), ())[0] == original
    data = payload(original, summary=[statement()])
    first = validate_report(data, original, "receipt:1")
    second = validate_report(data, original, "receipt:2")
    assert first.report_id != second.report_id
    assert first.sections.summary[0].statement_id == second.sections.summary[0].statement_id
    with pytest.raises(ValueError):
        validate_report(data, original, "assertion:1")


@pytest.mark.parametrize(
    "data", [b'{"version":1,"version":1}', b"\xff", b"[" * 1000, b'"' + b"x" * 65536 + b'"']
)
def test_bounded_duplicate_key_safe_json(data: bytes) -> None:
    task = report_tasks((LEAF,), (evidence(),), ())[0]
    with pytest.raises(ValueError):
        validate_report(data, task, "receipt:1")


def test_cache_key_rejects_malformed_digests() -> None:
    for value in ("x", "A" * 64, "z" * 64):
        with pytest.raises(ValueError):
            report_cache_key("c", value, (), "f" * 64)
    with pytest.raises(ValueError):
        report_cache_key("c", "a" * 64, ("invalid",), "f" * 64)


def test_task_cannot_promote_counter_evidence_or_drop_selected_findings_limitations() -> None:
    item = evidence(counter=True)
    task = report_tasks((LEAF,), (item,), (finding(item),))[0]
    with pytest.raises(ValueError):
        replace(
            task,
            bundle=replace(
                task.bundle, assertion_ids=task.allowed_assertion_ids, counter_assertion_ids=()
            ),
        )
    assert "eligibility_not_diagnosis" in task.required_limitations


def test_child_statement_inventory_closes_through_three_report_levels() -> None:
    item = evidence()
    leaf_task = report_tasks((LEAF,), (item,), ())[0]
    leaf_done = finished(leaf_task, summary=[statement()])
    (overview,) = report_tasks((LEAF,), (item,), (), completed_reports=(leaf_done,))
    child_ref = asdict(overview.allowed_child_statement_refs[0])
    complete = finished(
        overview, summary=[statement(assertion_refs=[], child_statement_refs=[child_ref])]
    )
    assert overview.child_citations[0].assertion_ids == ("assertion:1",)
    assert overview.child_citations[0].source_refs == ("source",)
    assert complete.report.sections.summary[0].assertion_refs == ()
    assert (
        complete.report.sections.summary[0].child_statement_refs[0].report_id
        == leaf_done.report.report_id
    )
    assert report_tasks((LEAF,), (item,), (), completed_reports=(leaf_done, complete)) == ()


@pytest.mark.parametrize(
    "mutation", ["source", "counter", "finding", "profile", "report", "foreign"]
)
def test_completed_children_require_current_exact_task_and_content(mutation: str) -> None:
    item = evidence()
    task = report_tasks((LEAF,), (item,), ())[0]
    done = finished(task, summary=[statement()])
    kwargs: dict[str, Any] = {"completed_reports": (done,)}
    findings: tuple[DiagnosticFinding, ...] = ()
    if mutation == "source":
        item = replace(
            item, assertions=(replace(item.assertions[0], evidence_refs=("new-source",)),)
        )
    elif mutation == "counter":
        item = evidence(counter=True)
    elif mutation == "finding":
        findings = (finding(item),)
    elif mutation == "profile":
        kwargs["profile"] = ReportProfile(prompt_version="new-prompt")
    elif mutation == "report":
        with pytest.raises(ValueError):
            replace(done, report=replace(done.report, title="Tampered content"))
        return
    else:
        with pytest.raises(ValueError):
            report_tasks((replace(LEAF, community_id="foreign"),), (item,), (), **kwargs)
        return
    with pytest.raises(ValueError):
        report_tasks((LEAF,), (item,), findings, **kwargs)


def test_child_hypothesis_cannot_become_a_parent_fact() -> None:
    item = evidence()
    task = report_tasks((LEAF,), (item,), ())[0]
    done = finished(task, summary=[statement("hypothesis")])
    (overview,) = report_tasks((LEAF,), (item,), (), completed_reports=(done,))
    with pytest.raises(ValueError):
        validate_report(
            payload(
                overview,
                summary=[
                    statement(
                        assertion_refs=[],
                        child_statement_refs=[asdict(overview.allowed_child_statement_refs[0])],
                    )
                ],
            ),
            overview,
            "receipt:parent",
        )


def test_source_stated_hypothesis_cannot_become_fact() -> None:
    item = evidence()
    item = replace(item, assertions=(replace(item.assertions[0], epistemic_status="hypothesized"),))
    task = report_tasks((LEAF,), (item,), ())[0]
    with pytest.raises(ValueError):
        validate_report(payload(task, summary=[statement()]), task, "receipt:report")
    assert (
        finished(task, summary=[statement("hypothesis")]).report.sections.summary[0].kind
        == "hypothesis"
    )


def test_scope_and_unknown_failed_children_are_not_accepted() -> None:
    item = evidence()
    other = replace(evidence(suffix="2"), scope=GraphScope("workspace", "alien"))
    with pytest.raises(ValueError):
        report_tasks((LEAF,), (item, other), ())
    with pytest.raises(ValueError):
        report_tasks((LEAF,), (item,), (), failed_community_ids=("absent",))


def test_schema_rejects_provider_execution_and_numeric_ranking_fields() -> None:
    from importlib.resources import files

    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/report-v1.json").read_bytes()
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    task = report_tasks((LEAF,), (evidence(),), ())[0]
    value = json.loads(payload(task, summary=[statement()]))
    assert validator.is_valid(value)
    for key in ("executed", "effective", "personality_score"):
        row = {**value, key: 10}
        assert not validator.is_valid(row)
        with pytest.raises(ValueError):
            validate_report(json.dumps(row).encode(), task, "receipt:report")


def report_request(harness: Any, task: ReportTask) -> Any:
    """Compose the closed Task 8 request at the authorized host test boundary."""
    from importlib.resources import files

    from capicove.graph_build.codec import canonical_bytes, fingerprint

    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/report-v1.json").read_bytes()
    )
    wire = json.loads(harness.request.canonical_payload)
    segments = (("report-input", "source", json.dumps(asdict(task))),)
    wire.update(
        role="report",
        schema=schema,
        schema_version="report-v1",
        segments=segments,
        prompt="Produce a source-cited inventory from the fixed task; no tools.",
        profile_fingerprint=task.profile_fingerprint,
    )
    return replace(
        harness.request,
        role="report",
        schema_version="report-v1",
        schema_payload=canonical_bytes(schema),
        canonical_payload=canonical_bytes(wire),
        request_fingerprint=fingerprint(wire),
        input_segments=segments,
        prompt_text=wire["prompt"],
        profile_fingerprint=task.profile_fingerprint,
    )


def test_task8_replay_rechecks_current_sources_and_uses_zero_provider_calls(tmp_path: Path) -> None:
    from capicove.graph_build.inference import ValidationResult
    from capicove.graph_build.receipts import receipt_key
    from tests.unit.graph_build.test_inference import Harness, raw

    task = report_tasks((LEAF,), (evidence(),), ())[0]
    data = payload(task, summary=[statement()])
    harness = Harness(tmp_path, [raw(data)])
    request = report_request(harness, task)
    key = receipt_key(SCOPE, request.role, request.request_fingerprint)

    def validator(data: bytes) -> ValidationResult:
        validate_report(data, task, key)
        return ValidationResult(data)

    first = harness.runner.run(SCOPE, request, task.source_refs, validator)
    replay = harness.runner.run(SCOPE, request, task.source_refs, validator)
    assert first.status == replay.status == "succeeded"
    assert len(harness.generator.requests) == 1
    harness.gate.denied_sources.add("source")
    with pytest.raises(PermissionError):
        harness.runner.run(SCOPE, request, task.source_refs, validator)
    assert len(harness.generator.requests) == 1


@pytest.mark.parametrize("change", ["evidence", "counter", "child", "failed_child"])
def test_task8_cache_invalidates_actual_report_dependencies(tmp_path: Path, change: str) -> None:
    from capicove.graph_build.inference import ValidationResult
    from capicove.graph_build.receipts import receipt_key
    from tests.unit.graph_build.test_inference import Harness, raw

    item = evidence()
    first = report_tasks((LEAF,), (item,), ())[0]
    if change == "evidence":
        item = replace(
            item, assertions=(replace(item.assertions[0], qualifiers=(("time_text", "later"),)),)
        )
        second = report_tasks((LEAF,), (item,), ())[0]
    elif change == "counter":
        second = report_tasks((LEAF,), (evidence(counter=True),), ())[0]
    else:
        done = finished(first, summary=[statement()])
        first = report_tasks((LEAF,), (item,), (), completed_reports=(done,))[0]
        if change == "child":
            changed = finished(
                done.task, summary=[statement(text="A differently worded source account.")]
            )
            second = report_tasks((LEAF,), (item,), (), completed_reports=(changed,))[0]
        else:
            second = report_tasks((LEAF,), (item,), (), failed_community_ids=(LEAF.community_id,))[
                0
            ]
    harness = Harness(tmp_path, [])
    for task in (first, second):
        sections: dict[str, list[dict[str, Any]]] = {"summary": [statement()]}
        if task.bundle.counter_assertion_ids:
            sections["counter_evidence"] = [
                statement(local_id="counter", assertion_refs=["counter:1"])
            ]
        if task.required_limitations:
            sections["limitations"] = [
                statement("limitation", local_id="limit" + str(i), text=text, assertion_refs=[])
                for i, text in enumerate(task.required_limitations)
            ]
        data = payload(task, **sections)
        request = report_request(harness, task)
        key = receipt_key(SCOPE, request.role, request.request_fingerprint)
        harness.generator.actions = iter([raw(data)])

        def validator(data: bytes, task: ReportTask = task, key: str = key) -> ValidationResult:
            validate_report(data, task, key)
            return ValidationResult(data)

        assert harness.runner.run(SCOPE, request, task.source_refs, validator).status == "succeeded"
    assert len(harness.generator.requests) == 2
    assert (
        harness.generator.requests[0].request_fingerprint
        != harness.generator.requests[1].request_fingerprint
    )


def test_task8_report_failure_can_retry_and_token_preflight_prevents_dispatch(
    tmp_path: Path,
) -> None:
    from capicove.graph_build.inference import ValidationResult
    from capicove.graph_build.receipts import receipt_key
    from tests.unit.graph_build.test_inference import Harness, raw

    task = report_tasks((LEAF,), (evidence(),), ())[0]
    harness = Harness(tmp_path, [raw(b"{"), raw(b"{")])
    request = report_request(harness, task)
    key = receipt_key(SCOPE, request.role, request.request_fingerprint)

    def validator(data: bytes) -> ValidationResult:
        try:
            validate_report(data, task, key)
        except ValueError:
            return ValidationResult(None, ("invalid_report",), "structural")
        return ValidationResult(data)

    failure = harness.runner.run(SCOPE, request, task.source_refs, validator)
    assert (failure.status, failure.call_count, failure.repair_count) == ("failed", 2, 1)
    harness.generator.actions = iter([raw(payload(task, summary=[statement()]))])
    assert harness.runner.run(SCOPE, request, task.source_refs, validator).status == "succeeded"
    assert len(harness.generator.requests) == 3
    harness.tokenizer.extra = request.inference_policy.context_window_tokens
    with pytest.raises(ValueError, match="context_window_exceeded"):
        harness.runner.run(
            SCOPE,
            request,
            task.source_refs,
            validator,
            mode="regenerate",
            generation_id="preflight-test",
        )
    assert len(harness.generator.requests) == 3


def test_completed_report_remains_usable_near_original_wire_limit() -> None:
    base = report_tasks((LEAF,), (evidence(),), ())[0]
    limit = len(payload(base, summary=[statement()])) + 1
    task = report_tasks((LEAF,), (evidence(),), (), profile=ReportProfile(max_codepoints=limit))[0]
    done = finished(task, summary=[statement()])
    assert (
        report_tasks((LEAF,), (evidence(),), (), completed_reports=(done,), profile=task.profile)[
            0
        ].report_kind
        == "overview"
    )


def test_child_citation_freezes_closure_and_task_rejects_changed_source_path() -> None:
    task = report_tasks((LEAF,), (evidence(),), ())[0]
    done = finished(task, summary=[statement()])
    (parent,) = report_tasks((LEAF,), (evidence(),), (), completed_reports=(done,))
    refs = list(parent.child_citations[0].assertions)
    citation = replace(parent.child_citations[0], assertions=refs)  # type: ignore[arg-type]
    refs.clear()
    assert citation.assertion_ids == ("assertion:1",)
    with pytest.raises(ValueError):
        replace(parent, child_citations=(replace(citation, source_refs=("alien",)),))


def test_actual_transitive_child_paths_keep_original_sources() -> None:
    root = Community("root", None, 0, ("skill", "unused"), ())
    leaf = Community("leaf", "root", 1, ("skill",), ())
    other = Community("other", "root", 1, ("unused",), ())
    rows = (root, leaf, other)
    first = report_tasks(rows, (evidence(),), ())
    done = finished(first[0], summary=[statement()])
    (parent,) = report_tasks(
        rows, (evidence(),), (), completed_reports=(done,), failed_community_ids=("other",)
    )
    parent_done = finished(
        parent,
        summary=[
            statement(
                assertion_refs=[],
                child_statement_refs=[asdict(parent.allowed_child_statement_refs[0])],
            )
        ],
        limitations=[
            statement(
                "limitation", local_id="limit", text="partial_child_coverage", assertion_refs=[]
            )
        ],
    )
    (overview,) = report_tasks(
        rows,
        (evidence(),),
        (),
        completed_reports=(done, parent_done),
        failed_community_ids=("other",),
    )
    citation = next(item for item in overview.child_citations if item.kind == "fact")
    assert citation.ref.report_id == parent_done.report.report_id
    assert citation.assertion_ids == ("assertion:1",)
    assert citation.source_refs == ("source",)


def test_permuted_community_members_and_finding_refs_keep_task_identity() -> None:
    item = evidence(counter=True)
    community = replace(LEAF, member_ids=("skill", "extra"), issue_codes=("one", "two"))
    original = report_tasks((community,), (item,), (finding(item),))[0]
    shuffled = replace(
        community,
        member_ids=tuple(reversed(community.member_ids)),
        issue_codes=tuple(reversed(community.issue_codes)),
    )
    assert report_tasks((shuffled,), (item,), (finding(item),))[0] == original


def test_cross_community_findings_survive_in_overview() -> None:
    item = evidence(target="cross-community-target")
    diagnostic = finding(item)
    leaf_task = report_tasks((LEAF,), (item,), (diagnostic,))[0]
    assert leaf_task.findings == ()
    done = finished(
        leaf_task,
        limitations=[statement("limitation", text="insufficient_evidence", assertion_refs=[])],
    )
    (overview,) = report_tasks((LEAF,), (item,), (diagnostic,), completed_reports=(done,))
    assert overview.allowed_finding_ids == ("finding:1",)
    assert overview.allowed_assertion_ids == ("assertion:1",)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_assertions", True),
        ("max_response_bytes", 65537),
        ("max_child_statements", 0),
        ("version", 2),
    ],
)
def test_report_profile_cannot_expand_or_coerce_closed_limits(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        replace(ReportProfile(), **{field: value})


def two_leaf_reports(profile: ReportProfile) -> tuple[Any, ...]:
    """Two independent source snapshots, with authored distinct child prose."""
    root = Community("root", None, 0, ("a", "b"), ())
    rows = (
        root,
        Community("left", "root", 1, ("a",), ()),
        Community("right", "root", 1, ("b",), ()),
    )
    a, b = evidence("a", "1"), evidence("b", "2")
    b = replace(b, assertions=(replace(b.assertions[0], evidence_refs=("other-source",)),))
    inputs = (a, b)
    tasks = report_tasks(rows, inputs, (), profile=profile)
    done = tuple(
        finished(
            task,
            summary=[
                statement(
                    text="CHILD_PROSE_" + str(i), assertion_refs=list(task.allowed_assertion_ids)
                )
            ],
        )
        for i, task in enumerate(tasks)
    )
    parent = report_tasks(rows, inputs, (), completed_reports=done, profile=profile)[0]
    return rows, inputs, done, parent


def test_low_child_budget_omits_unselected_prose_and_retains_content_identity() -> None:
    profile = ReportProfile(max_assertions=1, max_child_statements=1)
    rows, inputs, done, parent = two_leaf_reports(profile)
    encoded = json.dumps(asdict(parent))
    selected = parent.allowed_child_statement_refs[0].report_id
    for child in done:
        text = child.report.sections.summary[0].text
        assert (text in encoded) == (child.report.report_id == selected)
    unselected = next(child for child in done if child.report.report_id != selected)
    changed = finished(
        unselected.task,
        summary=[
            statement(
                text="CHANGED_OMITTED_PROSE",
                assertion_refs=list(unselected.task.allowed_assertion_ids),
            )
        ],
    )
    replacements = tuple(changed if child is unselected else child for child in done)
    updated = report_tasks(rows, inputs, (), completed_reports=replacements, profile=profile)[0]
    assert parent.cache_key != updated.cache_key


@pytest.mark.parametrize("mutation", ["empty", "kind", "repoint"])
def test_child_closure_cannot_be_forged_by_replacing_parallel_metadata(mutation: str) -> None:
    _, _, _, parent = two_leaf_reports(ReportProfile())
    citation = parent.child_citations[0]
    changes: dict[str, Any]
    if mutation == "empty":
        changes = {"assertion_ids": (), "finding_ids": (), "source_refs": ()}
    elif mutation == "kind":
        changes = {"kind": "limitation"}
    else:
        changes = {
            "assertion_ids": parent.child_citations[1].assertion_ids,
            "source_refs": parent.child_citations[1].source_refs,
        }
    with pytest.raises(ValueError):
        forged = replace(citation, **changes)
        replace(parent, child_citations=(forged, *parent.child_citations[1:]))


def test_source_snapshot_cannot_drift_from_selected_report_evidence() -> None:
    task = report_tasks((LEAF,), (evidence(),), ())[0]
    with pytest.raises(ValueError):
        replace(
            task,
            source_assertions=(
                replace(task.source_assertions[0], evidence_refs=("alien-source",)),
            ),
        )


def test_child_finding_limitations_survive_parent_finding_budget_and_next_level() -> None:
    profile = ReportProfile(max_findings=1)
    rows, inputs, _, _ = two_leaf_reports(profile)
    findings = (
        finding(inputs[0], finding_id="finding:a", limitations=("limit-a",)),
        finding(inputs[1], finding_id="finding:b", limitations=("limit-b",)),
    )
    tasks = report_tasks(rows, inputs, findings, profile=profile)
    done = tuple(
        finished(
            task,
            summary=[
                statement(
                    "hypothesis", assertion_refs=[], finding_refs=list(task.allowed_finding_ids)
                )
            ],
            limitations=[
                statement(
                    "limitation",
                    local_id="limit",
                    text=task.findings[0].limitations[0],
                    assertion_refs=[],
                )
            ],
        )
        for task in tasks
    )
    (parent,) = report_tasks(rows, inputs, findings, completed_reports=done, profile=profile)
    assert parent.allowed_finding_ids == ("finding:a",)
    assert {"limit-a", "limit-b"} <= set(parent.required_limitations)
    claim = statement(
        "hypothesis",
        assertion_refs=[],
        child_statement_refs=[asdict(ref) for ref in parent.allowed_child_statement_refs],
    )
    notices = [
        statement("limitation", local_id="l" + str(i), text=limit, assertion_refs=[])
        for i, limit in enumerate(parent.required_limitations)
    ]
    with pytest.raises(ValueError):
        validate_report(
            payload(
                parent,
                summary=[claim],
                limitations=[row for row in notices if row["text"] != "limit-b"],
            ),
            parent,
            "receipt:parent",
        )
    parent_done = finished(parent, summary=[claim], limitations=notices)
    (overview,) = report_tasks(
        rows, inputs, findings, completed_reports=(*done, parent_done), profile=profile
    )
    assert {"limit-a", "limit-b"} <= set(overview.required_limitations)


def test_source_free_tasks_materialize_without_provider_or_fabricated_sources(
    tmp_path: Path,
) -> None:
    from capicove.graph_build.reports import materialize_inventory_report
    from tests.unit.graph_build.test_inference import Harness

    harness = Harness(tmp_path, [])
    task = report_tasks((LEAF,), (), ())[0]
    assert task.requires_generation is False
    first = materialize_inventory_report(task)
    assert first == materialize_inventory_report(task)
    assert task.scope is None and task.source_refs == ()
    assert first.sections.limitations[0].text == "insufficient_evidence"
    done = CompletedReport(first, task)
    (overview,) = report_tasks((LEAF,), (), (), completed_reports=(done,))
    assert overview.requires_generation is False
    assert overview.child_citations == ()
    assert materialize_inventory_report(overview).community_id is None
    assert harness.generator.requests == []
    with pytest.raises(ValueError):
        materialize_inventory_report(report_tasks((LEAF,), (evidence(),), ())[0])


def test_selected_child_metadata_cannot_drift_from_completed_child_binding() -> None:
    item = evidence(counter=True)
    task = report_tasks((LEAF,), (item,), ())[0]
    done = finished(
        task,
        summary=[statement()],
        counter_evidence=[
            statement(
                local_id="counter", text="A contrary observation.", assertion_refs=["counter:1"]
            )
        ],
    )
    (parent,) = report_tasks((LEAF,), (item,), (), completed_reports=(done,))
    index = next(i for i, row in enumerate(parent.child_citations) if row.counter_assertion_ids)
    for changes in ({"counter_assertion_ids": ()}, {"section": "strengths"}):
        changed = list(parent.child_citations)
        changed[index] = replace(changed[index], **changes)
        with pytest.raises(ValueError):
            replace(parent, child_citations=tuple(changed))


def test_low_budget_task8_payload_and_replay_authorize_every_retained_child_text(
    tmp_path: Path,
) -> None:
    from capicove.graph_build.inference import ValidationResult
    from capicove.graph_build.receipts import receipt_key
    from tests.unit.graph_build.test_inference import Harness, raw

    _, _, done, task = two_leaf_reports(ReportProfile(max_assertions=1, max_child_statements=1))
    harness = Harness(tmp_path, [])
    request = report_request(harness, task)
    model_input = json.loads(request.canonical_payload)["segments"][0][2]
    for child in done:
        if child.report.sections.summary[0].text in model_input:
            assert set(child.task.source_refs) <= set(task.source_refs)
    assert model_input.count("CHILD_PROSE_") == 1
    notices = [
        statement("limitation", local_id=f"limit:{i}", text=text, assertion_refs=[])
        for i, text in enumerate(task.required_limitations)
    ]
    data = payload(task, summary=[statement()], limitations=notices)
    harness.generator.actions = iter([raw(data)])
    receipt = receipt_key(SCOPE, request.role, request.request_fingerprint)

    def validator(raw_payload: bytes) -> ValidationResult:
        validate_report(raw_payload, task, receipt)
        return ValidationResult(raw_payload)

    assert harness.runner.run(SCOPE, request, task.source_refs, validator).status == "succeeded"
    assert harness.runner.run(SCOPE, request, task.source_refs, validator).status == "succeeded"
    assert len(harness.generator.requests) == 1
    harness.gate.denied_sources.update(task.child_citations[0].source_refs)
    with pytest.raises(PermissionError):
        harness.runner.run(SCOPE, request, task.source_refs, validator)
    assert len(harness.generator.requests) == 1


def test_selected_sourced_limitations_remain_required_in_parent_output() -> None:
    task = report_tasks((LEAF,), (evidence(),), ())[0]
    done = finished(
        task, limitations=[statement("limitation", text="This source covers one occasion.")]
    )
    (parent,) = report_tasks((LEAF,), (evidence(),), (), completed_reports=(done,))
    assert "This source covers one occasion." in parent.required_limitations
    with pytest.raises(ValueError):
        validate_report(payload(parent, summary=[statement()]), parent, "receipt:parent")
