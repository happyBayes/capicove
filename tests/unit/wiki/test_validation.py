import json
from datetime import date
from pathlib import Path

import pytest

from capicove.wiki import WikiService, WikiValidationError


def _initialized(tmp_path: Path) -> tuple[WikiService, Path]:
    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 9, 1))
    return service, tmp_path / "wiki"


def test_lint_accepts_an_ingested_page(tmp_path: Path) -> None:
    service, _ = _initialized(tmp_path)
    source = tmp_path / "alpha.md"
    source.write_text("# Alpha\n\nLocal knowledge.\n", encoding="utf-8")
    service.ingest(source, ingested_on=date(2026, 9, 1))

    report = service.lint()

    assert report.valid is True
    assert report.findings == ()


def test_lint_reports_malformed_frontmatter(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "pages" / "broken.md").write_text("---\nid: [\n---\nBody\n", encoding="utf-8")

    report = service.lint()

    assert report.valid is False
    assert [(item.severity.value, item.code, item.path) for item in report.findings] == [
        ("error", "page_frontmatter_invalid", "pages/broken.md")
    ]


def _inferred_fact_ledger(wiki: Path, input_fact: object) -> Path:
    (wiki / "pages" / "subject.md").write_text(
        "---\nid: page:SUBJECT0001\ntitle: Subject\nkind: concept\n"
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\nSubject.\n",
        encoding="utf-8",
    )
    (wiki / "graph" / "ontology.yaml").write_text(
        "schemaVersion: 1\nentityTypes: {}\npredicates:\n"
        "  displayName:\n    kind: data\n    valueType: string\n"
        "labels:\n  namespaces: []\n",
        encoding="utf-8",
    )
    base = {
        "factId": "fact:FACT000002",
        "subjectId": "page:SUBJECT0001",
        "predicate": "displayName",
        "value": "Subject",
        "valueType": "string",
        "status": "current",
        "knowledge": "asserted",
        "rationale": "Reviewed source statement",
    }
    inferred = {
        **base,
        "factId": "fact:FACT000001",
        "knowledge": "inferred",
        "rule": "display-name-copy",
        "inputFacts": [input_fact],
    }
    ledger = wiki / "graph" / "facts.jsonl"
    ledger.write_text(json.dumps(inferred) + "\n" + json.dumps(base) + "\n", encoding="utf-8")
    return ledger


@pytest.mark.parametrize("input_fact", ["not-a-fact", "fact:MISSING001", ["fact:FACT000002"]])
def test_lint_rejects_invalid_or_unresolved_inference_inputs(
    tmp_path: Path, input_fact: object
) -> None:
    service, wiki = _initialized(tmp_path)
    ledger = _inferred_fact_ledger(wiki, input_fact)
    original = ledger.read_bytes()

    report = service.lint()

    assert [(item.code, item.path) for item in report.findings] == [
        ("fact_inference_invalid", "graph/facts.jsonl:1")
    ]
    with pytest.raises(WikiValidationError):
        service.build(built_on=date(2026, 9, 1))
    assert ledger.read_bytes() == original
    assert not (wiki / "_derived" / "search.sqlite3").exists()


def test_lint_accepts_inference_input_declared_later_in_the_ledger(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    _inferred_fact_ledger(wiki, "fact:FACT000002")

    assert service.lint().valid is True


def test_lint_rejects_a_yaml_escaped_surrogate_title(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    page = wiki / "pages" / "invalid-title.md"
    page.write_text(
        '---\nid: page:INVALIDTITLE01\ntitle: "\\uD800"\nkind: concept\n'
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\nBody.\n",
        encoding="utf-8",
    )
    original = page.read_bytes()

    report = service.lint()

    assert report.valid is False
    assert [(item.code, item.path) for item in report.findings] == [
        ("page_title_invalid", "pages/invalid-title.md")
    ]
    assert page.read_bytes() == original


def test_lint_reports_an_unreadable_log(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "log.md").write_bytes(b"\xff")

    report = service.lint()

    assert report.valid is False
    assert ("log_invalid", "log.md") in {(item.code, item.path) for item in report.findings}


def test_lint_rejects_a_timestamp_where_an_iso_date_is_required(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nDated content.\n", encoding="utf-8")
    result = service.ingest(source, ingested_on=date(2026, 9, 1))
    page = wiki / result.page_path
    rendered = page.read_text(encoding="utf-8")
    page.write_text(
        rendered.replace("created: 2026-09-01", "created: 2026-09-01T00:00:00", 1),
        encoding="utf-8",
    )

    report = service.lint()

    assert ("page_created_invalid", result.page_path) in {
        (item.code, item.path) for item in report.findings
    }


def test_lint_reports_duplicate_ids_and_broken_wikilinks(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    page = (
        "---\nid: page:DUPLICATE01\ntitle: {title}\nkind: concept\n"
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\n{body}\n"
    )
    (wiki / "pages" / "alpha.md").write_text(
        page.format(title="Alpha", body="See [[Missing Page]]."), encoding="utf-8"
    )
    (wiki / "pages" / "beta.md").write_text(
        page.format(title="Beta", body="Beta."), encoding="utf-8"
    )

    report = service.lint()

    assert [(item.code, item.path) for item in report.findings] == [
        ("wikilink_broken", "pages/alpha.md"),
        ("page_id_duplicate", "pages/beta.md"),
    ]


def test_lint_reports_unknown_predicate_and_unresolved_fact_object(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    page = (
        "---\nid: page:SUBJECT0001\ntitle: Subject\nkind: entity\nentityTypes: []\n"
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\nSubject.\n"
    )
    (wiki / "pages" / "subject.md").write_text(page, encoding="utf-8")
    (wiki / "graph" / "facts.jsonl").write_text(
        '{"factId":"fact:FACT000001","status":"current","knowledge":"asserted",'
        '"subjectId":"page:SUBJECT0001","predicate":"unknown",'
        '"objectId":"page:MISSING0001","sources":[]}\n',
        encoding="utf-8",
    )

    report = service.lint()

    assert [(item.code, item.path) for item in report.findings] == [
        ("fact_object_unresolved", "graph/facts.jsonl:1"),
        ("fact_predicate_unknown", "graph/facts.jsonl:1"),
        ("fact_provenance_invalid", "graph/facts.jsonl:1"),
    ]


def test_lint_rejects_incomplete_ontology_and_malformed_graph_ledgers(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "graph" / "ontology.yaml").write_text("schemaVersion: 1\n", encoding="utf-8")
    (wiki / "graph" / "facts.jsonl").write_text(
        '{"factId":"fact:FACT000001","subjectId":"page:MISSING0001",'
        '"predicate":"unknown","value":"x","valueType":"string"}\n',
        encoding="utf-8",
    )
    (wiki / "graph" / "identity-events.jsonl").write_text(
        '{"eventId":"identity:EVENT00001","type":"MERGED_INTO"}\n'
        '{"eventId":"identity:EVENT00001","type":"MERGED_INTO"}\n',
        encoding="utf-8",
    )

    codes = {item.code for item in service.lint().findings}

    assert "ontology_entity_types_invalid" in codes
    assert "ontology_predicates_invalid" in codes
    assert "ontology_labels_invalid" in codes
    assert "fact_status_invalid" in codes
    assert "fact_knowledge_invalid" in codes
    assert "fact_provenance_invalid" in codes
    assert "identity_event_id_duplicate" in codes
    assert "identity_event_rationale_invalid" in codes
    assert "identity_event_date_invalid" in codes
    assert "identity_event_merge_invalid" in codes


def test_lint_recomputes_content_addressed_source_hash(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nOriginal.\n", encoding="utf-8")
    result = service.ingest(source, ingested_on=date(2026, 9, 1))
    digest = result.source_id.removeprefix("source:sha256:")
    (wiki / "raw" / f"{digest}.md").write_text("tampered", encoding="utf-8")

    report = service.lint()

    assert ("source_hash_mismatch", f"raw/{digest}.md") in {
        (item.code, item.path) for item in report.findings
    }


def test_lint_rejects_nested_and_duplicate_raw_captures(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    source = tmp_path / "source.md"
    source.write_text("# Source\n\nOriginal.\n", encoding="utf-8")
    result = service.ingest(source, ingested_on=date(2026, 9, 1))
    digest = result.source_id.removeprefix("source:sha256:")
    direct = wiki / "raw" / f"{digest}.md"
    nested = wiki / "raw" / "nested" / f"{digest}.md"
    nested.parent.mkdir()
    nested.write_bytes(direct.read_bytes())

    findings = {(item.code, item.path) for item in service.lint().findings}

    assert ("source_path_invalid", f"raw/nested/{digest}.md") in findings
    assert ("source_id_duplicate", f"raw/nested/{digest}.md") in findings


def test_lint_reports_unhashable_identity_id_without_crashing(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "graph" / "identity-events.jsonl").write_text(
        '{"eventId":["bad"],"type":"UNKNOWN"}\n', encoding="utf-8"
    )

    codes = {item.code for item in service.lint().findings}

    assert "identity_event_id_invalid" in codes
    assert "identity_event_type_invalid" in codes


def test_lint_reports_unhashable_graph_references_without_crashing(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "graph" / "facts.jsonl").write_text(
        '{"factId":"fact:FACT000001","status":"current","knowledge":"asserted",'
        '"subjectId":["bad"],"predicate":"unknown","value":"x",'
        '"valueType":"string","rationale":"reviewed"}\n',
        encoding="utf-8",
    )
    (wiki / "graph" / "identity-events.jsonl").write_text(
        '{"eventId":"identity:EVENT00001","type":"MERGED_INTO",'
        '"fromId":["bad"],"toId":{"bad":true},"rationale":"reviewed",'
        '"effectiveAt":"2026-09-01"}\n',
        encoding="utf-8",
    )

    codes = {item.code for item in service.lint().findings}

    assert "fact_subject_unresolved" in codes
    assert "identity_event_merge_invalid" in codes


def test_lint_reports_container_enum_values_without_crashing(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "pages" / "bad-kind.md").write_text(
        "---\nid: page:BADKIND0001\ntitle: Bad kind\nkind: []\n"
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (wiki / "graph" / "ontology.yaml").write_text(
        "schemaVersion: 1\n"
        "entityTypes: {}\n"
        "predicates:\n"
        "  broken:\n"
        "    kind: []\n"
        "    domain: []\n"
        "labels:\n"
        "  namespaces: []\n",
        encoding="utf-8",
    )
    (wiki / "graph" / "facts.jsonl").write_text(
        '{"factId":"fact:FACT000001","status":[],"knowledge":{},'
        '"subjectId":"page:BADKIND0001","predicate":"broken",'
        '"value":"x","valueType":"string","rationale":"reviewed"}\n',
        encoding="utf-8",
    )

    codes = {item.code for item in service.lint().findings}

    assert "page_kind_invalid" in codes
    assert "ontology_predicate_invalid" in codes
    assert "fact_status_invalid" in codes
    assert "fact_knowledge_invalid" in codes


def test_lint_enforces_fact_shape_declared_by_predicate(tmp_path: Path) -> None:
    service, wiki = _initialized(tmp_path)
    (wiki / "graph" / "ontology.yaml").write_text(
        "schemaVersion: 1\n"
        "entityTypes: {}\n"
        "predicates:\n"
        "  displayName:\n"
        "    kind: data\n"
        "    domain: []\n"
        "    valueType: string\n"
        "labels:\n"
        "  namespaces: []\n",
        encoding="utf-8",
    )
    (wiki / "pages" / "subject.md").write_text(
        "---\nid: page:SUBJECT0001\ntitle: Subject\nkind: concept\n"
        "created: 2026-09-01\nupdated: 2026-09-01\nsources: []\n---\n\nSubject.\n",
        encoding="utf-8",
    )
    (wiki / "graph" / "facts.jsonl").write_text(
        '{"factId":"fact:FACT000001","status":"current","knowledge":"asserted",'
        '"subjectId":"page:SUBJECT0001","predicate":"displayName",'
        '"objectId":"page:SUBJECT0001","rationale":"reviewed"}\n'
        '{"factId":"fact:FACT000002","status":"current","knowledge":"asserted",'
        '"subjectId":"page:SUBJECT0001","predicate":"displayName",'
        '"value":7,"valueType":"integer","rationale":"reviewed"}\n',
        encoding="utf-8",
    )

    findings = {(item.code, item.path) for item in service.lint().findings}

    assert ("fact_object_shape_invalid", "graph/facts.jsonl:1") in findings
    assert ("fact_value_type_invalid", "graph/facts.jsonl:2") in findings
