"""Deterministic validation for standalone local Wikis."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from capicove.books.tasks import AtomicClaim
from capicove.contracts import DocumentParseResult
from capicove.documents import document_parse_result_from_json
from capicove.wiki.citations import validate_book_citations
from capicove.wiki.filesystem import (
    WikiDependencyError,
    WikiError,
    WikiFormatError,
    jsonl_records,
    load_yaml_mapping,
    page_paths,
    read_log,
    read_markdown,
    scan,
)
from capicove.wiki.models import (
    ClaimCitation,
    WikiFinding,
    WikiFindingSeverity,
    WikiLintReport,
)


_KINDS = frozenset({"entity", "concept", "comparison", "query", "summary"})
_PAGE_ID = re.compile(r"^page:[A-Z0-9]{10,64}$")
_FACT_ID = re.compile(r"^fact:[A-Z0-9]{6,64}$")
_EVENT_ID = re.compile(r"^(?:event|identity):[A-Z0-9]{6,64}$")
_SOURCE_ID = re.compile(r"^source:sha256:([0-9a-f]{64})$")
_PAGE_FILENAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
_WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_SEVERITY_ORDER = {
    WikiFindingSeverity.ERROR: 0,
    WikiFindingSeverity.WARNING: 1,
    WikiFindingSeverity.STYLE: 2,
}


def lint(wiki: Path) -> WikiLintReport:
    """Validate one Wiki without modifying durable or derived content."""
    findings: list[WikiFinding] = []
    try:
        inventory = scan(wiki)
    except WikiError as error:
        findings.append(_finding("wiki_format_invalid", ".", str(error)))
        return _report(findings)
    try:
        read_log(wiki)
    except WikiFormatError as error:
        findings.append(_finding("log_invalid", "log.md", str(error)))
    ontology: dict[str, Any] = {}
    try:
        ontology = load_yaml_mapping(wiki / "graph" / "ontology.yaml")
        if ontology.get("schemaVersion") != 1:
            findings.append(
                _finding(
                    "ontology_schema_invalid",
                    "graph/ontology.yaml",
                    "ontology schemaVersion must be 1",
                )
            )
        entity_types, predicates = _validate_ontology(ontology, findings)
    except WikiDependencyError:
        raise
    except WikiFormatError as error:
        findings.append(_finding("ontology_invalid", "graph/ontology.yaml", str(error)))
        entity_types = {}
        predicates = {}

    source_ids = _validate_sources(wiki, findings, schema_version=inventory.schema_version)

    pages: dict[str, tuple[str, dict[str, Any], str]] = {}
    page_names: set[str] = set()
    page_titles: set[str] = set()
    for path in page_paths(wiki):
        relative = path.relative_to(wiki).as_posix()
        if _PAGE_FILENAME.fullmatch(path.name) is None:
            findings.append(
                _finding(
                    "page_filename_invalid", relative, "page filename must be lowercase kebab-case"
                )
            )
        try:
            record = read_markdown(path)
        except WikiDependencyError:
            raise
        except WikiFormatError as error:
            findings.append(_finding("page_frontmatter_invalid", relative, str(error)))
            continue
        metadata = record.metadata
        page_id = metadata.get("id")
        title = metadata.get("title")
        if not isinstance(page_id, str) or _PAGE_ID.fullmatch(page_id) is None:
            findings.append(_finding("page_id_invalid", relative, "page id is missing or invalid"))
        elif page_id in pages:
            findings.append(
                _finding("page_id_duplicate", relative, f"duplicate page id: {page_id}")
            )
        else:
            pages[page_id] = (relative, metadata, record.body)
        if not isinstance(title, str) or not title.strip():
            findings.append(
                _finding("page_title_invalid", relative, "page title is missing or blank")
            )
        else:
            try:
                title.encode("utf-8")
            except UnicodeEncodeError:
                findings.append(
                    _finding(
                        "page_title_invalid", relative, "page title must contain valid Unicode"
                    )
                )
            else:
                page_titles.add(title.casefold())
        page_kind = metadata.get("kind")
        if not isinstance(page_kind, str) or page_kind not in _KINDS:
            findings.append(
                _finding("page_kind_invalid", relative, "page kind is missing or invalid")
            )
        for field in ("created", "updated"):
            if not _valid_date(metadata.get(field)):
                findings.append(
                    _finding(f"page_{field}_invalid", relative, f"page {field} must be an ISO date")
                )
        sources = metadata.get("sources")
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            findings.append(
                _finding("page_sources_invalid", relative, "page sources must be a string list")
            )
        else:
            for source_id in sources:
                if _SOURCE_ID.fullmatch(source_id) is None or source_id not in source_ids:
                    findings.append(
                        _finding(
                            "page_source_unresolved",
                            relative,
                            f"source does not resolve: {source_id}",
                        )
                    )
        page_entity_types = metadata.get("entityTypes", [])
        if page_kind == "entity":
            if not isinstance(page_entity_types, list) or any(
                not isinstance(item, str) or item not in entity_types for item in page_entity_types
            ):
                findings.append(
                    _finding(
                        "page_entity_types_invalid",
                        relative,
                        "entity page types must be a list of registered entity types",
                    )
                )
        elif "entityTypes" in metadata:
            findings.append(
                _finding(
                    "page_entity_types_invalid",
                    relative,
                    "entityTypes is allowed only on entity pages",
                )
            )
        page_names.add(path.stem.casefold())

    for _, (relative, _, body) in sorted(pages.items()):
        for target in sorted(set(_WIKILINK.findall(body)), key=str.casefold):
            key = target.strip().casefold()
            if key not in page_names and key not in page_titles:
                findings.append(
                    _finding("wikilink_broken", relative, f"wikilink does not resolve: {target}")
                )

    try:
        facts = jsonl_records(wiki / "graph" / "facts.jsonl")
    except WikiFormatError as error:
        findings.append(_finding("facts_jsonl_invalid", "graph/facts.jsonl", str(error)))
        facts = ()
    fact_ids = {
        fact["factId"]
        for fact in facts
        if isinstance(fact.get("factId"), str) and _FACT_ID.fullmatch(fact["factId"]) is not None
    }
    seen_facts: set[str] = set()
    for number, fact in enumerate(facts, 1):
        fact_path = f"graph/facts.jsonl:{number}"
        fact_id = fact.get("factId")
        if not isinstance(fact_id, str) or _FACT_ID.fullmatch(fact_id) is None:
            findings.append(_finding("fact_id_invalid", fact_path, "fact id is missing or invalid"))
        elif fact_id in seen_facts:
            findings.append(
                _finding("fact_id_duplicate", fact_path, f"duplicate fact id: {fact_id}")
            )
        else:
            seen_facts.add(fact_id)
        subject_id = fact.get("subjectId")
        if not isinstance(subject_id, str) or subject_id not in pages:
            findings.append(
                _finding("fact_subject_unresolved", fact_path, "fact subject does not resolve")
            )
        object_id = fact.get("objectId")
        has_object = isinstance(object_id, str)
        has_value = "value" in fact and isinstance(fact.get("valueType"), str)
        if has_object == has_value:
            findings.append(
                _finding(
                    "fact_object_shape_invalid",
                    fact_path,
                    "fact requires exactly one object form",
                )
            )
        elif has_object and object_id not in pages:
            findings.append(
                _finding("fact_object_unresolved", fact_path, "fact object does not resolve")
            )
        predicate_name = fact.get("predicate")
        predicate = predicates.get(predicate_name) if isinstance(predicate_name, str) else None
        if not isinstance(predicate, dict):
            findings.append(
                _finding("fact_predicate_unknown", fact_path, "fact predicate is not registered")
            )
        elif predicate.get("kind") == "object" and (not has_object or has_value):
            findings.append(
                _finding(
                    "fact_object_shape_invalid",
                    fact_path,
                    "object predicate requires objectId only",
                )
            )
        elif predicate.get("kind") == "data":
            if has_object or not has_value:
                findings.append(
                    _finding(
                        "fact_object_shape_invalid",
                        fact_path,
                        "data predicate requires value and valueType only",
                    )
                )
            elif fact.get("valueType") != predicate.get("valueType"):
                findings.append(
                    _finding(
                        "fact_value_type_invalid",
                        fact_path,
                        "fact valueType does not match the predicate declaration",
                    )
                )
        status = fact.get("status")
        if not isinstance(status, str) or status not in {"current", "superseded", "retracted"}:
            findings.append(
                _finding("fact_status_invalid", fact_path, "fact status is missing or invalid")
            )
        knowledge = fact.get("knowledge")
        if not isinstance(knowledge, str) or knowledge not in {"asserted", "inferred"}:
            findings.append(
                _finding(
                    "fact_knowledge_invalid",
                    fact_path,
                    "fact knowledge is missing or invalid",
                )
            )
        fact_sources = fact.get("sources")
        has_sources = isinstance(fact_sources, list) and bool(fact_sources)
        has_rationale = isinstance(fact.get("rationale"), str) and bool(fact["rationale"].strip())
        if not has_sources and not has_rationale:
            findings.append(
                _finding(
                    "fact_provenance_invalid",
                    fact_path,
                    "fact requires source provenance or a reviewable rationale",
                )
            )
        if fact_sources is not None:
            if not isinstance(fact_sources, list) or any(
                not isinstance(item, str)
                or _SOURCE_ID.fullmatch(item) is None
                or item not in source_ids
                for item in fact_sources
            ):
                findings.append(
                    _finding(
                        "fact_sources_invalid",
                        fact_path,
                        "fact sources must resolve to content-addressed captures",
                    )
                )
        if knowledge == "inferred" and (
            not isinstance(fact.get("rule"), str)
            or not fact["rule"].strip()
            or not isinstance(fact.get("inputFacts"), list)
            or not fact["inputFacts"]
            or any(
                not isinstance(item, str)
                or _FACT_ID.fullmatch(item) is None
                or item not in fact_ids
                for item in fact["inputFacts"]
            )
        ):
            findings.append(
                _finding(
                    "fact_inference_invalid",
                    fact_path,
                    "inferred fact requires a rule and resolvable inputFacts IDs",
                )
            )
        for field in ("assertedAt", "validFrom", "validTo"):
            if field in fact and not _valid_date(fact[field]):
                findings.append(
                    _finding("fact_date_invalid", fact_path, f"fact {field} must be an ISO date")
                )
        if (
            _valid_date(fact.get("validFrom"))
            and _valid_date(fact.get("validTo"))
            and str(fact["validFrom"]) > str(fact["validTo"])
        ):
            findings.append(
                _finding("fact_date_interval_invalid", fact_path, "validFrom exceeds validTo")
            )

    _validate_identity_events(wiki, pages, findings)
    if inventory.schema_version == 2:
        _validate_v2_citations(wiki, pages, source_ids, findings)
    return _report(findings)


def _validate_v2_citations(
    wiki: Path,
    pages: dict[str, tuple[str, dict[str, Any], str]],
    source_ids: set[str],
    findings: list[WikiFinding],
) -> None:
    representations: dict[str, DocumentParseResult] = {}
    for path in sorted((wiki / "representations").glob("*.json")):
        relative = path.relative_to(wiki).as_posix()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            result = document_parse_result_from_json(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            findings.append(
                _finding(
                    "representation_invalid",
                    relative,
                    f"book representation is invalid: {error}",
                )
            )
            continue
        if path.stem != result.representation_sha256:
            findings.append(
                _finding(
                    "representation_identity_mismatch",
                    relative,
                    "representation filename does not match its identity",
                )
            )
            continue
        representations[result.representation_sha256] = result
    try:
        records = jsonl_records(wiki / "graph" / "citations.jsonl")
    except WikiFormatError as error:
        findings.append(_finding("citations_jsonl_invalid", "graph/citations.jsonl", str(error)))
        return
    expected = {
        "citation_id",
        "claim_id",
        "page_id",
        "book_id",
        "source_sha256",
        "representation_sha256",
        "page_index",
        "printed_page_label",
        "block_ids",
        "evidence_text_sha256",
        "quotation",
        "submission_version",
    }
    grouped: dict[str, list[ClaimCitation]] = {}
    for number, record in enumerate(records, 1):
        citation_path = f"graph/citations.jsonl:{number}"
        if set(record) != expected:
            findings.append(
                _finding(
                    "citation_record_shape_invalid",
                    citation_path,
                    "citation record fields are missing or unknown",
                )
            )
            continue
        try:
            label = record["printed_page_label"]
            quotation = record["quotation"]
            blocks = record["block_ids"]
            if not isinstance(blocks, list) or any(not isinstance(item, str) for item in blocks):
                raise ValueError("block_ids must be a string array")
            citation = ClaimCitation(
                str(record["citation_id"]),
                str(record["claim_id"]),
                str(record["page_id"]),
                str(record["book_id"]),
                str(record["source_sha256"]),
                str(record["representation_sha256"]),
                _strict_int(record["page_index"]),
                None if label is None else str(label),
                tuple(blocks),
                str(record["evidence_text_sha256"]),
                None if quotation is None else str(quotation),
                _strict_int(record["submission_version"]),
            )
        except (TypeError, ValueError) as error:
            findings.append(
                _finding(
                    "citation_record_invalid",
                    citation_path,
                    f"citation record is invalid: {error}",
                )
            )
            continue
        if citation.page_id not in pages:
            findings.append(
                _finding(
                    "citation_page_unresolved",
                    citation_path,
                    "citation Wiki page does not resolve",
                )
            )
        if f"source:sha256:{citation.source_sha256}" not in source_ids:
            findings.append(
                _finding(
                    "citation_source_unresolved",
                    citation_path,
                    "citation source does not resolve",
                )
            )
        grouped.setdefault(citation.representation_sha256, []).append(citation)
    for representation_sha256, citations in grouped.items():
        resolved_result = representations.get(representation_sha256)
        if resolved_result is None:
            findings.append(
                _finding(
                    "citation_representation_unresolved",
                    "graph/citations.jsonl",
                    f"citation representation does not resolve: {representation_sha256}",
                )
            )
            continue
        claim_blocks: dict[str, list[str]] = {}
        for citation in citations:
            claim_blocks.setdefault(citation.claim_id, []).extend(citation.block_ids)
        claims = tuple(
            AtomicClaim(claim_id, "citation ledger claim", tuple(dict.fromkeys(block_ids)))
            for claim_id, block_ids in claim_blocks.items()
        )
        report = validate_book_citations(tuple(citations), claims, resolved_result)
        for item in report.findings:
            findings.append(
                _finding(
                    item.code,
                    "graph/citations.jsonl",
                    item.message,
                )
            )


def _strict_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("value must be an integer")
    return value


def _validate_ontology(
    ontology: dict[str, Any], findings: list[WikiFinding]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = "graph/ontology.yaml"
    entity_types = ontology.get("entityTypes")
    if not isinstance(entity_types, dict):
        findings.append(
            _finding("ontology_entity_types_invalid", path, "entityTypes must be a mapping")
        )
        entity_types = {}
    predicates = ontology.get("predicates")
    if not isinstance(predicates, dict):
        findings.append(
            _finding("ontology_predicates_invalid", path, "predicates must be a mapping")
        )
        predicates = {}
    labels = ontology.get("labels")
    if (
        not isinstance(labels, dict)
        or not isinstance(labels.get("namespaces"), list)
        or any(
            not isinstance(item, str) or not item.strip() for item in labels.get("namespaces", [])
        )
    ):
        findings.append(
            _finding("ontology_labels_invalid", path, "labels.namespaces must be a string list")
        )
    for name, declaration in entity_types.items():
        if not isinstance(name, str) or not name or not isinstance(declaration, dict):
            findings.append(
                _finding(
                    "ontology_entity_type_invalid",
                    path,
                    "entity type declarations must be named mappings",
                )
            )
            continue
        parents = declaration.get("parents", [])
        if not isinstance(parents, list) or any(
            not isinstance(parent, str) or parent not in entity_types for parent in parents
        ):
            findings.append(
                _finding(
                    "ontology_entity_type_invalid",
                    path,
                    f"entity type parents are invalid: {name}",
                )
            )
    for name, declaration in predicates.items():
        if not isinstance(name, str) or not name or not isinstance(declaration, dict):
            findings.append(
                _finding(
                    "ontology_predicate_invalid",
                    path,
                    "predicate declarations must be named mappings",
                )
            )
            continue
        kind = declaration.get("kind")
        if not isinstance(kind, str) or kind not in {"object", "data"}:
            findings.append(
                _finding("ontology_predicate_invalid", path, f"predicate kind is invalid: {name}")
            )
        for field in ("domain", "range") if kind == "object" else ("domain",):
            values = declaration.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(item, str) or item not in entity_types for item in values
            ):
                findings.append(
                    _finding(
                        "ontology_predicate_invalid",
                        path,
                        f"predicate {field} is invalid: {name}",
                    )
                )
        if kind == "data" and not isinstance(declaration.get("valueType"), str):
            findings.append(
                _finding(
                    "ontology_predicate_invalid",
                    path,
                    f"data predicate requires valueType: {name}",
                )
            )
    return entity_types, predicates


def _validate_sources(
    wiki: Path,
    findings: list[WikiFinding],
    *,
    schema_version: int,
) -> set[str]:
    source_ids: set[str] = set()
    seen_digests: set[str] = set()
    raw_root = wiki / "raw"
    for path in sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(raw_root).as_posix(),
    ):
        relative = path.relative_to(wiki).as_posix()
        if path.parent != raw_root:
            findings.append(
                _finding(
                    "source_path_invalid",
                    relative,
                    "raw captures must be direct children of the raw directory",
                )
            )
        suffixes = "md|pdf" if schema_version == 2 else "md"
        match = re.fullmatch(rf"([0-9a-f]{{64}})\.(?:{suffixes})", path.name)
        if match is None:
            findings.append(
                _finding(
                    "source_filename_invalid",
                    relative,
                    "raw source filename must be a lowercase SHA-256 digest with a supported suffix",
                )
            )
            continue
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            findings.append(_finding("source_unreadable", relative, "raw source is unreadable"))
            continue
        expected = match.group(1)
        if expected in seen_digests:
            findings.append(
                _finding(
                    "source_id_duplicate",
                    relative,
                    f"duplicate content-addressed source: source:sha256:{expected}",
                )
            )
        seen_digests.add(expected)
        source_ids.add(f"source:sha256:{expected}")
        if actual != expected:
            findings.append(
                _finding(
                    "source_hash_mismatch",
                    relative,
                    "raw source bytes do not match the content-addressed filename",
                )
            )
    return source_ids


def _validate_identity_events(
    wiki: Path,
    pages: dict[str, tuple[str, dict[str, Any], str]],
    findings: list[WikiFinding],
) -> None:
    try:
        events = jsonl_records(wiki / "graph" / "identity-events.jsonl")
    except WikiFormatError as error:
        findings.append(
            _finding("identity_events_jsonl_invalid", "graph/identity-events.jsonl", str(error))
        )
        return
    string_ids = [
        event_id for event in events if isinstance((event_id := event.get("eventId")), str)
    ]
    duplicate_ids = {value for value, count in Counter(string_ids).items() if count > 1}
    for number, event in enumerate(events, 1):
        path = f"graph/identity-events.jsonl:{number}"
        event_id = event.get("eventId")
        if not isinstance(event_id, str) or _EVENT_ID.fullmatch(event_id) is None:
            findings.append(
                _finding("identity_event_id_invalid", path, "identity event id is invalid")
            )
        if isinstance(event_id, str) and event_id in duplicate_ids:
            findings.append(
                _finding(
                    "identity_event_id_duplicate", path, f"duplicate identity event: {event_id}"
                )
            )
        if not isinstance(event.get("rationale"), str) or not event["rationale"].strip():
            findings.append(
                _finding(
                    "identity_event_rationale_invalid",
                    path,
                    "identity event requires a rationale",
                )
            )
        if not _valid_date(event.get("effectiveAt")):
            findings.append(
                _finding(
                    "identity_event_date_invalid",
                    path,
                    "identity event requires an ISO effectiveAt date",
                )
            )
        event_type = event.get("type")
        if event_type == "MERGED_INTO":
            source = event.get("fromId")
            target = event.get("toId")
            if (
                not isinstance(source, str)
                or not isinstance(target, str)
                or source not in pages
                or target not in pages
                or source == target
            ):
                findings.append(
                    _finding(
                        "identity_event_merge_invalid",
                        path,
                        "merge event requires distinct, resolvable fromId and toId",
                    )
                )
        elif event_type == "SPLIT":
            source = event.get("fromId")
            targets = event.get("toIds")
            if (
                not isinstance(source, str)
                or source not in pages
                or not isinstance(targets, list)
                or len(targets) < 2
                or any(not isinstance(target, str) or target not in pages for target in targets)
            ):
                findings.append(
                    _finding(
                        "identity_event_split_invalid",
                        path,
                        "split event requires a resolvable source and at least two targets",
                    )
                )
        else:
            findings.append(
                _finding("identity_event_type_invalid", path, "identity event type is invalid")
            )


def _valid_date(value: object) -> bool:
    if type(value) is date:
        return True
    if not isinstance(value, str):
        return False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


def _finding(code: str, path: str, message: str) -> WikiFinding:
    return WikiFinding(WikiFindingSeverity.ERROR, code, path, message)


def _report(findings: list[WikiFinding]) -> WikiLintReport:
    ordered = tuple(
        sorted(
            findings,
            key=lambda item: (_SEVERITY_ORDER[item.severity], item.path, item.code, item.message),
        )
    )
    return WikiLintReport(ordered)


__all__ = ["lint"]
