"""Closed JSON codecs for durable book workflow values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from capicove.books.models import (
    BookSection,
    BookStructure,
    StructureConfidence,
    StructureMethod,
    StructureReviewTask,
)
from capicove.books.review import StructureReviewSubmission
from capicove.books.tasks import (
    AtomicClaim,
    SummaryEvidence,
    SummarySubmission,
    SummaryTask,
)


def book_structure_to_json(value: BookStructure) -> dict[str, object]:
    """Encode one deterministic book structure."""
    return {
        "source_sha256": value.source_sha256,
        "representation_sha256": value.representation_sha256,
        "profile_version": value.profile_version,
        "sections": [
            {
                "section_id": item.section_id,
                "exact_title": item.exact_title,
                "normalized_title": item.normalized_title,
                "level": item.level,
                "parent_id": item.parent_id,
                "children_ids": list(item.children_ids),
                "start_page_index": item.start_page_index,
                "end_page_index": item.end_page_index,
                "printed_start_label": item.printed_start_label,
                "printed_end_label": item.printed_end_label,
                "heading_block_ids": list(item.heading_block_ids),
                "method": item.method.value,
                "confidence": item.confidence.value,
                "reason_codes": list(item.reason_codes),
                "warnings": list(item.warnings),
            }
            for item in value.sections
        ],
        "root_ids": list(value.root_ids),
        "warnings": list(value.warnings),
    }


def book_structure_from_json(value: object) -> BookStructure:
    """Decode one closed book structure mapping."""
    root = _mapping(value, "book structure")
    _fields(
        root,
        {
            "source_sha256",
            "representation_sha256",
            "profile_version",
            "sections",
            "root_ids",
            "warnings",
        },
    )
    sections: list[BookSection] = []
    section_fields = {
        "section_id",
        "exact_title",
        "normalized_title",
        "level",
        "parent_id",
        "children_ids",
        "start_page_index",
        "end_page_index",
        "printed_start_label",
        "printed_end_label",
        "heading_block_ids",
        "method",
        "confidence",
        "reason_codes",
        "warnings",
    }
    for raw in _list(root["sections"], "sections"):
        item = _mapping(raw, "section")
        _fields(item, section_fields)
        parent = item["parent_id"]
        start_label = item["printed_start_label"]
        end_label = item["printed_end_label"]
        sections.append(
            BookSection(
                _string(item["section_id"], "section_id"),
                _string(item["exact_title"], "exact_title"),
                _string(item["normalized_title"], "normalized_title"),
                _integer(item["level"], "level"),
                None if parent is None else _string(parent, "parent_id"),
                _strings(item["children_ids"], "children_ids"),
                _integer(item["start_page_index"], "start_page_index"),
                _integer(item["end_page_index"], "end_page_index"),
                None if start_label is None else _string(start_label, "printed_start_label"),
                None if end_label is None else _string(end_label, "printed_end_label"),
                _strings(item["heading_block_ids"], "heading_block_ids"),
                StructureMethod(_string(item["method"], "method")),
                StructureConfidence(_string(item["confidence"], "confidence")),
                _strings(item["reason_codes"], "reason_codes"),
                _strings(item["warnings"], "warnings"),
            )
        )
    return BookStructure(
        _string(root["source_sha256"], "source_sha256"),
        _string(root["representation_sha256"], "representation_sha256"),
        _integer(root["profile_version"], "profile_version"),
        tuple(sections),
        _strings(root["root_ids"], "root_ids"),
        _strings(root["warnings"], "warnings"),
    )


def summary_task_to_json(value: SummaryTask) -> dict[str, object]:
    """Encode one bounded summary task."""
    return {
        "task_id": value.task_id,
        "section_id": value.section_id,
        "objective": value.objective,
        "evidence": [
            {
                "citation_id": item.citation_id,
                "page_index": item.page_index,
                "printed_page_label": item.printed_page_label,
                "block_type": item.block_type,
                "exact_text": item.exact_text,
                "exact_text_sha256": item.exact_text_sha256,
            }
            for item in value.evidence
        ],
        "allowed_citation_ids": list(value.allowed_citation_ids),
        "dependency_task_ids": list(value.dependency_task_ids),
        "output_contract_version": value.output_contract_version,
        "idempotency_key": value.idempotency_key,
    }


def structure_review_task_to_json(value: StructureReviewTask) -> dict[str, object]:
    """Encode one bounded low-confidence structure review task."""
    return {
        "task_id": value.task_id,
        "section_id": value.section_id,
        "candidate_heading_block_ids": list(value.candidate_heading_block_ids),
        "context_block_ids": list(value.context_block_ids),
        "allowed_parent_ids": list(value.allowed_parent_ids),
        "permitted_operations": list(value.permitted_operations),
    }


def structure_review_task_from_json(value: object) -> StructureReviewTask:
    """Decode one closed structure review task mapping."""
    root = _mapping(value, "structure review task")
    _fields(
        root,
        {
            "task_id",
            "section_id",
            "candidate_heading_block_ids",
            "context_block_ids",
            "allowed_parent_ids",
            "permitted_operations",
        },
    )
    return StructureReviewTask(
        _string(root["task_id"], "task_id"),
        _string(root["section_id"], "section_id"),
        _strings(root["candidate_heading_block_ids"], "candidate_heading_block_ids"),
        _strings(root["context_block_ids"], "context_block_ids"),
        _strings(root["allowed_parent_ids"], "allowed_parent_ids"),
        _strings(root["permitted_operations"], "permitted_operations"),
    )


def structure_review_submission_to_json(
    value: StructureReviewSubmission,
) -> dict[str, object]:
    """Encode one caller-owned bounded structure correction."""
    return {
        "task_id": value.task_id,
        "operation": value.operation,
        "selected_block_ids": list(value.selected_block_ids),
        "target_section_id": value.target_section_id,
    }


def structure_review_submission_from_json(value: object) -> StructureReviewSubmission:
    """Decode one closed structure correction mapping."""
    root = _mapping(value, "structure review submission")
    _fields(
        root,
        {"task_id", "operation", "selected_block_ids", "target_section_id"},
    )
    target = root["target_section_id"]
    return StructureReviewSubmission(
        _string(root["task_id"], "task_id"),
        _string(root["operation"], "operation"),
        _strings(root["selected_block_ids"], "selected_block_ids"),
        None if target is None else _string(target, "target_section_id"),
    )


def summary_task_from_json(value: object) -> SummaryTask:
    """Decode one closed summary task mapping."""
    root = _mapping(value, "summary task")
    _fields(
        root,
        {
            "task_id",
            "section_id",
            "objective",
            "evidence",
            "allowed_citation_ids",
            "dependency_task_ids",
            "output_contract_version",
            "idempotency_key",
        },
    )
    evidence: list[SummaryEvidence] = []
    for raw in _list(root["evidence"], "evidence"):
        item = _mapping(raw, "summary evidence")
        _fields(
            item,
            {
                "citation_id",
                "page_index",
                "printed_page_label",
                "block_type",
                "exact_text",
                "exact_text_sha256",
            },
        )
        label = item["printed_page_label"]
        evidence.append(
            SummaryEvidence(
                _string(item["citation_id"], "citation_id"),
                _integer(item["page_index"], "page_index"),
                None if label is None else _string(label, "printed_page_label"),
                _string(item["block_type"], "block_type"),
                _string(item["exact_text"], "exact_text", allow_empty=True),
                _string(item["exact_text_sha256"], "exact_text_sha256"),
            )
        )
    return SummaryTask(
        _string(root["task_id"], "task_id"),
        _string(root["section_id"], "section_id"),
        _string(root["objective"], "objective"),
        tuple(evidence),
        _strings(root["allowed_citation_ids"], "allowed_citation_ids"),
        _strings(root["dependency_task_ids"], "dependency_task_ids"),
        _integer(root["output_contract_version"], "output_contract_version"),
        _string(root["idempotency_key"], "idempotency_key"),
    )


def summary_submission_to_json(value: SummarySubmission) -> dict[str, object]:
    """Encode one caller-owned structured summary."""
    return {
        "task_id": value.task_id,
        "idempotency_key": value.idempotency_key,
        "abstract": value.abstract,
        "key_points": list(value.key_points),
        "claims": [
            {
                "claim_id": item.claim_id,
                "text": item.text,
                "citation_ids": list(item.citation_ids),
                "quotation": item.quotation,
            }
            for item in value.claims
        ],
        "provider": value.provider,
        "model": value.model,
    }


def summary_submission_from_json(value: object) -> SummarySubmission:
    """Decode one closed summary submission mapping."""
    root = _mapping(value, "summary submission")
    _fields(
        root,
        {
            "task_id",
            "idempotency_key",
            "abstract",
            "key_points",
            "claims",
            "provider",
            "model",
        },
    )
    claims: list[AtomicClaim] = []
    for raw in _list(root["claims"], "claims"):
        item = _mapping(raw, "atomic claim")
        _fields(item, {"claim_id", "text", "citation_ids", "quotation"})
        quotation = item["quotation"]
        claims.append(
            AtomicClaim(
                _string(item["claim_id"], "claim_id"),
                _string(item["text"], "text"),
                _strings(item["citation_ids"], "citation_ids"),
                None if quotation is None else _string(quotation, "quotation"),
            )
        )
    provider = root["provider"]
    model = root["model"]
    return SummarySubmission(
        _string(root["task_id"], "task_id"),
        _string(root["idempotency_key"], "idempotency_key"),
        _string(root["abstract"], "abstract"),
        _strings(root["key_points"], "key_points"),
        tuple(claims),
        None if provider is None else _string(provider, "provider"),
        None if model is None else _string(model, "model"),
    )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], dict(value))


def _fields(value: Mapping[str, object], expected: set[str]) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(_string(item, name) for item in _list(value, name))


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


__all__ = [
    "book_structure_from_json",
    "book_structure_to_json",
    "structure_review_submission_from_json",
    "structure_review_submission_to_json",
    "structure_review_task_from_json",
    "structure_review_task_to_json",
    "summary_submission_from_json",
    "summary_submission_to_json",
    "summary_task_from_json",
    "summary_task_to_json",
]
