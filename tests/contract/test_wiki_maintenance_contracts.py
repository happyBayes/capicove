"""Closed-wire and immutable-value tests for maintenance contracts."""

import importlib
import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest


def codec() -> Any:
    return importlib.import_module("capicove.wiki.maintenance.codec")


def span_payload() -> Any:
    return {
        "kind": "markdown_span",
        "source_id": "source:sha256:" + "a" * 64,
        "source_sha256": "a" * 64,
        "representation_sha256": "b" * 64,
        "start_char": 0,
        "end_char": 2,
        "exact_sha256": "c" * 64,
        "page_index": None,
        "block_id": None,
    }


def envelope(payload: Any = None) -> Any:
    return {
        "schema_version": 1,
        "record_type": "evidence_span",
        "payload": span_payload() if payload is None else payload,
    }


def test_maintenance_codec_is_publicly_available() -> None:
    import capicove.wiki as wiki

    assert hasattr(wiki, "decode_maintenance_record")


def test_exact_canonical_encoding_and_roundtrip() -> None:
    value = envelope()
    raw = json.dumps(value).encode()
    record = codec().decode_record(raw)
    expected = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert codec().encode_record(record) == expected
    assert codec().decode_record(expected) == record
    with pytest.raises(FrozenInstanceError):
        record.payload.end_char = 9


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"unexpected": 1},
        {"record_type": "unknown"},
    ],
)
def test_rejects_invalid_envelope(change: Any) -> None:
    value = envelope()
    value.update(change)
    with pytest.raises(ValueError):
        codec().decode_record(json.dumps(value).encode())


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "unknown"},
        {"end_char": 0},
        {"start_char": True},
        {"start_char": -1},
        {"page_index": 0},
        {"source_sha256": "z" * 64},
        {"source_id": "source:sha256:" + "d" * 64},
        {"extra": "ignored"},
        {"block_id": "unexpected"},
    ],
)
def test_rejects_invalid_evidence(change: Any) -> None:
    value = span_payload()
    value.update(change)
    with pytest.raises(ValueError):
        codec().decode_record(json.dumps(envelope(value)).encode())


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"x":NaN}',
        b'{"x":1.5}',
        b'{"x":Infinity}',
        b"\xff",
        b" " * (1048576 + 1),
        b"[]",
        b'{"x":"\\ud800"}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_rejects_malformed_or_unbounded_json(raw: Any) -> None:
    with pytest.raises(ValueError):
        codec().decode_record(raw)


def test_hash_ignores_object_key_order() -> None:
    one = codec().decode_record(json.dumps(envelope()).encode())
    two = codec().decode_record(json.dumps(envelope(), sort_keys=True).encode())
    assert codec().record_digest(one) == codec().record_digest(two)


def test_direct_construction_cannot_bypass_validation() -> None:
    contracts = importlib.import_module("capicove.contracts.wiki_maintenance")
    with pytest.raises(ValueError):
        contracts.EvidenceSpan(**{**span_payload(), "end_char": True})
    assert contracts.RECORD_TYPES.keys() >= {
        "evidence_span",
        "managed_claim",
        "page_revision",
        "compilation_task",
        "compilation_submission",
        "proposal",
        "review",
        "publication",
        "source_policy_event",
        "answer_task",
        "answer_submission",
        "answer_result",
    }


def records() -> Any:
    h = "a" * 64
    task_id = "task:" + h
    claim = {
        "claim_id": "claim:" + h,
        "text": "A supported claim.",
        "evidence": [span_payload()],
        "status": "current",
        "knowledge": "asserted",
        "premise_ids": [],
        "derivation": None,
    }
    page = {
        "page_id": "page:ABCDEFGHIJ",
        "page_path": "pages/concepts/example.md",
        "parent_sha256": None,
        "title": "Example",
        "kind": "concept",
        "aliases": ["example"],
        "sections": [{"heading": "Evidence", "claim_ids": [claim["claim_id"]]}],
        "claims": [claim],
        "relations": [],
        "source_ids": [span_payload()["source_id"]],
        "rendering_version": 1,
    }
    alias = {"alias": "e1", "span": span_payload(), "text": "AB"}
    source = {
        key: span_payload()[key] for key in ("source_id", "source_sha256", "representation_sha256")
    }
    task = {
        "task_id": task_id,
        "task_version": 1,
        "workspace_id": "test",
        "wiki_revision": 0,
        "policy_revision": 0,
        "sources": [source],
        "evidence": [alias],
        "candidates": [],
        "permitted_page_ids": [],
        "discovery_policy": "lexical-v1",
        "coverage": "complete_inventory",
        "max_output_bytes": 1048576,
        "previous_task_id": None,
        "affected_page_ids": [],
        "deferred_page_count": 0,
        "deferred_page_sha256": None,
    }
    submission = {
        "task_id": task_id,
        "task_version": 1,
        "idempotency_key": "one",
        "dispositions": ["new"],
        "rationale": "New evidence.",
        "pages": [page],
        "impact": [],
        "provenance": None,
    }
    draft_claim = {
        "draft_id": "claim-one",
        "text": "A supported claim.",
        "evidence_aliases": ["e1"],
        "status": "current",
        "knowledge": "asserted",
        "premise_ids": [],
        "derivation": None,
    }
    draft_page = {
        "draft_id": "page-one",
        "operation": "create",
        "page_id": None,
        "title": "Example",
        "kind": "concept",
        "aliases": ["example"],
        "sections": [{"heading": "Evidence", "claim_ids": ["claim-one"]}],
        "claims": [draft_claim],
        "relations": [],
    }
    submission["pages"] = [draft_page]
    answer_claim = {
        "text": "A supported claim.",
        "citations": [{"alias": "e1", "quote": "AB"}],
        "status": "current",
    }
    return {
        "evidence_span": span_payload(),
        "claim_draft": draft_claim,
        "page_draft": draft_page,
        "managed_claim": claim,
        "page_revision": page,
        "compilation_task": task,
        "compilation_submission": submission,
        "proposal": {
            "proposal_id": "proposal:" + h,
            "workspace_id": "test",
            "submission": submission,
            "wiki_revision": 0,
            "policy_revision": 0,
            "rendered_sha256": h,
        },
        "review": {
            "review_id": "review:" + h,
            "proposal_id": "proposal:" + h,
            "proposal_sha256": h,
            "rendered_sha256": h,
            "wiki_revision": 0,
            "policy_revision": 0,
            "reviewer_id": "operator",
            "decisions": [
                {"subject_id": claim["claim_id"], "verdict": "supported", "reason": "Read source."}
            ],
            "accepted": True,
        },
        "publication": {
            "publication_id": "publication:" + h,
            "parent_id": None,
            "proposal_id": "proposal:" + h,
            "review_id": "review:" + h,
            "wiki_revision": 1,
            "policy_revision": 0,
            "changes": [
                {
                    "relative_path": "pages/concepts/example.md",
                    "before_sha256": None,
                    "after_sha256": h,
                }
            ],
            "state_sha256": h,
            "status": "complete",
            "origin": "reviewed",
        },
        "source_policy_event": {
            "event_id": "policy:" + h,
            "workspace_id": "test",
            "source_id": source["source_id"],
            "policy_revision": 1,
            "reason": "Withdraw source.",
            "idempotency_key": "withdraw-one",
            "action": "withdraw",
        },
        "answer_task": {
            "task_id": task_id,
            "workspace_id": "test",
            "wiki_revision": 0,
            "policy_revision": 0,
            "question": "What is supported?",
            "evidence": [alias],
            "claims": [claim],
            "include_history": False,
        },
        "answer_submission": {"task_id": task_id, "claims": [answer_claim]},
        "answer_result": {
            "task_id": task_id,
            "wiki_revision": 0,
            "policy_revision": 0,
            "status": "answered",
            "claims": [answer_claim],
            "semantic_status": "not_verified",
            "omitted_claims": 0,
            "evidence": [alias],
        },
    }


@pytest.mark.parametrize("name", list(records()))
def test_every_record_roundtrips_and_rejects_extra_fields(name: Any) -> None:
    payload = records()[name]
    value = {"schema_version": 1, "record_type": name, "payload": payload}
    record = codec().decode_record(json.dumps(value).encode())
    assert codec().decode_record(codec().encode_record(record)) == record
    payload["permission"] = "admin"
    with pytest.raises(ValueError):
        codec().decode_record(json.dumps(value).encode())


def test_citation_order_changes_digest() -> None:
    payload = records()["managed_claim"]
    other = {**span_payload(), "start_char": 1, "end_char": 3}
    payload["evidence"].append(other)
    value = {"schema_version": 1, "record_type": "managed_claim", "payload": payload}
    one = codec().record_digest(codec().decode_record(json.dumps(value).encode()))
    payload["evidence"].reverse()
    two = codec().record_digest(codec().decode_record(json.dumps(value).encode()))
    assert one != two


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("page_revision", "aliases", [""]),
        ("page_revision", "aliases", ["x" * 2049]),
        ("compilation_task", "evidence", []),
        ("compilation_submission", "pages", []),
        ("managed_claim", "premise_ids", ["not-an-id"]),
    ],
)
def test_nested_limits_and_required_compilation_content(name: Any, field: Any, value: Any) -> None:
    payload = records()[name]
    payload[field] = value
    with pytest.raises(ValueError):
        codec().decode_record(
            json.dumps({"schema_version": 1, "record_type": name, "payload": payload}).encode()
        )


def test_ports_cannot_return_approval_objects() -> None:
    ports = importlib.import_module("capicove.wiki.maintenance.ports")
    from typing import get_type_hints

    assert get_type_hints(ports.CompilationGenerator.generate)["return"] is bytes
    assert get_type_hints(ports.AnswerGenerator.generate)["return"] is bytes


def test_submission_rejects_host_owned_page_paths() -> None:
    submission = records()["compilation_submission"]
    submission["pages"] = [records()["page_revision"]]
    with pytest.raises(ValueError):
        codec().decode_record(
            json.dumps(
                {
                    "schema_version": 1,
                    "record_type": "compilation_submission",
                    "payload": submission,
                }
            ).encode()
        )


def test_task_requires_explicit_permitted_page_ids() -> None:
    task = records()["compilation_task"]
    task.pop("permitted_page_ids")
    with pytest.raises(ValueError):
        codec().decode_record(
            json.dumps(
                {"schema_version": 1, "record_type": "compilation_task", "payload": task}
            ).encode()
        )
