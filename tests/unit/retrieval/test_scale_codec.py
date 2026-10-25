"""Strict wire-contract tests for personal scale precheck evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from typing import cast

import pytest

import capicove.retrieval.scale_codec as scale_codec
from capicove.retrieval.scale import (
    PersonalScalePrecheckVerdict,
    evaluate_personal_scale_precheck,
    report_sha256,
)
from capicove.retrieval.scale_codec import (
    MAX_PERSONAL_SCALE_REPORT_BYTES,
    MAX_PERSONAL_SCALE_VERDICT_BYTES,
    decode_personal_scale_precheck_verdict,
    decode_personal_scale_report,
    encode_personal_scale_precheck_verdict,
    encode_personal_scale_report,
    verdict_sha256,
)
from tests.unit.retrieval.test_scale import NOW, _bindings, _report


def _canonical_bytes(document: object) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _document(payload: bytes) -> dict[str, object]:
    return cast(dict[str, object], json.loads(payload))


def _verdict() -> PersonalScalePrecheckVerdict:
    bindings = _bindings()
    return evaluate_personal_scale_precheck(_report(bindings), bindings, evaluated_at=NOW)


def test_report_and_verdict_round_trip_byte_for_byte() -> None:
    report = _report(_bindings())
    verdict = _verdict()

    report_payload = encode_personal_scale_report(report)
    verdict_payload = encode_personal_scale_precheck_verdict(verdict)

    assert report_payload.endswith(b"\n")
    assert not report_payload.endswith(b"\n\n")
    assert verdict_payload.endswith(b"\n")
    assert encode_personal_scale_report(decode_personal_scale_report(report_payload)) == (
        report_payload
    )
    assert (
        encode_personal_scale_precheck_verdict(
            decode_personal_scale_precheck_verdict(verdict_payload)
        )
        == verdict_payload
    )


def test_report_encoder_normalizes_allowed_integral_float_values() -> None:
    report = _report(_bindings())
    first_build_stage = replace(report.build_stages[0], elapsed_seconds=2)
    integral_report = replace(
        report,
        cold_start_ms=1,
        build_stages=(first_build_stage, *report.build_stages[1:]),
    )

    payload = encode_personal_scale_report(integral_report)

    assert b'"cold_start_ms":1.0' in payload
    assert b'"elapsed_seconds":2.0' in payload
    assert encode_personal_scale_report(decode_personal_scale_report(payload)) == payload


def test_report_uses_real_personal_semantic_index_shape() -> None:
    document = _document(encode_personal_scale_report(_report(_bindings())))

    assert "ann_policy" not in document
    raw_policy = cast(list[object], document["semantic_index_policy"])
    first = cast(dict[str, object], raw_policy[0])
    assert first == {
        "exact_candidate_depth": 10_000,
        "index_format": "numpy-flat-f32-v1",
        "index_parameters_sha256": first["index_parameters_sha256"],
        "lane": "instance",
    }


def test_report_and_verdict_hash_exact_canonical_bytes() -> None:
    report = _report(_bindings())
    verdict = _verdict()

    assert report_sha256(report) == hashlib.sha256(encode_personal_scale_report(report)).hexdigest()
    assert (
        verdict_sha256(verdict)
        == hashlib.sha256(encode_personal_scale_precheck_verdict(verdict)).hexdigest()
    )


def test_report_and_verdict_digests_drift_with_evidence() -> None:
    report = _report(_bindings())
    verdict = _verdict()

    changed_report = replace(report, report_id=f"{report.report_id}-tampered")
    changed_verdict = replace(
        verdict, evaluated_at=verdict.evaluated_at + timedelta(microseconds=1)
    )

    assert report_sha256(changed_report) != report_sha256(report)
    assert verdict_sha256(changed_verdict) != verdict_sha256(verdict)


@pytest.mark.parametrize("field", ("report_id", "schema_version"))
def test_report_decoder_rejects_unknown_and_missing_fields(field: str) -> None:
    document = _document(encode_personal_scale_report(_report(_bindings())))
    if field == "report_id":
        document["unexpected"] = document.pop(field)
        match = "unknown personal scale report fields"
    else:
        del document[field]
        match = "missing personal scale report fields"

    with pytest.raises(ValueError, match=match):
        decode_personal_scale_report(_canonical_bytes(document))


def test_report_decoder_rejects_unknown_nested_field() -> None:
    document = _document(encode_personal_scale_report(_report(_bindings())))
    hardware = cast(dict[str, object], document["hardware"])
    hardware["unexpected"] = "value"

    with pytest.raises(ValueError, match="unknown hardware observation fields"):
        decode_personal_scale_report(_canonical_bytes(document))


def test_decoders_reject_duplicate_fields() -> None:
    report_payload = encode_personal_scale_report(_report(_bindings()))
    verdict_payload = encode_personal_scale_precheck_verdict(_verdict())
    duplicate_report = report_payload.replace(
        b'{"activation_set_id":',
        b'{"activation_set_id":"duplicate","activation_set_id":',
        1,
    )
    duplicate_verdict = verdict_payload.replace(
        b'{"evaluated_at":',
        b'{"evaluated_at":"duplicate","evaluated_at":',
        1,
    )

    with pytest.raises(ValueError, match="duplicate JSON field"):
        decode_personal_scale_report(duplicate_report)
    with pytest.raises(ValueError, match="duplicate JSON field"):
        decode_personal_scale_precheck_verdict(duplicate_verdict)


@pytest.mark.parametrize("variant", ("no-newline", "leading-space", "extra-newline"))
def test_report_decoder_rejects_noncanonical_whitespace(variant: str) -> None:
    payload = encode_personal_scale_report(_report(_bindings()))
    changed = {
        "no-newline": payload[:-1],
        "leading-space": b" " + payload,
        "extra-newline": payload + b"\n",
    }[variant]

    with pytest.raises(ValueError, match="canonical JSON encoding"):
        decode_personal_scale_report(changed)


def test_verdict_decoder_rejects_noncanonical_key_order() -> None:
    payload = encode_personal_scale_precheck_verdict(_verdict())
    document = _document(payload)
    reversed_document = dict(reversed(tuple(document.items())))
    noncanonical = (json.dumps(reversed_document, separators=(",", ":")) + "\n").encode()

    with pytest.raises(ValueError, match="canonical JSON encoding"):
        decode_personal_scale_precheck_verdict(noncanonical)


def test_report_decoder_rejects_nonfinite_numbers() -> None:
    payload = encode_personal_scale_report(_report(_bindings()))
    nonfinite = payload.replace(b'"bytes_per_vector":1200.0', b'"bytes_per_vector":NaN', 1)

    assert nonfinite != payload
    with pytest.raises(ValueError, match="non-finite JSON number"):
        decode_personal_scale_report(nonfinite)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("schema_version", 2, "schema_version must be 1"),
        ("unit_count", "10000", "unit_count must be an integer"),
    ),
)
def test_report_decoder_rejects_wrong_schema_and_type(
    field: str,
    value: object,
    match: str,
) -> None:
    document = _document(encode_personal_scale_report(_report(_bindings())))
    document[field] = value

    with pytest.raises(ValueError, match=match):
        decode_personal_scale_report(_canonical_bytes(document))


def test_report_decoder_rejects_wrong_enum() -> None:
    document = _document(encode_personal_scale_report(_report(_bindings())))
    policy = cast(list[dict[str, object]], document["semantic_index_policy"])
    policy[0]["lane"] = "balanced"

    with pytest.raises(ValueError, match="valid RetrievalLane"):
        decode_personal_scale_report(_canonical_bytes(document))


def test_report_decoder_rejects_boolean_impersonating_integer() -> None:
    document = _document(encode_personal_scale_report(_report(_bindings())))
    document["unit_count"] = True

    with pytest.raises(ValueError, match="unit_count must be an integer"):
        decode_personal_scale_report(_canonical_bytes(document))


def test_verdict_decoder_rejects_unknown_missing_schema_and_type() -> None:
    payload = encode_personal_scale_precheck_verdict(_verdict())
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({**_document(payload), "unexpected": 1}, "unknown personal scale"),
        (
            {key: value for key, value in _document(payload).items() if key != "tier"},
            "missing personal scale",
        ),
        ({**_document(payload), "schema_version": 2}, "schema_version must be 1"),
        ({**_document(payload), "passed": 1}, "passed must be a boolean"),
    )

    for document, match in cases:
        with pytest.raises(ValueError, match=match):
            decode_personal_scale_precheck_verdict(_canonical_bytes(document))


def test_decoders_reject_oversized_documents_before_parsing() -> None:
    with pytest.raises(ValueError, match=f"exceeds {MAX_PERSONAL_SCALE_REPORT_BYTES} bytes"):
        decode_personal_scale_report(b" " * (MAX_PERSONAL_SCALE_REPORT_BYTES + 1))
    with pytest.raises(ValueError, match=f"exceeds {MAX_PERSONAL_SCALE_VERDICT_BYTES} bytes"):
        decode_personal_scale_precheck_verdict(b" " * (MAX_PERSONAL_SCALE_VERDICT_BYTES + 1))


def test_report_encoder_enforces_boundary_without_large_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(_bindings())
    payload = encode_personal_scale_report(report)
    monkeypatch.setattr(scale_codec, "MAX_PERSONAL_SCALE_REPORT_BYTES", len(payload))

    assert encode_personal_scale_report(report) == payload

    monkeypatch.setattr(scale_codec, "MAX_PERSONAL_SCALE_REPORT_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="personal scale report exceeds"):
        encode_personal_scale_report(report)


def test_verdict_encoder_enforces_boundary_without_large_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verdict = _verdict()
    payload = encode_personal_scale_precheck_verdict(verdict)
    monkeypatch.setattr(scale_codec, "MAX_PERSONAL_SCALE_VERDICT_BYTES", len(payload))

    assert encode_personal_scale_precheck_verdict(verdict) == payload

    monkeypatch.setattr(scale_codec, "MAX_PERSONAL_SCALE_VERDICT_BYTES", len(payload) - 1)
    with pytest.raises(ValueError, match="personal scale precheck verdict exceeds"):
        encode_personal_scale_precheck_verdict(verdict)
