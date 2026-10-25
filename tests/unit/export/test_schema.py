"""Tests for the packaged version 1 Agent trace JSON Schema."""

from __future__ import annotations

import json
from importlib import resources
from typing import cast

import pytest
from jsonschema import Draft202012Validator, ValidationError  # type: ignore[import-untyped]

from capicove.traces.codec import encode_trace_record
from tests.unit.traces.test_codec import make_record


def _schema() -> dict[str, object]:
    return cast(
        dict[str, object],
        json.loads(
            resources.files("capicove.export.schemas")
            .joinpath("capicove-agent-trace-v1.schema.json")
            .read_text(encoding="utf-8")
        ),
    )


def test_packaged_schema_is_valid_and_accepts_a_canonical_record() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(encode_trace_record(make_record()))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"occurred_at": "2026-08-18T12:30:04"}),
        lambda value: value.update({"record_type": "unknown"}),
    ],
)
def test_schema_rejects_version_one_contract_violations(mutate: object) -> None:
    value = encode_trace_record(make_record())
    mutate(value)  # type: ignore[operator]

    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(value)
