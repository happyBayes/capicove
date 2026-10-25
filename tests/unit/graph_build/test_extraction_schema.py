"""Exercise the packaged model-response schema with authored local spans."""

import copy
import json
from importlib.resources import files
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import pytest

from capicove.contracts.graph_build import RelativeSpan
from capicove.graph_build.codec import graph_from_json, graph_to_json


def response() -> dict[str, Any]:
    return {
        "entities": [
            {
                "local_id": "e1",
                "name": "学习",
                "entity_type": "SKILL",
                "description": "A source-mentioned skill",
                "anchor_role": "core",
                "source_spans": [
                    {"segment_id": "core", "start_char": 0, "end_char": 2, "quote": "学习"}
                ],
                "extraction_confidence": 0.8,
                "attribution_ref": None,
                "identity_keys": [],
            }
        ],
        "relationships": [],
    }


def validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(
        files("capicove.graph_build").joinpath("schemas/extraction-v1.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_model_schema_and_relative_span_contract_agree() -> None:
    payload = response()
    validator().validate(payload)
    span = RelativeSpan(**payload["entities"][0]["source_spans"][0])
    assert graph_from_json(graph_to_json(span)) == span
    validator().validate({"entities": [], "relationships": []})


@pytest.mark.parametrize(
    "field,value",
    [
        ("entity_type", "UNKNOWN"),
        ("extraction_confidence", True),
        ("extraction_confidence", 1.1),
        ("tool", "execute"),
        ("identity_keys", [{"namespace": "model", "value": "claimed", "authoritative": True}]),
    ],
)
def test_model_schema_rejects_open_fields_and_authority_escalation(field: str, value: Any) -> None:
    payload = response()
    payload["entities"][0][field] = value
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(payload)


def test_model_schema_rejects_absolute_coordinates_and_excess_records() -> None:
    payload = response()
    payload["entities"][0]["source_spans"][0]["evidence_ref"] = "old-document"
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(payload)
    payload = response()
    payload["entities"] *= 65
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(payload)


def test_negative_relationship_preserves_attribution_and_closed_qualifiers() -> None:
    payload = response()
    payload["relationships"] = [
        {
            "local_id": "r1",
            "source_local_id": "e1",
            "target_local_id": "e1",
            "predicate": "RELATED_TO",
            "description": "Source-stated denial",
            "source_spans": copy.deepcopy(payload["entities"][0]["source_spans"]),
            "epistemic_status": "explicit",
            "polarity": "negative",
            "extraction_confidence": 0.7,
            "attribution_ref": "speaker",
            "qualifiers": [["time_text", "Monday"]],
        }
    ]
    validator().validate(payload)
    payload["relationships"][0]["qualifiers"] = [["command", "run"]]
    with pytest.raises(jsonschema.ValidationError):
        validator().validate(payload)
