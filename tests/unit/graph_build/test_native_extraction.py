"""Native extraction derives coordinates only from unique exact source quotes."""

import hashlib
import json
from typing import Any

import pytest

from capicove.graph_build.extraction import validate_extraction
from tests.unit.graph_build.test_extraction import unit


def validate(payload: dict[str, Any]):  # type: ignore[no-untyped-def]
    raw = json.dumps(payload).encode()
    return validate_extraction(
        raw,
        unit(),
        "personal-growth-v1",
        hashlib.sha256(raw).hexdigest(),
        schema_version="extraction-native-v1",
    )


def test_native_quote_binding_preserves_original_response_identity() -> None:
    text = next(segment.text for segment in unit().segments if segment.role == "core")
    payload = {
        "entities": [
            {
                "name": "topic",
                "entity_type": "CONCEPT",
                "description": "A cited topic",
                "quote": text,
            }
        ],
        "relationships": [],
    }
    result = validate(payload)
    assert len(result.entities) == 1 and result.rejected_count == 0
    assert result.result_fingerprint == hashlib.sha256(json.dumps(payload).encode()).hexdigest()


@pytest.mark.parametrize("quote", ["invented text", "", 123])
def test_native_binding_never_repairs_or_invents_quotes(quote: object) -> None:
    result = validate(
        {
            "entities": [
                {
                    "name": "topic",
                    "entity_type": "CONCEPT",
                    "description": "A cited topic",
                    "quote": quote,
                }
            ],
            "relationships": [],
        }
    )
    assert not result.entities and result.rejected_count == 1


def test_unknown_schema_fails_closed() -> None:
    from dataclasses import replace

    from capicove.graph_build.extraction import extraction_request
    from tests.unit.graph_build.test_extraction import build_profile

    with pytest.raises(ValueError, match="unsupported_extraction_schema"):
        extraction_request(unit(), replace(build_profile(), extraction_schema_version="unknown"))
    raw = b'{"entities":[],"relationships":[]}'
    with pytest.raises(ValueError, match="unsupported_extraction_schema"):
        validate_extraction(
            raw,
            unit(),
            "personal-growth-v1",
            hashlib.sha256(raw).hexdigest(),
            schema_version="unknown",
        )


def test_native_relationship_can_declare_only_literally_quoted_endpoints() -> None:
    raw = {
        "entities": [],
        "relationships": [
            {
                "source": "Alice",
                "target": "Python",
                "description": "The source mentions Alice and Python.",
                "quote": "Alice Python",
                "polarity": "positive",
            }
        ],
    }
    result = validate(raw)
    assert len(result.entities) == 2 and len(result.relationships) == 1
    raw["relationships"][0]["target"] = "Secret"
    denied = validate(raw)
    assert not denied.relationships
    assert all(row.name != "Secret" for row in denied.entities)
