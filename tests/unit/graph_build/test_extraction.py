"""Untrusted extraction results bind only to current source evidence."""

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    GraphScope,
    InputSegment,
    RelativeSpan,
    SourceSpan,
    TextUnit,
    ValidatedExtraction,
)
from capicove.contracts.retrieval import CharacterRange, TextChunkLocator, TextOrigin
from capicove.graph_build.extraction import (
    bind_relative_span,
    bind_segment_span,
    extraction_request,
    validate_extraction,
)
from capicove.graph_build.profiles import (
    ChunkingProfile,
    ExtractionLimits,
    GraphBuildProfile,
    InferencePolicy,
)


def unit(offset: int = 100) -> TextUnit:
    text = "Alice Python"
    return TextUnit(
        "unit",
        "document",
        GraphScope("workspace", "private"),
        "a" * 64,
        "representation",
        TextChunkLocator(TextOrigin.DOCUMENT, CharacterRange(offset, offset + len(text))),
        (
            InputSegment(
                "core",
                "core",
                text,
                (SourceSpan("source", "representation", offset, offset + len(text)),),
            ),
        ),
        (),
        0,
        12,
        "b" * 64,
    )


def span(start: int, end: int, quote: str, segment: str = "core") -> dict[str, Any]:
    return {"segment_id": segment, "start_char": start, "end_char": end, "quote": quote}


def response() -> dict[str, Any]:
    return {
        "entities": [
            {
                "local_id": "person",
                "name": "Alice",
                "entity_type": "PERSON",
                "description": "Named person",
                "source_spans": [span(0, 5, "Alice")],
                "anchor_role": "core",
                "extraction_confidence": 0.8,
                "identity_keys": [],
                "attribution_ref": None,
            },
            {
                "local_id": "skill",
                "name": "Python",
                "entity_type": "SKILL",
                "description": "Named skill",
                "source_spans": [span(6, 12, "Python")],
                "anchor_role": "core",
                "extraction_confidence": 0.7,
                "identity_keys": [],
                "attribution_ref": None,
            },
        ],
        "relationships": [
            {
                "local_id": "relation",
                "source_local_id": "person",
                "target_local_id": "skill",
                "predicate": "HAS_SKILL",
                "description": "Source-stated denial",
                "source_spans": [span(0, 12, "Alice Python")],
                "epistemic_status": "explicit",
                "polarity": "negative",
                "extraction_confidence": 0.6,
                "attribution_ref": "person",
                "qualifiers": [["time_text", "Monday"]],
            },
        ],
    }


def validate(data: dict[str, Any], current: TextUnit | None = None) -> ValidatedExtraction:
    payload = json.dumps(data).encode()
    return validate_extraction(
        payload, current or unit(), "personal-growth-v1", hashlib.sha256(payload).hexdigest()
    )


def build_profile() -> GraphBuildProfile:
    return GraphBuildProfile(
        profile_id="personal-growth-local",
        ontology_version="personal-growth-v1",
        chunking=ChunkingProfile(),
        inference=InferencePolicy(
            provider_id="local",
            model_id="qwen",
            model_revision="sha256:model",
            tokenizer_id="qwen-tokenizer",
            tokenizer_revision="sha256:tokenizer",
            input_role="user",
            context_window_tokens=8192,
        ),
        generation_profile_id="deterministic-json-v1",
        identity_embedding_space="bge-m3-v1",
        retrieval_embedding_space="bge-m3-v1",
    )


def test_cached_quote_rebinds_to_current_occurrence() -> None:
    quote = RelativeSpan("core", 0, 2, "学习")
    assert bind_relative_span("学习 Python", 100, quote) == (100, 102)
    with pytest.raises(ValueError):
        bind_relative_span("工作 Python", 100, quote)


@pytest.mark.parametrize("start", [-1, True, 0.5])
def test_source_start_must_be_a_nonnegative_codepoint_offset(start: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        bind_relative_span("abc", start, RelativeSpan("core", 0, 1, "a"))


def test_relative_binding_rejects_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        bind_relative_span("abc", 0, RelativeSpan("core", 3, 4, "x"))


def test_discontinuous_segment_binding_preserves_each_source_piece() -> None:
    segment = InputSegment(
        "context",
        "context",
        "Alice",
        (
            SourceSpan("first", "r1", 100, 102, "b1", 0),
            SourceSpan("second", "r2", 900, 903, "b2", 1),
        ),
    )
    bound = bind_segment_span(segment, RelativeSpan("context", 1, 4, "lic"))
    assert bound == (
        SourceSpan("first", "r1", 101, 102, "b1", 0),
        SourceSpan("second", "r2", 900, 902, "b2", 1),
    )
    with pytest.raises(ValueError):
        bind_segment_span(segment, RelativeSpan("other", 1, 4, "lic"))


def test_valid_negative_relation_rebinds_endpoints_and_attribution() -> None:
    result = validate(response())
    assert result.rejected_count == 0
    person, skill = result.entities
    relation = result.relationships[0]
    assert relation.source_mention_id == person.mention_id
    assert relation.target_mention_id == skill.mention_id
    assert relation.attribution_ref == person.mention_id
    assert relation.polarity == "negative"
    assert relation.qualifiers == (("time_text", "Monday"),)
    assert person.source_spans == (SourceSpan("source", "representation", 100, 105),)
    assert result == validate(response())


def test_same_local_ids_cannot_reuse_previous_document_coordinates_or_content() -> None:
    original = validate(response())
    moved = validate(response(), unit(300))
    changed = response()
    changed["entities"][0]["description"] = "Changed model interpretation"
    regenerated = validate(changed)
    assert moved.entities[0].source_spans[0].start_char == 300
    assert (
        len(
            {
                original.entities[0].mention_id,
                moved.entities[0].mention_id,
                regenerated.entities[0].mention_id,
            }
        )
        == 3
    )


def test_empty_result_is_successful_not_failed_extraction() -> None:
    result = validate({"entities": [], "relationships": []})
    assert result.entities == ()
    assert result.relationships == ()
    assert result.rejected_count == 0
    assert result.issue_codes == ()


def test_extraction_request_excludes_occurrence_identity_and_absolute_coordinates() -> None:
    first = extraction_request(unit(offset=100), build_profile())
    second = extraction_request(unit(offset=900), build_profile())
    assert first.request_fingerprint == second.request_fingerprint
    assert first.input_segments == (("core", "core", "Alice Python"),)
    assert "document" not in first.canonical_payload.decode("utf-8")
    assert "900" not in second.canonical_payload.decode("utf-8")


def test_extraction_request_remaps_occurrence_derived_segment_ids() -> None:
    first_unit = unit()
    first_unit = replace(
        first_unit,
        segments=(replace(first_unit.segments[0], segment_id="private-doc:core"),),
    )
    second_unit = replace(
        first_unit,
        segments=(replace(first_unit.segments[0], segment_id="other-doc:core"),),
    )
    first = extraction_request(first_unit, build_profile())
    second = extraction_request(second_unit, build_profile())
    assert first.request_fingerprint == second.request_fingerprint
    assert b"private-doc" not in first.canonical_payload


def test_validator_rebinds_request_local_segment_ids_to_original_source_map() -> None:
    current = unit()
    current = replace(
        current,
        segments=(replace(current.segments[0], segment_id="private-doc:core"),),
    )
    data = response()
    for entity in data["entities"]:
        entity["source_spans"][0]["segment_id"] = "core"
    data["relationships"][0]["source_spans"][0]["segment_id"] = "core"
    assert len(validate(data, current).relationships) == 1


def test_extraction_request_enforces_profile_request_byte_limit() -> None:
    current = build_profile()
    limits = replace(current.extraction_limits, max_request_bytes=100)
    with pytest.raises(ValueError, match="request_too_large"):
        extraction_request(unit(), replace(current, extraction_limits=limits))


@pytest.mark.parametrize(
    "family",
    [
        "profile_id",
        "embedding_spaces",
        "resolution",
        "community",
        "report",
        "diagnostic",
    ],
)
def test_extraction_request_ignores_other_stage_policy_families(family: str) -> None:
    current = build_profile()
    if family == "profile_id":
        changed = replace(current, profile_id="renamed-run-profile")
    elif family == "embedding_spaces":
        changed = replace(
            current,
            identity_embedding_space="identity-v2",
            retrieval_embedding_space="retrieval-v2",
        )
    elif family == "resolution":
        changed = replace(
            current,
            resolution=replace(current.resolution, top_k=current.resolution.top_k + 1),
        )
    elif family == "community":
        changed = replace(
            current,
            community=replace(current.community, max_size=current.community.max_size + 1),
        )
    elif family == "report":
        changed = replace(
            current,
            report_prompt_version="report-v2",
            report_schema_version="report-v2",
        )
    else:
        changed = replace(
            current,
            diagnostic_prompt_version="diagnostic-v2",
            diagnostic_schema_version="diagnostic-v2",
        )
    assert extraction_request(unit(), current) == extraction_request(unit(), changed)


@pytest.mark.parametrize(
    "family",
    ["chunking", "inference", "generation", "limits", "prompt", "schema", "validation"],
)
def test_extraction_request_changes_with_extraction_policy_families(family: str) -> None:
    current = build_profile()
    if family == "chunking":
        changed = replace(
            current,
            chunking=replace(current.chunking, target_tokens=current.chunking.target_tokens - 1),
        )
    elif family == "inference":
        changed = replace(
            current,
            inference=replace(current.inference, model_revision="sha256:new-model"),
        )
    elif family == "generation":
        changed = replace(current, generation_profile_id="deterministic-json-v2")
    elif family == "limits":
        changed = replace(
            current,
            extraction_limits=replace(current.extraction_limits, max_entities=63),
        )
    elif family == "prompt":
        changed = replace(current, extraction_prompt_version="extract-v2")
    elif family == "schema":
        changed = replace(current, extraction_schema_version="extraction-native-v1")
    else:
        changed = replace(current, extraction_validation_version=2)
    baseline = extraction_request(unit(), current)
    candidate = extraction_request(unit(), changed)
    assert candidate.profile_fingerprint != baseline.profile_fingerprint
    assert candidate.request_fingerprint != baseline.request_fingerprint


def test_structured_request_rejects_tampered_payload_or_exposed_fields() -> None:
    request = extraction_request(unit(), build_profile())
    with pytest.raises(ValueError):
        replace(request, canonical_payload=b'{"changed":true}')
    with pytest.raises(ValueError):
        replace(request, role="system")
    injected = json.loads(request.canonical_payload)
    injected["tool"] = "run"
    injected_payload = json.dumps(injected, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        replace(
            request,
            canonical_payload=injected_payload,
            request_fingerprint=hashlib.sha256(injected_payload).hexdigest(),
        )


def test_validator_applies_the_same_profile_record_limits() -> None:
    data = response()
    payload = json.dumps(data).encode()
    with pytest.raises(ValueError, match="invalid_array"):
        validate_extraction(
            payload,
            unit(),
            "personal-growth-v1",
            hashlib.sha256(payload).hexdigest(),
            ExtractionLimits(max_entities=1),
        )


@pytest.mark.parametrize("change", ["heading", "validation", "prompt", "schema", "model"])
def test_request_fingerprint_covers_every_semantic_input(change: str) -> None:
    current_unit = unit()
    current_profile = build_profile()
    baseline = extraction_request(current_unit, current_profile).request_fingerprint
    if change == "heading":
        candidate = extraction_request(
            replace(current_unit, section_path=("New heading",)), current_profile
        )
    elif change == "validation":
        candidate = extraction_request(
            current_unit, replace(current_profile, extraction_validation_version=2)
        )
    elif change == "prompt":
        candidate = extraction_request(
            current_unit, replace(current_profile, extraction_prompt_version="extract-v2")
        )
    elif change == "schema":
        candidate = extraction_request(
            current_unit, replace(current_profile, extraction_schema_version="extraction-native-v1")
        )
    else:
        candidate = extraction_request(
            current_unit,
            replace(
                current_profile,
                inference=replace(current_profile.inference, model_revision="sha256:new"),
            ),
        )
    assert candidate.request_fingerprint != baseline


@pytest.mark.parametrize(
    "field,value",
    [
        ("entity_type", "UNKNOWN"),
        ("name", ""),
        ("name", "x" * 4097),
        ("extraction_confidence", True),
        ("extraction_confidence", 10**400),
        ("name", "\ud800"),
        ("tool", "run"),
        ("identity_keys", [{"namespace": "id", "value": "claimed", "authoritative": True}]),
        ("source_spans", [span(0, 5, "Wrong")]),
        ("source_spans", [span(0, 5, "Alice", "unknown")]),
        ("attribution_ref", "external-person"),
    ],
)
def test_bad_entity_rejects_dependent_relations_without_dropping_valid_entities(
    field: str, value: Any
) -> None:
    data = response()
    data["entities"][0][field] = value
    result = validate(data)
    assert [e.name for e in result.entities] == ["Python"]
    assert result.relationships == ()
    assert result.rejected_count == 2
    assert result.issue_codes


@pytest.mark.parametrize(
    "field,value",
    [
        ("predicate", "ABOUT_CONCEPT"),
        ("target_local_id", "missing"),
        ("polarity", "neutral"),
        ("extraction_confidence", 10**400),
        ("description", "\ud800"),
        ("qualifiers", [["time_text", "\ud800"]]),
        ("attribution_ref", "skill"),
        ("qualifiers", [["tool", "run"]]),
        ("qualifiers", [["time_text", "one"], ["time_text", "two"]]),
    ],
)
def test_invalid_relation_is_not_accepted_as_supported_fact(field: str, value: Any) -> None:
    data = response()
    data["relationships"][0][field] = value
    result = validate(data)
    assert len(result.entities) == 2
    assert result.relationships == ()
    assert result.rejected_count == 1


def test_context_anchor_needs_a_core_supported_relation() -> None:
    current = unit()
    current = replace(
        current,
        segments=(
            InputSegment(
                "context", "context", "Alice", (SourceSpan("context-source", "r", 900, 905),)
            ),
            *current.segments,
        ),
    )
    data = response()
    data["entities"][0]["source_spans"] = [span(0, 5, "Alice", "context:0")]
    data["entities"][0]["anchor_role"] = "context_anchor"
    assert len(validate(data, current).entities) == 2
    data["relationships"][0]["source_spans"] = [span(0, 5, "Alice", "context:0")]
    result = validate(data, current)
    assert [e.name for e in result.entities] == ["Python"]
    assert result.rejected_count == 2


def test_core_claim_cannot_mislabel_context_as_core() -> None:
    current = unit()
    current = replace(
        current,
        segments=(
            InputSegment("context", "context", "Alice", (SourceSpan("e", "r", 0, 5),)),
            *current.segments,
        ),
    )
    data = response()
    data["entities"][0]["source_spans"] = [span(0, 5, "Alice", "context:0")]
    assert validate(data, current).rejected_count == 2


def test_duplicate_local_ids_are_rejected_instead_of_choosing_an_endpoint() -> None:
    data = response()
    data["entities"].append(copy.deepcopy(data["entities"][0]))
    with pytest.raises(ValueError):
        validate(data)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"entities":',
        b'{"entities":[],"entities":[],"relationships":[]}',
        b'{"entities":[],"relationships":[],"tools":[]}',
        b'{"entities":[],"relationships":[],"x":NaN}',
        b"\xff",
        b" " * (262144 + 1),
    ],
)
def test_malformed_or_oversized_response_is_not_empty_success(payload: bytes) -> None:
    with pytest.raises(ValueError):
        validate_extraction(
            payload, unit(), "personal-growth-v1", hashlib.sha256(payload).hexdigest()
        )


def test_counts_and_result_digest_are_verified_before_acceptance() -> None:
    data = response()
    data["entities"] *= 33
    with pytest.raises(ValueError):
        validate(data)
    payload = b'{"entities":[],"relationships":[]}'
    with pytest.raises(ValueError):
        validate_extraction(payload, unit(), "personal-growth-v1", "c" * 64)
    with pytest.raises(ValueError):
        validate_extraction(payload, unit(), "other", hashlib.sha256(payload).hexdigest())


def test_native_prompt_uses_exact_quotes_and_closed_general_types() -> None:
    from dataclasses import replace

    request = extraction_request(
        unit(),
        replace(
            build_profile(),
            extraction_prompt_version="extract-native-v1",
            extraction_schema_version="extraction-native-v1",
        ),
    )
    assert "CONCEPT" in request.prompt_text and "RELATED_TO" in request.prompt_text
    assert "Do not calculate character offsets" in request.prompt_text
    assert json.loads(request.canonical_payload)["prompt"] == request.prompt_text
