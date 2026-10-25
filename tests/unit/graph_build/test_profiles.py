"""Behavioral tests for closed graph-build and inference policies."""

import json
from dataclasses import replace
from typing import Any

import pytest

import capicove.graph_build.profiles as policies
from capicove.graph_build.profiles import (
    ChunkingProfile,
    GraphBuildProfile,
    InferencePolicy,
    graph_build_profile_fingerprint,
    graph_build_profile_from_json,
    graph_build_profile_to_json,
)


def profile() -> GraphBuildProfile:
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


def test_profile_fingerprint_changes_with_semantic_inference_policy() -> None:
    current = profile()
    changed = profile()
    changed = GraphBuildProfile(
        **{
            field: getattr(changed, field)
            for field in changed.__dataclass_fields__
            if field != "inference"
        },
        inference=InferencePolicy(
            provider_id="local",
            model_id="qwen",
            model_revision="sha256:new-model",
            tokenizer_id="qwen-tokenizer",
            tokenizer_revision="sha256:tokenizer",
            input_role="user",
            context_window_tokens=8192,
        ),
    )
    assert graph_build_profile_fingerprint(current) != graph_build_profile_fingerprint(changed)


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
def test_extraction_profile_fingerprint_ignores_other_stage_policy_families(
    family: str,
) -> None:
    current = profile()
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
    assert policies.extraction_profile_fingerprint(current) == (
        policies.extraction_profile_fingerprint(changed)
    )


@pytest.mark.parametrize(
    "family",
    [
        "chunking",
        "ontology",
        "inference",
        "generation",
        "limits",
        "prompt",
        "schema",
        "validation",
    ],
)
def test_extraction_profile_fingerprint_covers_extraction_inputs(family: str) -> None:
    current = profile()
    if family == "chunking":
        changed = replace(
            current,
            chunking=replace(current.chunking, target_tokens=current.chunking.target_tokens - 1),
        )
    elif family == "ontology":
        changed = replace(current, ontology_version="personal-growth-v2")
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
        changed = replace(current, extraction_schema_version="extraction-v2")
    else:
        changed = replace(current, extraction_validation_version=2)
    assert policies.extraction_profile_fingerprint(current) != (
        policies.extraction_profile_fingerprint(changed)
    )


def test_graph_build_profile_round_trips_through_closed_json() -> None:
    current = profile()
    assert graph_build_profile_from_json(graph_build_profile_to_json(current)) == current


def test_community_profile_uses_closed_nested_louvain_parameters() -> None:
    payload = json.loads(graph_build_profile_to_json(profile()))
    assert payload["community"] == {
        "max_size": 50,
        "max_depth": 3,
        "support_cap": 20,
        "version": 2,
        "partition_profile": {
            "tag": "louvain",
            "resolution": 1.0,
            "threshold": 1e-7,
            "max_level": 10,
            "seed": 42,
            "version": 1,
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolution", 2.0),
        ("threshold", 1e-5),
        ("max_level", 4),
        ("seed", 7),
    ],
)
def test_nested_partition_parameters_round_trip_and_invalidate_fingerprint(
    field: str,
    value: Any,
) -> None:
    current = profile()
    partition = replace(current.community.partition_profile, **{field: value})
    changed = replace(current, community=replace(current.community, partition_profile=partition))
    assert graph_build_profile_from_json(graph_build_profile_to_json(changed)) == changed
    assert graph_build_profile_fingerprint(current) != graph_build_profile_fingerprint(changed)


@pytest.mark.parametrize(
    "nested,field,value",
    [
        (False, "seed", 42),
        (False, "iterations", 2),
        (False, "resolution", 1.0),
        (False, "version", 1),
        (False, "version", True),
        (False, "version", 3),
        (False, "extra", {}),
        (True, "tag", "leiden"),
        (True, "version", 2),
        (True, "version", True),
        (True, "kwargs", {}),
        (True, "iterations", 2),
        (True, "max_level", None),
        (True, "max_level", 0),
        (True, "max_level", True),
        (True, "seed", -1),
        (True, "seed", True),
        (True, "resolution", 0),
        (True, "threshold", -1),
        (True, "threshold", True),
        (True, "threshold", float("inf")),
        (True, "resolution", float("nan")),
    ],
)
def test_decoder_rejects_obsolete_and_malformed_community_profiles(
    nested: bool,
    field: str,
    value: Any,
) -> None:
    payload = json.loads(graph_build_profile_to_json(profile()))
    target = payload["community"]
    if nested:
        target = target.setdefault("partition_profile", {})
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        graph_build_profile_from_json(json.dumps(payload))


def test_partition_variant_cannot_accept_arbitrary_tag_or_dictionary() -> None:
    with pytest.raises(TypeError):
        policies.LouvainPartitionProfile(tag="leiden")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        policies.CommunityProfile(partition_profile={})  # type: ignore[arg-type]


@pytest.mark.parametrize("nested", [False, True])
def test_every_nested_community_field_is_required(nested: bool) -> None:
    baseline = json.loads(graph_build_profile_to_json(profile()))
    target = baseline["community"]
    if nested:
        target = target["partition_profile"]
    for field in tuple(target):
        missing = target.pop(field)
        with pytest.raises((TypeError, ValueError)):
            graph_build_profile_from_json(json.dumps(baseline))
        target[field] = missing


def test_duplicate_partition_tag_is_rejected() -> None:
    payload = graph_build_profile_to_json(profile()).replace(
        '"tag":"louvain"', '"tag":"louvain","tag":"louvain"'
    )
    with pytest.raises(ValueError):
        graph_build_profile_from_json(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_size", 0),
        ("max_size", True),
        ("max_depth", -1),
        ("max_depth", True),
        ("support_cap", 0),
        ("support_cap", True),
    ],
)
def test_community_hierarchy_and_support_limits_reject_invalid_values(
    field: str, value: Any
) -> None:
    with pytest.raises((TypeError, ValueError)):
        policies.CommunityProfile(**{field: value})


@pytest.mark.parametrize(
    "payload",
    [
        graph_build_profile_to_json(profile())[:-1] + ',"tool":"run"}',
        graph_build_profile_to_json(profile()).replace('"version":1', '"version":1,"version":1', 1),
        graph_build_profile_to_json(profile()).replace('"total_calls":3', '"total_calls":true'),
    ],
)
def test_graph_build_profile_decoder_rejects_open_or_ambiguous_json(payload: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        graph_build_profile_from_json(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_output_tokens", 0),
        ("max_output_tokens", True),
        ("temperature", float("nan")),
        ("total_calls", 0),
        ("max_repairs", 4),
        ("context_window_tokens", 4096),
    ],
)
def test_inference_policy_rejects_unsafe_or_incompatible_limits(field: str, value: object) -> None:
    values = {
        "provider_id": "local",
        "model_id": "qwen",
        "model_revision": "sha256:model",
        "tokenizer_id": "qwen-tokenizer",
        "tokenizer_revision": "sha256:tokenizer",
        "input_role": "user",
        "context_window_tokens": 8192,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        InferencePolicy(**values)  # type: ignore[arg-type]
