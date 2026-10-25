from dataclasses import FrozenInstanceError, fields

import pytest

from capicove.contracts import Episode, EpisodeStep, StepStatus, ToolCall


def make_step(sequence: int) -> EpisodeStep:
    return EpisodeStep(
        step_id=f"step-{sequence}",
        sequence=sequence,
        observation_refs=(f"artifact-{sequence}",),
        action="invoke_tool",
        tool_call=ToolCall(
            request_id=f"request-{sequence}",
            tool_id="capicove.search",
            tool_version=1,
            arguments={"query": "capicove"},
        ),
        result_ref=f"evidence-{sequence}",
        status=StepStatus.SUCCEEDED,
    )


def make_episode(*steps: EpisodeStep) -> Episode:
    return Episode(
        episode_id="episode-1",
        goal="Find grounded evidence.",
        environment_id="environment-1",
        environment_version=1,
        tool_catalog_hash="a" * 64,
        active_asset_versions={"grounded-search": 1},
        steps=steps,
    )


def test_episode_requires_contiguous_step_order() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        make_episode(make_step(0), make_step(2))


def test_trace_records_are_immutable() -> None:
    step = make_step(0)
    with pytest.raises(FrozenInstanceError):
        step.sequence = 1  # type: ignore[misc]


def test_trace_mappings_are_defensive_copies() -> None:
    arguments: dict[str, object] = {"query": "capicove"}
    versions = {"grounded-search": 1}
    call = ToolCall(
        request_id="request-1",
        tool_id="capicove.search",
        tool_version=1,
        arguments=arguments,
    )
    episode = Episode(
        episode_id="episode-1",
        goal="Find grounded evidence.",
        environment_id="environment-1",
        environment_version=1,
        tool_catalog_hash="a" * 64,
        active_asset_versions=versions,
    )

    arguments["query"] = "mutated"
    versions["grounded-search"] = 2

    assert call.arguments["query"] == "capicove"
    assert episode.active_asset_versions["grounded-search"] == 1


def test_episode_accepts_ordered_steps_and_optional_outcome() -> None:
    episode = make_episode(make_step(0), make_step(1))

    assert tuple(step.sequence for step in episode.steps) == (0, 1)
    assert episode.outcome is None
    assert episode.reward is None


def test_episode_rejects_invalid_catalog_hash() -> None:
    with pytest.raises(ValueError, match="tool_catalog_hash"):
        Episode(
            episode_id="episode-1",
            goal="Find grounded evidence.",
            environment_id="environment-1",
            environment_version=1,
            tool_catalog_hash="not-a-sha256",
            active_asset_versions={},
        )


def test_trace_schema_does_not_capture_hidden_reasoning() -> None:
    field_names = {field.name for field in fields(EpisodeStep)}

    assert "chain_of_thought" not in field_names
    assert "reasoning" not in field_names
