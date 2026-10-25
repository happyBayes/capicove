"""Model-backed skill adapters keep authorization and response validation explicit."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from capicove.evolution.providers import SanitizedExample
from capicove.runtime.evolution_generation import (
    CompatibleExperienceExtractor,
    CompatibleSkillExecutor,
)


def test_network_requires_callsite_authorization() -> None:
    with pytest.raises(ValueError, match="network"):
        CompatibleSkillExecutor(Path("/nonexistent"))
    with pytest.raises(ValueError, match="network"):
        CompatibleExperienceExtractor(Path("/nonexistent"))


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from capicove.graph_build import remote_generation

    state: dict[str, Any] = {"payload": b'{"output":"hello"}', "finish": "stop"}
    profile = SimpleNamespace(
        provider_id="test-profile", model="test-model", model_revision="rev", context_limit=8192
    )
    monkeypatch.setattr(remote_generation, "load_remote_generation_profile", lambda path: profile)
    monkeypatch.setattr(
        remote_generation,
        "OpenAICompatibleTokenizer",
        lambda profile: SimpleNamespace(tokenizer_id="tokenizer", tokenizer_revision="rev"),
    )

    class Generator:
        def __init__(self, profile: Any):
            pass

        def generate(self, request: Any, *, control: Any) -> Any:
            state["request"] = request
            return SimpleNamespace(payload=state["payload"], finish_reason=state["finish"])

    monkeypatch.setattr(remote_generation, "OpenAICompatibleGenerator", Generator)
    return state


def test_executor_uses_pinned_profile_and_never_receives_expected_answer(
    stub: dict[str, Any],
) -> None:
    executor = CompatibleSkillExecutor(Path("profile.json"), allow_network=True)
    assert executor.execute("task", "summarize", "Use citations.") == "hello"
    assert "test-profile" in executor.evaluator_id
    assert b"expected_output" not in stub["request"].canonical_payload


@pytest.mark.parametrize(
    "payload,finish",
    [
        (b'{"output":"hello"}', "length"),
        (b'{"output":12}', "stop"),
        (b'{"output":"hello","extra":1}', "stop"),
    ],
)
def test_invalid_or_truncated_response_fails(
    stub: dict[str, Any], payload: bytes, finish: str
) -> None:
    stub.update(payload=payload, finish=finish)
    with pytest.raises(ValueError):
        CompatibleSkillExecutor(Path("profile"), allow_network=True).execute("task", "echo", "")


def test_extractor_does_not_accept_invented_sources(stub: dict[str, Any]) -> None:
    stub["payload"] = (
        b'{"lessons":[{"episode_id":"invented","lesson":"Verify citations.","applicability":"Reports","exclusion":"","attribution":"skill_gap","confidence":0.9}]}'
    )
    with pytest.raises(ValueError):
        CompatibleExperienceExtractor(Path("profile"), allow_network=True).extract(
            (SanitizedExample("real", "reports", ("Check citations",), "succeeded", False),)
        )


def test_extractor_returns_validated_lesson_and_rejects_sensitive_input(
    stub: dict[str, Any],
) -> None:
    stub["payload"] = (
        b'{"lessons":[{"episode_id":"real","lesson":"Verify citations.","applicability":"Reports","exclusion":"","attribution":"skill_gap","confidence":0.9}]}'
    )
    extractor = CompatibleExperienceExtractor(Path("profile"), allow_network=True)
    lessons = extractor.extract(
        (SanitizedExample("real", "reports", ("Verify citations",), "failed", False),)
    )
    assert lessons[0].episode_id == "real"
    assert not lessons[0].correction
    with pytest.raises(ValueError):
        extractor.extract(
            (SanitizedExample("real", "reports", ("token=private",), "failed", False),)
        )


def test_duplicate_response_fields_fail_closed(stub: dict[str, Any]) -> None:
    stub["payload"] = b'{"output":"first","output":"second"}'
    with pytest.raises(ValueError, match="duplicate"):
        CompatibleSkillExecutor(Path("profile"), allow_network=True).execute("task", "echo", "")
