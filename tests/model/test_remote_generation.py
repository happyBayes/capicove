"""Real user-configured compatible LLM acceptance; sends only authored evidence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from capicove.contracts.graph_build import ValidatedExtraction
from capicove.graph_build.extraction import extraction_request, validate_extraction
from capicove.graph_build.inference import DispatchControl, InferenceRunner, ValidationResult
from capicove.graph_build.profiles import StructuredRequest
from capicove.graph_build.remote_generation import (
    OpenAICompatibleGenerator,
    OpenAICompatibleProfile,
    OpenAICompatibleTokenizer,
)
from tests.unit.graph_build.test_extraction import build_profile, unit
from tests.unit.graph_build.test_inference import SCOPE, SOURCES, Harness


pytestmark = pytest.mark.model_small


@pytest.fixture(scope="module")
def profile() -> OpenAICompatibleProfile:
    path = os.environ.get("CAPICOVE_REMOTE_GENERATION_QUALIFICATION")
    if not path:
        pytest.skip("Select a compatible LLM configuration and credential environment variable")
    result = OpenAICompatibleProfile.from_mapping(json.loads(Path(path).read_text()))
    result.tokenizer_assets.verify()
    return result


def request_for(profile: OpenAICompatibleProfile) -> StructuredRequest:
    tokenizer = OpenAICompatibleTokenizer(profile)
    template = build_profile()
    policy = replace(
        template.inference,
        provider_id=profile.provider_id,
        model_id=profile.model,
        model_revision=profile.model_revision,
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_revision=tokenizer.tokenizer_revision,
        context_window_tokens=profile.context_limit,
        max_output_tokens=2048,
        call_deadline_seconds=90,
    )
    return extraction_request(unit(), replace(template, inference=policy))


def test_real_generation_validates_evidence_and_replays_without_credentials(
    profile: OpenAICompatibleProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = Harness(tmp_path, [])
    generator = OpenAICompatibleGenerator(profile)
    runner = InferenceRunner(
        generator=generator,
        tokenizer=OpenAICompatibleTokenizer(profile),
        gate=harness.gate,
        artifact_factory=lambda refs: harness.artifacts,
        receipts=harness.receipts,
        clock=time.monotonic,
        wall_clock=time.time,
        sleeper=time.sleep,
        cancelled=lambda: False,
    )
    accepted: list[ValidatedExtraction] = []

    def validate(payload: bytes) -> ValidationResult:
        result = validate_extraction(
            payload, unit(), "personal-growth-v1", hashlib.sha256(payload).hexdigest()
        )
        assert any(entity.name == "Alice" for entity in result.entities)
        assert all(
            span.evidence_ref == "source"
            for entity in result.entities
            for span in entity.source_spans
        )
        accepted.append(result)
        return ValidationResult(payload, result.issue_codes, rejected_count=result.rejected_count)

    request = request_for(profile)
    receipt = runner.run(SCOPE, request, SOURCES, validate)
    assert receipt.status == "succeeded"
    assert accepted
    monkeypatch.delenv(profile.api_key_env, raising=False)
    replay = runner.run(SCOPE, request, SOURCES, validate, mode="replay")
    assert replay == receipt
    harness.gate.denied = True
    with pytest.raises(PermissionError):
        runner.run(SCOPE, request, SOURCES, validate, mode="replay")


def test_real_server_rejects_wrong_credentials(
    profile: OpenAICompatibleProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(profile.api_key_env, "authored-invalid-qualification-credential")
    with pytest.raises(ValueError, match=r"remote_http_(401|403)"):
        OpenAICompatibleGenerator(profile).generate(
            request_for(profile),
            control=DispatchControl(time.monotonic() + 30, time.monotonic, lambda: False),
        )
