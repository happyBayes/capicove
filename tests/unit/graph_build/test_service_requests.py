"""Service request roles compose with the local adapter without graph storage."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.graph_build._adapters.local_generation import (
    LocalStructuredGenerator,
    LocalTextTokenizer,
)
from capicove.graph_build.service import BuildPolicy, GraphBuildService
from tests.unit.graph_build.test_extraction import build_profile
from tests.unit.graph_build.test_local_generation import Runtime, profile_at, request_for


@pytest.mark.parametrize(
    "role,accepted",
    [
        ("user", True),
        ("diagnostic", True),
        ("report", True),
        ("system", False),
        ("tool", False),
        ("assistant", False),
    ],
)
def test_service_roles_obey_the_local_adapter_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, accepted: bool
) -> None:
    runtime = Runtime(monkeypatch)
    local = profile_at(tmp_path / "assets")
    # Request composition needs policy only, not a build workspace or run database.
    service: Any = object.__new__(GraphBuildService)
    service.policy = BuildPolicy(subject_ref="subject:explicit")
    service.subject_resolver = None
    service.source_provenance = None
    service.inference_runner = None
    profile = replace(build_profile(), inference=request_for(local).inference_policy)
    request = (
        request_for(local)
        if role == "user"
        else service._model_request(role, {}, profile, "a" * 64)
    )
    tokenizer = LocalTextTokenizer(local)
    generator = LocalStructuredGenerator(local)
    if accepted:
        assert tokenizer.count_request(request) > 0
        assert generator.generate(request, control=runtime.control()).finish_reason == "stop"
        assert [item["role"] for item in runtime.tokenizer.messages] == ["system", "user"]
    else:
        with pytest.raises(ValueError, match="unsupported_local_decoding_policy"):
            generator.generate(request, control=runtime.control())
        assert not runtime.kwargs
        with pytest.raises(ValueError, match="unsupported_local_decoding_policy"):
            tokenizer.count_request(request)
