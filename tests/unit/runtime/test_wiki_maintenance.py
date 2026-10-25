"""Optional generation never bypasses live authorization and source fences."""

from pathlib import Path

import pytest

from capicove.contracts.wiki_maintenance import AnswerSubmission, AnswerTask, MaintenanceRecord
from capicove.wiki.maintenance.codec import encode_record
from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


class Generator:
    calls = 0

    def generate(self, task: AnswerTask) -> bytes:
        self.calls += 1
        return encode_record(
            MaintenanceRecord(
                schema_version=1,
                record_type="answer_submission",
                payload=AnswerSubmission(task_id=task.task_id, claims=()),
            )
        )


def test_absent_and_denied_generator_have_no_effects(tmp_path: Path) -> None:
    from capicove.runtime.wiki_maintenance import ask

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    before = snapshot(service.wiki)
    assert ask(service, "Plants").status == "evidence_only"
    generator = Generator()
    with pytest.raises(PermissionError):
        ask(service, "Plants", generator=generator)
    assert generator.calls == 0
    assert snapshot(service.wiki) == before


def test_generation_rechecks_policy_after_call(tmp_path: Path) -> None:
    from capicove.runtime.wiki_maintenance import GenerationAuthorization, ask

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()

    class WithdrawingGenerator(Generator):
        def generate(self, task: AnswerTask) -> bytes:
            service.withdraw(
                task.evidence[0].span.source_id,
                reason="Withdraw during generation",
                idempotency_key="mid-call",
            )
            return super().generate(task)

    with pytest.raises(ValueError, match="task"):
        ask(
            service,
            "Plants",
            generator=WithdrawingGenerator(),
            authorization=GenerationAuthorization(profile_id="synthetic", allow_egress=True),
        )


def test_malformed_and_oversized_output_are_rejected(tmp_path: Path) -> None:
    from capicove.runtime.wiki_maintenance import GenerationAuthorization, ask

    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()

    class InvalidGenerator(Generator):
        def generate(self, task: AnswerTask) -> bytes:
            return b"x" * 65537

    before = snapshot(service.wiki)
    with pytest.raises(ValueError, match="capacity"):
        ask(
            service,
            "Plants",
            generator=InvalidGenerator(),
            authorization=GenerationAuthorization(profile_id="synthetic", allow_egress=True),
        )
    assert snapshot(service.wiki) == before


def test_structured_adapter_rejects_truncation_and_profile_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace

    import capicove.graph_build.remote_generation as remote
    from capicove.runtime.wiki_maintenance import GenerationAuthorization, StructuredWikiGenerator

    @dataclass(frozen=True)
    class Profile:
        provider_id: str = "fixture"
        model: str = "fixture-model"
        model_revision: str = "pinned-revision"
        context_limit: int = 32000

    profile = Profile()
    monkeypatch.setattr(remote, "load_remote_generation_profile", lambda path: profile)

    class Tokenizer:
        tokenizer_id = "fixture-tokenizer"
        tokenizer_revision = "fixture-revision"

        def __init__(self, value: object) -> None:
            pass

        def count_request_controlled(self, request: object, *, control: object) -> int:
            return 100

    monkeypatch.setattr(remote, "OpenAICompatibleTokenizer", Tokenizer)

    class Provider:
        def __init__(self, value: object) -> None:
            pass

        def generate(self, request: object, *, control: object) -> object:
            return SimpleNamespace(payload=b"{}", finish_reason="length")

    monkeypatch.setattr(remote, "OpenAICompatibleGenerator", Provider)
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    task = service.answer_task("Plants")
    adapter = StructuredWikiGenerator(
        tmp_path / "profile.json", GenerationAuthorization(profile_id="fixture", allow_egress=True)
    )
    with pytest.raises(ValueError, match="incomplete"):
        adapter.generate(task)
    profile = Profile(model_revision="changed")
    with pytest.raises(ValueError, match="configuration_changed"):
        adapter.generate(task)
