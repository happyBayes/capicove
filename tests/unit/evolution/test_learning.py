"""Exercise deterministic evidence learning without a model or network."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import pytest

from capicove.contracts import PrivacyClass
from capicove.evolution.learning import batch_from_json, batch_to_json, distill, propose
from capicove.traces import EpisodeCompleted, EpisodeFeedback, EpisodeStarted, EpisodeStatus
from capicove.traces.codec import trace_record_sha256
from capicove.traces.events import TracePayload, TraceRecordType
from capicove.traces.models import TraceRecord
from capicove.traces.ports import DEFAULT_TRACE_QUERY, TraceQuery, TraceSelectionSummary


class Reader:
    def __init__(self, records: list[TraceRecord]):
        self.records = records

    def open_snapshot(self, query: TraceQuery = DEFAULT_TRACE_QUERY, /) -> Reader:
        return self

    @property
    def summary(self) -> TraceSelectionSummary:
        return TraceSelectionSummary(1, 0, 0)

    def __enter__(self) -> Reader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def iter_records(self) -> Iterator[TraceRecord]:
        return iter(self.records)


def episode(identity: str = "one", *, correction: bool = True, **fields: Any) -> list[TraceRecord]:
    feedback = dict(
        task_signature="reports",
        lesson="Check every citation.",
        applicability="When writing reports",
        exclusion="None",
        attribution="skill_gap",
        confidence=0.95,
    )
    feedback.update(fields)
    payloads: list[TracePayload] = [
        EpisodeStarted(
            "secret raw goal /Users/private", "workspace", "actor", "environment", 1, "a" * 64, {}
        ),
        EpisodeCompleted(EpisodeStatus.SUCCEEDED),
        EpisodeFeedback("correction" if correction else "lesson", feedback, "user"),
    ]
    records: list[TraceRecord] = []
    for index, payload in enumerate(payloads):
        records.append(
            TraceRecord(
                schema_version=1,
                record_version=1,
                record_type=[
                    TraceRecordType.EPISODE_STARTED,
                    TraceRecordType.EPISODE_COMPLETED,
                    TraceRecordType.EPISODE_FEEDBACK,
                ][index],
                record_id=f"{identity}-{index}",
                episode_id=identity,
                trace_id=identity,
                sequence=index,
                occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
                payload=payload,
                previous_record_sha256=trace_record_sha256(records[-1]) if records else None,
            )
        )
    return records


def test_correction_beats_success_and_roundtrips() -> None:
    (batch,) = distill(Reader(episode()))
    assert batch.lessons[0].correction
    assert batch_from_json(batch_to_json(batch)) == batch
    proposal = propose(batch, name="report-writing")
    assert proposal.action == "create"
    assert "Check every citation." in proposal.content
    assert "secret raw goal" not in proposal.content
    assert proposal.source_episode_ids == ("one",)


def test_support_counts_episodes_not_feedback_records() -> None:
    (batch,) = distill(Reader(episode(correction=False)))
    assert propose(batch, name="reports").action == "no_change"
    (batch,) = distill(Reader([r for i in range(3) for r in episode(str(i), correction=False)]))
    assert propose(batch, name="reports").action == "create"


@pytest.mark.parametrize("attribution", ["uncertain", "environment_failure", "agent_misuse"])
def test_attribution_blocks_changes(attribution: str) -> None:
    (batch,) = distill(Reader(episode(attribution=attribution)))
    assert propose(batch, name="reports").action == "no_change"


def test_invalid_and_restricted_chains_are_excluded() -> None:
    records = episode()
    assert distill(Reader(records[:1])) == ()
    assert (
        distill(
            Reader([records[0], replace(records[1], previous_record_sha256="0" * 64), records[2]])
        )
        == ()
    )
    records[0] = replace(records[0], privacy_class=PrivacyClass.RESTRICTED)
    assert distill(Reader(records)) == ()


@pytest.mark.parametrize(
    "lesson", ["/Users/private/file", "token=abcdef", "https://secret.test/a", "x" * 3000]
)
def test_unsafe_or_unbounded_feedback_is_excluded(lesson: str) -> None:
    assert distill(Reader(episode(lesson=lesson))) == ()


def test_identity_is_deterministic_and_tampering_rejected() -> None:
    (first,) = distill(Reader(episode("a") + episode("b")))
    (second,) = distill(Reader(episode("b") + episode("a")))
    assert first == second
    with pytest.raises(ValueError):
        batch_from_json(batch_to_json(first).replace("Check every citation.", "Wrong"))


def test_patch_preserves_baseline_and_covered_guidance_is_no_change() -> None:
    (batch,) = distill(Reader(episode()))
    proposal = propose(batch, name="reports", baseline_content="---\nname: reports\n---\nExisting")
    assert proposal.action == "patch"
    assert "Existing" in proposal.content
    assert propose(batch, name="reports", baseline_content=proposal.content).action == "no_change"


def test_versions_require_resolution() -> None:
    records = episode()
    start = records[0].payload
    assert isinstance(start, EpisodeStarted)
    records[0] = replace(records[0], payload=replace(start, active_asset_versions={"reports": 1}))
    for index in range(1, len(records)):
        records[index] = replace(
            records[index], previous_record_sha256=trace_record_sha256(records[index - 1])
        )
    assert distill(Reader(records)) == ()
    (batch,) = distill(
        Reader(records), resolve_version=lambda name, version: name == "reports" and version == 1
    )
    assert batch.skill_versions == (("reports", 1),)


def test_extractor_gets_only_bounded_visible_evidence_and_cannot_invent_corrections() -> None:
    from capicove.evolution.learning import Lesson
    from capicove.evolution.providers import SanitizedExample
    from capicove.traces import MessageObserved, MessageRole, TextContent

    class Extractor:
        def extract(self, examples: tuple[SanitizedExample, ...]) -> tuple[Lesson, ...]:
            assert examples[0].messages == (
                "Citations need verification.",
                "I verified the citations.",
            )
            return (
                Lesson(
                    "Check citations.",
                    "Reports",
                    "",
                    "skill_gap",
                    1.0,
                    examples[0].episode_id,
                    False,
                ),
            )

    records = episode()[:2]
    records[1] = replace(
        records[1],
        record_type=TraceRecordType.MESSAGE_OBSERVED,
        payload=MessageObserved(
            MessageRole.USER,
            (TextContent("Citations need verification."), TextContent("token=private")),
        ),
    )
    records.append(
        replace(
            episode()[1],
            sequence=2,
            record_id="terminal",
            previous_record_sha256=trace_record_sha256(records[1]),
        )
    )
    assert distill(Reader(records), task_signature="reports", extractor=Extractor()) == ()
    records.insert(
        2,
        replace(
            records[1],
            record_id="answer",
            sequence=2,
            payload=MessageObserved(
                MessageRole.ASSISTANT, (TextContent("I verified the citations."),)
            ),
            previous_record_sha256=trace_record_sha256(records[1]),
        ),
    )
    records[3] = replace(
        records[3], sequence=3, previous_record_sha256=trace_record_sha256(records[2])
    )
    (batch,) = distill(Reader(records), task_signature="reports", extractor=Extractor())
    assert batch.extractor_id.endswith("Extractor")
    assert batch_from_json(batch_to_json(batch)) == batch
    assert propose(batch, name="reports").action == "no_change"


def test_patch_requires_matching_version_when_baseline_is_identified() -> None:
    (batch,) = distill(Reader(episode()))
    assert (
        propose(batch, name="reports", baseline_content="Existing", baseline_version=1).action
        == "no_change"
    )


def test_patch_counts_only_matching_episode_versions() -> None:
    def versioned(identity: str, version: int, correction: bool = False) -> list[TraceRecord]:
        records = episode(identity, correction=correction)
        start = records[0].payload
        assert isinstance(start, EpisodeStarted)
        records[0] = replace(
            records[0], payload=replace(start, active_asset_versions={"reports": version})
        )
        for index in range(1, len(records)):
            records[index] = replace(
                records[index], previous_record_sha256=trace_record_sha256(records[index - 1])
            )
        return records

    (batch,) = distill(
        Reader(versioned("a", 1) + versioned("b", 1) + versioned("c", 2)),
        resolve_version=lambda name, version: True,
    )
    assert (
        propose(batch, name="reports", baseline_content="Existing", baseline_version=2).action
        == "no_change"
    )
    (batch,) = distill(
        Reader(episode("old") + versioned("new", 2, correction=True)),
        resolve_version=lambda name, version: True,
    )
    proposal = propose(batch, name="reports", baseline_content="Existing", baseline_version=2)
    assert proposal.action == "patch"
    assert proposal.source_episode_ids == ("new",)
    assert batch_from_json(batch_to_json(batch)) == batch


def test_repeated_low_confidence_claims_do_not_create_guidance() -> None:
    (batch,) = distill(Reader([r for i in range(3) for r in episode(str(i), confidence=0.1)]))
    assert propose(batch, name="reports").action == "no_change"


def test_snapshot_limits_fail_closed() -> None:
    with pytest.raises(ValueError, match="limit"):
        distill(Reader(episode()), max_records=2)
    with pytest.raises(ValueError, match="limit"):
        distill(Reader(episode("a") + episode("b")), max_episodes=1)


def test_explicit_feedback_policy_is_part_of_identity() -> None:
    (batch,) = distill(Reader(episode()))
    assert batch.extraction_policy == "bounded-visible-feedback-v1"
    assert batch.extractor_id == "explicit-feedback-v1"
