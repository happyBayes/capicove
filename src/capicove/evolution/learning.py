"""Deterministic learning from explicit, bounded, trace-linked feedback.

This adapter deliberately does not infer instructions from arbitrary tool output.
Feedback uses lesson, applicability, exclusion, attribution, confidence and
 task_signature fields. Hosts explicitly choose which feedback to record.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from capicove.contracts import PrivacyClass
from capicove.evolution.providers import ExperienceExtractor, SanitizedExample
from capicove.traces.codec import canonical_json_bytes, trace_record_sha256
from capicove.traces.content import MessageRole, TextContent
from capicove.traces.events import (
    EpisodeCompleted,
    EpisodeFeedback,
    EpisodeStarted,
    EpisodeStatus,
    MessageObserved,
    ToolCallObserved,
    ToolResultObserved,
)
from capicove.traces.models import TraceRecord
from capicove.traces.ports import DEFAULT_TRACE_QUERY, TraceQuery, TraceReader


_UNSAFE = re.compile(
    r"(?:https?://|file:|(?:^|\s)(?:/|~/|[A-Za-z]:\\)|"
    r"(?i:password|secret|token|api[_ -]?key)\s*[:=]|"
    r"-----BEGIN|\bsk-[A-Za-z0-9]|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,})"
)
_ATTRIBUTIONS = {"skill_gap", "agent_misuse", "environment_failure", "uncertain"}


def sanitize_text(value: object, *, empty: bool = False) -> str:
    """Reject rather than forward unbounded or recognizable sensitive text."""
    if not isinstance(value, str) or len(value.encode("utf-8")) > 2048:
        raise ValueError("feedback text must be bounded")
    value = " ".join(value.split())
    if (not value and not empty) or _UNSAFE.search(value):
        raise ValueError("feedback contains unsupported sensitive text")
    return value


@dataclass(frozen=True, slots=True)
class Lesson:
    """One explicit feedback observation, never an inferred success claim."""

    lesson: str
    applicability: str
    exclusion: str
    attribution: str
    confidence: float
    episode_id: str
    correction: bool
    skill_versions: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        for key in ("lesson", "applicability", "exclusion"):
            if sanitize_text(getattr(self, key), empty=key == "exclusion") != getattr(self, key):
                raise ValueError("lesson must contain canonical text")
        if self.attribution not in _ATTRIBUTIONS:
            raise ValueError("unsupported attribution")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be between zero and one")
        if not isinstance(self.correction, bool) or not self.episode_id:
            raise ValueError("invalid lesson identity")
        if not isinstance(self.skill_versions, tuple) or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or not pair[0]
            or type(pair[1]) is not int
            or pair[1] < 1
            for pair in self.skill_versions
        ):
            raise ValueError("invalid skill version identity")


@dataclass(frozen=True, slots=True)
class EvidenceBatch:
    """Immutable sanitized experience with deterministic evidence identity."""

    batch_id: str
    task_signature: str
    lessons: tuple[Lesson, ...]
    source_episode_ids: tuple[str, ...]
    record_hashes: tuple[str, ...]
    skill_versions: tuple[tuple[str, int], ...] = ()

    extraction_policy: str = "bounded-visible-feedback-v1"
    extractor_id: str = "explicit-feedback-v1"

    def __post_init__(self) -> None:
        if sanitize_text(self.task_signature) != self.task_signature or len(self.lessons) > 50:
            raise ValueError("invalid task signature or batch size")
        if not self.lessons or tuple(sorted({x.episode_id for x in self.lessons})) != (
            self.source_episode_ids
        ):
            raise ValueError("source identities must match lessons")
        if not self.record_hashes or any(
            re.fullmatch("[0-9a-f]{64}", value) is None for value in self.record_hashes
        ):
            raise ValueError("invalid evidence hashes")
        if self.extraction_policy != "bounded-visible-feedback-v1" or not self.extractor_id:
            raise ValueError("invalid extraction provenance")
        if self.batch_id != _identity(self):
            raise ValueError("evidence identity mismatch")


def _identity(batch: EvidenceBatch) -> str:
    payload = asdict(batch)
    payload.pop("batch_id")
    return "experience:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _make_batch(
    signature: str,
    lessons: tuple[Lesson, ...],
    hashes: tuple[str, ...],
    versions: tuple[tuple[str, int], ...],
    extractor_id: str,
) -> EvidenceBatch:
    values: dict[str, Any] = dict(
        task_signature=signature,
        lessons=lessons,
        source_episode_ids=tuple(sorted({x.episode_id for x in lessons})),
        record_hashes=hashes,
        skill_versions=versions,
        extraction_policy="bounded-visible-feedback-v1",
        extractor_id=extractor_id,
    )
    digest_values = dict(values, lessons=tuple(asdict(x) for x in lessons))
    identity = "experience:" + hashlib.sha256(canonical_json_bytes(digest_values)).hexdigest()
    return EvidenceBatch(batch_id=identity, **values)


def _eligible(records: list[TraceRecord], resolve: Callable[[str, int], bool] | None) -> bool:
    if not records or not isinstance(records[0].payload, EpisodeStarted):
        return False
    completed = False
    previous = None
    for index, record in enumerate(records):
        if (
            record.sequence != index
            or record.previous_record_sha256 != previous
            or record.privacy_class in {PrivacyClass.RESTRICTED, PrivacyClass.CONFIDENTIAL}
            or record.trace_id != records[0].trace_id
        ):
            return False
        if index and isinstance(record.payload, EpisodeStarted):
            return False
        if completed and not isinstance(record.payload, EpisodeFeedback):
            return False
        if isinstance(record.payload, EpisodeCompleted):
            if record.payload.status is EpisodeStatus.INCOMPLETE:
                return False
            completed = True
        previous = trace_record_sha256(record)
    assets = records[0].payload.active_asset_versions
    return completed and all(resolve is not None and resolve(k, v) for k, v in assets.items())


def _observable_pair(records: list[TraceRecord]) -> bool:
    """Require a visible host action/result, not a bare success assertion."""
    user_seen = False
    calls: set[tuple[str | None, str]] = set()
    for record in records:
        payload = record.payload
        if isinstance(payload, MessageObserved):
            if payload.role is MessageRole.USER:
                user_seen = True
            elif payload.role is MessageRole.ASSISTANT and user_seen:
                return True
        elif isinstance(payload, ToolCallObserved):
            calls.add((record.request_id, payload.tool_id))
        elif isinstance(payload, ToolResultObserved):
            if (record.request_id, payload.tool_id) in calls:
                return True
    return False


def distill(
    reader: TraceReader,
    query: TraceQuery = DEFAULT_TRACE_QUERY,
    *,
    task_signature: str | None = None,
    resolve_version: Callable[[str, int], bool] | None = None,
    extractor: ExperienceExtractor | None = None,
    max_records: int = 10000,
    max_episodes: int = 1000,
) -> tuple[EvidenceBatch, ...]:
    """Read a fixed snapshot and extract explicit feedback from eligible Episodes.

    Invalid Episodes are excluded even when the caller requests restricted or
    incomplete records. Version resolution is mandatory for referenced assets.
    At most fifty canonical episode observations enter each task batch.
    """
    if (
        type(max_records) is not int
        or type(max_episodes) is not int
        or min(max_records, max_episodes) < 1
    ):
        raise ValueError("snapshot limits must be positive integers")
    extractor_id = "explicit-feedback-v1"
    if extractor is not None:
        extractor_id = getattr(extractor, "extractor_id", None) or (
            type(extractor).__module__ + "." + type(extractor).__qualname__
        )
        if not isinstance(extractor_id, str) or len(extractor_id) > 512:
            raise ValueError("invalid extractor identity")
    episodes: dict[str, list[TraceRecord]] = defaultdict(list)
    with reader.open_snapshot(query) as snapshot:
        for count, record in enumerate(snapshot.iter_records(), start=1):
            if count > max_records or (
                record.episode_id not in episodes and len(episodes) >= max_episodes
            ):
                raise ValueError("trace snapshot exceeds configured limit")
            episodes[record.episode_id].append(record)
    groups: dict[str, list[tuple[Lesson, tuple[str, ...], tuple[tuple[str, int], ...]]]] = (
        defaultdict(list)
    )
    used_extractors: dict[str, set[str]] = defaultdict(set)
    for records in episodes.values():
        records.sort(key=lambda item: item.sequence)
        if not _eligible(records, resolve_version):
            continue
        start = records[0].payload
        assert isinstance(start, EpisodeStarted)
        if (
            extractor is not None
            and task_signature is not None
            and _observable_pair(records)
            and not any(isinstance(r.payload, EpisodeFeedback) for r in records)
        ):
            messages: list[str] = []
            for record in records[:20]:
                if isinstance(record.payload, MessageObserved) and record.payload.role in {
                    MessageRole.USER,
                    MessageRole.ASSISTANT,
                }:
                    for block in record.payload.content:
                        if isinstance(block, TextContent):
                            try:
                                messages.append(sanitize_text(block.text))
                            except ValueError:
                                continue
            visible = tuple(messages)[:16]
            if visible:
                terminal = next(
                    r.payload for r in records if isinstance(r.payload, EpisodeCompleted)
                )
                example = SanitizedExample(
                    records[0].episode_id,
                    sanitize_text(task_signature),
                    visible,
                    terminal.status.value,
                    False,
                )
                extracted = extractor.extract((example,))
                if len(extracted) > 1:
                    raise ValueError("extractor must return at most one lesson per episode")
                for lesson in extracted:
                    # Model output cannot invent sources or an explicit user correction.
                    if lesson.episode_id != example.episode_id or lesson.correction:
                        raise ValueError("extractor changed authoritative evidence identity")
                    Lesson(**asdict(lesson))
                    lesson = replace(
                        lesson, skill_versions=tuple(sorted(start.active_asset_versions.items()))
                    )
                    used_extractors[task_signature].add(extractor_id)
                    groups[task_signature].append(
                        (
                            lesson,
                            tuple(trace_record_sha256(r) for r in records),
                            tuple(sorted(start.active_asset_versions.items())),
                        )
                    )
        feedback = [r.payload for r in records if isinstance(r.payload, EpisodeFeedback)]
        corrections = [f for f in feedback if f.kind == "correction" and f.provenance == "user"]
        # The latest explicit correction supersedes earlier positive observations.
        selected = corrections[-1:] if corrections else feedback[-1:]
        for item in selected:
            if item.kind not in {"correction", "lesson"}:
                continue
            value = item.value
            try:
                signature = sanitize_text(value.get("task_signature", task_signature))
                confidence = value.get("confidence")
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    continue
                lesson = Lesson(
                    sanitize_text(value.get("lesson")),
                    sanitize_text(value.get("applicability")),
                    sanitize_text(value.get("exclusion", ""), empty=True),
                    str(value.get("attribution", "uncertain")),
                    float(confidence),
                    records[0].episode_id,
                    bool(corrections),
                    tuple(sorted(start.active_asset_versions.items())),
                )
            except (ValueError, TypeError):
                continue
            if task_signature is not None and signature != task_signature:
                continue
            used_extractors[signature].add("explicit-feedback-v1")
            groups[signature].append(
                (
                    lesson,
                    tuple(trace_record_sha256(r) for r in records),
                    tuple(sorted(start.active_asset_versions.items())),
                )
            )
    result = []
    for signature, items in sorted(groups.items()):
        items.sort(key=lambda item: item[0].episode_id)
        if len(items) > 50:
            items = items[:25] + items[-25:]
        result.append(
            _make_batch(
                signature,
                tuple(x[0] for x in items),
                tuple(sorted({h for x in items for h in x[1]})),
                tuple(sorted({v for x in items for v in x[2]})),
                "+".join(sorted(used_extractors[signature])),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ProposedSkill:
    """A declarative proposal that never activates or writes a skill."""

    action: str
    name: str
    content: str
    source_episode_ids: tuple[str, ...]
    reason: str
    batch_id: str


def propose(
    batch: EvidenceBatch,
    *,
    name: str,
    baseline_content: str | None = None,
    baseline_version: int | None = None,
) -> ProposedSkill:
    """Materialize supported rules, preserving an existing document for patches."""
    if re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name) is None:
        raise ValueError("skill name must be a lowercase slug")
    support: dict[tuple[str, str, str], list[Lesson]] = defaultdict(list)
    for item in batch.lessons:
        if (
            baseline_content is not None
            and baseline_version is not None
            and (name, baseline_version) not in item.skill_versions
        ):
            continue
        if item.attribution == "skill_gap" and item.confidence >= 0.7:
            support[item.lesson, item.applicability, item.exclusion].append(item)
    accepted = []
    sources: set[str] = set()
    for key, observations in sorted(support.items()):
        if len({x.episode_id for x in observations}) >= 3 or any(
            x.correction and x.confidence >= 0.9 for x in observations
        ):
            if baseline_content and key[0] in baseline_content:
                continue
            accepted.append(f"- {key[0]} Apply: {key[1]}. Exclude: {key[2] or 'None'}.")
            sources.update(x.episode_id for x in observations)
    action = "patch" if baseline_content is not None else "create"
    content = baseline_content or (
        f"---\nname: {name}\ndescription: "
        f"{json.dumps('Guidance for ' + batch.task_signature)}\n---\n\n# {name}\n"
    )
    if not accepted:
        return ProposedSkill(
            "no_change", name, "", (), "Insufficient or covered skill-gap evidence", batch.batch_id
        )
    content = content.rstrip() + "\n\n## Learned guidance\n\n" + "\n".join(accepted) + "\n"
    return ProposedSkill(
        action, name, content, tuple(sorted(sources)), "Supported explicit feedback", batch.batch_id
    )


def batch_to_json(batch: EvidenceBatch) -> str:
    """Serialize an immutable batch without raw trace payloads."""
    return canonical_json_bytes(asdict(batch)).decode("utf-8")


def batch_from_json(content: str) -> EvidenceBatch:
    """Load and verify a previously stored batch's deterministic identity."""
    try:
        value = json.loads(content)
        if not isinstance(value, Mapping) or set(value) != {
            "batch_id",
            "task_signature",
            "lessons",
            "source_episode_ids",
            "record_hashes",
            "skill_versions",
            "extraction_policy",
            "extractor_id",
        }:
            raise ValueError("batch must be an object")
        return EvidenceBatch(
            batch_id=value["batch_id"],
            task_signature=value["task_signature"],
            lessons=tuple(
                Lesson(
                    **dict(
                        item, skill_versions=tuple(tuple(v) for v in item.get("skill_versions", ()))
                    )
                )
                for item in value["lessons"]
            ),
            source_episode_ids=tuple(value["source_episode_ids"]),
            record_hashes=tuple(value["record_hashes"]),
            skill_versions=tuple(tuple(x) for x in value["skill_versions"]),
            extraction_policy=value["extraction_policy"],
            extractor_id=value["extractor_id"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid evidence batch") from error
