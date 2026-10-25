"""Offline benchmark preparation and paired reporting, never an unofficial QA judge.

Gold labels stay in evaluator-only output. Official score generation requires the
pinned upstream evaluator; this module does not infer correctness from retrieval.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PreparedCampaign:
    ingestion: tuple[dict[str, Any], ...]
    questions: tuple[dict[str, Any], ...]
    evaluation: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid dataset string")
    return value


def _history_text(value: object) -> str:
    # Preserve empty official turns; do not invent or silently drop text.
    if not isinstance(value, str):
        raise ValueError("invalid history content")
    return value


def _answer(value: object) -> str | int | float:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    raise ValueError("invalid evaluator answer")


def _manifest(
    profile: str,
    revision: str,
    seed: str,
    ingestion: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    groups: Mapping[str, str],
) -> PreparedCampaign:
    _string(revision)
    _string(seed)
    entries = [
        {
            "question_id": q["question_id"],
            "group": groups[q["question_id"]],
            "split": "development"
            if int(_hash([seed, groups[q["question_id"]]])[:8], 16) % 100 < 20
            else "acceptance",
            "category": labels[i]["category"],
        }
        for i, q in enumerate(questions)
    ]
    manifest = {
        "version": 1,
        "profile": profile,
        "dataset_revision": revision,
        "seed": seed,
        "text_only": True,
        "ingestion_sha256": _hash(ingestion),
        "questions_sha256": _hash(questions),
        "labels_sha256": _hash(labels),
        "cases": entries,
        "qualification": "PREPARED_NOT_RUN",
        "split_note": "Group-hash assignment may not achieve exact 20/80 or category balance; inspect before freezing.",
    }
    manifest["manifest_sha256"] = _hash(manifest)
    return PreparedCampaign(tuple(ingestion), tuple(questions), tuple(labels), manifest)


def prepare_longmemeval(
    raw: Sequence[Mapping[str, Any]], *, dataset_revision: str, seed: str
) -> PreparedCampaign:
    """Allowlist history fields and group connected supporting-session families."""
    ingestion: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    support: dict[str, str] = {}
    parents: dict[str, str] = {}

    def root(identity: str) -> str:
        while parents[identity] != identity:
            identity = parents[identity]
        return identity

    for item in raw:
        identity = _string(item["question_id"])
        if identity in parents:
            raise ValueError("duplicate question_id")
        parents[identity] = identity
        ids, dates, histories = (
            item["haystack_session_ids"],
            item["haystack_dates"],
            item["haystack_sessions"],
        )
        if (
            not all(isinstance(v, list) for v in (ids, dates, histories))
            or not len(ids) == len(dates) == len(histories)
            or not ids
        ):
            raise ValueError("misaligned history")
        answer_ids = item["answer_session_ids"]
        if not isinstance(answer_ids, list) or any(x not in ids for x in answer_ids):
            raise ValueError("invalid evidence session ids")
        sessions: list[dict[str, Any]] = []
        for session_index, (session_id, date, turns) in enumerate(
            zip(ids, dates, histories, strict=True)
        ):
            _string(session_id)
            _string(date)
            if not isinstance(turns, list) or not turns:
                raise ValueError("invalid history turns")
            clean = [
                {"role": _string(turn["role"]), "content": _history_text(turn["content"])}
                for turn in turns
            ]
            if any(t["role"] not in {"user", "assistant", "system"} for t in clean):
                raise ValueError("invalid role")
            sessions.append(
                {
                    "session_id": session_id,
                    "session_index": session_index,
                    "timestamp": date,
                    "turns": clean,
                }
            )
            if session_id in answer_ids:
                fingerprint = _hash(clean)
                if fingerprint in support:
                    a, b = root(identity), root(support[fingerprint])
                    parents[max(a, b)] = min(a, b)
                support[fingerprint] = identity
        ingestion.append({"history_id": identity, "sessions": sessions})
        questions.append(
            {
                "question_id": identity,
                "history_id": identity,
                "question": _string(item["question"]),
                "question_date": _string(item["question_date"]),
            }
        )
        labels.append(
            {
                "question_id": identity,
                "answer": _answer(item["answer"]),
                "category": _string(item["question_type"]),
                "answer_session_ids": list(answer_ids),
                "abstention": identity.endswith("_abs"),
            }
        )
    if not questions:
        raise ValueError("empty dataset")
    groups = {identity: _hash(["history-family", root(identity)]) for identity in parents}
    return _manifest("longmemeval", dataset_revision, seed, ingestion, questions, labels, groups)


def prepare_locomo(
    raw: Sequence[Mapping[str, Any]], *, dataset_revision: str, seed: str
) -> PreparedCampaign:
    """Keep entire conversations together; explicitly prepare a text-only profile."""
    ingestion: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    seen: set[str] = set()
    for item in raw:
        identity = _string(item["sample_id"])
        if identity in seen:
            raise ValueError("duplicate sample_id")
        seen.add(identity)
        conversation = item["conversation"]
        sessions = []
        keys = sorted(
            (k for k in conversation if re.fullmatch(r"session_\d+", k)),
            key=lambda k: int(k.split("_")[1]),
        )
        if not keys:
            raise ValueError("empty conversation")
        for key in keys:
            turns = [
                {
                    "speaker": _string(t["speaker"]),
                    "message_id": _string(t["dia_id"]),
                    "text": _string(t["text"]),
                }
                for t in conversation[key]
            ]
            sessions.append(
                {
                    "session_id": key,
                    "timestamp": _string(conversation[key + "_date_time"]),
                    "turns": turns,
                }
            )
        ingestion.append({"history_id": identity, "sessions": sessions})
        for i, qa in enumerate(item["qa"]):
            question_id = f"{identity}:{i}"
            category = qa["category"]
            if type(category) is not int or category not in {1, 2, 3, 4, 5}:
                raise ValueError("unknown LoCoMo category")
            questions.append(
                {
                    "question_id": question_id,
                    "history_id": identity,
                    "question": _string(qa["question"]),
                }
            )
            # Adversarial cases can omit an answer; the official evaluator owns
            # that category's scoring semantics, not this preparation adapter.
            labels.append(
                {
                    "question_id": question_id,
                    "answer": qa.get("answer"),
                    "category": category,
                    "evidence": qa.get("evidence", []),
                }
            )
            groups[question_id] = _hash(["conversation", identity])
    if not questions:
        raise ValueError("empty dataset")
    return _manifest("locomo", dataset_revision, seed, ingestion, questions, labels, groups)


def summarize_paired(
    ids: Sequence[str],
    baseline: Mapping[str, float | bool],
    candidate: Mapping[str, float | bool],
    groups: Mapping[str, str],
    *,
    seed: int,
) -> dict[str, Any]:
    """Summarize upstream scores; missing predictions are zero and incomplete."""
    expected = set(ids)
    if (
        not ids
        or len(expected) != len(ids)
        or set(groups) != expected
        or (set(baseline) | set(candidate)) - expected
    ):
        raise ValueError("invalid result identities")
    for values in (baseline, candidate):
        for value in values.values():
            if (
                not isinstance(value, (int, float, bool))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("invalid upstream score")
    baseline_mean = sum(baseline.get(i, 0) for i in ids) / len(ids)
    candidate_mean = sum(candidate.get(i, 0) for i in ids) / len(ids)
    grouped: dict[str, list[str]] = {}
    for identity in ids:
        grouped.setdefault(groups[identity], []).append(identity)
    # Reproducible statistical resampling; never used for keys or tokens.
    rng = random.Random(seed)  # nosec B311
    names = sorted(grouped)
    deltas = []
    for _ in range(2000):
        sample = [i for name in rng.choices(names, k=len(names)) for i in grouped[name]]
        deltas.append(
            sum(float(candidate.get(i, 0)) - float(baseline.get(i, 0)) for i in sample)
            / len(sample)
        )
    deltas.sort()
    missing_baseline, missing_candidate = (
        sorted(expected - baseline.keys()),
        sorted(expected - candidate.keys()),
    )
    return {
        "status": "INCOMPLETE" if missing_baseline or missing_candidate else "SCORED_NOT_PROMOTED",
        "count": len(ids),
        "group_count": len(names),
        "baseline_score": baseline_mean,
        "candidate_score": candidate_mean,
        "delta": candidate_mean - baseline_mean,
        "paired_cluster_bootstrap_95ci": [deltas[49], deltas[1949]],
        "seed": seed,
        "bootstrap_samples": 2000,
        "missing_baseline": missing_baseline,
        "missing_candidate": missing_candidate,
        "uncertainty_warning": "Few independent groups; exploratory interval"
        if len(names) < 20
        else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare a caller-provided local benchmark without network or model calls."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--dataset", required=True, choices=["longmemeval", "locomo"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    if args.input.stat().st_size > 384 * 1024 * 1024:
        raise ValueError("input exceeds the 384 MiB preparation profile")
    raw_bytes = args.input.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, list):
        raise ValueError("dataset must be a list")
    prepare = prepare_longmemeval if args.dataset == "longmemeval" else prepare_locomo
    campaign = prepare(raw, dataset_revision=args.revision, seed=args.seed)
    campaign.manifest["source_file_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    campaign.manifest.pop("manifest_sha256")
    campaign.manifest["manifest_sha256"] = _hash(campaign.manifest)
    # Output is a fresh private directory. A partial write has no COMPLETE marker
    # and must never be consumed as a frozen campaign.
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, content in (
        ("ingestion.json", campaign.ingestion),
        ("questions.json", campaign.questions),
        ("evaluation-only.json", campaign.evaluation),
        ("manifest.json", campaign.manifest),
    ):
        target = args.output / name
        with target.open("x", encoding="utf-8") as stream:
            target.chmod(0o600)
            json.dump(content, stream, ensure_ascii=False, indent=2)
    (args.output / "COMPLETE").write_text(campaign.manifest["manifest_sha256"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
