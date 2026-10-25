"""Offline, authored retrieval qualification through public Tools across processes.

Expected answers below are fixed independently of the implementation. No labels
are passed to indexing/extraction. This is not a model/LoCoMo/LongMemEval score.
Run: python scripts/benchmarks/temporal_memory_long_horizon.py --output REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from capicove.contracts import InvocationContext, ToolStatus
from capicove.runtime.memory_config import DEFAULT_PREDICATES
from capicove.runtime.memory_runtime import build_memory_runtime
from capicove.tools.memory_tools import encode_memory_tool_result


def run_phase(root: Path, phase: str, output: Path) -> None:
    handlers = build_memory_runtime(
        root / "store",
        workspace="evaluation",
        scopes=frozenset({"journal", "other"}),
        ontology_version="personal-v1",
        predicates=DEFAULT_PREDICATES,
    )

    def call(operation: str, **request: Any) -> dict[str, Any]:
        context = InvocationContext(
            uuid4().hex,
            uuid4().hex,
            "evaluation",
            "offline-evaluator",
            "local",
            datetime.now(UTC) + timedelta(seconds=30),
            frozenset({"memory:read", "memory:write", "memory:erase"}),
        )
        result = handlers[operation](context, dict(scope="journal", **request))
        if result.status is not ToolStatus.SUCCEEDED:
            raise RuntimeError(encode_memory_tool_result(result)["error"])
        return encode_memory_tool_result(result)["result"]

    def fact(name: str, city: str, start: str | None, key: str, **extra: Any) -> dict[str, Any]:
        return call(
            "assert_fact",
            subject_name=name,
            subject_type="person",
            predicate="city",
            object={"kind": "string", "value": city},
            valid_from=start,
            event_time=start or "2018-01-01T00:00:00Z",
            evidence_text=f"{name} reported moving to {city}.",
            idempotency_key=key,
            pinned=True,
            **extra,
        )

    if phase == "ingest":
        alice = fact("Alice", "Paris", None, "alice-original")
        old_known = call("get_fact_history", fact_id=alice["fact_id"])["facts"][0]["knowledge"][
            "start"
        ]
        fact("Alice", "Rome", "2024-01-01T00:00:00Z", "alice-current")
        fact("Alice", "Nairobi", "2035-01-01T00:00:00Z", "alice-future")
        fact("Alice", "Oslo", "2020-01-01T00:00:00Z", "alice-backfill")
        boris = fact("Boris", "Berlin", "2019-01-01T00:00:00Z", "boris-old")
        boris_known = call("get_fact_history", fact_id=boris["fact_id"])["facts"][0]["knowledge"][
            "start"
        ]
        fact(
            "Boris", "Lyon", "2019-01-01T00:00:00Z", "boris-corrected", supersedes=boris["fact_id"]
        )
        for name, city in (
            ("Camila", "Lima"),
            ("Diego", "Quito"),
            ("Esther", "Seoul"),
            ("Farah", "Cairo"),
        ):
            fact(name, city, "2021-01-01T00:00:00Z", name)
        # Persist meaningful irrelevant sessions after the desired facts. A recent-only
        # implementation cannot answer the historical semantic questions below.
        for i in range(240):
            day = datetime(2010, 1, 1, tzinfo=UTC) + timedelta(days=i * 20)
            call(
                "remember_episode",
                content=f"Session {i}: the workshop inventory contains {i + 7} ceramic tiles and a blue notebook.",
                event_time=day.isoformat(),
                idempotency_key=f"session-{i}",
                pinned=True,
            )
        call(
            "remember_episode",
            content="小王确认下次课程安排在周四下午。",
            event_time="2025-01-01T00:00:00Z",
            idempotency_key="cjk",
            pinned=True,
        )
        (root / "observed-times.json").write_text(
            json.dumps({"old_known": old_known, "boris_known": boris_known})
        )
        output.write_text(
            json.dumps(
                {
                    "episodes": 251,
                    "distractor_sessions": 240,
                    "world_time_span": "2010-2035",
                    "status": "ingested",
                }
            )
        )
        return

    clocks = json.loads((root / "observed-times.json").read_text())
    cases = [
        (
            "current_after_backfill",
            "Alice city",
            "2026-01-01T00:00:00Z",
            None,
            "Alice reported moving to Rome.",
        ),
        (
            "negative_infinity",
            "Alice city",
            "2019-01-01T00:00:00Z",
            None,
            "Alice reported moving to Paris.",
        ),
        (
            "backfilled_world",
            "Alice city",
            "2021-01-01T00:00:00Z",
            None,
            "Alice reported moving to Oslo.",
        ),
        (
            "adjacent_boundary",
            "Alice city",
            "2024-01-01T00:00:00Z",
            None,
            "Alice reported moving to Rome.",
        ),
        (
            "future_explicit",
            "Alice city",
            "2036-01-01T00:00:00Z",
            None,
            "Alice reported moving to Nairobi.",
        ),
        (
            "former_belief",
            "Alice city",
            "2026-01-01T00:00:00Z",
            clocks["old_known"],
            "Alice reported moving to Paris.",
        ),
        (
            "correction",
            "Boris city",
            "2026-01-01T00:00:00Z",
            None,
            "Boris reported moving to Lyon.",
        ),
        (
            "before_correction",
            "Boris city",
            "2026-01-01T00:00:00Z",
            clocks["boris_known"],
            "Boris reported moving to Berlin.",
        ),
        (
            "other_entity",
            "Camila city",
            "2026-01-01T00:00:00Z",
            None,
            "Camila reported moving to Lima.",
        ),
    ]
    observations = []
    for name, query, valid, known, expected in cases:
        result = call(
            "recall",
            text=query,
            valid_at=valid,
            known_at=known,
            types=["semantic"],
            top_k=5,
            max_tokens=12000,
        )
        hits = result["hits"]
        passed = len(hits) == 1 and hits[0]["text"] == expected and bool(hits[0]["citations"])
        if passed:
            span = hits[0]["citations"][0]
            passed = (
                span["reference"]["content_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
                and span["start"] == 0
                and span["end"] == len(expected)
            )
        observations.append(
            {"case": name, "passed": passed, "returned_texts": [h["text"] for h in hits]}
        )
    unknown = call("recall", text="unmentioned observatory penguin")
    observations.append({"case": "unknown_abstention", "passed": unknown["hits"] == []})
    cjk = call("recall", text="周四", types=["episodic"])
    observations.append(
        {
            "case": "cjk_evidence",
            "passed": bool(cjk["hits"])
            and cjk["hits"][0]["text"] == "小王确认下次课程安排在周四下午。",
        }
    )
    bounded = call("recall", text="Alice city", max_characters=20, max_tokens=20)
    observations.append(
        {"case": "bounded_context", "passed": bounded["hits"] == [] and bounded["characters"] <= 20}
    )
    call("erase_scope", idempotency_key="erase-evaluation")
    context = InvocationContext(
        "erased",
        "erased",
        "evaluation",
        "offline-evaluator",
        "local",
        datetime.now(UTC) + timedelta(seconds=30),
        frozenset({"memory:read"}),
    )
    erased = handlers["recall"](context, {"scope": "journal", "text": "Alice"})
    observations.append(
        {
            "case": "erasure_denial",
            "passed": erased.status is ToolStatus.FAILED and erased.result is None,
        }
    )
    report = {
        "qualification": "authored_offline_retrieval_not_model_answer_accuracy",
        "process_restart": True,
        "cases": observations,
        "passed": sum(c["passed"] for c in observations),
        "total": len(observations),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "World time is simulated; this is not a months-long live deployment.",
            "No semantic model, embedding model, LoCoMo or LongMemEval dataset was run.",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--phase", choices=("ingest", "score"))
    args = parser.parse_args()
    if args.phase:
        run_phase(args.root, args.phase, args.output)
        return
    with tempfile.TemporaryDirectory(prefix="capicove-memory-evaluation-") as directory:
        for phase in ("ingest", "score"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--phase",
                    phase,
                    "--root",
                    directory,
                    "--output",
                    str(args.output.resolve()),
                ],
                check=True,
                timeout=120,
            )


if __name__ == "__main__":
    main()
