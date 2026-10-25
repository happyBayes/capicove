"""Opt-in synthetic SQLite capacity qualification; no model or network access.

Seed canonical repository events in bounded batches, then measure public service
operations. This measures engineering behavior, never model answer accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from capicove.contracts.knowledge import Predicate, digest, micros
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.source import SQLiteTextEvidenceStore


def qualify(root: Path, count: int, *, reuse: bool = False) -> dict[str, Any]:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(root.iterdir()) and not reuse:
        raise ValueError("qualification directory must be empty")
    source = SQLiteTextEvidenceStore(root / "evidence.sqlite3", root / "privacy")
    repo = SQLiteMemoryRepository(root / "memory.sqlite3")
    definitions = (
        Predicate("parent", ("person",), "entity", ("person",), "many", "static"),
        *(
            Predicate("value" + str(i), ("person",), "literal", ("int64",), "one", "static")
            for i in range(1, 5)
        ),
    )
    service = TemporalKnowledgeService(
        repo, source, ontology_version="fixture-v1", predicates=definitions
    )
    source_root = Path(__file__).resolve().parents[2]
    source_manifest = {
        str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for package in ("knowledge", "source", "agent", "contracts")
        for path in sorted((source_root / "src" / "capicove" / package).rglob("*.py"))
    }
    run_id = uuid4().hex
    started = time.perf_counter()
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    for batch in range(count if reuse else 0, count, 500):
        revision, recorded = repo.head("benchmark")
        events: list[tuple[str, dict[str, Any]]] = []
        indexes: list[tuple[str, str, str]] = []
        for number in range(batch, min(count, batch + 500)):
            entity, episode = f"entity-{number}", f"episode-{number}"
            statements = [f"Synthetic entity {number} parent entity {max(0, number - 1)}."] + [
                f"Synthetic entity {number} value{i} is {number + i}." for i in range(1, 5)
            ]
            content = "\n".join(statements + statements)
            evidence = source.accept_text(workspace="benchmark", scope="fixture", text=content)
            ep = {
                "id": episode,
                "workspace": "benchmark",
                "scope": "fixture",
                "episode_type": "document",
                "evidence_refs": [evidence.content_sha256],
                "content_sha256": evidence.content_sha256,
                "event_time": micros(epoch + timedelta(microseconds=number)),
                "recorded": micros(recorded),
                "source_ref": "synthetic-capacity-fixture",
                "schema_version": 1,
                "content_length": len(content),
            }
            events += [
                (
                    "EntityResolved",
                    {
                        "id": entity,
                        "name": f"Synthetic entity {number}",
                        "entity_type": "person",
                        "ontology_version": "fixture-v1",
                    },
                ),
                ("EpisodeAccepted", ep),
                ("EpisodeIndexed", {"id": episode}),
            ]
            indexes.append(("episode", episode, content))
            offset = 0
            repeated_offset = sum(len(line) + 1 for line in statements)
            for i, definition in enumerate(definitions):
                obj = (
                    {"kind": "entity", "value": f"entity-{max(0, number - 1)}"}
                    if i == 0
                    else {"kind": "int64", "value": number + i}
                )
                support = [
                    {
                        "digest": evidence.content_sha256,
                        "start": offset,
                        "end": offset + len(statements[i]),
                    },
                    {
                        "digest": evidence.content_sha256,
                        "start": offset + repeated_offset,
                        "end": offset + repeated_offset + len(statements[i]),
                    },
                ]
                offset += len(statements[i]) + 1
                fact = f"fact-{number}-{i}"
                body = {
                    "id": fact,
                    "version": fact + "-v1",
                    "subject": entity,
                    "predicate": definition.name,
                    "object": obj,
                    "asserted_validity": [None, None],
                    "validity": [None, None],
                    "recorded": micros(recorded),
                    "assertion_hash": digest(
                        [entity, definition.name, obj, [None, None], "fixture-v1"]
                    ),
                    "conflict_key": digest([entity, definition.name, "fixture-v1"]),
                    "ontology_version": "fixture-v1",
                    "authority": 5,
                    "extraction_confidence": None,
                    "evidence": support,
                    "state": "accepted",
                    "parent_version_id": None,
                }
                events += [
                    ("FactProposed", {"id": fact, "assertion_hash": body["assertion_hash"]}),
                    ("FactAccepted", body),
                ]
                indexes.append(("fact", fact, statements[i]))
        repo.commit(
            workspace="benchmark",
            scope="fixture",
            key=f"seed-{batch}",
            request={"batch": batch},
            expected_revision=revision,
            recorded=recorded,
            events=events,
            indexes=indexes,
            receipt={"seeded": min(count, batch + 500)},
        )
        if batch % 10000 == 0:
            print(f"seeded {min(count, batch + 500)} episodes", flush=True)
    seed_seconds = time.perf_counter() - started
    measurements: dict[str, list[float]] = {
        "episode_commit": [],
        "fact_lookup": [],
        "lexical": [],
        "hybrid": [],
    }

    def timed(name: str, call: Any) -> None:
        start = time.perf_counter()
        call()
        measurements[name].append((time.perf_counter() - start) * 1000)

    def writer() -> None:
        for i in range(100):
            timed(
                "episode_commit",
                lambda i=i: service.remember_episode(
                    workspace="benchmark",
                    scope="probe",
                    content=f"writer probe {run_id}-{i}" + " synthetic" * 50,
                    event_time=epoch,
                    idempotency_key=f"probe-{run_id}-{i}",
                ),
            )

    def reader(i: int) -> None:
        n = (i * 7919) % count
        timed(
            "fact_lookup",
            lambda: service.current_facts(
                workspace="benchmark", scope="fixture", subject=f"entity-{n}", predicate="value1"
            ),
        )
        q = MemoryQuery("benchmark", "fixture", text=f"entity {n}", top_k=20)
        timed("lexical", lambda: repo.candidates(q, lane="lexical"))
        timed(
            "hybrid",
            lambda: service.recall(
                MemoryQuery(
                    "benchmark",
                    "fixture",
                    text=f"entity {n}",
                    center_entities=(f"entity-{n}",),
                    top_k=20,
                )
            ),
        )

    with ThreadPoolExecutor(max_workers=10) as warmup_pool:
        list(warmup_pool.map(reader, range(50)))
        cold_measurements = {name: list(values) for name, values in measurements.items()}
        for values in measurements.values():
            values.clear()
    with ThreadPoolExecutor(max_workers=11) as pool:
        writing = pool.submit(writer)
        list(pool.map(reader, range(50)))
        writing.result()
    with sqlite3.connect(root / "memory.sqlite3") as db:
        sizes = {
            table: db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            for table in ("episodes", "fact_versions", "fact_evidence", "memory_events")
        }
    repo.close()
    start = time.perf_counter()
    recovered = SQLiteMemoryRepository(root / "memory.sqlite3")
    recovery = time.perf_counter() - start
    recovered.close()
    targets = {"episode_commit": 75, "fact_lookup": 100, "lexical": 150, "hybrid": 250}
    latencies = {
        name: {
            "p50_ms": statistics.median(values),
            "p95_ms": sorted(values)[int(len(values) * 0.95) - 1],
            "samples": len(values),
            "target_ms": targets[name],
        }
        for name, values in measurements.items()
    }
    required_scale = {"episodes": 100000, "fact_versions": 500000, "fact_evidence": 1000000}
    scale_met = all(sizes[name] >= minimum for name, minimum in required_scale.items())
    result = {
        "dataset": "authored synthetic; not a semantic accuracy benchmark",
        "qualification": "required_scale" if scale_met else "smoke_only",
        "required_scale": required_scale,
        "source_sha256": source_manifest,
        "reused_fixture": reuse,
        "hardware": platform.platform(),
        "machine": platform.machine(),
        "sqlite": sqlite3.sqlite_version,
        "one_writer": True,
        "concurrent_readers": 10,
        "state": "50 explicit read warmups; restart verifies complete ledger",
        "cold_read_p95_ms": {
            name: sorted(values)[int(len(values) * 0.95) - 1]
            for name, values in cold_measurements.items()
            if values
        },
        "counts": sizes,
        "seed_seconds": seed_seconds,
        "latencies": latencies,
        "restart_seconds": recovery,
        "restart_target_seconds": 5,
        "all_initial_targets_met": scale_met
        and all(v["p95_ms"] <= v["target_ms"] for v in latencies.values())
        and recovery <= 5,
    }
    (root / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--reuse", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.episodes <= 100000:
        parser.error("episodes must be between 1 and 100000")
    qualify(arguments.output, arguments.episodes, reuse=arguments.reuse)
