"""Evaluate local code search stages against frozen, symbol-labelled queries (RFC-0033).

Labels name a relative path and optionally a symbol, never a generated chunk ID, so
chunker changes cannot move the ground truth. The corpus is an operator-selected Git
repository at a recorded commit; nothing is uploaded or downloaded. Results are
evaluation evidence for the measured snapshot only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess  # nosec B404
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _host(workspace: Path, repo: Path, policy: str, model_config: Path | None) -> Any:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config

    hybrid = policy in {"code-hybrid-v1", "code-relations-v1"}
    profile = "local-code-hybrid-v1" if hybrid else "local-code-lexical-v1"

    body = {
        "schema_version": 2,
        "server_name": "capicove-eval",
        "workspace_id": "evaluation",
        "actor_id": "evaluator",
        "environment_id": "local",
        "workspace_root": str(workspace),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.get_evidence", "capicove.search"],
        "trace_store": None,
        "capabilities": {
            "retrieval": {
                "schema_version": 2,
                "source_roots": {"repo": str(repo)},
                "index_root": str(workspace / "index"),
                "corpus_ref": "corpus:evaluation",
                "retrieval_profile_ref": profile,
                "max_source_bytes": 1024 * 1024,
                "model_config_path": None if model_config is None else str(model_config),
                "model_config_sha256": None
                if model_config is None
                else hashlib.sha256(model_config.read_bytes()).hexdigest(),
                "model_binding": None if model_config is None else "code",
                "max_units": 10_000,
                "max_vector_bytes": 0 if model_config is None else 64 * 1024 * 1024,
                "code_policy_ref": policy,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    config = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(config, WorkspaceHostConfiguration)
    return config


def _relevant(metadata: dict[str, Any], label: dict[str, Any]) -> bool:
    if metadata["relative_path"] != label["path"]:
        return False
    symbol = label.get("symbol")
    if symbol is None:
        return True
    return any(item["name"] == symbol for item in metadata.get("definitions", ())) or (
        symbol in metadata["text"]
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run import/build once, then every frozen query cold and warm."""
    from capicove.contracts import InvocationContext
    from capicove.contracts.retrieval import SearchRequest, TargetType, TextQuery
    from capicove.runtime import code_retrieval

    queries = json.loads(args.queries.read_text())
    if queries.get("schema") != "capicove.code-search-eval.v1":
        raise ValueError("unsupported query file schema")
    commit = (
        subprocess.run(  # nosec B603
            [args.git, "-C", str(args.repo), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
        )
        .stdout.decode()
        .strip()
    )
    with tempfile.TemporaryDirectory(prefix="capicove-code-eval-") as scratch:
        if args.workspace is not None:
            # A persistent workspace keeps the vector cache, so a later policy on
            # the same snapshot reuses embeddings of unchanged chunks.
            workspace = args.workspace.resolve()
            workspace.mkdir(mode=0o700, parents=True, exist_ok=True)
        else:
            workspace = Path(scratch).resolve() / "workspace"
            workspace.mkdir(mode=0o700)
        host = _host(workspace, args.repo.resolve(), args.policy, args.model_config)
        started = time.perf_counter()
        report = code_retrieval.import_repository(host, "repo", git=args.git)
        imported = time.perf_counter()
        built = code_retrieval.build_index(host)
        finished = time.perf_counter()
        index_bytes = _directory_bytes(workspace / "index" / "generations")
        context = InvocationContext(
            request_id="eval",
            trace_id="eval",
            workspace_id="evaluation",
            actor_id="evaluator",
            environment_id="local",
            deadline=datetime.now(UTC) + timedelta(hours=1),
            permissions=frozenset({"evidence:read"}),
        )
        results = []
        cold: list[float] = []
        warm: list[float] = []
        opened = time.perf_counter()
        service = code_retrieval.open_code_retrieval(host)
        # The first search also loads a hybrid profile's resident encoder.
        service.search(
            SearchRequest(TextQuery("warm up"), frozenset({TargetType.TEXT_CHUNK}), top_k=1)
        )
        encoder_load = time.perf_counter() - opened
        for item in queries["queries"]:
            request = SearchRequest(
                TextQuery(item["query"]),
                frozenset({TargetType.TEXT_CHUNK}),
                top_k=10,
                retrieval_profile_ref="local-default",
            )
            first = time.perf_counter()
            result = service.search(request)
            cold.append(time.perf_counter() - first)
            for _ in range(args.repeats):
                again = time.perf_counter()
                service.search(request)
                warm.append(time.perf_counter() - again)
            ranked = []
            context_bytes = sum(len((hit.snippet or "").encode()) for hit in result.hits)
            for hit in result.hits:
                metadata = dict(
                    service.read_evidence(context, hit.evidence_id, ("metadata",))["metadata"]
                )
                ranked.append(metadata)
            labels = item.get("relevant", [])
            context_labels = item.get("context", [])
            context_found = sum(
                any(_relevant(metadata, label) for metadata in ranked[:10])
                for label in context_labels
            )
            off_target = sum(
                not any(_relevant(metadata, label) for label in (*labels, *context_labels))
                for metadata in ranked[:10]
            )
            relevant_ranks = [
                rank
                for rank, metadata in enumerate(ranked, start=1)
                if any(_relevant(metadata, label) for label in labels)
            ]
            top = ranked[0] if ranked else None
            results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "status": result.status.value,
                    "hits": len(ranked),
                    "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
                    "definition_top1": bool(
                        top is not None
                        and item.get("definition") is not None
                        and any(
                            d["name"] == item["definition"]
                            and top["relative_path"] == labels[0]["path"]
                            for d in top.get("definitions", ())
                        )
                    ),
                    "top_paths": [metadata["relative_path"] for metadata in ranked[:3]],
                    "top10": [
                        [metadata["relative_path"], metadata.get("start_line")]
                        for metadata in ranked[:10]
                    ],
                    "context_bytes": context_bytes,
                    "context_labels": len(context_labels),
                    "context_found": context_found,
                    "off_target_hits": off_target,
                }
            )
        service.close()
    answerable = [row for row in results if row["category"] != "no_answer"]
    definition_rows = [
        row for row, item in zip(results, queries["queries"], strict=True) if item.get("definition")
    ]
    by_category: dict[str, dict[str, float]] = {}
    for category in sorted({row["category"] for row in answerable}):
        rows = [row for row in answerable if row["category"] == category]
        by_category[category] = {
            "queries": len(rows),
            "recall_at_10": sum(row["first_relevant_rank"] is not None for row in rows) / len(rows),
            "mrr": statistics.fmean(
                1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0.0
                for row in rows
            ),
        }

    def percentile(values: list[float], fraction: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]

    return {
        "schema": "capicove.code-search-eval-results.v1",
        "stage": {
            "code-symbols-v1": "S1",
            "code-ast-v1": "S2",
            "code-hybrid-v1": "S3",
            "code-relations-v1": "S4",
        }[args.policy],
        "relations": built.get("relations"),
        "policy": args.policy,
        "parse_outcomes": built.get("parse_outcomes"),
        "completed_at": datetime.now(UTC).isoformat(),
        "corpus": {
            "repository_commit": commit,
            "selected_files": report["selected"],
            "excluded": report["excluded"],
        },
        "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "generation_id": built["generation_id"],
        "units": built["units"],
        "build": {
            "import_seconds": imported - started,
            "build_seconds": finished - imported,
            "index_bytes": index_bytes,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "latency": {
            "cold_p50_seconds": statistics.median(cold),
            "cold_p95_seconds": percentile(cold, 0.95),
            "warm_p50_seconds": statistics.median(warm),
            "warm_p95_seconds": percentile(warm, 0.95),
            "warm_samples": len(warm),
            "cold_includes": "first search per query: generation digest verification and search",
            "service_open_and_first_search_seconds": encoder_load,
        },
        "quality": {
            "definition_top1": sum(row["definition_top1"] for row in definition_rows)
            / len(definition_rows)
            if definition_rows
            else None,
            "recall_at_10": sum(row["first_relevant_rank"] is not None for row in answerable)
            / len(answerable),
            "mrr": statistics.fmean(
                1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0.0
                for row in answerable
            ),
            "by_category": by_category,
            "no_answer_false_positive_queries": sum(
                row["hits"] > 0 for row in results if row["category"] == "no_answer"
            ),
            "no_answer_queries": sum(row["category"] == "no_answer" for row in results),
            "mean_context_bytes": statistics.fmean(row["context_bytes"] for row in results),
            "context_coverage": sum(row["context_found"] for row in results)
            / max(1, sum(row["context_labels"] for row in results)),
            "context_labelled_queries": sum(row["context_labels"] > 0 for row in results),
            "mean_off_target_hits": statistics.fmean(row["off_target_hits"] for row in answerable),
            "natural_language_recall_at_10": (
                sum(
                    row["first_relevant_rank"] is not None
                    for row in answerable
                    if row["category"] in {"english", "chinese"}
                )
                / max(1, sum(row["category"] in {"english", "chinese"} for row in answerable))
            ),
        },
        "host": {
            "platform": sys.platform,
            "cpus": os.cpu_count(),
            "python": sys.version.split()[0],
        },
        "rows": results,
    }


def main() -> int:
    """Run one evaluation and write the results file exclusively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--git", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--policy",
        choices=("code-symbols-v1", "code-ast-v1", "code-hybrid-v1", "code-relations-v1"),
        default="code-symbols-v1",
    )
    parser.add_argument(
        "--model-config", type=Path, help="Required for code-hybrid-v1 and code-relations-v1."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workspace", type=Path, help="Optional persistent private workspace (keeps vectors)."
    )
    args = parser.parse_args()
    results = evaluate(args)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: results[k] for k in ("units", "build", "latency", "quality")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
