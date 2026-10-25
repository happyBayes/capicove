"""Score text-to-speech-audio retrieval on a frozen, locally synthesized spoken corpus.

Clips are synthesized at run time with FFmpeg's built-in ``flite`` voice from the
frozen sentences, together with tone and noise distractors; nothing is downloaded.
The corpus is synthetic speech: results measure real cross-modal inference on these
inputs, not representative recording quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _host(root: Path, media: Path, model_config: Path, ffmpeg: str, ffprobe: str) -> Any:
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config

    body = {
        "schema_version": 2,
        "server_name": "capicove-eval",
        "workspace_id": "evaluation",
        "actor_id": "evaluator",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.search.multimodal"],
        "trace_store": None,
        "capabilities": {
            "multimodal": {
                "schema_version": 1,
                "profile": "audio-local-v1",
                "source_roots": {"media": str(media)},
                "index_root": str(root / "index"),
                "staging_root": str(root / "staging"),
                "corpus_ref": "corpus:speech-eval",
                "model_config": str(model_config),
                "model_binding": "unified",
                "ffmpeg": ffmpeg,
                "ffprobe": ffprobe,
                "max_source_bytes": 64 * 1024 * 1024,
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    config = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(config, WorkspaceHostConfiguration)
    return config


def _synthesize(ffmpeg: str, media: Path, clips: list[dict[str, str]]) -> dict[str, str]:
    names = {}
    for clip in clips:
        text = media.parent / f"{clip['id']}.txt"
        text.write_text(clip["text"], encoding="utf-8")
        target = media / f"{clip['id']}.wav"
        subprocess.run(  # nosec B603 - fixed argument list
            [
                *(ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", f"flite=textfile={text}"),
                *("-ar", "16000", "-ac", "1", str(target)),
            ],
            check=True,
            timeout=120,
        )
        names[target.name] = clip["id"]
    distractors = {
        "tone-440": "sine=frequency=440:duration=4",
        "tone-880": "sine=frequency=880:duration=4",
        "noise-white": "anoisesrc=color=white:duration=4:amplitude=0.3",
        "noise-pink": "anoisesrc=color=pink:duration=4:amplitude=0.3",
    }
    for name, source in distractors.items():
        target = media / f"{name}.wav"
        subprocess.run(  # nosec B603 - fixed argument list
            [
                *(ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", source),
                *("-ar", "16000", "-ac", "1", str(target)),
            ],
            check=True,
            timeout=60,
        )
        names[target.name] = name
    return names


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Synthesize, index, search every frozen query and score by clip identity."""
    from capicove.contracts import InvocationContext
    from capicove.contracts.retrieval import (
        DetailLevel,
        FullHit,
        SearchRequest,
        TargetType,
        TextQuery,
    )
    from capicove.runtime import unified_retrieval as runtime

    frozen = json.loads(args.labels.read_text())
    if frozen.get("schema") != "capicove.audio-speech-eval.v1":
        raise ValueError("unsupported label file schema")
    with tempfile.TemporaryDirectory(prefix="capicove-speech-eval-") as scratch:
        root = Path(scratch).resolve() / "workspace"
        media = root / "media"
        media.mkdir(parents=True)
        os.chmod(root, 0o700)
        names = _synthesize(args.ffmpeg, media, frozen["clips"])
        host = _host(root, media, args.model_config, args.ffmpeg, args.ffprobe)
        started = time.perf_counter()
        for name in sorted(names):
            runtime.import_media(host, "media", name)
        built = runtime.build_generation(host)
        runtime.activate_generation(host, built["generation_id"])
        build_seconds = time.perf_counter() - started
        context = InvocationContext(
            request_id="eval",
            trace_id="eval",
            workspace_id="evaluation",
            actor_id="evaluator",
            environment_id="local",
            deadline=datetime.now(UTC) + timedelta(hours=1),
            permissions=frozenset({"evidence:read"}),
        )
        service = runtime.open_unified_retrieval(host)
        rows = []
        latencies = []
        try:
            for item in frozen["queries"]:
                request = SearchRequest(
                    TextQuery(item["query"]),
                    frozenset({TargetType.AUDIO_SEGMENT}),
                    top_k=5,
                    detail_level=DetailLevel.FULL,
                    retrieval_profile_ref="audio-local-v1",
                )
                began = time.perf_counter()
                result = service.search(request)
                latencies.append(time.perf_counter() - began)
                ranked = []
                for hit in result.hits:
                    source = service.read_evidence(context, hit.evidence_id, ("metadata",))[
                        "metadata"
                    ]["source_ref"]
                    score = hit.lane_scores[0].raw_score if isinstance(hit, FullHit) else None
                    ranked.append((names[source.split(":", 1)[1]], score))
                ranks = [
                    rank
                    for rank, (clip, _) in enumerate(ranked, start=1)
                    if clip in item["relevant"]
                ]
                rows.append(
                    {
                        "id": item["id"],
                        "answerable": bool(item["relevant"]),
                        "first_relevant_rank": ranks[0] if ranks else None,
                        "top": [clip for clip, _ in ranked],
                        "top_score": ranked[0][1] if ranked else None,
                    }
                )
        finally:
            service.close()
    answerable = [row for row in rows if row["answerable"]]
    return {
        "schema": "capicove.audio-speech-eval-results.v1",
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256(args.model_config.read_bytes()).hexdigest(),
        "completed_at": datetime.now(UTC).isoformat(),
        "units": built["units"],
        "clips": len(names),
        "build_seconds": build_seconds,
        "query_p50_seconds": statistics.median(latencies),
        "quality": {
            "recall_at_1": sum(row["first_relevant_rank"] == 1 for row in answerable)
            / len(answerable),
            "recall_at_5": sum(row["first_relevant_rank"] is not None for row in answerable)
            / len(answerable),
            "mrr": statistics.fmean(
                1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0.0
                for row in answerable
            ),
            "random_recall_at_1": 1 / len(names),
            "mean_top_score_answerable": statistics.fmean(
                row["top_score"] for row in answerable if row["top_score"] is not None
            ),
            "mean_top_score_no_answer": statistics.fmean(
                row["top_score"]
                for row in rows
                if not row["answerable"] and row["top_score"] is not None
            ),
        },
        "rows": rows,
    }


def main() -> int:
    """Run the evaluation and write the results file exclusively."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = evaluate(args)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(results["quality"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
